#!/usr/bin/env python
"""Smoke-test a deployed FRAME site (or a local one). Standard library only.

    python deploy/verify_deployment.py https://frame.example.com
    python deploy/verify_deployment.py https://frame.example.com --api https://api.example.com   # API on another origin
    python deploy/verify_deployment.py http://127.0.0.1:8000 --api-only                          # backend only, no frontend

Exit code 0 = no FAIL, 1 = at least one FAIL. WARN means "works, but not as fast/safe as it should be".
It only issues GET requests plus one read-only POST /api/predict (a prediction; nothing is stored).
"""
from __future__ import annotations

import argparse
import gzip
import json
import re
import sys
import urllib.error
import urllib.request

SAMPLE = {"title_year": 2010, "duration": 120, "genres": ["Drama"], "country": "USA", "content_rating": "PG-13"}
SPA_ROUTES = ["/", "/predict", "/performance", "/explain", "/explore", "/about"]
results: list[tuple[str, str, str]] = []


def record(level: str, name: str, detail: str = "") -> None:
    results.append((level, name, detail))
    print(f"{level:5s} {name}" + (f" - {detail}" if detail else ""))


def fetch(url: str, method: str = "GET", body: dict | None = None, headers: dict | None = None, timeout: int = 30):
    """Return (status, headers, raw_bytes). No automatic decompression, so Content-Encoding stays visible."""
    data = json.dumps(body).encode() if body is not None else None
    h = {"Accept-Encoding": "br, gzip", "User-Agent": "frame-verify/1.0", **(headers or {})}
    if data:
        h["Content-Type"] = "application/json"
    req = urllib.request.Request(url, data=data, method=method, headers=h)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.status, {k.lower(): v for k, v in r.headers.items()}, r.read()
    except urllib.error.HTTPError as e:
        return e.code, {k.lower(): v for k, v in e.headers.items()}, e.read()
    except Exception as e:  # DNS, TLS, refused, timeout
        return 0, {}, str(e).encode()


def jbody(raw: bytes):
    try:
        return json.loads(raw)
    except Exception:
        return None


def check_api(api: str, origin: str | None) -> None:
    s, h, raw = fetch(api + "/api/health")
    j = jbody(raw)
    if s == 200 and j and j.get("status") == "ok":
        m = j.get("model", {})
        level = "PASS" if m.get("estimator") == "RandomForestClassifier" else "FAIL"
        record(level, "GET /api/health", f"{m.get('estimator')} classes={m.get('classes')} rows={j.get('dataset', {}).get('rows')}")
    else:
        record("FAIL", "GET /api/health", f"status {s}; body {raw[:120]!r} (503 = model not loaded; check backend logs and FRAME_MODEL_PATH)")
        return

    s, h, raw = fetch(api + "/api/overview")
    enc = h.get("content-encoding", "none")
    size_kb = len(raw) / 1024
    record("PASS" if s == 200 else "FAIL", "GET /api/overview", f"{s}, {size_kb:.0f} KB on the wire, content-encoding={enc}")
    if s == 200 and enc == "none" and size_kb > 20:
        record("WARN", "API responses are not compressed", "this payload is ~170 KB raw / ~39 KB gzipped: enable gzip/br at the proxy (docs/DEPLOYMENT.md)")

    s, h, raw = fetch(api + "/api/predict", "POST", SAMPLE)
    j = jbody(raw)
    if s == 200 and j and j.get("label") in ("Flop", "Average", "Hit") and abs(sum(j["probabilities"].values()) - 1) < 1e-6:
        record("PASS", "POST /api/predict", f"label={j['label']} (class scores sum to 1)")
    else:
        record("FAIL", "POST /api/predict", f"status {s}; {raw[:160]!r}")

    s, h, raw = fetch(api + "/api/predict", "POST", {**SAMPLE, "gross": 1000000})
    record("PASS" if s == 422 else "FAIL", "POST /api/predict rejects a post-release field (gross)", f"status {s} (expected 422)")

    s, h, raw = fetch(api + "/api/does-not-exist")
    ct = h.get("content-type", "")
    record("PASS" if s == 404 and "json" in ct else "FAIL", "unknown /api path returns JSON 404", f"status {s}, content-type {ct} (HTML here means the SPA fallback is swallowing /api)")

    if origin:
        s, h, raw = fetch(api + "/api/predict", "OPTIONS", headers={"Origin": origin, "Access-Control-Request-Method": "POST", "Access-Control-Request-Headers": "content-type"})
        allow = h.get("access-control-allow-origin")
        record("PASS" if allow in (origin, "*") else "FAIL", f"CORS preflight for origin {origin}", f"status {s}, allow-origin={allow} (set FRAME_CORS_ORIGINS on the API)")


def check_site(site: str) -> None:
    for route in SPA_ROUTES:
        s, h, raw = fetch(site + route, headers={"Accept-Encoding": "gzip"})
        ok = s == 200 and "text/html" in h.get("content-type", "")
        record("PASS" if ok else "FAIL", f"GET {route}", f"{s}, {h.get('content-type', '?')}" + ("" if ok else " (deep links need the SPA fallback to index.html)"))
        if route == "/" and s == 200:
            cc = h.get("cache-control", "")
            record("PASS" if re.search(r"no-cache|no-store|max-age=0", cc) else "WARN", "index.html is not cached long-term", f"cache-control={cc or '(none)'}")
            html = (gzip.decompress(raw) if h.get("content-encoding") == "gzip" else raw).decode("utf8", "ignore")
    m = re.search(r'src="(/assets/index-[^"]+\.js)"', html)
    if not m:
        record("WARN", "could not find the entry bundle in index.html", "skipping asset header checks")
        return
    s, h, raw = fetch(site + m.group(1))  # advertises br+gzip; we only read the headers, never decode
    enc, cc = h.get("content-encoding", "none"), h.get("cache-control", "")
    record("PASS" if s == 200 else "FAIL", f"GET {m.group(1)}", f"{s}, {len(raw) / 1024:.0f} KB on the wire")
    record("PASS" if enc in ("br", "gzip", "zstd") else "WARN", "entry bundle is compressed", f"content-encoding={enc} (expected br or gzip; raw it is ~453 KB, gzip ~147 KB, brotli ~126 KB)")
    record("PASS" if "immutable" in cc and re.search(r"max-age=(\d{7,})", cc) else "WARN", "hashed assets are cached for a year", f"cache-control={cc or '(none)'}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("site", help="site origin, e.g. https://frame.example.com (no trailing slash)")
    ap.add_argument("--api", help="API origin if it differs from the site origin (default: same as site)")
    ap.add_argument("--api-only", action="store_true", help="skip the frontend checks")
    a = ap.parse_args()
    site = a.site.rstrip("/")
    api = (a.api or a.site).rstrip("/")
    print(f"Site: {site}\nAPI:  {api}\n")
    check_api(api, origin=site if a.api and a.api.rstrip("/") != site else None)
    if not a.api_only:
        check_site(site)
    fails = sum(1 for r in results if r[0] == "FAIL")
    warns = sum(1 for r in results if r[0] == "WARN")
    print(f"\n{len(results) - fails - warns} passed, {warns} warnings, {fails} failed")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
