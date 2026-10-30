"""Online Research Agent: bounded, read-only, pre-release context from Wikipedia.

Why "as of a cutoff": a film's CURRENT Wikipedia article is retrospective (box office, reviews, ratings). To keep this step honest about
what was knowable before release, every page is read as it was at its last revision BEFORE a cutoff date (1 January of the film's
planned release year, or now if that year has not started). Each source therefore carries the revision id and timestamp it came from
and links to that exact revision, so a reader can check it. A page that did not exist before the cutoff is excluded, never substituted
by today's version.

Safeguards, all enforced in code (the report generator never sees these URLs or retrieved text; movie mode receives only bounded,
untrusted excerpts with their revision provenance):
  * one allow-listed host over https, validated before every request and every redirect target; at most one redirect
  * hard caps on requests, sources, bytes per response, per-request time and total time
  * retrieved HTML is reduced to plain text (tables, citations, scripts dropped), stripped of control and bidi characters, and truncated
  * output is context only: nothing here is imported by, or passed to, the prediction pipeline
No function in this module raises; failures become a status.
"""
from __future__ import annotations

import json
import re
import time
from datetime import datetime, timezone
from html.parser import HTMLParser
from urllib.parse import quote, urljoin

import httpx

from . import config

API = "https://en.wikipedia.org/w/api.php"

# Article titles for the dataset's genre labels. Anything not listed is skipped and reported, never guessed.
GENRE_ARTICLES = {
    "Action": ("Action film",), "Adventure": ("Adventure film",), "Animation": ("Animation",), "Biography": ("Biographical film", "Biopic"),
    "Comedy": ("Comedy film",), "Crime": ("Crime film",), "Documentary": ("Documentary film",), "Drama": ("Drama (film and television)", "Drama film"),
    "Family": ("Family film",), "Fantasy": ("Fantasy film",), "Film-Noir": ("Film noir",), "History": ("Historical film", "History film"),
    "Horror": ("Horror film",), "Musical": ("Musical film",), "Mystery": ("Mystery film",), "Romance": ("Romance film",),
    "Sci-Fi": ("Science fiction film",), "Sport": ("Sports film", "Sport film"), "Thriller": ("Thriller film",), "War": ("War film",),
    "Western": ("Western (genre)", "Western film"),
}
_RETRY_NEXT_TITLE = ("redirect_page_at_cutoff", "no_revision_before_cutoff", "empty_excerpt")
_BENIGN_SKIPS = ("no_reference_article_for_genre",)

USE_NOTE = ("Contextual background only. It is not an input to the Random Forest prediction or report generator. Movie mode may receive "
            "bounded excerpts as untrusted reference data; they are not evidence that the prediction is right or wrong.")

_STATUS_TEXT = {
    "ok": "Retrieved pre-release background from Wikipedia.",
    "partial": "Retrieved some pre-release background; part of the research could not be completed.",
    "no_results": "The research ran but found no usable page versions from before the cutoff.",
    "unavailable": "Online research was unavailable, so no sources are shown.",
    "timeout": "Online research timed out, so no sources are shown.",
    "error": "Online research hit an error, so no sources are shown.",
    "disabled": "Online research is not enabled on this server.",
}


class _Fail(Exception):
    def __init__(self, kind: str, fatal: bool = False) -> None:
        super().__init__(kind)
        self.kind, self.fatal = kind, fatal


# ---------------------------------------------------------------- text hygiene
_CTRL = re.compile(r"[\x00-\x08\x0b-\x1f\x7f-\x9f​-‏‪-‮⁠-⁩﻿]")
_CITE = re.compile(r"\[(?:\d+|citation needed|note \d+)\]", re.I)


def clean_text(s: str, limit: int | None = None) -> str:
    s = _CTRL.sub("", s).replace("�", "")      # U+FFFD: a character the old page revision itself failed to encode
    s = _CITE.sub("", s)
    s = re.sub(r"\s+", " ", s).strip()
    if limit and len(s) > limit:
        s = s[:limit].rsplit(" ", 1)[0].rstrip(" ,;:-") + "…"
    return s


class _Lead(HTMLParser):
    """Plain text of the lead paragraphs only: drops tables (infoboxes), citations, scripts, styles and figures."""
    _SKIP = {"table", "style", "script", "sup", "figure", "math", "noscript"}

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.depth, self.in_p, self.buf, self.paras = 0, False, [], []

    def handle_starttag(self, tag, attrs):
        if tag in self._SKIP:
            self.depth += 1
        elif tag == "p" and self.depth == 0:
            self.in_p, self.buf = True, []

    def handle_endtag(self, tag):
        if tag in self._SKIP and self.depth:
            self.depth -= 1
        elif tag == "p" and self.in_p:
            self.in_p = False
            t = clean_text("".join(self.buf))
            if len(t) >= 30:
                self.paras.append(t)

    def handle_data(self, data):
        if self.in_p and self.depth == 0:
            self.buf.append(data)


def lead_excerpt(html: str, limit: int) -> str:
    p = _Lead()
    p.feed(html)
    out = ""
    for para in p.paras:
        out = f"{out} {para}".strip()
        if len(out) >= limit:
            break
    return clean_text(out, limit)


class _SectionText(HTMLParser):
    """Extract readable paragraphs and list items, omitting tables, citations and interface markup."""
    _SKIP = {"table", "style", "script", "sup", "figure", "math", "noscript", "nav"}
    _SKIP_CLASSES = {"mw-editsection", "references", "reflist", "mw-references-wrap", "refbegin"}

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.skip_stack: list[str] = []
        self.blocks: list[list] = []
        self.fragments: list[str] = []

    def handle_starttag(self, tag, attrs):
        classes = next((value for name, value in attrs if name == "class" and value), "")
        ident = next((value for name, value in attrs if name == "id" and value), "")
        if self.skip_stack:
            self.skip_stack.append(tag)
        elif tag in self._SKIP or self._SKIP_CLASSES & set(classes.split()) or ident.startswith("cite_note"):
            self.skip_stack.append(tag)       # includes the footnote list ("^ Author (date). Title. Retrieved ..."), which is not film evidence
        elif tag in ("p", "li"):
            self.blocks.append([tag, []])

    def handle_endtag(self, tag):
        if self.skip_stack:
            for index in range(len(self.skip_stack) - 1, -1, -1):
                if self.skip_stack[index] == tag:
                    del self.skip_stack[index:]
                    break
        elif tag in ("p", "li"):
            for index in range(len(self.blocks) - 1, -1, -1):
                if self.blocks[index][0] == tag:
                    completed = self.blocks[index:]
                    del self.blocks[index:]
                    for _, parts in completed:
                        text = clean_text("".join(parts))
                        if text and text not in self.fragments:
                            self.fragments.append(text)
                    break

    def handle_data(self, data):
        if not self.skip_stack and self.blocks:
            self.blocks[-1][1].append(data)


def section_excerpt(html: str, limit: int) -> str:
    parser = _SectionText()
    parser.feed(html)
    return clean_text(" ".join(parser.fragments), limit)


_SECTION_KINDS = ("cast", "production", "writing")


def _section_kinds(heading: str) -> tuple[str, ...]:
    words = set(_norm(heading).split())
    kinds = []
    if words & {"cast", "casting", "actors", "actresses"}:
        kinds.append("cast")
    if words & {"production", "development", "filming", "photography"}:
        kinds.append("production")
    if words & {"screenplay", "screenwriting", "writing", "writer", "writers"}:
        kinds.append("writing")
    return tuple(kinds)


def _section_plan(sections: list) -> tuple[dict[str, tuple[str, str]], list[str]]:
    selected: dict[str, tuple[str, str]] = {}
    for section in sections:
        if not isinstance(section, dict):
            continue
        heading, index = section.get("line"), section.get("index")
        if not isinstance(heading, str) or not isinstance(index, (str, int)):
            continue
        index = str(index)
        if not index.isdigit():
            continue
        for kind in _section_kinds(heading):
            if kind not in selected:
                selected[kind] = (clean_text(heading, 100), index)
    return selected, [kind for kind in _SECTION_KINDS if kind not in selected]


# ---------------------------------------------------------------- URL validation and HTTP
def validate_url(url: str) -> httpx.URL:
    """https only, exact allow-listed host, default port, no credentials, Wikipedia's /w/ path. Raises _Fail('url_not_allowed')."""
    try:
        u = httpx.URL(url)
    except Exception as exc:  # noqa: BLE001
        raise _Fail("url_not_allowed") from exc
    if (u.scheme != "https" or (u.host or "").lower() not in config.RESEARCH_HOSTS or u.port not in (None, 443)
            or u.userinfo or not u.path.startswith("/w/")):
        raise _Fail("url_not_allowed")
    return u


def _make_client() -> httpx.Client:
    return httpx.Client(follow_redirects=False, headers={"User-Agent": config.RESEARCH_USER_AGENT, "Accept": "application/json"},
                        timeout=httpx.Timeout(config.RESEARCH_TIMEOUT_S, connect=min(3.0, config.RESEARCH_TIMEOUT_S)))


class _State:
    def __init__(self) -> None:
        self.requests = 0
        self.deadline = time.monotonic() + config.RESEARCH_TOTAL_S


def _get_json(client: httpx.Client, st: _State, params: dict) -> dict:
    url = httpx.URL(API).copy_merge_params({**params, "format": "json", "formatversion": "2"})
    for hop in range(config.RESEARCH_MAX_REDIRECTS + 1):
        if st.requests >= config.RESEARCH_MAX_REQUESTS:
            raise _Fail("request_limit", fatal=True)
        if time.monotonic() >= st.deadline:
            raise _Fail("time_budget", fatal=True)
        validate_url(str(url))
        st.requests += 1
        try:
            with client.stream("GET", url) as r:
                if r.status_code in (301, 302, 303, 307, 308) and r.headers.get("location"):
                    if hop >= config.RESEARCH_MAX_REDIRECTS:
                        raise _Fail("redirect_limit")
                    url = httpx.URL(urljoin(str(url), r.headers["location"]))   # validated at the top of the next loop, before any request
                    continue
                if r.status_code == 429:
                    raise _Fail("rate_limited", fatal=True)
                if r.status_code >= 400:
                    raise _Fail(f"http_{r.status_code}")
                if "json" not in r.headers.get("content-type", "").lower():
                    raise _Fail("malformed_response")
                declared = r.headers.get("content-length")
                if declared and declared.isdigit() and int(declared) > config.RESEARCH_MAX_BYTES:
                    raise _Fail("response_too_large")
                buf = bytearray()
                for chunk in r.iter_bytes():
                    buf += chunk
                    if len(buf) > config.RESEARCH_MAX_BYTES:
                        raise _Fail("response_too_large")
        except httpx.TimeoutException as exc:
            raise _Fail("timeout", fatal=True) from exc
        except httpx.TransportError as exc:
            raise _Fail("connection_failed", fatal=True) from exc
        try:
            data = json.loads(bytes(buf))
        except Exception as exc:  # noqa: BLE001
            raise _Fail("malformed_response") from exc
        if not isinstance(data, dict):
            raise _Fail("malformed_response")
        return data
    raise _Fail("redirect_limit")


# ---------------------------------------------------------------- matching and as-of retrieval
def _norm(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", s.lower()).strip()


def title_matches(candidate: str, query: str, year: int) -> bool:
    """Strict: the page title must BE the queried title, optionally with a '(film)' / '(YYYY film)' qualifier within a year of the plan."""
    m = re.match(r"^(.*?)(?: \((.*)\))?$", candidate)
    base, qual = (m.group(1), m.group(2)) if m else (candidate, None)
    if _norm(base) != _norm(query):
        return False
    if qual is None:
        return True
    q = re.match(r"^(?:(\d{4}) )?(?:[A-Za-z]+ )?film$", qual)
    return bool(q) and (q.group(1) is None or abs(int(q.group(1)) - year) <= 1)


def _iso(dt: datetime) -> str:
    return dt.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _parse_ts(s: str) -> datetime:
    return datetime.fromisoformat(s.replace("Z", "+00:00"))


def _check_revision(parsed: dict, revid: int) -> dict:
    page = parsed.get("parse")
    if not isinstance(page, dict) or page.get("revid") != revid:
        raise _Fail("revision_mismatch")
    return page


def _page_asof(client, st, title: str, cutoff: datetime, retrieved: str, kind: str) -> tuple[dict, list[dict], str | None]:
    """Return bounded content from the last page revision before `cutoff`, plus optional-section issues."""
    data = _get_json(client, st, {"action": "query", "prop": "revisions", "titles": title, "rvlimit": "1", "rvdir": "older",
                                  "rvstart": _iso(cutoff), "rvprop": "ids|timestamp", "redirects": "0"})
    pages = (data.get("query") or {}).get("pages")
    page = pages[0] if isinstance(pages, list) and pages and isinstance(pages[0], dict) else None
    if page is None:
        raise _Fail("malformed_response")
    revs = page.get("revisions")
    if page.get("missing") or not revs:
        raise _Fail("no_revision_before_cutoff")
    rev = revs[0]
    revid, ts = rev.get("revid"), rev.get("timestamp")
    if not isinstance(revid, int) or not isinstance(ts, str):
        raise _Fail("malformed_response")
    try:
        if _parse_ts(ts) >= cutoff:                      # the API's answer is double-checked, never trusted blindly
            raise _Fail("revision_not_before_cutoff")
    except ValueError as exc:
        raise _Fail("malformed_response") from exc

    parsed = _get_json(client, st, {"action": "parse", "oldid": str(revid), "prop": "text", "section": "0",
                                    "disableeditsection": "1", "disablelimitreport": "1"})
    html = _check_revision(parsed, revid).get("text")
    if not isinstance(html, str):
        raise _Fail("malformed_response")
    if "redirectMsg" in html:
        raise _Fail("redirect_page_at_cutoff")
    excerpt = lead_excerpt(html, config.RESEARCH_EXCERPT_CHARS)
    if not excerpt:
        raise _Fail("empty_excerpt")
    shown = clean_text(str(page.get("title") or title), 200)
    url = f"https://en.wikipedia.org/w/index.php?title={quote(shown.replace(' ', '_'), safe='_()')}&oldid={revid}"
    validate_url(url)
    source = {"title": shown, "url": url, "host": "en.wikipedia.org", "kind": kind, "revision_id": revid, "revision_timestamp": ts,
              "retrieved_at": retrieved, "cutoff": _iso(cutoff), "excerpt": excerpt, "temporal_status": "version_before_cutoff", "use": "context_only"}
    source["sections"] = []
    source["unavailable_sections"] = []
    if kind != "film":
        return source, [], None

    issues: list[dict] = []
    fatal: str | None = None
    unavailable: list[dict] = []
    source["sections"].append({"kind": "introduction", "heading": "Introduction", "index": "0", "text": excerpt})
    source["unavailable_sections"] = unavailable
    try:
        section_data = _get_json(client, st, {"action": "parse", "oldid": str(revid), "prop": "sections"})
        sections = _check_revision(section_data, revid).get("sections")
        if not isinstance(sections, list):
            raise _Fail("malformed_response")
    except _Fail as exc:
        unavailable.extend({"kind": section_kind, "reason": exc.kind} for section_kind in _SECTION_KINDS)
        issues.append({"subject": f"{shown}: section discovery", "reason": exc.kind})
        return source, issues, exc.kind if exc.fatal else None

    selected, missing = _section_plan(sections)
    unavailable.extend({"kind": section_kind, "reason": "section_not_found"} for section_kind in missing)
    remaining = config.RESEARCH_SECTION_BUDGET_CHARS
    fetched_sections: dict[str, str | None] = {}
    for section_kind in _SECTION_KINDS:
        planned = selected.get(section_kind)
        if planned is None:
            continue
        heading, section_id = planned
        if section_id in fetched_sections:
            text = fetched_sections[section_id]
            if text:
                source["sections"].append({"kind": section_kind, "heading": heading, "index": section_id, "text": text})
            else:
                unavailable.append({"kind": section_kind, "reason": "no_readable_text"})
            continue
        if remaining <= 0:
            unavailable.append({"kind": section_kind, "reason": "section_text_limit"})
            continue
        try:
            section_data = _get_json(client, st, {"action": "parse", "oldid": str(revid), "prop": "text", "section": section_id,
                                                  "disableeditsection": "1", "disablelimitreport": "1"})
            section = _check_revision(section_data, revid)
            section_html = section.get("text")
            if not isinstance(section_html, str):
                raise _Fail("malformed_response")
            text = section_excerpt(section_html, min(config.RESEARCH_SECTION_CHARS, remaining))
            if not text:
                fetched_sections[section_id] = None
                unavailable.append({"kind": section_kind, "reason": "no_readable_text"})
                continue
            fetched_sections[section_id] = text
            source["sections"].append({"kind": section_kind, "heading": heading, "index": section_id, "text": text})
            remaining -= len(text)
        except _Fail as exc:
            unavailable.append({"kind": section_kind, "reason": exc.kind})
            issues.append({"subject": f"{shown}: {heading}", "reason": exc.kind})
            if exc.fatal:
                fatal = exc.kind
                break
    return source, issues, fatal


def _find_film(client, st, title: str, year: int) -> str | None:
    data = _get_json(client, st, {"action": "query", "list": "search", "srsearch": f"{title.replace(chr(34), ' ')} film", "srlimit": "5", "srnamespace": "0"})
    hits = (data.get("query") or {}).get("search")
    if not isinstance(hits, list):
        raise _Fail("malformed_response")
    for h in hits:
        t = h.get("title") if isinstance(h, dict) else None
        if isinstance(t, str) and title_matches(t, title, year):
            return t
    return None


def cutoff_for(year: int, now: datetime) -> datetime:
    """1 January of the planned release year (UTC), or now if that moment is still in the future."""
    return min(datetime(year, 1, 1, tzinfo=timezone.utc), now)


# ---------------------------------------------------------------- public entry point
def run(title: str | None, genres: list[str], year: int, now: datetime | None = None) -> dict:
    """Never raises. Returns the research block placed at evidence["research"]."""
    now = now or datetime.now(timezone.utc)
    cutoff = cutoff_for(year, now)
    base = {"provider": "wikipedia", "use": USE_NOTE, "retrieved_at": _iso(now), "cutoff": _iso(cutoff),
            "cutoff_rule": "Pages are read as they were before 1 January of the planned release year (or now, if that year has not started). Pages that did not yet exist are excluded.",
            "query": {"title": title, "genres": genres[:config.RESEARCH_MAX_GENRES]}, "max_requests": config.RESEARCH_MAX_REQUESTS}
    def result(status, reason, sources, skipped, requests):
        return {**base, "available": bool(sources), "status": status, "reason": reason, "message": _STATUS_TEXT[status],
                "sources": sources, "skipped": skipped, "requests_made": requests}

    if not config.RESEARCH_ENABLED:
        return result("disabled", "research_not_enabled", [], [], 0)
    st = _State()
    sources: list[dict] = []
    skipped: list[dict] = []
    errors: list[str] = []
    fatal: str | None = None
    try:
        client = _make_client()
        try:
            subjects: list[tuple[str, tuple[str, ...]]] = []
            if title:
                subjects.append(("film", (title,)))
            for g in genres[:config.RESEARCH_MAX_GENRES]:
                if g in GENRE_ARTICLES:
                    subjects.append(("genre", GENRE_ARTICLES[g]))
                else:
                    skipped.append({"subject": g, "reason": "no_reference_article_for_genre"})
            for kind, candidates in subjects:
                label = candidates[0]
                if len(sources) >= config.RESEARCH_MAX_SOURCES:
                    skipped.append({"subject": label, "reason": "source_limit"})
                    continue
                if kind == "genre" and st.requests >= config.RESEARCH_MAX_REQUESTS:
                    skipped.append({"subject": label, "reason": "request_limit"})
                    fatal = "request_limit"
                    break
                last: _Fail | None = None
                source: dict | None = None
                section_issues: list[dict] = []
                page_fatal: str | None = None
                for name in candidates:
                    try:
                        page_title = name
                        if kind == "film":
                            found = _find_film(client, st, name, year)
                            if not found:
                                raise _Fail("no_matching_film_page")
                            page_title = found
                        source, section_issues, page_fatal = _page_asof(client, st, page_title, cutoff, _iso(datetime.now(timezone.utc)), kind)
                        last = None
                        break
                    except _Fail as f:
                        last = f
                        if f.fatal or f.kind not in _RETRY_NEXT_TITLE:
                            break
                if source is not None:
                    sources.append(source)
                    skipped.extend(section_issues)
                    if page_fatal:
                        fatal = page_fatal
                        break
                if last is not None:
                    skipped.append({"subject": label, "reason": last.kind})
                    if last.fatal:
                        fatal = last.kind
                        break
                    if last.kind in ("malformed_response", "response_too_large", "redirect_limit", "url_not_allowed") or last.kind.startswith("http_"):
                        errors.append(last.kind)
        finally:
            client.close()
    except Exception:  # noqa: BLE001 - research must never break a run; details are not exposed
        return result("error", "internal_error", sources, skipped, st.requests)

    incomplete = [x["reason"] for x in skipped if x["reason"] not in _BENIGN_SKIPS]     # anything requested but not retrieved
    if sources:
        status, reason = ("partial", fatal or (errors[0] if errors else incomplete[0])) if (fatal or errors or incomplete) else ("ok", None)
    elif fatal:
        status, reason = ("timeout" if fatal == "timeout" else "unavailable"), fatal
    elif errors:
        status, reason = "error", errors[0]
    else:
        status, reason = "no_results", "no_pre_cutoff_pages"
    return result(status, reason, sources, skipped, st.requests)


def disabled_block(reason: str = "research_failed") -> dict:
    """Used by the orchestrator if the research step itself could not run at all."""
    return {"provider": "wikipedia", "use": USE_NOTE, "available": False, "status": "error", "reason": reason, "message": _STATUS_TEXT["error"],
            "sources": [], "skipped": [], "requests_made": 0, "max_requests": config.RESEARCH_MAX_REQUESTS, "query": {"title": None, "genres": []}}
