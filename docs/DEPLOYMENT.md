# Deploying FRAME (runbook)

No hosting provider has been selected, and the repository contains no host-specific configuration (no Dockerfile, no CI deploy,
no platform manifest). This document gives the architecture the app is built for, the exact commands, example configuration for
common setups, and a script to verify a deployment. Anything marked **verified** was run during Phase 5 from a clean clone of the
repository; anything marked **example** is configuration you should adapt and test on your host.

Nothing here changes the model, the API contract or any saved artifact.

## 1. Architecture

```
 Browser ──HTTPS──▶  Reverse proxy / static host  ──┬─ /api/*  ─▶  uvicorn (FastAPI)  127.0.0.1:8000
                      (TLS, compression, caching)   │                reads model, dataset, artifacts from disk
                                                    └─ everything else ─▶ frontend/dist (static files, SPA fallback)
```

- **Frontend:** a static single-page app (`frontend/dist`). All API calls are relative (`/api/...`), so the recommended setup is
  **one public origin**: the proxy serves the static files and forwards `/api/*` to the API. No CORS configuration is needed then.
- **Backend:** one FastAPI process group (uvicorn) that is read-only apart from `POST /api/predict` and `POST /api/sensitivity`,
  which compute and return a result and store nothing. There is no database, no session state, and no authentication.
- **Alternative (split origins):** host the static site and the API on different origins. Build the frontend with
  `VITE_API_URL=https://api.example.com` and set `FRAME_CORS_ORIGINS=https://site.example.com` on the API (see section 9).

## 2. What the API needs at runtime

The API loads everything from the repository checkout at startup (`lifespan` in `backend/app/main.py`):

| Needed | Path (relative to the repo root) | Notes |
|---|---|---|
| Python package | `backend/` (the saved pipeline references `backend.ml.features`) | The pipeline is unpickled by import path, so the `backend` package must be importable. Start uvicorn from the repo root. |
| Production model | `pre_release_movie_classifier.joblib` (20 MB) | Random Forest. Override with `FRAME_MODEL_PATH`. |
| Feature schema | `backend/artifacts/prerelease_schema.json` | The API checks the model's inputs against it at startup. |
| Saved metrics/explanations | `backend/artifacts/model_metrics.json`, `explain.json` | Served by `/api/model-metrics` and `/api/explain`. |
| Dataset | `dataset/movie_metadata.csv` | Override with `FRAME_DATA_PATH`. |

All of these are committed, and a clean `git clone` contains them (**verified**). The rollback model and the original post-release
model are not needed to serve and must not be pointed at: the API refuses a model that does not match the schema and reports
`503 degraded` (**verified**), so a wrong `FRAME_MODEL_PATH` fails loudly instead of serving wrong predictions.

**Python and packages.** Verified on Python 3.13.1. Install `backend/requirements-api.txt` (pinned to the verified versions).
`scikit-learn` must match the version that serialized the model (1.9.1). Serving does **not** need lightgbm, shap, imbalanced-learn,
xgboost, catboost, nbformat or matplotlib: every endpoint was exercised with those imports blocked, and a fresh virtualenv built
from `requirements-api.txt` alone served all of them (**verified**). `backend/requirements.txt` is the full development set.

**Memory and CPU.** One worker is about 200 MB resident after startup. Each additional uvicorn worker loads its own copy of the
model and dataset, so budget about 250 MB per worker. Two workers is a sensible default for a small host; one is enough for a demo.

## 3. Environment variables

| Variable | Where | Required | Meaning |
|---|---|---|---|
| `FRAME_CORS_ORIGINS` | API, runtime | Only for split origins | Comma-separated browser origins allowed to call the API. Default is the two local dev origins. Set it to the real site origin(s), e.g. `https://frame.example.com`. |
| `FRAME_MODEL_PATH` | API, runtime | No | Default `pre_release_movie_classifier.joblib` in the repo root. Do not point it at `final_movie_classifier.joblib`. |
| `FRAME_DATA_PATH` | API, runtime | No | Default `dataset/movie_metadata.csv`. |
| `FRAME_SCHEMA_PATH` | API, runtime | No | Default `backend/artifacts/prerelease_schema.json`. |
| `VITE_API_URL` | Frontend, **build time** | Only for split origins | API origin baked into the bundle. Public by nature: never put a credential in it. |
| `VITE_API_PROXY` | Dev server and `vite preview` only | No | Where `npm run dev` / `npm run preview` proxy `/api`. Ignored by `vite build`; it has no effect in production. |

Keep API keys out of source, logs, responses, and frontend build variables. `.env` files are git-ignored;
`backend/.env.example` and `frontend/.env.example` are templates only.

## 4. Build and start commands

**Backend** (run from the repo root; Linux shown, on Windows use `.venv\Scripts\...`):

```bash
python3.13 -m venv .venv
.venv/bin/pip install -r backend/requirements-api.txt
FRAME_CORS_ORIGINS=https://frame.example.com \
  .venv/bin/uvicorn backend.app.main:app --host 127.0.0.1 --port 8000 --workers 2 --proxy-headers
```

Bind to `127.0.0.1` when a proxy is on the same host; use `0.0.0.0` only inside a container or private network. With
`--proxy-headers` uvicorn trusts forwarded headers from loopback only; set `--forwarded-allow-ips` if the proxy is elsewhere.

**Frontend:**

```bash
cd frontend
npm ci
npm run build            # tsc -b && vite build  ->  frontend/dist/   (add VITE_API_URL=... only for split origins)
```

Publish the contents of `frontend/dist/` to the static host or the proxy's document root. A clean `npm ci && npm run build` from a
fresh clone succeeded and produced the same entry bundle as the working copy (**verified**).

**Example (not run): systemd unit for the API**

```ini
[Unit]
Description=FRAME API
After=network.target

[Service]
User=frame
WorkingDirectory=/srv/frame
Environment=FRAME_CORS_ORIGINS=https://frame.example.com
ExecStart=/srv/frame/.venv/bin/uvicorn backend.app.main:app --host 127.0.0.1 --port 8000 --workers 2 --proxy-headers
Restart=on-failure
RestartSec=3

[Install]
WantedBy=multi-user.target
```

## 5. Routing between the frontend and the API

| Request | Goes to | Notes |
|---|---|---|
| `/api/*` (for example `/api/health`, `/api/overview`, `/api/predict`) | the API, path unchanged | Do not strip the `/api` prefix: the API's routes include it. |
| any other path (`/`, `/predict`, `/performance`, `/explain`, `/explore`, `/about`, `/assets/*`) | static files, falling back to `index.html` | Deep links such as `/performance` are client-side routes and must return `index.html` with status 200. |

`/api/health` and `/api/overview` were checked end to end through a proxy in front of a clean-clone API: both returned 200 with the
expected bodies (`RandomForestClassifier`, 4,919 rows). An unknown `/api/...` path returns a JSON 404 (not the HTML page), which
proves the SPA fallback is not swallowing API routes (**verified**, and checked by `deploy/verify_deployment.py`).

FastAPI also serves interactive docs at `/docs`, `/redoc` and `/openapi.json` at the **root**, outside `/api`. With the routing
above they are not reachable through the proxy. Keep it that way unless you want them public.

## 6. Compression and caching

The repository does not compress anything itself: the API has no compression middleware and the app has no static file server.
Check what your host already does before adding anything:

```bash
curl -sI -H "Accept-Encoding: br, gzip" https://YOUR-HOST/assets/<an index-*.js file> | grep -i "content-encoding\|cache-control"
curl -sI -H "Accept-Encoding: gzip"     https://YOUR-HOST/api/overview               | grep -i "content-encoding"
```

If `content-encoding: br` or `gzip` is already present, do nothing. Most managed static hosts compress text assets automatically.

Evidence that compression matters here (measured in this project):

| Payload | Raw | Compressed |
|---|---|---|
| entry bundle `index-*.js` | 453 KB | 147 KB gzip / ~126 KB Brotli |
| all static text assets | 1,065 KB | 328 KB gzip-9 |
| `/api/overview` | 170 KB | 39 KB gzip |
| `/api/explain` | 138 KB | 31 KB gzip |
| `/api/model-metrics` | 75 KB | 14 KB gzip |
| `/api/filters`, `/api/predict/schema`, movie lists | 3-6 KB | 1-2 KB |

And Lighthouse mobile scores on Home were 59-66 without compression versus 71-81 with it (see the results below).

Recommendations:

1. **Compress text assets at the host or proxy, Brotli preferred, gzip as fallback.**
2. **Compress API JSON at the proxy** too (`application/json`). Do this in the reverse proxy rather than in the app. FastAPI's
   `GZipMiddleware` would work if the API is exposed directly with no proxy, but it is deliberately not enabled here so that a host
   which already compresses does not compress twice. Add it only if `verify_deployment.py` reports uncompressed API responses and
   you have no proxy.
3. **Cache `assets/*` for a year (`public, max-age=31536000, immutable`)** - the files are content-hashed.
4. **Never cache `index.html` long-term** (`Cache-Control: no-cache`), so a new release is picked up immediately.

**Example (not run): nginx**

```nginx
server {
    listen 443 ssl http2;
    server_name frame.example.com;
    root /srv/frame/frontend/dist;

    gzip on;
    gzip_types text/css application/javascript application/json image/svg+xml;
    gzip_min_length 1024;
    # brotli on; brotli_types text/css application/javascript application/json image/svg+xml;   # needs the ngx_brotli module

    location /assets/ { add_header Cache-Control "public, max-age=31536000, immutable"; try_files $uri =404; }
    location /api/    { proxy_pass http://127.0.0.1:8000; proxy_set_header Host $host; proxy_set_header X-Forwarded-Proto $scheme;
                        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for; }
    location /        { add_header Cache-Control "no-cache"; try_files $uri /index.html; }
}
```

**Example (not run): Caddy**

```caddy
frame.example.com {
    encode zstd gzip
    root * /srv/frame/frontend/dist
    handle /api/* { reverse_proxy 127.0.0.1:8000 }
    @assets path /assets/*
    header @assets Cache-Control "public, max-age=31536000, immutable"
    header /index.html Cache-Control "no-cache"
    try_files {path} /index.html
    file_server
}
```

`vite preview` and `vite dev` do not compress and are not production servers; Lighthouse scores taken against them are pessimistic.

## 7. Health checks and startup

- `GET /api/health` returns **200** with `{"status":"ok", ...}` when the model is loaded, and **503** with `{"status":"degraded"}` when
  it is not (wrong or missing model, schema mismatch). Use it for load-balancer and container health checks; it needs no
  authentication and has no side effects. (**verified**: both the 200 and the 503 case.)
- The API loads the dataset and model during startup, so the port starts accepting connections only after loading finishes
  (about 3 seconds here). A connection error during that window is normal; allow a start period before treating it as unhealthy.
- A container health check example: `HEALTHCHECK CMD python -c "import urllib.request,sys;sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/api/health').status==200 else 1)"`.
- If the API is down or unreachable, every page shows a readable error with a "Try again" button and recovers when the API returns
  (**verified** for HTTP 500 and for an unreachable API on all six routes).

## 8. Security notes

- Terminate TLS at the proxy/host. Serve the site and API over HTTPS only.
- The API has no authentication and the two POST endpoints do CPU work. If the site is public, add rate limiting at the proxy.
- `POST /api/predict` rejects unknown fields, including post-release inputs such as `gross`, with HTTP 422 (**verified**).
- The model is a pickle (`joblib`). It is loaded only from the repository path or `FRAME_MODEL_PATH`; never point that at a file you do
  not control, and never accept a model path from a request (the API does not).
- No secrets are stored in the repo or the bundle (section 3).

## 9. CORS

Needed only when the site and the API are on different origins. `FRAME_CORS_ORIGINS` is a comma-separated allow-list; the API allows
`GET` and `POST` and any request header. Verified behaviour (in-process preflight against the real app):

| `FRAME_CORS_ORIGINS` | Request from `http://127.0.0.1:4301` | Result |
|---|---|---|
| `http://127.0.0.1:4301` | `OPTIONS /api/predict` | 200, `access-control-allow-origin` echoes the origin |
| `http://example.com` | `OPTIONS /api/predict` | 400, no allow-origin header (the browser blocks the call) |

With a mismatched allow-list the browser console shows a CORS error and the UI shows its "Cannot reach the FRAME API" state.

## 10. Verifying a deployment

Run the bundled script (standard library only; it makes GET requests plus one read-only prediction):

```bash
python deploy/verify_deployment.py https://frame.example.com
python deploy/verify_deployment.py https://frame.example.com --api https://api.example.com    # split origins (also checks CORS)
python deploy/verify_deployment.py http://127.0.0.1:8000 --api-only                           # backend only
```

It checks: `/api/health` is 200 with the Random Forest and 4,919 rows; `/api/overview` is 200 and reports its size and
`content-encoding`; `/api/predict` returns a valid label whose class scores sum to 1; a post-release field is rejected with 422; an
unknown `/api` path returns JSON 404; all six routes return HTML 200; `index.html` is not cached long-term; the entry bundle is
compressed and cached for a year. It exits non-zero on any failure; warnings (for example uncompressed API responses) do not fail it.
Run against a clean-clone deployment it gave 15 passed, 1 expected warning (the test proxy did not compress API JSON), 0 failed.

Manual checks:

```bash
curl -s https://frame.example.com/api/health
curl -s -X POST https://frame.example.com/api/predict -H "Content-Type: application/json" \
  -d '{"title_year":2010,"duration":120,"genres":["Drama"],"country":"USA","content_rating":"PG-13"}'
curl -sI -H "Accept-Encoding: br, gzip" https://frame.example.com/api/overview | grep -i content-encoding
```

Then open the site in a browser with the developer console open and visit each of the six routes (`/`, `/predict`, `/performance`,
`/explain`, `/explore`, `/about`): there should be no red errors and no failed requests in the Network tab; submit the predictor once
("A real Average" then "Predict category") and confirm a label and the "Class scores" appear.

## 11. What was verified in Phase 5, and what was not

**Verified (from a clean clone of the pushed repository):**

- Fresh virtualenv from `requirements-api.txt` only; the API started with 2 workers and served every endpoint.
- `npm ci && npm run build` succeeded; all six routes load from the production build, route chunks and lazy charts included.
- Browser suites against that production build, with the API behind a Brotli-compressing proxy: 29 functional checks, 20 chart /
  duplicate-request / layout-stability checks, and 59 error-state checks (HTTP 500 and unreachable API on all six routes with
  recovery, a 503 health response, and predictor 503 / 422 / network / 500 failures with recovery). 0 failures.
- Backend tests: 47 passed, 8 skipped.
- Model artifacts, schema, saved metrics and the original notebook were not modified.

**Not verified / open items:**

- **No host was selected or deployed to.** The nginx, Caddy and systemd snippets are examples that have not been executed.
- **HTTPS/TLS, a real domain and real-host compression headers** can only be confirmed on the target host; use `verify_deployment.py`.
- **Cross-origin mode in a real browser** (`VITE_API_URL` + `FRAME_CORS_ORIGINS`) was verified at the HTTP level (build bakes the URL;
  preflight allowed/denied as in section 9) but a full browser run of that mode did not complete here. Use it only after running
  `verify_deployment.py --api` against it. Same-origin is the verified path.
- **Operating system:** everything was run on Windows 11. The pinned packages have Linux wheels, but a Linux host has not been
  exercised here.
- **Sub-path hosting** (site not at the domain root) is unsupported without changing Vite's `base` and the `routePreload` plugin.
- `/performance` stays the slowest route on throttled mobile (see below).

## 12. How the performance numbers were measured

The same harness was used for every before/after comparison so that conditions match: a small static server that adds Brotli and
immutable caching headers and proxies `/api` to the real backend, Lighthouse CLI against Microsoft Edge (headless), default
simulated "mobile" throttling (slow 4G, 4x CPU slowdown) and the "desktop" preset, three runs per cell, median reported, and the
builds being compared were run alternately (A, B, A, B, ...) so background load on the machine affects both equally.
Lighthouse scores vary with machine load; treat differences of one or two points as noise.

## 13. Measured results (before vs after the Phase 4 changes)

Lighthouse performance score, median of 3 interleaved runs, simulated mobile throttling, text assets served with Brotli
("before" = the build at the start of Phase 4, "after" = the build in this repository). LCP in seconds.

| Route | Mobile score | Mobile LCP | Mobile FCP | Desktop score |
|---|---|---|---|---|
| `/` | 71 → 81 | 5.98 → 4.32 | 2.99 → 1.66 | 98 → 99 |
| `/predict` | 85 → 87 | 3.57 → 3.62 | 2.91 → 2.06 | 99 → 100 |
| `/performance` | 74 → 73 | 4.00 → 4.06 | 2.91 → 2.09 | 99 → 99 |
| `/explain` | 78 → 80 | 4.28 → 4.27 | 2.84 → 2.06 | 99 → 99 |
| `/explore` | 89 → 94 | 3.10 → 2.80 | 2.61 → 1.66 | 100 → 100 |
| `/about` | 85 → 91 | 3.52 → 3.24 | 2.53 → 1.73 | 99 → 100 |

Home transfers 512 KB instead of 781 KB. The same comparison with **no compression** (what `vite preview` does) gives Home 59 → 66,
Explore 70 → 74 and Performance 61 → 64 on mobile, so a host that compresses is worth more than any change in the app.

What changed and why (each kept only where an interleaved A/B or a trace showed it helped):

- Recharts is no longer pulled in by the routes that only show charts further down. Overview, Performance, Explain and the
  Predictor result load their chart code as separate chunks, with placeholders of the exact final height so nothing moves.
- Fonts are preloaded from `index.html`. Fonts normally start downloading only when text using them first renders, so they
  arrived late and the swap reflowed the page (a 0.24 layout shift on Explore under throttling, now 0).
- The current route's chunk is preloaded in parallel with the entry bundle (3 points on Home under applied throttling, neutral
  under Lighthouse's simulated throttling).
- Identical concurrent GET requests share one request (`/api/health` and `/api/overview` were each fetched twice).

Tried and rejected because they showed no measurable gain: removing the hero paragraph's fade-in, removing the grain overlay,
changing the JS compile target, a smaller Motion feature set, and prefetching the Overview chart chunk at idle (now fetched on
first scroll or interaction instead, so it costs nothing until someone reads).
