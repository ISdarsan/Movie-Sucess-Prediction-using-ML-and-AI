# Deploying the FRAME site (frontend delivery notes)

This covers how the built site should be served so that the performance work in the app is not undone by the host.
It does not touch the model, the API contract or any saved artifact.

## What the build produces

```
cd frontend
npm install
npm run build        # tsc -b && vite build  ->  frontend/dist/
```

`dist/` is a static site (about 1.3 MB on disk, 26 files in `assets/`). Every file under `assets/` has a content hash in its
name, so it never changes at a given URL. `index.html` is the only file that changes between releases.

| Chunk | Raw | Gzip | Loaded |
|---|---|---|---|
| `index-*.js` (React, router, Motion, shell) | 453 KB | 147 KB | every page |
| `index-*.css` | 70 KB | 16 KB | every page |
| `CartesianChart-*.js` (Recharts) | 338 KB | 99 KB | only when a chart is on screen or about to be |
| Route chunks (`Overview`, `Predictor`, ...) | 12–32 KB | 4–10 KB | the route being visited |
| Chart modules (`OverviewCharts`, `ModelCharts`, `insight`) | 9–17 KB | 4–6 KB | with their route's charts |
| Fonts (three Latin `woff2` files) | 41 / 40 / 30 KB | already compressed | preloaded from `index.html` |

The build also writes a small inline script and font `preload` links into `index.html` (see `routePreload` in
`frontend/vite.config.ts`). The script requests the current route's chunk in parallel with the entry bundle. It assumes the site
is served from the domain root (`/assets/...`). If you host it under a sub-path, set Vite's `base` and update that plugin.

## What the host must do

The repository contains no deployment configuration, and the FastAPI app serves only `/api/*` (no static files, no compression
middleware). Compression and caching therefore depend entirely on the platform you choose. Check what it already does before
adding anything:

```
curl -sI -H "Accept-Encoding: br, gzip" https://YOUR-HOST/assets/<an index-*.js file> | grep -i "content-encoding\|cache-control"
curl -sI -H "Accept-Encoding: gzip"     https://YOUR-HOST/api/overview               | grep -i "content-encoding"
```

If `content-encoding: br` (or `gzip`) is already present, do nothing. Most managed static hosts (Netlify, Vercel, Cloudflare Pages,
GitHub Pages) compress text assets automatically; verify it, because the saving is large.

1. **Compress text assets, Brotli preferred.** The static text assets total 1,065 KB raw and 328 KB with gzip level 9. The entry
   bundle is 147 KB gzipped and about 126 KB with Brotli.
2. **Cache hashed assets for a year, never cache `index.html`.**
   `assets/*` → `Cache-Control: public, max-age=31536000, immutable`; `index.html` → `Cache-Control: no-cache`.
3. **Single-page-app fallback.** Any path that is not a file (for example `/performance`) must return `index.html` with status 200.
4. **Proxy `/api` to the FastAPI app** so the browser sees one origin (otherwise set `VITE_API_URL` at build time and
   `FRAME_CORS_ORIGINS` on the API).
5. **Compress API JSON at the proxy.** The API does not compress its own responses. Measured payloads on this dataset:

   | Endpoint | Raw | Gzip |
   |---|---|---|
   | `/api/overview` | 170 KB | 39 KB |
   | `/api/explain` | 138 KB | 31 KB |
   | `/api/model-metrics` | 75 KB | 14 KB |
   | `/api/filters`, `/api/predict/schema`, movie lists | 3–6 KB | 1–2 KB |

   Prefer doing this in the reverse proxy. If the API is exposed directly with no proxy, FastAPI's `GZipMiddleware` is a one-line
   alternative; it is deliberately **not** enabled in this repository, so that hosts that already compress are not double-compressed.

### nginx

```nginx
gzip on;
gzip_types text/css application/javascript application/json image/svg+xml;
gzip_min_length 1024;
# brotli on; brotli_types ...;   # needs the ngx_brotli module

location /assets/ { add_header Cache-Control "public, max-age=31536000, immutable"; try_files $uri =404; }
location /api/    { proxy_pass http://127.0.0.1:8000; }
location /        { add_header Cache-Control "no-cache"; try_files $uri /index.html; }
```

### Caddy

```caddy
example.com {
    encode zstd gzip
    root * /srv/frame/dist
    handle /api/* { reverse_proxy 127.0.0.1:8000 }
    @assets path /assets/*
    header @assets Cache-Control "public, max-age=31536000, immutable"
    header /index.html Cache-Control "no-cache"
    try_files {path} /index.html
    file_server { precompressed br gzip }   # only if you precompress; otherwise omit
}
```

`vite preview` and `vite dev` do not compress. Lighthouse scores taken against them are pessimistic compared with a real host.

## How the numbers in this project were measured

The same harness was used for every before/after comparison so that conditions match: a small static server that adds Brotli and
immutable caching headers and proxies `/api` to the real backend, Lighthouse CLI against Microsoft Edge (headless), default
simulated "mobile" throttling (slow 4G, 4x CPU slowdown) and the "desktop" preset, three runs per cell, median reported, and the
builds being compared were run alternately (A, B, A, B, ...) so background load on the machine affects both equally.
Lighthouse scores vary with machine load; treat differences of one or two points as noise.

## Measured results (before vs after the Phase 4 changes)

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
