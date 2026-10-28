import tailwindcss from '@tailwindcss/vite'
import react from '@vitejs/plugin-react'
import { defineConfig, loadEnv, type Plugin } from 'vite'

const ROUTE_CHUNKS: Record<string, string> = { '/': 'Overview', '/predict': 'Predictor', '/performance': 'Performance', '/explain': 'Explain', '/explore': 'Explore', '/about': 'About' }

/** Route pages are lazy chunks, so the browser would only request one after the entry bundle has run. This adds a tiny inline script that
 *  issues modulepreload for the current route's chunk (and its static dependencies) while the entry is still downloading. */
function routePreload(): Plugin {
  return {
    name: 'route-preload',
    apply: 'build',
    transformIndexHtml: {
      order: 'post',
      handler(html, ctx) {
        const bundle = ctx.bundle
        if (!bundle) return html
        const map: Record<string, string[]> = {}
        for (const [route, page] of Object.entries(ROUTE_CHUNKS)) {
          const chunk = Object.values(bundle).find((c) => c.type === 'chunk' && c.facadeModuleId?.replaceAll('\\', '/').endsWith(`/src/pages/${page}.tsx`))
          if (chunk?.type === 'chunk') map[route] = [chunk.fileName, ...chunk.imports.filter((f) => !bundle[f] || (bundle[f].type === 'chunk' && !bundle[f].isEntry))]
        }
        const script = `<script>(function(){var m=${JSON.stringify(map)},p=location.pathname.replace(/[/]+$/,'')||'/',f=m[p];if(!f)return;f.forEach(function(u){var l=document.createElement('link');l.rel='modulepreload';l.href='/'+u;document.head.appendChild(l)})})()</script>`
        // Fonts are otherwise requested only once text using them is first rendered, so they arrive late and the swap reflows the page.
        const fonts = Object.keys(bundle).filter((f) => /-latin-wght-normal-[\w-]+\.woff2$/.test(f))
          .map((f) => `<link rel="preload" as="font" type="font/woff2" crossorigin href="/${f}" />`)
        return html.replace('</head>', `    ${[...fonts, script].join('\n    ')}
  </head>`)
      },
    },
  }
}

// The dev server proxies /api to the FastAPI backend, so the browser sees a single origin.
export default defineConfig(({ mode }) => {
  const env = loadEnv(mode, process.cwd(), '')
  const target = env.VITE_API_PROXY || 'http://127.0.0.1:8000'
  return {
    plugins: [react(), tailwindcss(), routePreload()],
    server: { port: 5173, proxy: { '/api': { target, changeOrigin: true } } },
    preview: { port: 4173, proxy: { '/api': { target, changeOrigin: true } } },
    build: { chunkSizeWarningLimit: 700 },
  }
})
