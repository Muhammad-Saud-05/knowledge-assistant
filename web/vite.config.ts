import react from '@vitejs/plugin-react'
import { defineConfig } from 'vitest/config'

// The back end (see the main README). The browser only ever talks to the dev server
// on port 5173; the proxy below forwards /api to the back end, so the page and the
// API look like one origin and no cross-site request is involved.
const API_ORIGIN = 'http://127.0.0.1:8000'

export default defineConfig({
  plugins: [react()],
  server: {
    host: '127.0.0.1', // localhost only, like the back end
    port: 5173,
    strictPort: true, // fail instead of silently moving to another port the API does not allow
    proxy: {
      '/api': {
        target: API_ORIGIN,
        // Answers stream for a long time (and may wait in a queue first): never time out.
        timeout: 0,
        proxyTimeout: 0,
        configure: (proxy) => {
          // Make sure nothing between the browser and the server holds a stream back.
          proxy.on('proxyRes', (proxyRes) => {
            const type = String(proxyRes.headers['content-type'] ?? '')
            if (type.startsWith('text/event-stream')) {
              proxyRes.headers['cache-control'] = 'no-cache'
              proxyRes.headers['x-accel-buffering'] = 'no'
            }
          })
        },
      },
    },
  },
  test: {
    environment: 'jsdom',
    include: ['src/**/*.test.{ts,tsx}'],
  },
})
