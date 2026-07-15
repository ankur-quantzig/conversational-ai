import { defineConfig, loadEnv } from 'vite'
import react from '@vitejs/plugin-react'

// Content-Security-Policy for the production build. Dev mode is excluded
// because @vitejs/plugin-react injects an inline preamble script.
// frame-ancestors/X-Frame-Options must be set as HTTP headers by the host —
// meta CSP cannot express them.
function cspPlugin(env) {
  let apiOrigin = ''
  try {
    apiOrigin = new URL(env.VITE_API_BASE_URL).origin
  } catch {
    // Relative API base (default /api) is covered by 'self'.
  }
  const remote = apiOrigin ? ` ${apiOrigin}` : ''
  const csp = [
    "default-src 'self'",
    "script-src 'self'",
    `connect-src 'self'${remote}`,
    "img-src 'self' data:",
    `media-src 'self'${remote}`,
    "style-src 'self' 'unsafe-inline'",
    "font-src 'self'",
    "object-src 'none'",
    "base-uri 'self'",
    "form-action 'self'",
  ].join('; ')

  return {
    name: 'inject-csp',
    apply: 'build',
    transformIndexHtml(html) {
      return html.replace(
        '<meta charset="UTF-8" />',
        `<meta charset="UTF-8" />\n    <meta http-equiv="Content-Security-Policy" content="${csp}" />`,
      )
    },
  }
}

export default defineConfig(({ mode }) => {
  const env = loadEnv(mode, process.cwd(), '')
  return {
    plugins: [react(), cspPlugin(env)],
  }
})
