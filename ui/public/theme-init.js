// Applies the saved/system theme before React renders to avoid a flash of the
// wrong theme, and to keep native form controls in sync on first paint.
// Loaded as a classic blocking script so it runs before first paint.
try {
  var theme = new URLSearchParams(window.location.search).get('theme') || window.localStorage.getItem('insight-copilot-theme')
  if (theme !== 'dark' && theme !== 'light') {
    theme = window.matchMedia('(prefers-color-scheme: dark)').matches ? 'dark' : 'light'
  }
  document.documentElement.dataset.theme = theme
} catch (error) {
  /* default to light */
}
