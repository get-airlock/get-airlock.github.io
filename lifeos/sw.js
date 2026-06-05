// LifeOS service worker — PWA shell, network-first for pages so the UI is never stale.
// v0.2.0 (new design): HTML = network-first (fresh always), static = cache-first (offline).

const CACHE_VERSION = 'lifeos-v0.2.0';
const SHELL_ASSETS = [
  '/lifeos/',
  '/lifeos/index.html',
  '/lifeos/create.html',
  '/lifeos/memory.html',
  '/lifeos/wallet.html',
  '/lifeos/you.html',
  '/lifeos/kyi/arrive.html',
  '/lifeos/kyi/continue.html',
  '/lifeos/kyi/contribute.html',
  '/lifeos/css/lifeos.css',
  '/lifeos/js/store.js',
  '/lifeos/js/ui.js',
  '/lifeos/js/memory.js',
  '/lifeos/js/companion.js',
  '/lifeos/manifest.json',
];

self.addEventListener('install', (event) => {
  event.waitUntil(
    caches.open(CACHE_VERSION)
      .then((cache) => cache.addAll(SHELL_ASSETS).catch(() => {})) // tolerate any 404 in list
      .then(() => self.skipWaiting())
  );
});

self.addEventListener('activate', (event) => {
  event.waitUntil(
    caches.keys()
      .then((keys) => Promise.all(keys.filter((k) => k !== CACHE_VERSION).map((k) => caches.delete(k))))
      .then(() => self.clients.claim())
  );
});

self.addEventListener('fetch', (event) => {
  const req = event.request;
  if (req.method !== 'GET') return;

  const isHTML = req.mode === 'navigate' || (req.headers.get('accept') || '').includes('text/html');

  if (isHTML) {
    // Network-first: always serve fresh pages; fall back to cache only when offline.
    event.respondWith(
      fetch(req)
        .then((response) => {
          if (response.ok && req.url.startsWith(self.location.origin)) {
            const clone = response.clone();
            caches.open(CACHE_VERSION).then((cache) => cache.put(req, clone));
          }
          return response;
        })
        .catch(() => caches.match(req).then((c) => c || caches.match('/lifeos/index.html')))
    );
    return;
  }

  // Static assets (css/js/fonts/img): cache-first for speed + offline.
  event.respondWith(
    caches.match(req).then((cached) =>
      cached ||
      fetch(req).then((response) => {
        if (response.ok && req.url.startsWith(self.location.origin)) {
          const clone = response.clone();
          caches.open(CACHE_VERSION).then((cache) => cache.put(req, clone));
        }
        return response;
      })
    )
  );
});
