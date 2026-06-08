// LifeOS service worker — PWA shell, network-first for pages so the UI is never stale.
// v0.3.0: light.html is the one true app. Precache the LIGHT shell + icons so an installed,
// offline launch (e.g. flaky bar wifi) opens the app instead of a blank redirect chain.

const CACHE_VERSION = 'lifeos-v0.3.0';
const APP = '/lifeos/light.html';
const SHELL_ASSETS = [
  '/lifeos/',
  '/lifeos/index.html',
  '/lifeos/light.html',
  '/lifeos/manifest.json',
  '/lifeos/icons/icon-180.png',
  '/lifeos/icons/icon-192.png',
  '/lifeos/icons/icon-512.png',
  '/lifeos/icons/icon.svg',
];

self.addEventListener('install', (event) => {
  event.waitUntil(
    caches.open(CACHE_VERSION)
      // add each asset independently — one 404 can't stop the rest (addAll is all-or-nothing)
      .then((cache) => Promise.all(SHELL_ASSETS.map((u) => cache.add(u).catch(() => {}))))
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
    // Offline fallback is the APP (light.html), never the redirect-only index.
    event.respondWith(
      fetch(req)
        .then((response) => {
          if (response.ok && req.url.startsWith(self.location.origin)) {
            const clone = response.clone();
            caches.open(CACHE_VERSION).then((cache) => cache.put(req, clone));
          }
          return response;
        })
        .catch(() => caches.match(req).then((c) => c || caches.match(APP) || caches.match('/lifeos/index.html')))
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
