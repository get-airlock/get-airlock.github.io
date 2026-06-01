// LifeOS service worker — minimal caching for Add-to-Home-Screen + offline shell
// Built 2026-05-30 for Rika June 1 demo

const CACHE_VERSION = 'lifeos-v0.1.0';
const SHELL_ASSETS = [
  '/lifeos/',
  '/lifeos/index.html',
  '/lifeos/kyi/',
  '/lifeos/kyi/arrive.html',
  '/lifeos/kyi/continue.html',
  '/lifeos/kyi/contribute.html',
  '/lifeos/memory.html',
  '/lifeos/create.html',
  '/lifeos/wallet.html',
  '/lifeos/css/lifeos.css',
  '/lifeos/js/memory.js',
  '/lifeos/js/companion.js',
  '/lifeos/js/nav.js',
  '/lifeos/manifest.json'
];

self.addEventListener('install', (event) => {
  event.waitUntil(
    caches.open(CACHE_VERSION).then((cache) => cache.addAll(SHELL_ASSETS)).then(() => self.skipWaiting())
  );
});

self.addEventListener('activate', (event) => {
  event.waitUntil(
    caches.keys().then((keys) =>
      Promise.all(keys.filter((k) => k !== CACHE_VERSION).map((k) => caches.delete(k)))
    ).then(() => self.clients.claim())
  );
});

self.addEventListener('fetch', (event) => {
  if (event.request.method !== 'GET') return;
  event.respondWith(
    caches.match(event.request).then((cached) => {
      if (cached) return cached;
      return fetch(event.request).then((response) => {
        if (response.ok && event.request.url.startsWith(self.location.origin)) {
          const clone = response.clone();
          caches.open(CACHE_VERSION).then((cache) => cache.put(event.request, clone));
        }
        return response;
      }).catch(() => caches.match('/lifeos/index.html'));
    })
  );
});
