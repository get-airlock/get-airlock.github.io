// GRAVITY PWA — network-first for HTML, cache-first for static assets.
const CACHE = 'gravity-v0.3.0';
const ASSETS = ['./index.html', './manifest.webmanifest', './icon.svg', './weavemind.js'];

self.addEventListener('install', e => {
  self.skipWaiting();
  e.waitUntil(caches.open(CACHE).then(c => c.addAll(ASSETS)));
});
self.addEventListener('activate', e => {
  e.waitUntil(caches.keys().then(ks =>
    Promise.all(ks.filter(k => k !== CACHE).map(k => caches.delete(k)))).then(() => self.clients.claim()));
});
self.addEventListener('fetch', e => {
  const req = e.request;
  const isHTML = req.mode === 'navigate' || (req.headers.get('accept') || '').includes('text/html');
  if (isHTML) {
    e.respondWith(fetch(req).then(r => {
      const copy = r.clone(); caches.open(CACHE).then(c => c.put(req, copy)); return r;
    }).catch(() => caches.match(req).then(m => m || caches.match('./index.html'))));
  } else {
    e.respondWith(caches.match(req).then(m => m || fetch(req)));
  }
});
