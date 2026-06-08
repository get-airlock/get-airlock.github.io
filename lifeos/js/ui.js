/* LifeOS — shared UI chrome: aurora background, grain, bottom nav */
(function () {
  const ICONS = {
    home:   '<path d="M4 11.5 12 4l8 7.5"/><path d="M6 10v9h12v-9"/>',
    create: '<path d="M12 5v14M5 12h14"/>',
    memory: '<path d="M12 3a4 4 0 0 0-4 4 4 4 0 0 0-1 7.9V18a3 3 0 0 0 6 0V3Z"/><path d="M12 3a4 4 0 0 1 4 4 4 4 0 0 1 1 7.9V18a3 3 0 0 1-6 0"/>',
    wallet: '<rect x="3" y="6" width="18" height="13" rx="3"/><path d="M3 10h18M16 14.5h.01"/>',
    you:    '<circle cx="12" cy="8.5" r="3.5"/><path d="M5.5 19a6.5 6.5 0 0 1 13 0"/>',
  };
  const ITEMS = [
    { id: 'home',   href: 'index.html',  label: 'Home' },
    { id: 'create', href: 'create.html', label: 'Create' },
    { id: 'memory', href: 'memory.html', label: 'Memory' },
    { id: 'wallet', href: 'wallet.html', label: 'Wallet' },
    { id: 'you',    href: 'you.html',    label: 'You' },
  ];

  const LifeOSUI = {
    mount(active) {
      // Aurora + grain (prepended so they sit behind everything)
      if (!document.querySelector('.aurora')) {
        const aurora = document.createElement('div');
        aurora.className = 'aurora';
        aurora.setAttribute('aria-hidden', 'true');
        aurora.innerHTML = '<div class="blob b1"></div><div class="blob b2"></div><div class="blob b3"></div><div class="blob b4"></div>';
        const grain = document.createElement('div');
        grain.className = 'grain';
        grain.setAttribute('aria-hidden', 'true');
        document.body.prepend(grain);
        document.body.prepend(aurora);
      }
      // Bottom nav
      const nav = document.createElement('nav');
      nav.className = 'nav';
      nav.innerHTML = ITEMS.map((it) =>
        `<a href="${it.href}" class="${it.id === active ? 'active' : ''}" aria-label="${it.label}">
           <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round">${ICONS[it.id]}</svg>
         </a>`).join('');
      const mountPoint = document.getElementById('bottom-nav');
      if (mountPoint) mountPoint.replaceWith(nav);
      else document.body.appendChild(nav);
    },
  };

  window.LifeOSUI = LifeOSUI;
})();
