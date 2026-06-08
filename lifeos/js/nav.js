// LifeOS Bottom Nav — progressive unlock per Sisyphus L2.3
// Built 2026-05-30 for Rika June 1 demo

const Nav = {
  render(activeSurface) {
    const surfaces = [
      { key: 'home',   label: 'Home',   icon: '✦', href: '/lifeos/' },
      { key: 'memory', label: 'Memory', icon: '◎', href: '/lifeos/memory.html' },
      { key: 'create', label: 'Create', icon: '◇', href: '/lifeos/create.html' },
      { key: 'wallet', label: 'Wallet', icon: '◈', href: '/lifeos/wallet.html' },
      { key: 'you',    label: 'You',    icon: '☺', href: '/lifeos/you.html' }
    ];

    const alwaysOpen = { home: true, you: true };
    const isUnlocked = (key) => alwaysOpen[key] ? true : (window.LifeOSMemory ? window.LifeOSMemory.isUnlocked(key) : true);

    const html = `
      <nav class="bottom-nav" aria-label="LifeOS navigation">
        ${surfaces.map((s) => {
          const locked = !isUnlocked(s.key);
          const active = s.key === activeSurface;
          const cls = ['', active ? 'active' : '', locked ? 'locked' : ''].filter(Boolean).join(' ');
          return `<a href="${locked ? '#' : s.href}" class="${cls}" data-key="${s.key}" ${locked ? 'aria-disabled="true"' : ''}>
            <span class="bottom-nav-icon">${s.icon}</span>
            <span class="bottom-nav-label">${s.label}</span>
          </a>`;
        }).join('')}
      </nav>`;

    const container = document.getElementById('bottom-nav') || document.body;
    if (container.id === 'bottom-nav') {
      container.outerHTML = html;
    } else {
      container.insertAdjacentHTML('beforeend', html);
    }
  }
};

if (typeof window !== 'undefined') {
  window.LifeOSNav = Nav;
}
