// ===== Fullpage sections selection (fixed) =====
const sections = Array.from(
  document.querySelectorAll('header.hero-section, section.project-section')
);

let currentSectionIndex = 0;
let isScrolling = false;

// Helper: clamp index
const clamp = (n, min, max) => Math.max(min, Math.min(max, n));

// Sync index to current viewport (on load / hash change / manual scroll)
const syncIndexToViewport = () => {
  if (location.hash) {
    const idx = sections.findIndex(s => '#' + s.id === location.hash);
    if (idx > -1) { currentSectionIndex = idx; return; }
  }
  const mid = window.scrollY + window.innerHeight / 2;
  const i = sections.findIndex(s => {
    const r = s.getBoundingClientRect();
    const top = r.top + window.scrollY;
    const bottom = top + r.height;
    return mid >= top && mid < bottom;
  });
  if (i > -1) currentSectionIndex = i;
};

window.addEventListener('load', syncIndexToViewport);
window.addEventListener('hashchange', syncIndexToViewport);
window.addEventListener('scroll', () => { if (!isScrolling) syncIndexToViewport(); }, { passive: true });

// Smooth scroll to a given section index
const smoothTo = (idx) => {
  const target = clamp(idx, 0, sections.length - 1);
  if (target === currentSectionIndex) {
    // still ensure alignment
    sections[target].scrollIntoView({ behavior: 'smooth', block: 'start' });
    return;
  }
  isScrolling = true;
  currentSectionIndex = target;
  sections[target].scrollIntoView({ behavior: 'smooth', block: 'start' });
  setTimeout(() => { isScrolling = false; }, 600); // slightly shorter lock for snappier feel
};

// Wheel navigation (fullpage)
window.addEventListener('wheel', (e) => {
  if (isScrolling) return;
  e.preventDefault(); // custom fullpage behavior
  const dir = e.deltaY > 0 ? 1 : -1;
  smoothTo(currentSectionIndex + dir);
}, { passive: false });

// Down-arrow buttons: jump to *next* section of the button's own section
document.querySelectorAll('.scroll-down-arrow').forEach((btn) => {
  btn.addEventListener('click', (e) => {
    e.preventDefault();
    e.stopPropagation();
    const parent = btn.closest('header.hero-section, section.project-section');
    const i = sections.indexOf(parent);
    smoothTo((i > -1 ? i : currentSectionIndex) + 1);
  });
});
