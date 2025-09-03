
(function () {
  // ------- Section collection (hero + full sections)
  const sections = Array.from(
    document.querySelectorAll('header.hero-section, section.project-section')
  );

  let currentSectionIndex = 0;

  // Clamp helper
  const clamp = (n, min, max) => Math.max(min, Math.min(max, n));

  // Sync current section based on viewport center
  const syncIndexToViewport = () => {
    if (!sections.length) return;
    const mid = window.scrollY + window.innerHeight / 2;
    let best = 0;
    let bestDist = Infinity;
    sections.forEach((s, i) => {
      const top = s.offsetTop;
      const bottom = top + s.offsetHeight;
      const center = (top + bottom) / 2;
      const d = Math.abs(center - mid);
      if (d < bestDist) {
        bestDist = d;
        best = i;
      }
    });
    currentSectionIndex = best;
    // 同步地址栏 hash
    history.replaceState(null, "", "#" + (sections[best].id || ""));
  };

  window.addEventListener("load", syncIndexToViewport);
  window.addEventListener("scroll", syncIndexToViewport, { passive: true });
  window.addEventListener("resize", syncIndexToViewport);

  // Smooth jump helper
  const jumpTo = (idx) => {
    const target = clamp(idx, 0, sections.length - 1);
    sections[target].scrollIntoView({ behavior: "smooth", block: "start" });
  };

  // ------- Top menu smooth scroll
  document.querySelectorAll('.top-nav a[href^="#"]').forEach((link) => {
    link.addEventListener("click", (e) => {
      const id = link.getAttribute("href");
      const target = id && document.querySelector(id);
      if (!target) return;
      e.preventDefault();
      target.scrollIntoView({ behavior: "smooth", block: "start" });
      history.replaceState(null, "", id);
    });
  });

  // ------- Down-arrow buttons
  document.querySelectorAll(".scroll-down-arrow").forEach((btn) => {
    btn.addEventListener("click", (e) => {
      e.preventDefault();
      const parent = btn.closest(
        "header.hero-section, section.project-section"
      );
      const idx = sections.indexOf(parent);
      jumpTo((idx > -1 ? idx : currentSectionIndex) + 1);
    });
  });

  // ------- Keyboard navigation
  window.addEventListener("keydown", (e) => {
    const nextKeys = new Set(["PageDown", "ArrowDown", " "]); // spacebar
    const prevKeys = new Set(["PageUp", "ArrowUp"]);
    const isShiftSpace = e.key === " " && e.shiftKey;

    if (nextKeys.has(e.key) && !e.shiftKey) {
      e.preventDefault();
      jumpTo(currentSectionIndex + 1);
    } else if (prevKeys.has(e.key) || isShiftSpace) {
      e.preventDefault();
      jumpTo(currentSectionIndex - 1);
    }
  });

  // ------- Touch swipe navigation
  let touchStartY = null;
  window.addEventListener(
    "touchstart",
    (e) => {
      if (e.changedTouches && e.changedTouches[0]) {
        touchStartY = e.changedTouches[0].clientY;
      }
    },
    { passive: true }
  );
  window.addEventListener(
    "touchend",
    (e) => {
      if (touchStartY == null) return;
      const y =
        e.changedTouches && e.changedTouches[0]
          ? e.changedTouches[0].clientY
          : touchStartY;
      const dy = touchStartY - y;
      touchStartY = null;
      const threshold = 60; // px
      if (Math.abs(dy) < threshold) return;
      jumpTo(currentSectionIndex + (dy > 0 ? 1 : -1));
    },
    { passive: true }
  );
})();


// Part 2 
(() => {
  const host =
    document.querySelector('#part2 .cmp-vert') ||
    document.querySelector('#part2 .cmp');
  if (!host) return;
  const slider = host.querySelector('.cmp-slider');
  const set = v => host.style.setProperty('--pos', v + '%');
  set(slider.value);
  slider.addEventListener('input', e => set(e.target.value));
})();
