(() => {
  const code = document.querySelector('meta[name="goatcounter-code"]')?.content.trim();
  const counters = Array.from(document.querySelectorAll("[data-visit-counter]"));

  if (!code || !counters.length) return;

  counters.forEach(async (counter) => {
    const requestedPath = counter.dataset.goatcounterPath || window.location.pathname;
    const path = requestedPath === "TOTAL" ? "TOTAL" : encodeURIComponent(requestedPath);

    try {
      const response = await fetch(`https://${code}.goatcounter.com/counter/${path}.json`);
      if (!response.ok) return;

      const data = await response.json();
      const value = counter.querySelector("[data-visit-count]");
      if (!value || !data.count) return;

      value.textContent = data.count;
      counter.hidden = false;
    } catch (_) {
      // Analytics should never get in the way of reading the site.
    }
  });
})();

