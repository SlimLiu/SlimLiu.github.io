(() => {
  const buttons = Array.from(document.querySelectorAll("[data-filter]"));
  const posts = Array.from(document.querySelectorAll("[data-post-category]"));
  const emptyState = document.querySelector("[data-empty-state]");

  if (!buttons.length || !posts.length) return;

  function applyFilter(filter) {
    let visibleCount = 0;

    posts.forEach((post) => {
      const visible = filter === "all" || post.dataset.postCategory === filter;
      post.hidden = !visible;
      if (visible) visibleCount += 1;
    });

    buttons.forEach((button) => {
      const active = button.dataset.filter === filter;
      button.classList.toggle("is-active", active);
      button.setAttribute("aria-pressed", String(active));
    });

    if (emptyState) emptyState.hidden = visibleCount !== 0;
  }

  buttons.forEach((button) => {
    button.addEventListener("click", () => applyFilter(button.dataset.filter));
  });

  applyFilter("all");
})();

