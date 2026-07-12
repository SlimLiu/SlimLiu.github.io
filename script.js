(() => {
  const root = document.documentElement;
  const toggle = document.querySelector(".language-toggle");
  const description = document.querySelector('meta[name="description"]');
  const ogDescription = document.querySelector('meta[property="og:description"]');
  const ogLocale = document.querySelector('meta[property="og:locale"]');

  const copy = {
    zh: {
      description:
        "刘盎希，西安交通大学人工智能专业本科生，关注多智能体决策、具身智能、博弈论与计算机视觉。",
      ogDescription: "一个不喜欢 AI 的 AI 专业学生。",
      locale: "zh_CN",
      label: "Switch to English",
    },
    en: {
      description:
        "Angxi Liu is an AI undergraduate at Xi’an Jiaotong University, interested in computer vision, sports, and literature.",
      ogDescription: "I’m an AI major who doesn’t like AI.",
      locale: "en_US",
      label: "切换到中文",
    },
  };

  function setLanguage(language, persist = true) {
    const next = language === "en" ? "en" : "zh";
    root.dataset.language = next;
    root.lang = next === "zh" ? "zh-CN" : "en";

    if (toggle) {
      toggle.setAttribute("aria-label", copy[next].label);
      toggle.title = copy[next].label;
    }

    if (description) description.content = copy[next].description;
    if (ogDescription) ogDescription.content = copy[next].ogDescription;
    if (ogLocale) ogLocale.content = copy[next].locale;
    if (persist) {
      try {
        localStorage.setItem("angxi-language-v2", next);
      } catch (_) {
        // The switch still works when browser storage is unavailable.
      }
    }
  }

  setLanguage(root.dataset.language, false);

  toggle?.addEventListener("click", () => {
    setLanguage(root.dataset.language === "zh" ? "en" : "zh");
  });

  const year = document.getElementById("current-year");
  if (year) year.textContent = new Date().getFullYear();
})();
