document.addEventListener("DOMContentLoaded", () => {
  const navLinks = Array.from(
    document.querySelectorAll('.section-nav a[href^="#"]')
  );

  const sections = navLinks
    .map((link) => {
      const id = link.getAttribute("href").slice(1);
      return document.getElementById(id);
    })
    .filter(Boolean);

  function setCurrentSection(id) {
    navLinks.forEach((link) => {
      const targetId = link.getAttribute("href").slice(1);
      link.classList.toggle("is-current", targetId === id);
    });
  }

  function getCurrentSectionByViewportCenter() {
    const viewportCenter = window.innerHeight / 2;

    let currentSection = null;

    for (const section of sections) {
      const rect = section.getBoundingClientRect();

      if (rect.top <= viewportCenter && rect.bottom >= viewportCenter) {
        currentSection = section;
        break;
      }
    }

    if (!currentSection) {
      let minDistance = Infinity;

      for (const section of sections) {
        const rect = section.getBoundingClientRect();
        const sectionCenter = rect.top + rect.height / 2;
        const distance = Math.abs(sectionCenter - viewportCenter);

        if (distance < minDistance) {
          minDistance = distance;
          currentSection = section;
        }
      }
    }

    if (currentSection) {
      setCurrentSection(currentSection.id);
    }
  }

  let ticking = false;

  function handleScroll() {
    if (!ticking) {
      window.requestAnimationFrame(() => {
        getCurrentSectionByViewportCenter();
        ticking = false;
      });
      ticking = true;
    }
  }

  window.addEventListener("scroll", handleScroll, { passive: true });
  window.addEventListener("resize", handleScroll);

  navLinks.forEach((link) => {
    link.addEventListener("click", () => {
      const id = link.getAttribute("href").slice(1);
      setCurrentSection(id);
    });
  });

  getCurrentSectionByViewportCenter();
});