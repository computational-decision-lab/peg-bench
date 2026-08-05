(function () {
  "use strict";

  var year = document.getElementById("year");
  if (year) year.textContent = String(new Date().getFullYear());

  var mobileMenu = document.querySelector(".mobile-menu");
  if (mobileMenu) {
    mobileMenu.addEventListener("click", function (event) {
      var anchor = event.target.closest("a[href^='#']");
      if (anchor) mobileMenu.open = false;
    });
  }

  var stage = document.getElementById("ranking-motion");
  if (!stage) return;

  var reduceMotion = window.matchMedia && window.matchMedia("(prefers-reduced-motion: reduce)").matches;
  if (reduceMotion || !window.lottie) return;

  var canvas = stage.querySelector(".motion-canvas");
  if (!canvas) return;

  var animation = window.lottie.loadAnimation({
    container: canvas,
    renderer: "svg",
    loop: false,
    autoplay: false,
    path: stage.getAttribute("data-lottie-src")
  });
  var observer;

  function restoreFallback() {
    stage.setAttribute("data-motion-state", "fallback");
    if (observer) observer.disconnect();
    animation.destroy();
  }

  animation.addEventListener("data_failed", restoreFallback);
  animation.addEventListener("error", restoreFallback);
  animation.addEventListener("DOMLoaded", function () {
    stage.setAttribute("data-motion-state", "ready");

    if (!("IntersectionObserver" in window)) {
      animation.play();
      return;
    }

    observer = new IntersectionObserver(function (entries) {
      if (entries.some(function (entry) {
        return entry.isIntersecting && entry.intersectionRatio >= 0.45;
      })) {
        animation.play();
        observer.disconnect();
      }
    }, { threshold: 0.45 });
    observer.observe(stage);
  });
}());
