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
  var fragmentRestoreCancelled = false;
  var fragmentScrollKeys = ["ArrowDown", "ArrowUp", "PageDown", "PageUp", "Home", "End", " ", "Spacebar"];

  function cancelFragmentRestore(event) {
    if (event.type === "keydown" && fragmentScrollKeys.indexOf(event.key) === -1) return;
    fragmentRestoreCancelled = true;
  }

  function allowFragmentRestore() {
    fragmentRestoreCancelled = false;
  }

  function stopTrackingFragmentRestore() {
    window.removeEventListener("wheel", cancelFragmentRestore);
    window.removeEventListener("touchmove", cancelFragmentRestore);
    window.removeEventListener("pointerdown", cancelFragmentRestore);
    window.removeEventListener("keydown", cancelFragmentRestore);
    window.removeEventListener("hashchange", allowFragmentRestore);
  }

  window.addEventListener("wheel", cancelFragmentRestore, { passive: true });
  window.addEventListener("touchmove", cancelFragmentRestore, { passive: true });
  window.addEventListener("pointerdown", cancelFragmentRestore);
  window.addEventListener("keydown", cancelFragmentRestore);
  window.addEventListener("hashchange", allowFragmentRestore);

  function restoreFragmentAfterLayoutChange() {
    stopTrackingFragmentRestore();
    if (fragmentRestoreCancelled || !window.location.hash) return;

    var id;
    try {
      id = decodeURIComponent(window.location.hash.slice(1));
    } catch (error) {
      return;
    }

    var target = document.getElementById(id);
    if (!target) return;
    var scrollTarget = target.tagName === "SECTION" ? target.querySelector(".numbered-heading") || target : target;

    window.requestAnimationFrame(function () {
      var root = document.documentElement;
      var previousScrollBehavior = root.style.scrollBehavior;
      root.style.scrollBehavior = "auto";
      scrollTarget.scrollIntoView({ block: "start" });
      root.style.scrollBehavior = previousScrollBehavior;
    });
  }

  function restoreFallback() {
    stopTrackingFragmentRestore();
    stage.setAttribute("data-motion-state", "fallback");
    if (observer) observer.disconnect();
    animation.destroy();
  }

  animation.addEventListener("data_failed", restoreFallback);
  animation.addEventListener("error", restoreFallback);
  animation.addEventListener("DOMLoaded", function () {
    stage.setAttribute("data-motion-state", "ready");
    restoreFragmentAfterLayoutChange();

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
