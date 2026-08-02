(function () {
  "use strict";

  var buttons = document.querySelectorAll("[data-view]");
  var rows = document.querySelectorAll("[data-panel]");

  function setView(view) {
    document.body.dataset.view = view;
    buttons.forEach(function (button) {
      var active = button.dataset.view === view;
      button.classList.toggle("is-active", active);
      button.setAttribute("aria-pressed", String(active));
    });

    rows.forEach(function (row) {
      row.hidden = false;
    });
  }

  buttons.forEach(function (button) {
    button.addEventListener("click", function () {
      setView(button.dataset.view);
    });
  });

  setView("prediction");

  var year = document.getElementById("year");
  if (year) year.textContent = String(new Date().getFullYear());
}());
