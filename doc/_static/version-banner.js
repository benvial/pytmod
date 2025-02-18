const wrapper = document.querySelector(".version-banner");
const close = document.querySelector(".version-banner-close");

if (wrapper) {
  const style = document.createElement("style");
  document.head.appendChild(style);

  function resize() {
    style.textContent = `:root{--sy-s-banner-height:${wrapper.clientHeight}px}`;
  }

  close.addEventListener("click", () => {
    wrapper.parentNode.removeChild(wrapper);
    document.head.removeChild(style);
  });

  resize();
  window.addEventListener("resize", resize);
}
