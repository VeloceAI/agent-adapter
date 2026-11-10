// Mermaid is 1.15 MB gzipped, so it loads only when a page first renders a diagram. This stand-in
// keeps the theme from fetching Mermaid off a CDN and keeps diagrams.js's render wrapper in the
// call path: the real Mermaid stays private and window.mermaid stays this object.
(function () {
  "use strict";
  const src = new URL("vendor/mermaid-11.17.2.min.js", document.currentScript.src).href;
  let config = null;
  let loading = null;

  function load() {
    loading = loading || new Promise((resolve, reject) => {
      const script = document.createElement("script");
      script.src = src;
      script.onload = () => {
        const real = window.mermaid;
        window.mermaid = stub;
        if (config) real.initialize(config);
        resolve(real);
      };
      script.onerror = () => {
        loading = null;
        reject(new Error("Agent Adapter docs: could not load " + src));
      };
      document.head.append(script);
    });
    return loading;
  }

  const stub = {
    initialize(options) {
      config = options;
    },
    render(...args) {
      return load().then((real) => real.render(...args));
    },
  };
  window.mermaid = stub;
})();
