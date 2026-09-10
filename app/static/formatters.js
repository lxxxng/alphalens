"use strict";

(() => {
  // Fiscal periods remain compact in storage and API filters. This formatter
  // makes the distinction from calendar dates explicit wherever users see one.
  function formatFiscalPeriod(value) {
    const match = String(value || "").trim().match(/^(\d{4})Q([1-4])$/i);

    if (!match) {
      return value || "Period unavailable";
    }

    return `FY${match[1]} Q${match[2]}`;
  }

  window.AlphaLensFormatters = Object.freeze({
    fiscalPeriod: formatFiscalPeriod,
  });
})();
