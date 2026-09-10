const form = document.querySelector("#research-form");
const answer = document.querySelector("#answer");
const sources = document.querySelector("#sources");
const sourceCount = document.querySelector("#source-count");
const marketPanel = document.querySelector("#market-panel");
const marketSnapshots = document.querySelector("#market-snapshots");
const marketCount = document.querySelector("#market-count");
const statusPill = document.querySelector("#status-pill");
const resultTitle = document.querySelector("#result-title");
const submitButton = document.querySelector("#submit-button");
const previewButton = document.querySelector("#preview-button");
const copyButton = document.querySelector("#copy-button");
const sampleButton = document.querySelector("#sample-button");
const questionInput = document.querySelector("#question");
const tickerPicker = document.querySelector("#ticker-picker");
const tickerPickerButton = document.querySelector("#ticker-picker-button");
const tickerPickerLabel = document.querySelector("#ticker-picker-label");
const tickerMenu = document.querySelector("#ticker-menu");
const tickerOptions = document.querySelector("#ticker-options");
const tickerChips = document.querySelector("#ticker-chips");
const tickerAutoInput = document.querySelector("#ticker-auto");
const fiscalPeriodSelect = document.querySelector("#fiscal-period");
const formTypeSelect = document.querySelector("#form-type");
const sectionKeySelect = document.querySelector("#section-key");
const openAIHealthButton = document.querySelector("#openai-health");
const openAIHealthLabel = document.querySelector("#openai-health-label");
const priceChart = document.querySelector("#price-chart");
const priceChartSvg = document.querySelector("#price-chart-svg");
const chartTooltip = document.querySelector("#chart-tooltip");
const chartEmpty = document.querySelector("#chart-empty");
const chartTicker = document.querySelector("#chart-ticker");
const chartPeriodReturn = document.querySelector("#chart-period-return");
const chartDateRange = document.querySelector("#chart-date-range");
const chartLegend = document.querySelector("#chart-legend");
const chartPeriodButtons = document.querySelectorAll("[data-period]");
const chartEventButtons = document.querySelectorAll("[data-event-filter]");
const chartCompanyLabel = document.querySelector("#chart-company-label");
const chartCompanyReturn = document.querySelector("#chart-company-return");
const chartBenchmarkLabel = document.querySelector("#chart-benchmark-label");
const chartBenchmarkReturn = document.querySelector("#chart-benchmark-return");
const chartRelativeReturn = document.querySelector("#chart-relative-return");
const chartEventCount = document.querySelector("#chart-event-count");
const marketEvents = document.querySelector("#market-events");
const researchOutput = document.querySelector(".research-output");
const historyList = document.querySelector("#history-list");
const historyRefreshButton = document.querySelector("#history-refresh");

// A few grounded examples make the UI useful immediately after startup.
// They also double as quick manual smoke tests for each retrieval mode.
const samples = [
  {
    question: "What did Walmart management say about margins on the earnings call?",
    ticker: "WMT",
    source_type: "transcripts",
  },
  {
    question: "What cybersecurity risks does NVIDIA face?",
    ticker: "NVDA",
    source_type: "filings",
    form_type: "10-K",
  },
  {
    question: "How has NVIDIA stock performed over the last year versus SPY?",
    ticker: "NVDA",
    source_type: "auto",
  },
  {
    question: "Compare Walmart and Costco margin commentary across filings and earnings calls.",
    ticker: "WMT",
    tickers: ["WMT", "COST"],
    source_type: "both",
  },
];

let sampleIndex = 0;
let metadataReady = false;
// Selection order is intentional: the first ticker drives metadata filters
// and headline KPIs while every selected ticker participates in comparison.
let availableTickers = [];
const initialTickerParameter = new URLSearchParams(
  window.location.search
).get("tickers");
const initialTickerMode = new URLSearchParams(
  window.location.search
).get("ticker_mode");
let selectedTickers = (
  initialTickerParameter
    ?.split(",")
    .map((ticker) => ticker.trim().toUpperCase())
    .filter(Boolean)
    .slice(0, 4)
) || ["WMT"];
let tickerResolutionTimer = null;
let tickerResolutionRequest = 0;
let selectedChartPeriod = "1Y";
let selectedEventFilter = "all";
let marketChartData = null;
let marketChartRequest = 0;

const SVG_NAMESPACE = "http://www.w3.org/2000/svg";
const COMPANY_CHART_COLORS = ["#087f5b", "#6d4aff", "#0f8ea8", "#c24156"];
const BENCHMARK_CHART_COLOR = "#d97706";
const EVENT_COLORS = {
  earnings: "#7c3aed",
  filing: "#2563eb",
};

function chartColor(ticker, fallbackIndex = 0) {
  if (ticker === marketChartData?.benchmark_ticker) {
    return BENCHMARK_CHART_COLOR;
  }

  const companyIndex = (marketChartData?.tickers || []).indexOf(ticker);
  const colorIndex = companyIndex >= 0 ? companyIndex : fallbackIndex;
  return COMPANY_CHART_COLORS[colorIndex % COMPANY_CHART_COLORS.length];
}

function setStatus(label, state = "") {
  statusPill.textContent = label;
  statusPill.className = `status-pill ${state}`.trim();
}

function setBusy(isBusy) {
  submitButton.disabled = isBusy;
  previewButton.disabled = isBusy;
}

function renderOpenAIHealth(data) {
  const status = data.status || "error";
  const labels = {
    ok: "OpenAI: Connected",
    not_configured: "OpenAI: Not configured",
    error: "OpenAI: Unavailable",
  };

  openAIHealthButton.className = `connection-status ${status}`;
  openAIHealthLabel.textContent = labels[status] || labels.error;

  const latency = data.latency_ms !== undefined
    ? ` (${data.latency_ms} ms)`
    : "";
  openAIHealthButton.title = `${data.message || "Connection check failed."}${latency}`;
}

async function checkOpenAIHealth() {
  openAIHealthButton.disabled = true;
  openAIHealthButton.className = "connection-status checking";
  openAIHealthLabel.textContent = "OpenAI: Checking";

  try {
    const response = await fetch("/api/health/openai");
    const data = await response.json();
    renderOpenAIHealth(data);
  } catch (error) {
    renderOpenAIHealth({
      status: "error",
      message: "The AlphaLens API could not run the connection check.",
    });
  } finally {
    openAIHealthButton.disabled = false;
  }
}

function formatHistoryDate(value) {
  return new Intl.DateTimeFormat("en-US", {
    month: "short",
    day: "numeric",
    hour: "numeric",
    minute: "2-digit",
  }).format(new Date(value));
}

function historyMetaItem(text, className = "") {
  const item = document.createElement("span");
  item.className = className;
  item.textContent = text;
  return item;
}

async function deleteSavedRun(runId) {
  const confirmed = window.confirm(
    "Delete this saved research run?"
  );

  if (!confirmed) {
    return;
  }

  try {
    const response = await fetch(
      `/api/research/history/${runId}`,
      { method: "DELETE" }
    );
    const data = await response.json();

    if (!response.ok) {
      throw new Error(data.detail || "Could not delete saved research.");
    }

    await loadResearchHistory();
    setStatus("Deleted");
  } catch (error) {
    setStatus("Delete failed", "error");
  }
}

function renderResearchHistory(items) {
  historyList.replaceChildren();

  if (!items.length) {
    const empty = document.createElement("p");
    empty.className = "history-empty";
    empty.textContent = "No saved research yet.";
    historyList.appendChild(empty);
    return;
  }

  for (const item of items) {
    const row = document.createElement("div");
    row.className = "history-item";

    const openButton = document.createElement("button");
    openButton.type = "button";
    openButton.className = "history-open";
    openButton.title = `Open saved run ${item.run_id}`;

    const meta = document.createElement("div");
    meta.className = "history-meta";
    meta.append(
      historyMetaItem(
        item.tickers?.join(", ") || "Research",
        "history-ticker"
      ),
      historyMetaItem(item.source_type),
      historyMetaItem(formatHistoryDate(item.created_at))
    );

    const question = document.createElement("span");
    question.className = "history-question";
    question.textContent = item.question;
    openButton.append(meta, question);
    openButton.addEventListener("click", () => {
      openSavedResearchRun(item.run_id);
    });

    const deleteButton = document.createElement("button");
    deleteButton.type = "button";
    deleteButton.className = "history-delete";
    deleteButton.textContent = "Delete";
    deleteButton.title = `Delete saved run ${item.run_id}`;
    deleteButton.addEventListener("click", () => {
      deleteSavedRun(item.run_id);
    });

    row.append(openButton, deleteButton);
    historyList.appendChild(row);
  }
}

async function loadResearchHistory() {
  historyRefreshButton.disabled = true;

  try {
    const response = await fetch("/api/research/history?limit=20");
    const data = await response.json();

    if (!response.ok) {
      throw new Error(data.detail || "Could not load research history.");
    }

    renderResearchHistory(data.runs || []);
  } catch (error) {
    historyList.replaceChildren();
    const message = document.createElement("p");
    message.className = "history-empty";
    message.textContent = "Research history is unavailable.";
    historyList.appendChild(message);
  } finally {
    historyRefreshButton.disabled = false;
  }
}

async function restoreSavedFilters(run) {
  form.elements.question.value = run.question;
  form.elements.top_k.value = run.top_k;
  setSourceType(run.source_type);

  const restoredTickers = (run.tickers || [])
    .filter((ticker) => ticker !== "SPY")
    .slice(0, 4);
  const tickers = restoredTickers.length
    ? restoredTickers
    : [run.ticker_filter].filter(Boolean);
  tickerAutoInput.checked = false;
  selectedTickers = tickers;
  syncTickerUrl();
  renderTickerPicker();
  const ticker = primaryTicker();

  if (metadataReady) {
    await loadFiltersForTicker(ticker);
  }

  setSelectValue(fiscalPeriodSelect, run.fiscal_period);
  setSelectValue(formTypeSelect, run.form_type);

  if (metadataReady) {
    await loadFilingSections(
      ticker,
      run.form_type || ""
    );
  }

  setSelectValue(sectionKeySelect, run.section_key);
  updateFilterState();
  await loadMarketChart();
}

async function openSavedResearchRun(runId) {
  setStatus("Loading", "loading");

  try {
    const response = await fetch(`/api/research/history/${runId}`);
    const data = await response.json();

    if (!response.ok) {
      throw new Error(data.detail || "Could not open saved research.");
    }

    await restoreSavedFilters(data);
    resultTitle.textContent = `Saved Run #${data.run_id}`;
    answer.textContent = data.answer;
    renderMarketContext(data.market_context || []);
    renderSources(data.sources || []);
    setStatus("History");

    if (window.innerWidth <= 900) {
      researchOutput.scrollIntoView({
        behavior: "smooth",
        block: "start",
      });
    }
  } catch (error) {
    setStatus("History error", "error");
  }
}

function sourceLabel(source) {
  if (source.source_type === "transcript") {
    return [
      source.ticker,
      source.fiscal_period,
      source.call_date,
    ].filter(Boolean).join(" | ");
  }

  return [
    source.ticker,
    source.form_type,
    source.filing_date,
    source.section_title,
  ].filter(Boolean).join(" | ");
}

function sourceDetail(source) {
  if (source.source_type === "transcript") {
    const speakers = source.speaker_names?.length
      ? `Speakers: ${source.speaker_names.join(", ")}`
      : "Speakers unavailable";

    return `${source.title || "Earnings call"} | ${speakers}`;
  }

  return [
    source.accession_number,
    source.section_key,
  ].filter(Boolean).join(" | ");
}

function sourceExcerpt(source) {
  return source.content || "No source text returned.";
}

function clearOptions(select, emptyLabel) {
  select.replaceChildren();

  const option = document.createElement("option");
  option.value = "";
  option.textContent = emptyLabel;
  select.appendChild(option);
}

function appendOption(select, value, label) {
  const option = document.createElement("option");
  option.value = value;
  option.textContent = label;
  select.appendChild(option);
}

function setSelectValue(select, value) {
  if (!value) {
    select.value = "";
    return;
  }

  const exists = [...select.options].some(
    (option) => option.value === value
  );

  if (exists) {
    select.value = value;
  }
}

function primaryTicker() {
  return selectedTickers[0] || "";
}

function setTickerMenuOpen(isOpen) {
  tickerMenu.hidden = !isOpen;
  tickerPickerButton.setAttribute("aria-expanded", String(isOpen));
}

function syncTickerUrl() {
  const url = new URL(window.location.href);

  if (selectedTickers.length) {
    url.searchParams.set("tickers", selectedTickers.join(","));
  } else {
    url.searchParams.delete("tickers");
  }

  if (tickerAutoInput.checked) {
    url.searchParams.set("ticker_mode", "auto");
  } else {
    url.searchParams.delete("ticker_mode");
  }

  window.history.replaceState({}, "", url);
}

async function applyTickerSelection(
  values,
  { manual = false, refresh = true } = {}
) {
  const previousPrimary = primaryTicker();
  const nextTickers = [...new Set(
    values
      .map((value) => value.trim().toUpperCase())
      .filter(Boolean)
  )].slice(0, 4);
  const changed = nextTickers.join(",") !== selectedTickers.join(",");

  if (manual) {
    tickerAutoInput.checked = false;
  }

  selectedTickers = nextTickers;
  syncTickerUrl();
  renderTickerPicker();

  if (changed && refresh && metadataReady) {
    if (previousPrimary !== primaryTicker()) {
      await loadFiltersForTicker(primaryTicker());
    }
    await loadMarketChart();
  }
}

async function resolveTickersFromQuestion({ refresh = true } = {}) {
  if (!tickerAutoInput.checked) {
    return selectedTickers;
  }

  const question = questionInput.value.trim();
  const requestId = ++tickerResolutionRequest;

  if (!question) {
    await applyTickerSelection([], { refresh });
    return [];
  }

  try {
    const params = new URLSearchParams({ question });
    const data = await fetchJson(
      `/api/metadata/resolve-tickers?${params}`
    );

    if (
      requestId !== tickerResolutionRequest
      || !tickerAutoInput.checked
      || question !== questionInput.value.trim()
    ) {
      return selectedTickers;
    }

    await applyTickerSelection(data.tickers || [], { refresh });
    return selectedTickers;
  } catch (error) {
    // Research requests still omit explicit tickers in Auto mode, allowing
    // the backend resolver to remain the authoritative fallback.
    return selectedTickers;
  }
}

function renderTickerPicker() {
  tickerPickerLabel.textContent = selectedTickers.length
    ? selectedTickers.join(", ")
    : "Auto-detect from question";
  tickerChips.replaceChildren();
  tickerOptions.replaceChildren();

  for (const ticker of selectedTickers) {
    const chip = document.createElement("span");
    chip.className = "ticker-chip";
    const label = document.createElement("strong");
    label.textContent = ticker;
    const remove = document.createElement("button");
    remove.type = "button";
    remove.textContent = "x";
    remove.title = `Remove ${ticker}`;
    remove.setAttribute("aria-label", `Remove ${ticker}`);
    remove.addEventListener("click", async () => {
      await applyTickerSelection(
        selectedTickers.filter((value) => value !== ticker),
        { manual: true }
      );
    });
    chip.append(label, remove);
    tickerChips.appendChild(chip);
  }

  for (const item of availableTickers) {
    const option = document.createElement("label");
    option.className = "ticker-option";
    option.setAttribute("role", "option");
    const input = document.createElement("input");
    input.type = "checkbox";
    input.value = item.ticker;
    input.checked = selectedTickers.includes(item.ticker);
    input.disabled = !input.checked && selectedTickers.length >= 4;
    option.setAttribute("aria-selected", String(input.checked));

    const identity = document.createElement("span");
    const ticker = document.createElement("strong");
    ticker.textContent = item.ticker;
    const company = document.createElement("small");
    company.textContent = item.company_name || item.ticker;
    identity.append(ticker, company);

    input.addEventListener("change", async () => {
      const next = input.checked
        ? [...selectedTickers, item.ticker]
        : selectedTickers.filter((value) => value !== item.ticker);
      await applyTickerSelection(next, { manual: true });
    });
    option.append(input, identity);
    tickerOptions.appendChild(option);
  }
}

function selectedSourceType() {
  return formDataSourceType(
    new FormData(form)
  );
}

function formDataSourceType(formData) {
  return formData.get("source_type") || "auto";
}

function updateFilterState() {
  const sourceType = selectedSourceType();
  const filingOnly = sourceType === "filings";
  const transcriptOnly = sourceType === "transcripts";

  fiscalPeriodSelect.disabled = filingOnly;
  formTypeSelect.disabled = transcriptOnly;
  sectionKeySelect.disabled = transcriptOnly;
}

async function fetchJson(url) {
  const response = await fetch(url);

  if (!response.ok) {
    throw new Error(`Metadata request failed: ${url}`);
  }

  return response.json();
}

async function loadTickers() {
  const data = await fetchJson("/api/metadata/tickers");
  availableTickers = data.tickers || [];
  const availableSymbols = new Set(
    availableTickers.map((item) => item.ticker)
  );
  selectedTickers = selectedTickers.filter(
    (ticker) => availableSymbols.has(ticker)
  );

  if (!selectedTickers.length && availableSymbols.has("WMT")) {
    selectedTickers = ["WMT"];
  }

  syncTickerUrl();
  renderTickerPicker();
}

async function loadTranscriptPeriods(ticker) {
  clearOptions(fiscalPeriodSelect, "Latest");

  if (!ticker) {
    return;
  }

  const params = new URLSearchParams({
    ticker,
  });

  const data = await fetchJson(
    `/api/metadata/transcript-periods?${params}`
  );

  for (const item of data.periods || []) {
    const label = [
      item.fiscal_period,
      item.call_date,
    ].filter(Boolean).join(" - ");

    appendOption(
      fiscalPeriodSelect,
      item.fiscal_period,
      label
    );
  }
}

async function loadFilingTypes(ticker) {
  const previousValue = formTypeSelect.value;
  clearOptions(formTypeSelect, "Any");

  const params = new URLSearchParams();

  if (ticker) {
    params.set("ticker", ticker);
  }

  const data = await fetchJson(
    `/api/metadata/filing-types?${params}`
  );

  for (const item of data.form_types || []) {
    appendOption(
      formTypeSelect,
      item.form_type,
      `${item.form_type} (${item.filing_count})`
    );
  }

  setSelectValue(
    formTypeSelect,
    previousValue
  );
}

async function loadFilingSections(ticker, formType) {
  const previousValue = sectionKeySelect.value;
  clearOptions(sectionKeySelect, "Any");

  const params = new URLSearchParams();

  if (ticker) {
    params.set("ticker", ticker);
  }

  if (formType) {
    params.set("form_type", formType);
  }

  const data = await fetchJson(
    `/api/metadata/filing-sections?${params}`
  );

  for (const item of data.sections || []) {
    appendOption(
      sectionKeySelect,
      item.section_key,
      `${item.section_title || item.section_key} (${item.chunk_count})`
    );
  }

  setSelectValue(
    sectionKeySelect,
    previousValue
  );
}

async function loadFiltersForTicker(ticker) {
  setStatus("Loading");

  await Promise.all([
    loadTranscriptPeriods(ticker),
    loadFilingTypes(ticker),
  ]);

  await loadFilingSections(
    ticker,
    formTypeSelect.value
  );

  updateFilterState();
  setStatus("Idle");
}

async function initializeMetadata() {
  try {
    await loadTickers();

    if (tickerAutoInput.checked) {
      await resolveTickersFromQuestion({ refresh: false });
    }

    await loadFiltersForTicker(primaryTicker());
    metadataReady = true;
  } catch (error) {
    // If metadata cannot load, the hard-coded fallback options still let
    // the page submit normal research requests.
    setStatus("Metadata unavailable", "error");
  }

  await loadMarketChart();
}

function formatPercent(value) {
  if (value === null || value === undefined) {
    return "N/A";
  }

  return `${(value * 100).toFixed(2)}%`;
}

function formatNumber(value, maximumFractionDigits = 2) {
  if (value === null || value === undefined) {
    return "N/A";
  }

  return new Intl.NumberFormat("en-US", {
    maximumFractionDigits,
  }).format(value);
}

function svgElement(name, attributes = {}) {
  const element = document.createElementNS(SVG_NAMESPACE, name);

  for (const [key, value] of Object.entries(attributes)) {
    element.setAttribute(key, value);
  }

  return element;
}

function chartLinePath(points, xScale, yScale) {
  return points.map((point, index) => {
    const command = index === 0 ? "M" : "L";
    const x = xScale(point.date).toFixed(2);
    const y = yScale(point.indexed_value - 100).toFixed(2);
    return `${command}${x},${y}`;
  }).join(" ");
}

function formatReturn(value, fractionDigits = 1) {
  if (!Number.isFinite(value)) {
    return "--";
  }

  const percentage = value * 100;
  const sign = percentage > 0 ? "+" : "";
  return `${sign}${percentage.toFixed(fractionDigits)}%`;
}

function setReturnClass(element, value) {
  element.className = Number.isFinite(value)
    ? (value >= 0 ? "positive" : "negative")
    : "";
}

function visibleMarketEvents() {
  const events = marketChartData?.events || [];

  if (selectedEventFilter === "all") {
    return events;
  }

  return events.filter(
    (event) => event.event_type === selectedEventFilter
  );
}

function formatChartDate(value) {
  return new Intl.DateTimeFormat("en-US", {
    month: "short",
    day: "numeric",
    year: "numeric",
    timeZone: "UTC",
  }).format(new Date(`${value}T00:00:00Z`));
}

function setChartLoading(message) {
  priceChartSvg.replaceChildren();
  chartLegend.replaceChildren();
  marketEvents.replaceChildren();
  chartTooltip.hidden = true;
  chartEmpty.hidden = false;
  chartEmpty.textContent = message;
  chartCompanyReturn.textContent = "--";
  chartBenchmarkReturn.textContent = "--";
  chartRelativeReturn.textContent = "--";
  chartEventCount.textContent = "--";
}

function renderChartLegend(series, events) {
  chartLegend.replaceChildren();

  series.forEach((item, index) => {
    const legendItem = document.createElement("span");
    const swatch = document.createElement("i");
    swatch.style.backgroundColor = chartColor(item.ticker, index);

    const label = document.createElement("strong");
    const first = item.points[0];
    const last = item.points.at(-1);
    const periodReturn = last.indexed_value / first.indexed_value - 1;
    label.textContent = `${item.ticker} ${formatReturn(periodReturn)}`;

    legendItem.append(swatch, label);
    chartLegend.appendChild(legendItem);
  });

  for (const [eventType, label] of [
    ["earnings", "Earnings call"],
    ["filing", "SEC filing"],
  ]) {
    if (!events.some((event) => event.event_type === eventType)) {
      continue;
    }

    const legendItem = document.createElement("span");
    legendItem.className = "event-legend-item";
    const marker = document.createElement("b");
    marker.style.backgroundColor = EVENT_COLORS[eventType];
    marker.textContent = eventType === "earnings" ? "E" : "F";
    legendItem.append(marker, label);
    chartLegend.appendChild(legendItem);
  }

  const note = document.createElement("span");
  note.className = "chart-index-note";
  note.textContent = "Adjusted close";
  chartLegend.appendChild(note);
}

function closestChartPoint(points, targetTime) {
  return points.reduce((best, point) => {
    const pointTime = Date.parse(`${point.date}T00:00:00Z`);
    const distance = Math.abs(pointTime - targetTime);
    return distance < best.distance ? { point, distance } : best;
  }, { point: points[0], distance: Infinity }).point;
}

function renderChartTooltip(date, rows, left, top) {
  const dateLabel = document.createElement("strong");
  dateLabel.textContent = formatChartDate(date);
  chartTooltip.replaceChildren(dateLabel);

  rows.forEach((row, index) => {
    const item = document.createElement("span");
    const swatch = document.createElement("i");
    swatch.style.backgroundColor = chartColor(row.ticker, index);
    item.append(
      swatch,
      `${row.ticker} ${formatReturn((row.point.indexed_value - 100) / 100)} · $${formatNumber(row.point.close)}`
    );
    chartTooltip.appendChild(item);
  });

  chartTooltip.hidden = false;
  chartTooltip.style.left = `${left}px`;
  chartTooltip.style.top = `${top}px`;
}

function renderEventTooltip(event, left, top) {
  const title = document.createElement("strong");
  title.textContent = `${event.ticker} · ${event.event_type === "earnings" ? "Earnings" : "SEC filing"} · ${event.label}`;
  const date = document.createElement("span");
  date.textContent = formatChartDate(event.date);
  const nextSession = document.createElement("span");
  nextSession.textContent = `Next session ${formatReturn(event.reaction_1d)}`;
  const fiveSessions = document.createElement("span");
  fiveSessions.textContent = `Five sessions ${formatReturn(event.reaction_5d)}`;
  chartTooltip.replaceChildren(title, date, nextSession, fiveSessions);
  chartTooltip.hidden = false;
  chartTooltip.style.left = `${left}px`;
  chartTooltip.style.top = `${top}px`;
}

function renderMarketEvents(events) {
  marketEvents.replaceChildren();

  if (!events.length) {
    const empty = document.createElement("p");
    empty.className = "market-events-empty";
    empty.textContent = "No earnings calls or SEC filings in this period.";
    marketEvents.appendChild(empty);
    return;
  }

  for (const event of [...events].reverse().slice(0, 8)) {
    const row = document.createElement(event.source_url ? "a" : "div");
    row.className = `market-event-row ${event.event_type}`;

    if (event.source_url) {
      row.href = event.source_url;
      row.target = "_blank";
      row.rel = "noreferrer";
    }

    const marker = document.createElement("span");
    marker.className = "market-event-icon";
    marker.textContent = event.event_type === "earnings" ? "E" : "F";

    const description = document.createElement("span");
    const title = document.createElement("strong");
    title.textContent = `${event.ticker} · ${event.label}`;
    const date = document.createElement("small");
    date.textContent = formatChartDate(event.date);
    description.append(title, date);

    const oneDay = document.createElement("span");
    oneDay.className = "event-reaction";
    oneDay.innerHTML = `<small>Next session</small><strong>${formatReturn(event.reaction_1d)}</strong>`;
    setReturnClass(oneDay.querySelector("strong"), event.reaction_1d);

    const fiveDay = document.createElement("span");
    fiveDay.className = "event-reaction";
    fiveDay.innerHTML = `<small>5 sessions</small><strong>${formatReturn(event.reaction_5d)}</strong>`;
    setReturnClass(fiveDay.querySelector("strong"), event.reaction_5d);

    row.append(marker, description, oneDay, fiveDay);
    marketEvents.appendChild(row);
  }
}

function renderMarketChart() {
  if (!marketChartData?.series?.length) {
    setChartLoading("No market history available.");
    return;
  }

  const width = Math.max(priceChart.clientWidth, 320);
  const height = Math.max(priceChart.clientHeight, 240);
  const margin = { top: 32, right: 24, bottom: 38, left: 58 };
  const plotWidth = width - margin.left - margin.right;
  const plotHeight = height - margin.top - margin.bottom;
  const allPoints = marketChartData.series.flatMap((item) => item.points);
  const timestamps = allPoints.map(
    (point) => Date.parse(`${point.date}T00:00:00Z`)
  );
  const values = allPoints.map((point) => point.indexed_value - 100);
  const minTime = Math.min(...timestamps);
  const maxTime = Math.max(...timestamps);
  const rawMin = Math.min(0, ...values);
  const rawMax = Math.max(0, ...values);
  const valuePadding = Math.max((rawMax - rawMin) * 0.08, 1);
  const minValue = rawMin - valuePadding;
  const maxValue = rawMax + valuePadding;

  const xScale = (date) => {
    const time = Date.parse(`${date}T00:00:00Z`);
    const ratio = maxTime === minTime
      ? 0
      : (time - minTime) / (maxTime - minTime);
    return margin.left + ratio * plotWidth;
  };
  const yScale = (value) => (
    margin.top
    + (maxValue - value) / (maxValue - minValue) * plotHeight
  );

  priceChartSvg.replaceChildren();
  priceChartSvg.setAttribute("viewBox", `0 0 ${width} ${height}`);
  chartEmpty.hidden = true;

  // Percentage-return labels avoid making the start-at-100 comparison look
  // like a dollar-price axis.
  for (let index = 0; index < 5; index += 1) {
    const ratio = index / 4;
    const value = maxValue - ratio * (maxValue - minValue);
    const y = margin.top + ratio * plotHeight;

    priceChartSvg.appendChild(svgElement("line", {
      x1: margin.left,
      x2: width - margin.right,
      y1: y,
      y2: y,
      class: "chart-grid-line",
    }));

    const label = svgElement("text", {
      x: margin.left - 9,
      y: y + 4,
      class: "chart-axis-label",
      "text-anchor": "end",
    });
    label.textContent = `${value > 0 ? "+" : ""}${value.toFixed(0)}%`;
    priceChartSvg.appendChild(label);
  }

  const primaryPoints = marketChartData.series[0].points;
  const dateLabels = [0, 0.25, 0.5, 0.75, 1].map(
    (ratio) => primaryPoints[
      Math.round(ratio * (primaryPoints.length - 1))
    ].date
  );

  dateLabels.forEach(
    (date, index) => {
      const label = svgElement("text", {
        x: xScale(date),
        y: height - 9,
        class: "chart-axis-label",
        "text-anchor": (
          index === 0
            ? "start"
            : (index === dateLabels.length - 1 ? "end" : "middle")
        ),
      });
      label.textContent = formatChartDate(date);
      priceChartSvg.appendChild(label);
    }
  );

  if (minValue <= 0 && maxValue >= 0) {
    priceChartSvg.appendChild(svgElement("line", {
      x1: margin.left,
      x2: width - margin.right,
      y1: yScale(0),
      y2: yScale(0),
      class: "chart-zero-line",
    }));
  }

  marketChartData.series.forEach((item, index) => {
    priceChartSvg.appendChild(svgElement("path", {
      d: chartLinePath(item.points, xScale, yScale),
      class: "chart-line",
      stroke: chartColor(item.ticker, index),
    }));
  });

  const hoverLine = svgElement("line", {
    class: "chart-hover-line",
    y1: margin.top,
    y2: height - margin.bottom,
  });
  hoverLine.style.visibility = "hidden";
  priceChartSvg.appendChild(hoverLine);

  const interaction = svgElement("rect", {
    x: margin.left,
    y: margin.top,
    width: plotWidth,
    height: plotHeight,
    class: "chart-interaction",
  });

  interaction.addEventListener("pointermove", (event) => {
    const bounds = priceChartSvg.getBoundingClientRect();
    const svgX = (event.clientX - bounds.left) * width / bounds.width;
    const ratio = Math.min(
      1,
      Math.max(0, (svgX - margin.left) / plotWidth)
    );
    const targetTime = minTime + ratio * (maxTime - minTime);
    const primaryPoint = closestChartPoint(
      marketChartData.series[0].points,
      targetTime
    );
    const primaryTime = Date.parse(`${primaryPoint.date}T00:00:00Z`);
    const lineX = xScale(primaryPoint.date);
    const tooltipRows = marketChartData.series.map((item) => ({
      ticker: item.ticker,
      point: closestChartPoint(item.points, primaryTime),
    }));

    hoverLine.setAttribute("x1", lineX);
    hoverLine.setAttribute("x2", lineX);
    hoverLine.style.visibility = "visible";

    renderChartTooltip(
      primaryPoint.date,
      tooltipRows,
      Math.max(8, Math.min(event.clientX - bounds.left + 12, bounds.width - 183)),
      Math.max(event.clientY - bounds.top - 70, 8)
    );
  });

  interaction.addEventListener("pointerleave", () => {
    hoverLine.style.visibility = "hidden";
    chartTooltip.hidden = true;
  });

  priceChartSvg.appendChild(interaction);

  const events = visibleMarketEvents().filter(
    (event) => event.plot_date
  );
  const eventStacks = new Map();

  for (const event of events) {
    const eventSeries = marketChartData.series.find(
      (series) => series.ticker === event.ticker
    ) || marketChartData.series[0];
    const point = closestChartPoint(
      eventSeries.points,
      Date.parse(`${event.plot_date}T00:00:00Z`)
    );
    const lineX = xScale(event.plot_date);
    const lineY = yScale(point.indexed_value - 100);
    const stackKey = `${event.plot_date}-${event.event_type}`;
    const stackIndex = eventStacks.get(stackKey) || 0;
    eventStacks.set(stackKey, stackIndex + 1);
    const direction = event.event_type === "earnings" ? -1 : 1;
    const markerY = Math.max(
      margin.top + 12,
      Math.min(
        height - margin.bottom - 12,
        lineY + direction * (18 + stackIndex * 21)
      )
    );

    priceChartSvg.appendChild(svgElement("line", {
      x1: lineX,
      x2: lineX,
      y1: lineY,
      y2: markerY,
      class: `event-stem ${event.event_type}`,
    }));

    const marker = svgElement("g", {
      class: `event-marker ${event.event_type}`,
      role: event.source_url ? "link" : "img",
      "aria-label": `${event.ticker} ${event.label} on ${event.date}`,
    });

    if (event.source_url) {
      marker.classList.add("linked");
      marker.setAttribute("tabindex", "0");
    }
    const markerCircle = svgElement("circle", {
      cx: lineX,
      cy: markerY,
      r: 10,
      fill: EVENT_COLORS[event.event_type],
    });
    markerCircle.style.stroke = chartColor(event.ticker);
    marker.appendChild(markerCircle);
    const glyph = svgElement("text", {
      x: lineX,
      y: markerY + 3.5,
      "text-anchor": "middle",
      class: "event-marker-glyph",
    });
    glyph.textContent = event.event_type === "earnings" ? "E" : "F";
    marker.appendChild(glyph);

    marker.addEventListener("pointerenter", () => {
      const bounds = priceChartSvg.getBoundingClientRect();
      renderEventTooltip(
        event,
        Math.max(8, Math.min(lineX / width * bounds.width + 12, bounds.width - 183)),
        Math.max(8, markerY / height * bounds.height - 86)
      );
    });
    marker.addEventListener("pointerleave", () => {
      chartTooltip.hidden = true;
    });

    // Event markers are real navigation controls, not just chart decoration.
    // Earnings open the local reader; filing events retain their SEC links.
    if (event.source_url) {
      const openEvent = () => {
        window.open(event.source_url, "_blank", "noopener,noreferrer");
      };

      marker.addEventListener("click", openEvent);
      marker.addEventListener("keydown", (keyboardEvent) => {
        if (keyboardEvent.key === "Enter" || keyboardEvent.key === " ") {
          keyboardEvent.preventDefault();
          openEvent();
        }
      });
    }

    priceChartSvg.appendChild(marker);
  }

  renderChartLegend(marketChartData.series, events);
  renderMarketEvents(events);
  chartEventCount.textContent = String(events.length);
}

function updateChartSummary(data) {
  const primarySeries = data.series[0];
  const first = primarySeries.points[0];
  const last = primarySeries.points[primarySeries.points.length - 1];
  const periodReturn = last.indexed_value / first.indexed_value - 1;
  const benchmarkSeries = data.series.find(
    (series) => series.ticker === data.benchmark_ticker
  );
  const benchmarkReturn = benchmarkSeries
    ? benchmarkSeries.points.at(-1).indexed_value / benchmarkSeries.points[0].indexed_value - 1
    : null;
  const relativeReturn = Number.isFinite(benchmarkReturn)
    ? periodReturn - benchmarkReturn
    : null;

  chartTicker.textContent = (data.tickers || [data.ticker]).join(" / ");
  chartPeriodReturn.textContent = `${data.ticker} ${formatPercent(periodReturn)} over ${data.period}`;
  chartPeriodReturn.className = periodReturn >= 0 ? "positive" : "negative";
  chartDateRange.textContent = `${formatChartDate(data.start_date)} - ${formatChartDate(data.end_date)}`;
  chartCompanyLabel.textContent = `${data.ticker} return`;
  chartCompanyReturn.textContent = formatReturn(periodReturn);
  chartBenchmarkLabel.textContent = `${data.benchmark_ticker} return`;
  chartBenchmarkReturn.textContent = formatReturn(benchmarkReturn);
  chartRelativeReturn.textContent = formatReturn(relativeReturn);
  setReturnClass(chartCompanyReturn, periodReturn);
  setReturnClass(chartBenchmarkReturn, benchmarkReturn);
  setReturnClass(chartRelativeReturn, relativeReturn);
}

async function loadMarketChart() {
  const ticker = primaryTicker();
  const requestId = ++marketChartRequest;

  if (!ticker) {
    marketChartData = null;
    setChartLoading("Select a ticker to load market data.");
    return;
  }

  chartTicker.textContent = selectedTickers.join(" / ");
  chartPeriodReturn.className = "";
  chartPeriodReturn.textContent = "Loading prices...";
  chartDateRange.textContent = "";
  setChartLoading("Loading market data...");

  try {
    const params = new URLSearchParams({
      ticker,
      period: selectedChartPeriod,
    });

    if (selectedTickers.length > 1) {
      params.set("tickers", selectedTickers.slice(1).join(","));
    }
    const response = await fetch(`/api/market/prices?${params}`);
    const data = await response.json();

    if (!response.ok) {
      throw new Error(data.detail || "Market history request failed.");
    }

    // Ignore a slower response when the user changes ticker or period while
    // the previous request is still in flight.
    if (requestId !== marketChartRequest) {
      return;
    }

    marketChartData = data;
    updateChartSummary(data);
    renderMarketChart();
  } catch (error) {
    if (requestId !== marketChartRequest) {
      return;
    }

    marketChartData = null;
    chartPeriodReturn.textContent = "Unavailable";
    setChartLoading(error.message);
  }
}

function metric(label, value) {
  const item = document.createElement("div");
  item.className = "metric";

  const labelElement = document.createElement("span");
  labelElement.textContent = label;

  const valueElement = document.createElement("strong");
  valueElement.textContent = value;

  item.append(labelElement, valueElement);
  return item;
}

function renderMarketContext(items) {
  marketSnapshots.replaceChildren();
  marketCount.textContent = String(items.length);
  marketPanel.hidden = items.length === 0;

  for (const item of items) {
    const card = document.createElement("article");
    card.className = "market-card";

    const title = document.createElement("div");
    title.className = "market-card-title";

    const ticker = document.createElement("h4");
    ticker.textContent = item.ticker;

    const date = document.createElement("span");
    date.textContent = item.latest_trading_date;

    title.append(ticker, date);

    const grid = document.createElement("div");
    grid.className = "metric-grid";

    const returns = item.returns || {};
    const benchmarkReturns = item.benchmark_returns || {};
    const relative = item.benchmark_relative_returns || {};

    grid.append(
      metric("Close", formatNumber(item.latest_close)),
      metric("1M", formatPercent(returns["1M"])),
      metric("3M", formatPercent(returns["3M"])),
      metric("1Y", formatPercent(returns["1Y"])),
      metric(`${item.benchmark_ticker} 1Y`, formatPercent(benchmarkReturns["1Y"])),
      metric(`1Y vs ${item.benchmark_ticker}`, formatPercent(relative["1Y"])),
      metric("Volatility", formatPercent(item.annualized_volatility)),
      metric("Avg Volume", formatNumber(item.average_volume_30d, 0)),
    );

    // Market snapshots are calculated facts, not generated prose. Showing
    // them separately lets users compare numbers without digging through
    // the natural-language answer.
    card.append(title, grid);
    marketSnapshots.appendChild(card);
  }
}

function renderSources(items) {
  sources.replaceChildren();
  sourceCount.textContent = String(items.length);

  for (const item of items) {
    // <details> gives us accessible expand/collapse behavior without
    // custom state management.
    const element = document.createElement("details");
    element.className = "source-item";

    const summary = document.createElement("summary");
    summary.className = "source-summary";

    const meta = document.createElement("div");
    meta.className = "source-meta";

    for (const value of [
      item.source,
      item.source_type,
      `chunk ${item.chunk_id}`,
      `score ${item.similarity_score}`,
    ]) {
      const tag = document.createElement("span");
      tag.className = "tag";
      tag.textContent = value;
      meta.appendChild(tag);
    }

    const summaryText = document.createElement("div");
    summaryText.className = "source-summary-text";

    const title = document.createElement("p");
    title.className = "source-title";
    title.textContent = sourceLabel(item);

    const detail = document.createElement("p");
    detail.className = "source-detail";
    detail.textContent = sourceDetail(item);

    summaryText.append(title, detail);
    summary.append(meta, summaryText);

    // The API returns the exact chunk sent to the answer generator.
    // Keeping it collapsed by default makes the page scan-friendly while
    // still making every citation auditable.
    const content = document.createElement("pre");
    content.className = "source-content";
    content.textContent = sourceExcerpt(item);

    const footer = document.createElement("div");
    footer.className = "source-footer";

    for (const value of [
      item.token_count ? `${item.token_count} tokens` : null,
      item.source_url || null,
    ].filter(Boolean)) {
      const span = document.createElement("span");
      span.textContent = value;
      footer.appendChild(span);
    }

    element.append(summary, content, footer);
    sources.appendChild(element);
  }
}

function payloadFromForm(formData) {
  // Only send optional filters when the user has supplied them. Empty
  // strings should mean "let the backend decide".
  const payload = {
    question: formData.get("question").trim(),
    source_type: formDataSourceType(formData),
    top_k: Number(formData.get("top_k") || 5),
  };

  const fiscalPeriod = (
    formData.get("fiscal_period") || ""
  ).trim().toUpperCase();
  const formType = (
    formData.get("form_type") || ""
  ).trim().toUpperCase();
  const sectionKey = (
    formData.get("section_key") || ""
  ).trim();

  if (!tickerAutoInput.checked && selectedTickers.length) {
    payload.tickers = [...selectedTickers];
  }

  if (fiscalPeriod) {
    payload.fiscal_period = fiscalPeriod;
  }

  if (formType) {
    payload.form_type = formType;
  }

  if (sectionKey) {
    payload.section_key = sectionKey;
  }

  return payload;
}

function setSourceType(value) {
  const radio = form.querySelector(`input[name="source_type"][value="${value}"]`);

  if (radio) {
    radio.checked = true;
  }
}

sampleButton.addEventListener("click", async () => {
  // Cycle through examples instead of replacing the form with a menu.
  // It keeps the interface small while covering the main source modes.
  const sample = samples[sampleIndex % samples.length];
  sampleIndex += 1;

  form.elements.question.value = sample.question;
  tickerAutoInput.checked = true;
  selectedTickers = sample.tickers || [sample.ticker];
  syncTickerUrl();
  renderTickerPicker();
  setSourceType(sample.source_type);

  if (metadataReady) {
    await loadFiltersForTicker(primaryTicker());
  }

  form.elements.fiscal_period.value = sample.fiscal_period || "";
  form.elements.form_type.value = sample.form_type || "";

  if (metadataReady) {
    await loadFilingSections(
      primaryTicker(),
      form.elements.form_type.value
    );
  }

  form.elements.section_key.value = sample.section_key || "";
  updateFilterState();
  await loadMarketChart();
});

tickerPickerButton.addEventListener("click", () => {
  setTickerMenuOpen(tickerMenu.hidden);
});

document.addEventListener("click", (event) => {
  if (!tickerPicker.contains(event.target)) {
    setTickerMenuOpen(false);
  }
});

tickerPicker.addEventListener("keydown", (event) => {
  if (event.key === "Escape") {
    setTickerMenuOpen(false);
    tickerPickerButton.focus();
  }
});

tickerAutoInput.checked = (
  initialTickerMode === "auto" || !initialTickerParameter
);
tickerAutoInput.addEventListener("change", async () => {
  if (tickerAutoInput.checked) {
    await resolveTickersFromQuestion();
  } else {
    syncTickerUrl();
  }
});

questionInput.addEventListener("input", () => {
  if (!tickerAutoInput.checked) {
    return;
  }

  window.clearTimeout(tickerResolutionTimer);
  tickerResolutionTimer = window.setTimeout(() => {
    resolveTickersFromQuestion();
  }, 450);
});

for (const button of chartPeriodButtons) {
  button.addEventListener("click", async () => {
    selectedChartPeriod = button.dataset.period;

    for (const periodButton of chartPeriodButtons) {
      const isActive = periodButton === button;
      periodButton.classList.toggle(
        "active",
        isActive
      );
      periodButton.setAttribute("aria-pressed", String(isActive));
    }

    await loadMarketChart();
  });
}

for (const button of chartEventButtons) {
  button.addEventListener("click", () => {
    selectedEventFilter = button.dataset.eventFilter;

    for (const eventButton of chartEventButtons) {
      const isActive = eventButton === button;
      eventButton.classList.toggle("active", isActive);
      eventButton.setAttribute("aria-pressed", String(isActive));
    }

    if (marketChartData) {
      renderMarketChart();
    }
  });
}

if ("ResizeObserver" in window) {
  new ResizeObserver(() => {
    if (marketChartData) {
      renderMarketChart();
    }
  }).observe(priceChart);
}

formTypeSelect.addEventListener("change", async () => {
  await loadFilingSections(
    primaryTicker(),
    formTypeSelect.value
  );
});

for (const radio of form.querySelectorAll("input[name='source_type']")) {
  radio.addEventListener("change", updateFilterState);
}

copyButton.addEventListener("click", async () => {
  const text = answer.textContent.trim();

  if (!text) {
    return;
  }

  try {
    await navigator.clipboard.writeText(text);
    setStatus("Copied");
  } catch (error) {
    setStatus("Copy failed", "error");
  }
});

// Metadata and provider health are independent startup checks. Running both
// immediately makes the console ready sooner and keeps one failure from
// hiding the other.
renderTickerPicker();
initializeMetadata();
checkOpenAIHealth();
loadResearchHistory();

openAIHealthButton.addEventListener("click", checkOpenAIHealth);
historyRefreshButton.addEventListener("click", loadResearchHistory);

async function runResearchRequest(mode) {
  if (tickerAutoInput.checked) {
    await resolveTickersFromQuestion();
  }

  const payload = payloadFromForm(
    new FormData(form)
  );

  const isPreview = mode === "preview";

  setBusy(true);
  setStatus("Running", "loading");
  resultTitle.textContent = isPreview ? "Previewing" : "Searching";
  answer.textContent = isPreview
    ? "Retrieving evidence only..."
    : "Retrieving evidence and generating the answer...";
  renderMarketContext([]);
  renderSources([]);

  try {
    // The static UI is served by the same FastAPI app, so a relative URL
    // works locally and keeps deployment simple later.
    const endpoint = isPreview
      ? "/api/retrieval/preview"
      : "/api/research";

    const response = await fetch(endpoint, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify(payload),
    });

    const data = await response.json();

    if (!response.ok) {
      throw new Error(data.detail || "Research request failed.");
    }

    if (tickerAutoInput.checked && data.tickers) {
      await applyTickerSelection(data.tickers);
    }

    resultTitle.textContent = "Complete";
    answer.textContent = isPreview
      ? [
          "Evidence preview loaded.",
          `${(data.sources || []).length} text source(s) found.`,
          `${(data.market_context || []).length} market snapshot(s) found.`,
          "",
          "Open the source cards below to inspect the exact retrieved chunks.",
        ].join("\n")
      : data.answer;
    renderMarketContext(data.market_context || []);
    renderSources(data.sources || []);
    setStatus("Done");

    if (!isPreview && data.run_id) {
      loadResearchHistory();
    }

    researchOutput.scrollIntoView({
      behavior: "smooth",
      block: "start",
    });
  } catch (error) {
    resultTitle.textContent = "Error";
    answer.textContent = error.message;
    setStatus("Error", "error");
  } finally {
    setBusy(false);
  }
}

form.addEventListener("submit", async (event) => {
  event.preventDefault();

  await runResearchRequest("answer");
});

previewButton.addEventListener("click", async () => {
  await runResearchRequest("preview");
});
