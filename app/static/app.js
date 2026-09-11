const form = document.querySelector("#research-form");
const appHeader = document.querySelector(".app-header");
const answer = document.querySelector("#answer");
const sources = document.querySelector("#sources");
const sourceCount = document.querySelector("#source-count");
const marketPanel = document.querySelector("#market-panel");
const marketSnapshots = document.querySelector("#market-snapshots");
const marketCount = document.querySelector("#market-count");
const statusPill = document.querySelector("#status-pill");
const resultTitle = document.querySelector("#result-title");
const resultScope = document.querySelector("#result-scope");
const resultCloseButton = document.querySelector("#result-close");
const workspaceTitle = document.querySelector("#workspace-title");
const workspaceViewButtons = document.querySelectorAll("[data-workspace-view]");
const marketView = document.querySelector("#market-view");
const researchView = document.querySelector("#research-view");
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
const priceChartPanel = document.querySelector(".price-chart-panel");
const chartPinButton = document.querySelector("#chart-pin");
const latestBriefScope = document.querySelector("#latest-brief-scope");
const latestBriefGenerateButton = document.querySelector("#latest-brief-generate");
const latestBriefCopyButton = document.querySelector("#latest-brief-copy");
const latestBriefStatus = document.querySelector("#latest-brief-status");
const latestBriefEmpty = document.querySelector("#latest-brief-empty");
const latestBriefOutput = document.querySelector("#latest-brief-output");
const latestBriefDetails = document.querySelector("#latest-brief-details");
const latestBriefSections = document.querySelector("#latest-brief-sections");
const latestBriefSources = document.querySelector("#latest-brief-sources");
const latestBriefSourceCount = document.querySelector("#latest-brief-source-count");
const signalSourceButtons = document.querySelectorAll("[data-signal-source]");
const signalTickerControls = document.querySelector("#signal-ticker-controls");
const signalSummary = document.querySelector("#signal-summary");
const signalBody = document.querySelector(".signal-body");
const signalPeriod = document.querySelector("#signal-period");
const signalScore = document.querySelector("#signal-score");
const signalChange = document.querySelector("#signal-change");
const signalCoverage = document.querySelector("#signal-coverage");
const signalTrend = document.querySelector("#signal-trend");
const signalTrendLegend = document.querySelector("#signal-trend-legend");
const signalEmpty = document.querySelector("#signal-empty");
const topicAudience = document.querySelector("#topic-audience");
const topicSignals = document.querySelector("#topic-signals");
const researchOutput = document.querySelector(".research-output");
const historyList = document.querySelector("#history-list");
const historyRefreshButton = document.querySelector("#history-refresh");
const { fiscalPeriod: formatFiscalPeriod } = window.AlphaLensFormatters;

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
const initialRunId = Number(
  new URLSearchParams(window.location.search).get("run_id") || 0
);
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
let selectedSignalSource = "transcripts";
let selectedSignalTicker = (
  new URLSearchParams(window.location.search).get("signal_ticker") || ""
).toUpperCase();
let compareSignalTickers = (
  new URLSearchParams(window.location.search).get("signal_view") === "compare"
);
let signalDataByTicker = new Map();
let signalRequest = 0;
let lastCopyText = answer.textContent.trim();
let resultAvailable = false;
let resultReturnFocus = null;
let latestBriefCopyText = "";
let latestBriefRequest = 0;

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

function setLatestBriefStatus(label, state = "") {
  latestBriefStatus.textContent = label;
  latestBriefStatus.className = `status-pill ${state}`.trim();
}

function setBusy(isBusy) {
  submitButton.disabled = isBusy;
  previewButton.disabled = isBusy;
}

function syncHeaderHeight() {
  document.documentElement.style.setProperty(
    "--header-height",
    `${appHeader.offsetHeight}px`
  );
}

function setWorkspaceView(view, scope = "") {
  const showResearch = view === "research";

  if (
    showResearch
    && marketView.hidden === false
    && document.activeElement instanceof HTMLElement
    && !researchOutput.contains(document.activeElement)
  ) {
    resultReturnFocus = document.activeElement;
  }

  if (scope) {
    resultScope.textContent = scope;
  }

  marketView.hidden = showResearch;
  researchView.hidden = !showResearch;
  document.body.classList.toggle("workspace-research", showResearch);
  workspaceTitle.textContent = showResearch ? "Research View" : "Market View";

  for (const button of workspaceViewButtons) {
    const isActive = button.dataset.workspaceView === view;
    button.classList.toggle("active", isActive);
    button.setAttribute("aria-selected", String(isActive));
    button.tabIndex = isActive ? 0 : -1;
  }

  document.querySelector("#research-tab")?.classList.toggle(
    "has-result",
    resultAvailable && !showResearch
  );
}

function showResearchView(scope) {
  resultAvailable = true;
  setWorkspaceView("research", scope);
}

function showMarketView() {
  setWorkspaceView("market");

  if (resultReturnFocus?.isConnected) {
    resultReturnFocus.focus({ preventScroll: true });
  }
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
  resetLatestBrief();
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
  await Promise.all([
    loadMarketChart(),
    loadSentimentSignals(),
  ]);
}

async function openSavedResearchRun(runId) {
  setStatus("Loading", "loading");
  showResearchView(`Saved research | Run ${runId}`);

  try {
    const response = await fetch(`/api/research/history/${runId}`);
    const data = await response.json();

    if (!response.ok) {
      throw new Error(data.detail || "Could not open saved research.");
    }

    await restoreSavedFilters(data);
    syncResearchRunUrl(data.run_id);
    resultTitle.textContent = `Saved Run #${data.run_id}`;
    resultScope.textContent = `${(data.tickers || []).join(", ") || "Research"} | ${data.source_type}`;
    showPlainAnswer(data.answer);
    renderMarketContext(data.market_context || []);
    renderSources(data.sources || []);
    setStatus("History");

  } catch (error) {
    resultTitle.textContent = "History Error";
    showPlainAnswer(error.message);
    setStatus("History error", "error");
  }
}

function sourceLabel(source) {
  if (source.source_type === "transcript") {
    return [
      source.ticker,
      formatFiscalPeriod(source.fiscal_period),
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

function syncResearchRunUrl(runId = null) {
  const url = new URL(window.location.href);

  if (runId) {
    url.searchParams.set("run_id", String(runId));
  } else {
    url.searchParams.delete("run_id");
  }

  window.history.replaceState({}, "", url);
}

// A ticker change invalidates the displayed brief so analysis from a previous
// company can never masquerade as current intelligence.
function resetLatestBrief() {
  latestBriefRequest += 1;
  latestBriefCopyText = "";
  latestBriefScope.textContent = selectedTickers.length
    ? `${selectedTickers.join(" / ")} | Latest earnings call + SEC filing`
    : "Select at least one company";
  latestBriefEmpty.textContent = selectedTickers.length
    ? "Generate a grounded brief from each selected company's latest earnings call and SEC filing."
    : "Select a company to prepare its latest event brief.";
  latestBriefEmpty.hidden = false;
  latestBriefOutput.hidden = true;
  latestBriefOutput.replaceChildren();
  latestBriefSections.replaceChildren();
  latestBriefSources.replaceChildren();
  latestBriefSourceCount.textContent = "0";
  latestBriefDetails.hidden = true;
  latestBriefDetails.open = false;
  latestBriefStatus.textContent = "Not generated";
  latestBriefStatus.className = "status-pill";
  latestBriefGenerateButton.disabled = !selectedTickers.length;
  latestBriefGenerateButton.textContent = selectedTickers.length > 1
    ? "Generate Comparison"
    : "Generate Latest Brief";
  latestBriefCopyButton.disabled = true;
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

  if (changed) {
    resetLatestBrief();
  }

  if (changed && refresh && metadataReady) {
    if (previousPrimary !== primaryTicker()) {
      await loadFiltersForTicker(primaryTicker());
    }
    await Promise.all([
      loadMarketChart(),
      loadSentimentSignals(),
    ]);
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

function setSignalSource(value) {
  selectedSignalSource = value;

  for (const button of signalSourceButtons) {
    const isActive = button.dataset.signalSource === value;
    button.classList.toggle("active", isActive);
    button.setAttribute("aria-pressed", String(isActive));
  }
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
  resetLatestBrief();
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
      formatFiscalPeriod(item.fiscal_period),
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

  await Promise.all([
    loadMarketChart(),
    loadSentimentSignals(),
  ]);
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

function formatSentimentScore(value) {
  if (!Number.isFinite(value)) {
    return "--";
  }

  const sign = value > 0 ? "+" : "";
  return `${sign}${value.toFixed(2)}`;
}

function setSentimentClass(element, value) {
  element.className = Number.isFinite(value)
    ? (value >= 0 ? "positive" : "negative")
    : "";
}

function transcriptManagementAggregate(call) {
  return call.groups?.find((group) => group.group === "management")
    || call.overall
    || {};
}

function signalRecordLabel(record) {
  if (selectedSignalSource === "transcripts") {
    return formatFiscalPeriod(record.fiscal_period);
  }

  return [record.form_type, record.filing_date].filter(Boolean).join(" | ");
}

function renderSignalTrend(records, selectedIndex, ticker) {
  signalTrend.replaceChildren();
  signalTrendLegend.replaceChildren();
  const points = records
    .map((record, index) => ({
      index,
      label: signalRecordLabel(record),
      score: selectedSignalSource === "transcripts"
        ? transcriptManagementAggregate(record).score
        : record.overall?.score,
    }))
    .filter((point) => Number.isFinite(point.score));

  if (!points.length) {
    return;
  }

  const width = 540;
  const height = 205;
  const margin = { top: 14, right: 14, bottom: 28, left: 31 };
  const plotWidth = width - margin.left - margin.right;
  const plotHeight = height - margin.top - margin.bottom;
  const x = (index) => margin.left + (
    records.length === 1 ? plotWidth / 2 : index * plotWidth / (records.length - 1)
  );
  const y = (score) => margin.top + (1 - score) * plotHeight / 2;
  signalTrend.setAttribute("viewBox", `0 0 ${width} ${height}`);
  signalTrend.setAttribute(
    "aria-label",
    `${ticker} sentiment history from ${points[0].label} to ${points.at(-1).label}`
  );

  for (const score of [1, 0.5, 0, -0.5, -1]) {
    signalTrend.appendChild(svgElement("line", {
      x1: margin.left,
      x2: width - margin.right,
      y1: y(score),
      y2: y(score),
      class: score === 0 ? "signal-zero-line" : "signal-grid-line",
    }));

    if ([1, 0, -1].includes(score)) {
      const label = svgElement("text", {
        x: margin.left - 7,
        y: y(score) + 3,
        class: "signal-axis-label",
        "text-anchor": "end",
      });
      label.textContent = score > 0 ? "+1" : String(score);
      signalTrend.appendChild(label);
    }
  }

  const path = points.map((point, index) => {
    const command = index === 0 ? "M" : "L";
    return `${command}${x(point.index).toFixed(2)},${y(point.score).toFixed(2)}`;
  }).join(" ");
  signalTrend.appendChild(svgElement("path", {
    d: path,
    class: "signal-line",
  }));

  for (const point of points) {
    const marker = svgElement("circle", {
      cx: x(point.index),
      cy: y(point.score),
      r: point.index === selectedIndex ? 5 : 3.5,
      class: `signal-point${point.index === selectedIndex ? " selected" : ""}`,
    });
    const title = svgElement("title");
    title.textContent = `${point.label}: ${formatSentimentScore(point.score)}`;
    marker.appendChild(title);
    signalTrend.appendChild(marker);
  }

  const labelIndexes = [...new Set([
    points[0].index,
    selectedIndex,
    points.at(-1).index,
  ])].filter((index) => index >= 0);

  for (const index of labelIndexes) {
    const record = records[index];
    const label = svgElement("text", {
      x: x(index),
      y: height - 7,
      class: "signal-axis-label",
      "text-anchor": index === 0
        ? "start"
        : (index === records.length - 1 ? "end" : "middle"),
    });
    label.textContent = selectedSignalSource === "transcripts"
      ? record.fiscal_period
      : record.filing_date;
    signalTrend.appendChild(label);
  }

  const legend = document.createElement("span");
  const swatch = document.createElement("i");
  swatch.style.backgroundColor = "#6842c2";
  const legendLabel = document.createElement("strong");
  legendLabel.textContent = ticker;
  legend.append(swatch, legendLabel);
  signalTrendLegend.appendChild(legend);
}

function renderTopicSignals(topics) {
  topicSignals.replaceChildren();

  if (!topics.length) {
    const empty = document.createElement("p");
    empty.className = "signal-empty";
    empty.textContent = "No topic-level scores are available for this event.";
    topicSignals.appendChild(empty);
    return;
  }

  for (const topic of topics.slice(0, 6)) {
    const row = document.createElement("div");
    row.className = "topic-row";
    const identity = document.createElement("div");
    identity.className = "topic-label";
    const label = document.createElement("strong");
    label.textContent = topic.topic_label;
    const detail = document.createElement("small");
    const delta = Number.isFinite(topic.score_change)
      ? `${formatSentimentScore(topic.score_change)} vs prior`
      : "No prior comparison";
    detail.textContent = `${topic.eligible_items || 0} passages | ${delta}`;
    identity.append(label, detail);

    const meter = document.createElement("div");
    meter.className = "topic-meter";
    const bar = document.createElement("span");
    const score = Number.isFinite(topic.score) ? topic.score : 0;
    const magnitude = Math.min(1, Math.abs(score)) * 50;
    bar.className = score >= 0 ? "positive" : "negative";
    bar.style.left = score >= 0 ? "50%" : `${50 - magnitude}%`;
    bar.style.width = `${magnitude}%`;
    meter.appendChild(bar);

    const value = document.createElement("span");
    value.className = "topic-value";
    value.textContent = formatSentimentScore(topic.score);
    row.append(identity, meter, value);
    topicSignals.appendChild(row);
  }
}

function renderSentimentSignals(data, ticker) {
  const records = selectedSignalSource === "transcripts"
    ? (data.calls || [])
    : (data.filings || []);

  if (!records.length) {
    signalSummary.hidden = true;
    signalBody.hidden = true;
    signalEmpty.hidden = false;
    signalTrend.replaceChildren();
    signalTrendLegend.replaceChildren();
    signalEmpty.textContent = `No scored ${selectedSignalSource === "transcripts" ? "calls" : "filings"} found for ${ticker}.`;
    return;
  }

  let selectedIndex = records.length - 1;
  const requestedPeriod = fiscalPeriodSelect.value;

  if (selectedSignalSource === "transcripts" && requestedPeriod) {
    const match = records.findIndex(
      (record) => record.fiscal_period === requestedPeriod
    );
    selectedIndex = match >= 0 ? match : selectedIndex;
  }

  const current = records[selectedIndex];
  const aggregate = selectedSignalSource === "transcripts"
    ? transcriptManagementAggregate(current)
    : (current.overall || {});
  const previous = selectedIndex > 0 ? records[selectedIndex - 1] : null;
  const previousAggregate = previous
    ? (selectedSignalSource === "transcripts"
      ? transcriptManagementAggregate(previous)
      : (previous.overall || {}))
    : null;
  const change = (
    Number.isFinite(aggregate.score)
    && Number.isFinite(previousAggregate?.score)
  ) ? aggregate.score - previousAggregate.score : null;

  signalSummary.hidden = false;
  signalBody.hidden = false;
  signalEmpty.hidden = true;
  signalPeriod.textContent = [
    signalRecordLabel(current),
    selectedSignalSource === "transcripts" ? current.call_date : null,
  ].filter(Boolean).join(" | ");
  signalScore.textContent = `${aggregate.label || "unscored"} ${formatSentimentScore(aggregate.score)}`;
  setSentimentClass(signalScore, aggregate.score);
  signalChange.textContent = Number.isFinite(change)
    ? `${formatSentimentScore(change)} points`
    : "No prior event";
  setSentimentClass(signalChange, change);
  signalCoverage.textContent = Number.isFinite(aggregate.coverage)
    ? `${(aggregate.coverage * 100).toFixed(0)}% (${aggregate.scored_items}/${aggregate.eligible_items})`
    : "--";
  topicAudience.textContent = selectedSignalSource === "transcripts"
    ? "Management commentary"
    : "SEC narrative sections";
  renderSignalTrend(records, selectedIndex, ticker);
  renderTopicSignals(current.topics || []);
}

function activeSignalTicker() {
  if (selectedTickers.includes(selectedSignalTicker)) {
    return selectedSignalTicker;
  }

  selectedSignalTicker = primaryTicker();
  return selectedSignalTicker;
}

function signalRecords(data) {
  return selectedSignalSource === "transcripts"
    ? (data?.calls || [])
    : (data?.filings || []);
}

function selectedSignalIndex(records) {
  if (selectedSignalSource !== "transcripts" || !fiscalPeriodSelect.value) {
    return records.length - 1;
  }

  const match = records.findIndex(
    (record) => record.fiscal_period === fiscalPeriodSelect.value
  );
  return match >= 0 ? match : records.length - 1;
}

function signalAggregate(record) {
  if (!record) {
    return {};
  }

  return selectedSignalSource === "transcripts"
    ? transcriptManagementAggregate(record)
    : (record.overall || {});
}

function signalTickerColor(ticker) {
  const index = Math.max(0, selectedTickers.indexOf(ticker));
  return COMPANY_CHART_COLORS[index % COMPANY_CHART_COLORS.length];
}

function syncSignalViewUrl() {
  const url = new URL(window.location.href);

  if (compareSignalTickers) {
    url.searchParams.set("signal_view", "compare");
    url.searchParams.delete("signal_ticker");
  } else {
    url.searchParams.delete("signal_view");
    url.searchParams.set("signal_ticker", activeSignalTicker());
  }

  window.history.replaceState({}, "", url);
}

function renderSignalTickerControls() {
  signalTickerControls.replaceChildren();
  signalTickerControls.hidden = selectedTickers.length <= 1;

  if (selectedTickers.length <= 1) {
    selectedSignalTicker = primaryTicker();
    compareSignalTickers = false;
    return;
  }

  for (const ticker of selectedTickers) {
    const button = document.createElement("button");
    button.type = "button";
    button.textContent = ticker;
    button.classList.toggle(
      "active",
      !compareSignalTickers && ticker === activeSignalTicker()
    );
    button.setAttribute(
      "aria-pressed",
      String(!compareSignalTickers && ticker === activeSignalTicker())
    );
    button.addEventListener("click", () => {
      selectedSignalTicker = ticker;
      compareSignalTickers = false;
      syncSignalViewUrl();
      renderSentimentCollection();
    });
    signalTickerControls.appendChild(button);
  }

  const compareButton = document.createElement("button");
  compareButton.type = "button";
  compareButton.className = `compare${compareSignalTickers ? " active" : ""}`;
  compareButton.textContent = "Compare";
  compareButton.setAttribute("aria-pressed", String(compareSignalTickers));
  compareButton.addEventListener("click", () => {
    compareSignalTickers = true;
    syncSignalViewUrl();
    renderSentimentCollection();
  });
  signalTickerControls.appendChild(compareButton);
}

function renderSignalComparisonTrend(entries) {
  signalTrend.replaceChildren();
  signalTrendLegend.replaceChildren();
  const series = entries.map(([ticker, data]) => ({
    ticker,
    color: signalTickerColor(ticker),
    points: signalRecords(data)
      .map((record) => ({
        key: selectedSignalSource === "transcripts"
          ? record.fiscal_period
          : record.filing_date,
        label: signalRecordLabel(record),
        score: signalAggregate(record).score,
      }))
      .filter((point) => point.key && Number.isFinite(point.score)),
  })).filter((item) => item.points.length);

  if (!series.length) {
    return;
  }

  const keys = [...new Set(
    series.flatMap((item) => item.points.map((point) => point.key))
  )].sort();
  const keyIndex = new Map(keys.map((key, index) => [key, index]));
  const width = 540;
  const height = 205;
  const margin = { top: 14, right: 14, bottom: 28, left: 31 };
  const plotWidth = width - margin.left - margin.right;
  const plotHeight = height - margin.top - margin.bottom;
  const x = (key) => margin.left + (
    keys.length === 1
      ? plotWidth / 2
      : keyIndex.get(key) * plotWidth / (keys.length - 1)
  );
  const y = (score) => margin.top + (1 - score) * plotHeight / 2;
  signalTrend.setAttribute("viewBox", `0 0 ${width} ${height}`);
  signalTrend.setAttribute(
    "aria-label",
    `Sentiment comparison for ${series.map((item) => item.ticker).join(", ")}`
  );

  for (const score of [1, 0.5, 0, -0.5, -1]) {
    signalTrend.appendChild(svgElement("line", {
      x1: margin.left,
      x2: width - margin.right,
      y1: y(score),
      y2: y(score),
      class: score === 0 ? "signal-zero-line" : "signal-grid-line",
    }));

    if ([1, 0, -1].includes(score)) {
      const label = svgElement("text", {
        x: margin.left - 7,
        y: y(score) + 3,
        class: "signal-axis-label",
        "text-anchor": "end",
      });
      label.textContent = score > 0 ? "+1" : String(score);
      signalTrend.appendChild(label);
    }
  }

  for (const item of series) {
    const path = item.points.map((point, index) => {
      const command = index === 0 ? "M" : "L";
      return `${command}${x(point.key).toFixed(2)},${y(point.score).toFixed(2)}`;
    }).join(" ");
    const pathElement = svgElement("path", {
      d: path,
      class: "signal-line",
    });
    pathElement.style.stroke = item.color;
    signalTrend.appendChild(pathElement);

    for (const point of item.points) {
      const marker = svgElement("circle", {
        cx: x(point.key),
        cy: y(point.score),
        r: 3.5,
        class: "signal-point",
      });
      marker.style.fill = item.color;
      const title = svgElement("title");
      title.textContent = `${item.ticker} | ${point.label}: ${formatSentimentScore(point.score)}`;
      marker.appendChild(title);
      signalTrend.appendChild(marker);
    }

    const legend = document.createElement("span");
    const swatch = document.createElement("i");
    swatch.style.backgroundColor = item.color;
    const label = document.createElement("strong");
    label.textContent = item.ticker;
    legend.append(swatch, label);
    signalTrendLegend.appendChild(legend);
  }

  for (const key of [...new Set([keys[0], keys.at(-1)])]) {
    const label = svgElement("text", {
      x: x(key),
      y: height - 7,
      class: "signal-axis-label",
      "text-anchor": key === keys[0] ? "start" : "end",
    });
    label.textContent = key;
    signalTrend.appendChild(label);
  }
}

function renderTopicComparison(currentEntries) {
  topicSignals.replaceChildren();
  const topics = new Map();

  for (const entry of currentEntries) {
    for (const topic of entry.record.topics || []) {
      if (!topics.has(topic.topic_key)) {
        topics.set(topic.topic_key, {
          label: topic.topic_label,
          totalItems: 0,
          values: new Map(),
        });
      }

      const aggregate = topics.get(topic.topic_key);
      aggregate.totalItems += topic.eligible_items || 0;
      aggregate.values.set(entry.ticker, topic);
    }
  }

  const ranked = [...topics.values()]
    .sort((left, right) => right.totalItems - left.totalItems)
    .slice(0, 6);

  if (!ranked.length) {
    renderTopicSignals([]);
    return;
  }

  for (const topic of ranked) {
    const row = document.createElement("div");
    row.className = "topic-comparison-row";
    const identity = document.createElement("div");
    identity.className = "topic-label";
    const label = document.createElement("strong");
    label.textContent = topic.label;
    const detail = document.createElement("small");
    detail.textContent = `${topic.totalItems} passages across companies`;
    identity.append(label, detail);
    const values = document.createElement("div");
    values.className = "topic-comparison-values";

    currentEntries.forEach((entry) => {
      const value = document.createElement("span");
      value.className = "topic-comparison-value";
      const swatch = document.createElement("i");
      swatch.style.backgroundColor = signalTickerColor(entry.ticker);
      const text = document.createElement("span");
      const score = topic.values.get(entry.ticker)?.score;
      text.textContent = `${entry.ticker} `;
      const scoreElement = document.createElement("strong");
      scoreElement.textContent = formatSentimentScore(score);
      text.appendChild(scoreElement);
      value.append(swatch, text);
      values.appendChild(value);
    });

    row.append(identity, values);
    topicSignals.appendChild(row);
  }
}

function renderSignalComparison(entries) {
  const currentEntries = entries.map(([ticker, data]) => {
    const records = signalRecords(data);
    const index = selectedSignalIndex(records);
    const record = records[index];
    const previous = index > 0 ? records[index - 1] : null;
    const aggregate = signalAggregate(record);
    const previousAggregate = signalAggregate(previous);
    const change = (
      Number.isFinite(aggregate.score)
      && Number.isFinite(previousAggregate.score)
    ) ? aggregate.score - previousAggregate.score : null;
    return { ticker, record, aggregate, change };
  }).filter((entry) => entry.record);

  if (!currentEntries.length) {
    renderSentimentSignals({}, selectedTickers.join(", "));
    return;
  }

  const scored = currentEntries.reduce(
    (total, entry) => total + (entry.aggregate.scored_items || 0),
    0
  );
  const eligible = currentEntries.reduce(
    (total, entry) => total + (entry.aggregate.eligible_items || 0),
    0
  );
  const changes = currentEntries
    .map((entry) => entry.change)
    .filter(Number.isFinite);
  const averageChange = changes.length
    ? changes.reduce((total, value) => total + value, 0) / changes.length
    : null;

  signalSummary.hidden = false;
  signalBody.hidden = false;
  signalEmpty.hidden = true;
  signalPeriod.textContent = selectedSignalSource === "transcripts"
    ? (fiscalPeriodSelect.value || "Latest call per company")
    : `Latest ${formTypeSelect.value || "SEC filing"} per company`;
  signalScore.textContent = `${currentEntries.length} companies`;
  signalScore.className = "";
  signalChange.textContent = Number.isFinite(averageChange)
    ? `${formatSentimentScore(averageChange)} average`
    : "No prior comparison";
  setSentimentClass(signalChange, averageChange);
  signalCoverage.textContent = eligible
    ? `${(scored / eligible * 100).toFixed(0)}% (${scored}/${eligible})`
    : "--";
  topicAudience.textContent = selectedSignalSource === "transcripts"
    ? "Latest management comparison"
    : "Latest SEC narrative comparison";
  renderSignalComparisonTrend(entries);
  renderTopicComparison(currentEntries);
}

function renderSentimentCollection() {
  renderSignalTickerControls();
  const entries = selectedTickers
    .map((ticker) => [ticker, signalDataByTicker.get(ticker)])
    .filter(([, data]) => data);

  if (compareSignalTickers && entries.length > 1) {
    renderSignalComparison(entries);
    return;
  }

  const ticker = activeSignalTicker();
  renderSentimentSignals(signalDataByTicker.get(ticker) || {}, ticker);
}

async function loadSentimentSignals() {
  const tickers = [...selectedTickers];
  const requestId = ++signalRequest;

  if (!tickers.length) {
    signalDataByTicker = new Map();
    renderSentimentCollection();
    return;
  }

  signalPeriod.textContent = "Loading";
  signalScore.textContent = "--";
  signalChange.textContent = "--";
  signalCoverage.textContent = "--";
  signalEmpty.hidden = true;

  try {
    const endpoint = selectedSignalSource === "transcripts"
      ? "/api/sentiment/transcripts"
      : "/api/sentiment/filings";
    const results = await Promise.all(tickers.map(async (ticker) => {
      const params = new URLSearchParams({ ticker });

      if (selectedSignalSource === "filings" && formTypeSelect.value) {
        params.set("form_type", formTypeSelect.value);
      }

      const response = await fetch(`${endpoint}?${params}`);
      const data = await response.json();

      if (!response.ok) {
        throw new Error(data.detail || `Sentiment request failed for ${ticker}.`);
      }

      return [ticker, data];
    }));

    if (requestId === signalRequest) {
      signalDataByTicker = new Map(results);
      renderSentimentCollection();
    }
  } catch (error) {
    if (requestId !== signalRequest) {
      return;
    }

    signalSummary.hidden = true;
    signalBody.hidden = true;
    signalTrend.replaceChildren();
    signalTrendLegend.replaceChildren();
    signalEmpty.hidden = false;
    signalEmpty.textContent = error.message;
  }
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

function eventDisplayLabel(event) {
  return event.event_type === "earnings"
    ? formatFiscalPeriod(event.label)
    : event.label;
}

function renderEventTooltip(event, left, top) {
  const title = document.createElement("strong");
  const eventType = event.event_type === "earnings"
    ? "Earnings"
    : "SEC filing";
  title.textContent = `${event.ticker} · ${eventType} · ${eventDisplayLabel(event)}`;
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
    title.textContent = `${event.ticker} · ${eventDisplayLabel(event)}`;
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
      "aria-label": `${event.ticker} ${eventDisplayLabel(event)} on ${event.date}`,
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

function renderSources(items, target = sources, countTarget = sourceCount) {
  target.replaceChildren();
  countTarget.textContent = String(items.length);

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
    target.appendChild(element);
  }
}

function showPlainAnswer(text) {
  answer.hidden = false;
  answer.textContent = text;
  lastCopyText = text.trim();
}

function appendBriefSection(container, title, items) {
  const section = document.createElement("section");
  section.className = "brief-section";
  const heading = document.createElement("h4");
  heading.textContent = title;
  const list = document.createElement("ul");
  const values = items?.length ? items : ["No material item identified."];

  for (const item of values) {
    const entry = document.createElement("li");
    entry.textContent = item;
    list.appendChild(entry);
  }

  section.append(heading, list);
  container.appendChild(section);
}

function renderEventBrief(brief) {
  latestBriefEmpty.hidden = true;
  latestBriefOutput.hidden = false;
  latestBriefOutput.replaceChildren();
  latestBriefSections.replaceChildren();

  const headline = document.createElement("h3");
  headline.className = "brief-headline";
  headline.textContent = brief.headline;
  const executiveSummary = document.createElement("p");
  executiveSummary.className = "brief-executive-summary";
  executiveSummary.textContent = brief.executive_summary;
  for (const [title, items] of [
    ["Key Developments", brief.key_developments],
    ["Topic Signals", brief.topic_signals],
    ["Market Reaction", brief.market_reaction],
    ["Risks", brief.risks],
    ["Watch Items", brief.watch_items],
    ["Limitations", brief.limitations],
  ]) {
    appendBriefSection(latestBriefSections, title, items);
  }

  latestBriefOutput.append(headline, executiveSummary);
  latestBriefDetails.hidden = false;
  latestBriefCopyText = [
    brief.headline,
    "",
    brief.executive_summary,
    ...[
      ["Key Developments", brief.key_developments],
      ["Topic Signals", brief.topic_signals],
      ["Market Reaction", brief.market_reaction],
      ["Risks", brief.risks],
      ["Watch Items", brief.watch_items],
      ["Limitations", brief.limitations],
    ].flatMap(([title, items]) => [
      "",
      title,
      ...(items || []).map((item) => `- ${item}`),
    ]),
  ].join("\n").trim();
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
  resetLatestBrief();
  setSourceType(sample.source_type);

  if (["transcripts", "filings"].includes(sample.source_type)) {
    setSignalSource(sample.source_type);
  }

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
  await Promise.all([
    loadMarketChart(),
    loadSentimentSignals(),
  ]);
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

  if (selectedSignalSource === "filings") {
    await loadSentimentSignals();
  }
});

fiscalPeriodSelect.addEventListener("change", () => {
  if (selectedSignalSource === "transcripts") {
    loadSentimentSignals();
  }
});

for (const button of signalSourceButtons) {
  button.addEventListener("click", async () => {
    setSignalSource(button.dataset.signalSource);
    await loadSentimentSignals();
  });
}

for (const radio of form.querySelectorAll("input[name='source_type']")) {
  radio.addEventListener("change", async () => {
    updateFilterState();

    if (["transcripts", "filings"].includes(radio.value)) {
      setSignalSource(radio.value);
      await loadSentimentSignals();
    }
  });
}

copyButton.addEventListener("click", async () => {
  const text = lastCopyText;

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

latestBriefCopyButton.addEventListener("click", async () => {
  if (!latestBriefCopyText) {
    return;
  }

  try {
    await navigator.clipboard.writeText(latestBriefCopyText);
    setLatestBriefStatus("Copied");
  } catch (error) {
    setLatestBriefStatus("Copy failed", "error");
  }
});

chartPinButton.addEventListener("click", () => {
  const pinned = priceChartPanel.classList.toggle("pinned");
  chartPinButton.setAttribute("aria-pressed", String(pinned));
  chartPinButton.textContent = pinned ? "Unpin Chart" : "Pin Chart";

  if (marketChartData) {
    window.requestAnimationFrame(renderMarketChart);
  }
});

resultCloseButton.addEventListener("click", showMarketView);

for (const button of workspaceViewButtons) {
  button.addEventListener("click", () => {
    setWorkspaceView(button.dataset.workspaceView);
  });

  button.addEventListener("keydown", (event) => {
    if (!["ArrowLeft", "ArrowRight"].includes(event.key)) {
      return;
    }

    event.preventDefault();
    const nextView = button.dataset.workspaceView === "market"
      ? "research"
      : "market";
    setWorkspaceView(nextView);
    document.querySelector(`[data-workspace-view='${nextView}']`)?.focus();
  });
}

document.addEventListener("keydown", (event) => {
  if (event.key === "Escape" && !researchView.hidden) {
    showMarketView();
  }
});

// Metadata and provider health are independent startup checks. Running both
// immediately makes the console ready sooner and keeps one failure from
// hiding the other.
renderTickerPicker();
resetLatestBrief();
syncHeaderHeight();
setWorkspaceView("market");
const metadataInitialization = initializeMetadata();
checkOpenAIHealth();
loadResearchHistory();

if (Number.isInteger(initialRunId) && initialRunId > 0) {
  metadataInitialization.then(() => openSavedResearchRun(initialRunId));
}

openAIHealthButton.addEventListener("click", checkOpenAIHealth);
historyRefreshButton.addEventListener("click", loadResearchHistory);

if ("ResizeObserver" in window) {
  new ResizeObserver(syncHeaderHeight).observe(appHeader);
} else {
  window.addEventListener("resize", syncHeaderHeight);
}

async function runResearchRequest(mode) {
  syncResearchRunUrl();

  if (tickerAutoInput.checked) {
    await resolveTickersFromQuestion();
  }

  const payload = payloadFromForm(
    new FormData(form)
  );

  const isPreview = mode === "preview";
  const scope = [
    selectedTickers.join(", ") || "Auto-detected company",
    payload.source_type,
    isPreview ? "Evidence preview" : "Generated answer",
  ].join(" | ");

  setBusy(true);
  setStatus("Running", "loading");
  showResearchView(scope);
  resultTitle.textContent = isPreview ? "Previewing" : "Searching";
  showPlainAnswer(isPreview
    ? "Retrieving evidence only..."
    : "Retrieving evidence and generating the answer...");
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
    showPlainAnswer(isPreview
      ? [
          "Evidence preview loaded.",
          `${(data.sources || []).length} text source(s) found.`,
          `${(data.market_context || []).length} market snapshot(s) found.`,
          "",
          "Open the source cards below to inspect the exact retrieved chunks.",
        ].join("\n")
      : data.answer);
    renderMarketContext(data.market_context || []);
    renderSources(data.sources || []);
    setStatus("Done");

    if (!isPreview && data.run_id) {
      syncResearchRunUrl(data.run_id);
      loadResearchHistory();
    }

  } catch (error) {
    resultTitle.textContent = "Error";
    showPlainAnswer(error.message);
    setStatus("Error", "error");
  } finally {
    setBusy(false);
  }
}

async function runLatestEventBrief() {
  const tickers = [...selectedTickers];

  if (!tickers.length) {
    latestBriefEmpty.textContent = "Select at least one company before generating a brief.";
    latestBriefEmpty.hidden = false;
    setLatestBriefStatus("Needs ticker", "error");
    return;
  }

  const selectionKey = tickers.join(",");
  const requestId = ++latestBriefRequest;
  // Latest Brief has a stable product scope; ad hoc query filters belong only
  // to the separate Research workflow.
  const payload = {
    tickers,
    event_type: "combined",
    top_k: 6,
  };

  latestBriefGenerateButton.disabled = true;
  latestBriefCopyButton.disabled = true;
  setLatestBriefStatus("Generating", "loading");
  latestBriefEmpty.textContent = "Collecting the latest event evidence, market reaction, and sentiment signals...";
  latestBriefEmpty.hidden = false;
  latestBriefOutput.hidden = true;
  latestBriefDetails.hidden = true;

  try {
    const response = await fetch("/api/briefs/generate", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    const data = await response.json();

    if (!response.ok) {
      throw new Error(data.detail || "Event brief request failed.");
    }

    if (
      requestId !== latestBriefRequest
      || selectionKey !== selectedTickers.join(",")
    ) {
      return;
    }

    renderEventBrief(data.brief);
    renderSources(
      data.sources || [],
      latestBriefSources,
      latestBriefSourceCount
    );
    latestBriefCopyButton.disabled = false;
    latestBriefGenerateButton.textContent = "Refresh Brief";
    setLatestBriefStatus("Current");
  } catch (error) {
    if (requestId !== latestBriefRequest) {
      return;
    }

    latestBriefEmpty.textContent = error.message;
    latestBriefEmpty.hidden = false;
    setLatestBriefStatus("Error", "error");
  } finally {
    if (requestId === latestBriefRequest) {
      latestBriefGenerateButton.disabled = false;
    }
  }
}

latestBriefGenerateButton.addEventListener("click", runLatestEventBrief);

form.addEventListener("submit", async (event) => {
  event.preventDefault();
  await runResearchRequest("answer");
});

previewButton.addEventListener("click", async () => {
  await runResearchRequest("preview");
});
