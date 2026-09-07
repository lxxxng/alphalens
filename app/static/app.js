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
const copyButton = document.querySelector("#copy-button");
const sampleButton = document.querySelector("#sample-button");

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
  },
  {
    question: "How has NVIDIA stock performed over the last year versus SPY?",
    ticker: "NVDA",
    source_type: "auto",
  },
  {
    question: "Compare Walmart filings and earnings call comments about margins.",
    ticker: "WMT",
    source_type: "both",
  },
];

let sampleIndex = 0;

function setStatus(label, state = "") {
  statusPill.textContent = label;
  statusPill.className = `status-pill ${state}`.trim();
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
    const relative = item.benchmark_relative_returns || {};

    grid.append(
      metric("Close", formatNumber(item.latest_close)),
      metric("1M", formatPercent(returns["1M"])),
      metric("3M", formatPercent(returns["3M"])),
      metric("1Y", formatPercent(returns["1Y"])),
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
    source_type: formData.get("source_type"),
    top_k: Number(formData.get("top_k") || 5),
  };

  const ticker = formData.get("ticker").trim().toUpperCase();
  const fiscalPeriod = formData.get("fiscal_period").trim().toUpperCase();

  if (ticker) {
    payload.ticker = ticker;
  }

  if (fiscalPeriod) {
    payload.fiscal_period = fiscalPeriod;
  }

  return payload;
}

function setSourceType(value) {
  const radio = form.querySelector(`input[name="source_type"][value="${value}"]`);

  if (radio) {
    radio.checked = true;
  }
}

sampleButton.addEventListener("click", () => {
  // Cycle through examples instead of replacing the form with a menu.
  // It keeps the interface small while covering the main source modes.
  const sample = samples[sampleIndex % samples.length];
  sampleIndex += 1;

  form.elements.question.value = sample.question;
  form.elements.ticker.value = sample.ticker;
  form.elements.fiscal_period.value = "";
  setSourceType(sample.source_type);
});

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

form.addEventListener("submit", async (event) => {
  event.preventDefault();

  const payload = payloadFromForm(
    new FormData(form)
  );

  submitButton.disabled = true;
  setStatus("Running", "loading");
  resultTitle.textContent = "Searching";
  answer.textContent = "Retrieving evidence and generating the answer...";
  renderMarketContext([]);
  renderSources([]);

  try {
    // The static UI is served by the same FastAPI app, so a relative URL
    // works locally and keeps deployment simple later.
    const response = await fetch("/api/research", {
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

    resultTitle.textContent = "Complete";
    answer.textContent = data.answer;
    renderMarketContext(data.market_context || []);
    renderSources(data.sources || []);
    setStatus("Done");
  } catch (error) {
    resultTitle.textContent = "Error";
    answer.textContent = error.message;
    setStatus("Error", "error");
  } finally {
    submitButton.disabled = false;
  }
});
