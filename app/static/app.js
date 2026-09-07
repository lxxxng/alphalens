const form = document.querySelector("#research-form");
const answer = document.querySelector("#answer");
const sources = document.querySelector("#sources");
const sourceCount = document.querySelector("#source-count");
const statusPill = document.querySelector("#status-pill");
const resultTitle = document.querySelector("#result-title");
const submitButton = document.querySelector("#submit-button");

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

function renderSources(items) {
  sources.replaceChildren();
  sourceCount.textContent = String(items.length);

  for (const item of items) {
    const element = document.createElement("article");
    element.className = "source-item";

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

    const title = document.createElement("p");
    title.className = "source-title";
    title.textContent = sourceLabel(item);

    const detail = document.createElement("p");
    detail.className = "source-detail";
    detail.textContent = sourceDetail(item);

    element.append(meta, title, detail);
    sources.appendChild(element);
  }
}

function payloadFromForm(formData) {
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

form.addEventListener("submit", async (event) => {
  event.preventDefault();

  const payload = payloadFromForm(
    new FormData(form)
  );

  submitButton.disabled = true;
  setStatus("Running", "loading");
  resultTitle.textContent = "Searching";
  answer.textContent = "Retrieving evidence and generating the answer...";
  renderSources([]);

  try {
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
