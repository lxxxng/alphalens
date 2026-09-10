const transcriptTitle = document.querySelector("#transcript-title");
const transcriptSubtitle = document.querySelector("#transcript-subtitle");
const transcriptFacts = document.querySelector("#transcript-facts");
const transcriptTurns = document.querySelector("#transcript-turns");
const transcriptError = document.querySelector("#transcript-error");
const transcriptSearch = document.querySelector("#transcript-search");
const turnCount = document.querySelector("#turn-count");
const copyButton = document.querySelector("#copy-transcript");
const researchLink = document.querySelector("#research-link");
const { fiscalPeriod: formatFiscalPeriod } = window.AlphaLensFormatters;

let transcript = null;

function transcriptIdFromPath() {
  const match = window.location.pathname.match(/\/transcripts\/(\d+)\/?$/);
  return match ? Number(match[1]) : null;
}

function formatDate(value) {
  if (!value) {
    return "Date unavailable";
  }

  return new Intl.DateTimeFormat("en-US", {
    year: "numeric",
    month: "short",
    day: "numeric",
    timeZone: "UTC",
  }).format(new Date(`${value}T00:00:00Z`));
}

function fact(label, value) {
  const item = document.createElement("div");
  item.className = "transcript-fact";
  const name = document.createElement("span");
  name.textContent = label;
  const content = document.createElement("strong");
  content.textContent = value;
  item.append(name, content);
  return item;
}

function speakerClass(turn) {
  const identity = [
    turn.speaker_name,
    turn.speaker_title,
    turn.speaker_role,
  ].filter(Boolean).join(" ").toLowerCase();

  if (identity.includes("operator")) {
    return "operator";
  }

  if (identity.includes("analyst")) {
    return "analyst";
  }

  return "management";
}

function renderTurns(query = "") {
  const normalizedQuery = query.trim().toLowerCase();
  const visibleTurns = transcript.turns.filter((turn) => (
    [
      turn.speaker_name,
      turn.speaker_title,
      turn.speaker_role,
      turn.content,
    ]
      .filter(Boolean)
      .join(" ")
      .toLowerCase()
      .includes(normalizedQuery)
  ));

  transcriptTurns.replaceChildren();

  if (!visibleTurns.length) {
    const empty = document.createElement("p");
    empty.className = "transcript-empty";
    empty.textContent = "No matching transcript turns.";
    transcriptTurns.appendChild(empty);
  }

  for (const turn of visibleTurns) {
    const item = document.createElement("article");
    item.className = `transcript-turn ${speakerClass(turn)}`;
    const speaker = document.createElement("div");
    speaker.className = "speaker";
    const name = document.createElement("strong");
    name.textContent = turn.speaker_name || "Unknown speaker";
    const detail = document.createElement("span");
    detail.textContent = [turn.speaker_title, turn.speaker_role]
      .filter(Boolean)
      .join(" | ") || `Turn ${turn.turn_index + 1}`;
    const content = document.createElement("p");
    content.className = "turn-content";
    content.textContent = turn.content;
    speaker.append(name, detail);
    item.append(speaker, content);
    transcriptTurns.appendChild(item);
  }

  turnCount.textContent = normalizedQuery
    ? `${visibleTurns.length} of ${transcript.turns.length} turns`
    : `${transcript.turns.length} turns`;
}

function renderTranscript(data) {
  transcript = data;
  const fiscalPeriod = formatFiscalPeriod(data.fiscal_period);
  transcriptTitle.textContent = data.title || `${data.ticker} Earnings Call`;
  transcriptSubtitle.textContent = `${data.ticker} | ${fiscalPeriod} | ${formatDate(data.call_date)}`;
  transcriptFacts.replaceChildren(
    fact("Ticker", data.ticker),
    fact("Fiscal period", fiscalPeriod),
    fact("Call date", formatDate(data.call_date)),
    fact("Provider", data.source_provider.replaceAll("_", " "))
  );
  researchLink.href = `/?tickers=${encodeURIComponent(data.ticker)}`;
  researchLink.textContent = `Research ${data.ticker}`;
  document.title = `${data.ticker} ${fiscalPeriod} | AlphaLens`;
  copyButton.disabled = false;
  renderTurns();
}

async function loadTranscript() {
  const transcriptId = transcriptIdFromPath();

  if (!transcriptId) {
    transcriptError.hidden = false;
    transcriptError.textContent = "Invalid transcript URL.";
    return;
  }

  try {
    const response = await fetch(`/api/transcripts/${transcriptId}`);
    const data = await response.json();

    if (!response.ok) {
      throw new Error(data.detail || "Transcript could not be loaded.");
    }

    renderTranscript(data);
  } catch (error) {
    transcriptTitle.textContent = "Transcript unavailable";
    turnCount.textContent = "Unavailable";
    transcriptError.hidden = false;
    transcriptError.textContent = error.message;
  }
}

transcriptSearch.addEventListener("input", () => {
  if (transcript) {
    renderTurns(transcriptSearch.value);
  }
});

copyButton.addEventListener("click", async () => {
  if (!transcript) {
    return;
  }

  const text = transcript.turns.map((turn) => (
    `${turn.speaker_name || "Unknown speaker"}\n${turn.content}`
  )).join("\n\n");

  try {
    await navigator.clipboard.writeText(text);
    copyButton.textContent = "Copied";
    window.setTimeout(() => {
      copyButton.textContent = "Copy";
    }, 1500);
  } catch (error) {
    copyButton.textContent = "Copy failed";
  }
});

loadTranscript();
