const refreshButton = document.querySelector("#refresh-button");
const gateTitle = document.querySelector("#gate-title");
const gateStatus = document.querySelector("#gate-status");
const updatedAt = document.querySelector("#updated-at");
const errorBanner = document.querySelector("#error-banner");
const retrievalRate = document.querySelector("#retrieval-rate");
const retrievalCount = document.querySelector("#retrieval-count");
const responseRate = document.querySelector("#response-rate");
const responseCount = document.querySelector("#response-count");
const checksRate = document.querySelector("#checks-rate");
const checksCount = document.querySelector("#checks-count");
const historyCount = document.querySelector("#history-count");
const dimensionList = document.querySelector("#dimension-list");
const judgeModel = document.querySelector("#judge-model");
const trendSvg = document.querySelector("#trend-svg");
const trendEmpty = document.querySelector("#trend-empty");
const caseTableBody = document.querySelector("#case-table-body");
const casesEmpty = document.querySelector("#cases-empty");
const caseFilterButtons = document.querySelectorAll("[data-case-filter]");
const caseInspector = document.querySelector("#case-inspector");
const inspectorSuite = document.querySelector("#inspector-suite");
const inspectorTitle = document.querySelector("#inspector-title");
const inspectorQuestion = document.querySelector("#inspector-question");
const inspectorFindings = document.querySelector("#inspector-findings");
const inspectorAnswerBlock = document.querySelector("#inspector-answer-block");
const inspectorAnswer = document.querySelector("#inspector-answer");
const closeInspector = document.querySelector("#close-inspector");
const runList = document.querySelector("#run-list");

const SVG_NAMESPACE = "http://www.w3.org/2000/svg";
const TREND_COLORS = {
  retrieval: "#2b6192",
  response: "#12684f",
};

let latestCases = [];
let selectedCaseId = null;
let caseFilter = "all";

function percent(value) {
  return Number.isFinite(value) ? `${Math.round(value * 100)}%` : "--";
}

function dateTime(value) {
  if (!value) {
    return "Unknown time";
  }

  return new Intl.DateTimeFormat("en-SG", {
    dateStyle: "medium",
    timeStyle: "short",
  }).format(new Date(value));
}

function summaryFor(report) {
  return report?.summary || null;
}

function setMetric(rateElement, countElement, summary) {
  if (!summary) {
    rateElement.textContent = "--";
    countElement.textContent = "No report";
    return;
  }

  rateElement.textContent = percent(summary.case_pass_rate);
  countElement.textContent = `${summary.passed_cases}/${summary.total_cases} passed`;
}

function renderOverview(data) {
  const retrieval = summaryFor(data.latest.retrieval);
  const response = summaryFor(data.latest.response);
  const reports = [data.latest.retrieval, data.latest.response].filter(Boolean);
  const summaries = [retrieval, response].filter(Boolean);
  const totalChecks = summaries.reduce(
    (total, summary) => total + (summary.total_checks || 0),
    0
  );
  const passedChecks = summaries.reduce(
    (total, summary) => total + (summary.passed_checks || 0),
    0
  );
  const failedCases = summaries.reduce(
    (total, summary) => total + (summary.failed_cases || 0),
    0
  );

  setMetric(retrievalRate, retrievalCount, retrieval);
  setMetric(responseRate, responseCount, response);
  checksRate.textContent = totalChecks ? percent(passedChecks / totalChecks) : "--";
  checksCount.textContent = totalChecks
    ? `${passedChecks}/${totalChecks} passed`
    : "No checks";
  historyCount.textContent = String(data.history.length);

  const latestTimestamp = reports
    .map((report) => report.generated_at)
    .filter(Boolean)
    .sort()
    .at(-1);

  if (!reports.length) {
    gateTitle.textContent = "No evaluation reports found";
    gateStatus.textContent = "No data";
    gateStatus.className = "status-badge loading";
  } else if (failedCases) {
    gateTitle.textContent = `${failedCases} latest case${failedCases === 1 ? "" : "s"} failed`;
    gateStatus.textContent = "Failed";
    gateStatus.className = "status-badge fail";
  } else {
    gateTitle.textContent = "Latest quality gate passed";
    gateStatus.textContent = "Passed";
    gateStatus.className = "status-badge pass";
  }

  updatedAt.textContent = latestTimestamp
    ? `Updated ${dateTime(latestTimestamp)}`
    : "Waiting for reports";
}

function renderDimensions(report) {
  dimensionList.replaceChildren();
  const scores = report?.summary?.average_judge_scores;
  judgeModel.textContent = report?.judge_model || "No judge report";

  const dimensions = [
    ["Groundedness", "groundedness"],
    ["Relevance", "relevance"],
    ["Completeness", "completeness"],
    ["Citation quality", "citation_quality"],
  ];

  for (const [label, key] of dimensions) {
    const score = scores?.[key];
    const row = document.createElement("div");
    row.className = "dimension-row";

    const name = document.createElement("span");
    name.textContent = label;

    const track = document.createElement("div");
    track.className = "score-track";
    const fill = document.createElement("div");
    fill.className = "score-fill";
    fill.style.width = Number.isFinite(score) ? `${(score / 5) * 100}%` : "0";
    track.appendChild(fill);

    const value = document.createElement("strong");
    value.textContent = Number.isFinite(score) ? score.toFixed(1) : "--";
    row.append(name, track, value);
    dimensionList.appendChild(row);
  }
}

function svgElement(name, attributes = {}) {
  const element = document.createElementNS(SVG_NAMESPACE, name);
  for (const [key, value] of Object.entries(attributes)) {
    element.setAttribute(key, String(value));
  }
  return element;
}

function renderTrend(history) {
  trendSvg.replaceChildren();
  const points = history
    .filter((item) => item.generated_at)
    .map((item) => ({ ...item, time: new Date(item.generated_at).getTime() }))
    .filter((item) => Number.isFinite(item.time))
    .sort((a, b) => a.time - b.time);

  trendEmpty.hidden = points.length > 0;
  if (!points.length) {
    return;
  }

  const width = 640;
  const height = 210;
  const margin = { top: 18, right: 18, bottom: 30, left: 42 };
  const plotWidth = width - margin.left - margin.right;
  const plotHeight = height - margin.top - margin.bottom;
  const times = points.map((item) => item.time);
  const minTime = Math.min(...times);
  const maxTime = Math.max(...times);
  const x = (time) => (
    minTime === maxTime
      ? margin.left + plotWidth / 2
      : margin.left + ((time - minTime) / (maxTime - minTime)) * plotWidth
  );
  const y = (rate) => margin.top + (1 - rate) * plotHeight;

  trendSvg.setAttribute("viewBox", `0 0 ${width} ${height}`);

  for (const rate of [0, 0.5, 1]) {
    trendSvg.appendChild(svgElement("line", {
      x1: margin.left,
      x2: width - margin.right,
      y1: y(rate),
      y2: y(rate),
      class: "trend-grid",
    }));
    const label = svgElement("text", {
      x: margin.left - 8,
      y: y(rate) + 4,
      "text-anchor": "end",
      class: "trend-label",
    });
    label.textContent = `${rate * 100}%`;
    trendSvg.appendChild(label);
  }

  for (const suiteType of ["retrieval", "response"]) {
    const suitePoints = points.filter((item) => item.suite_type === suiteType);
    if (!suitePoints.length) {
      continue;
    }

    const pathData = suitePoints.map((item, index) => {
      const command = index === 0 ? "M" : "L";
      return `${command}${x(item.time).toFixed(1)},${y(item.case_pass_rate).toFixed(1)}`;
    }).join(" ");
    trendSvg.appendChild(svgElement("path", {
      d: pathData,
      class: "trend-line",
      stroke: TREND_COLORS[suiteType],
    }));

    for (const item of suitePoints) {
      const point = svgElement("circle", {
        cx: x(item.time),
        cy: y(item.case_pass_rate),
        r: 4.5,
        fill: TREND_COLORS[suiteType],
        class: "trend-point",
      });
      const title = svgElement("title");
      title.textContent = `${suiteType}: ${percent(item.case_pass_rate)} on ${dateTime(item.generated_at)}`;
      point.appendChild(title);
      trendSvg.appendChild(point);
    }
  }

  for (const [time, anchor] of [[minTime, "start"], [maxTime, "end"]]) {
    const label = svgElement("text", {
      x: x(time),
      y: height - 8,
      "text-anchor": anchor,
      class: "trend-label",
    });
    label.textContent = new Intl.DateTimeFormat("en-SG", {
      month: "short",
      day: "numeric",
    }).format(new Date(time));
    trendSvg.appendChild(label);
  }
}

function caseScores(item) {
  const scores = item.result.judge?.scores;
  if (!scores) {
    return "Deterministic";
  }

  return [
    `G${scores.groundedness}`,
    `R${scores.relevance}`,
    `C${scores.completeness}`,
    `Q${scores.citation_quality}`,
  ].join(" ");
}

function renderCases() {
  caseTableBody.replaceChildren();
  const visible = latestCases.filter(
    (item) => caseFilter === "all" || !item.result.passed
  );
  casesEmpty.hidden = visible.length > 0;

  for (const item of visible) {
    const result = item.result;
    const row = document.createElement("tr");
    row.classList.toggle("selected", selectedCaseId === `${item.suite}:${result.id}`);

    const caseCell = document.createElement("td");
    const caseButton = document.createElement("button");
    caseButton.type = "button";
    caseButton.className = "case-link";
    caseButton.textContent = result.id;
    caseButton.addEventListener("click", () => inspectCase(item));
    caseCell.appendChild(caseButton);

    const suiteCell = document.createElement("td");
    suiteCell.textContent = item.suite;

    const resultCell = document.createElement("td");
    const badge = document.createElement("span");
    badge.className = `result-badge ${result.passed ? "pass" : "fail"}`;
    badge.textContent = result.passed ? "Pass" : "Fail";
    resultCell.appendChild(badge);

    const passedChecks = (result.checks || []).filter((check) => check.passed).length;
    const checksCell = document.createElement("td");
    checksCell.textContent = `${passedChecks}/${(result.checks || []).length}`;

    const scoreCell = document.createElement("td");
    scoreCell.className = "score-code";
    scoreCell.textContent = caseScores(item);

    const durationCell = document.createElement("td");
    durationCell.textContent = `${Math.round(result.duration_ms || 0)} ms`;

    row.append(caseCell, suiteCell, resultCell, checksCell, scoreCell, durationCell);
    caseTableBody.appendChild(row);
  }
}

function finding(text, failed = false) {
  const element = document.createElement("div");
  element.className = `finding${failed ? " fail" : ""}`;
  element.textContent = text;
  return element;
}

function inspectCase(item) {
  const result = item.result;
  selectedCaseId = `${item.suite}:${result.id}`;
  inspectorSuite.textContent = `${item.suite} evaluation`;
  inspectorTitle.textContent = result.id;
  inspectorQuestion.textContent = result.question;
  inspectorFindings.replaceChildren();

  const failures = (result.checks || []).filter((check) => !check.passed);
  if (!failures.length) {
    inspectorFindings.appendChild(finding("All deterministic and model checks passed."));
  } else {
    for (const check of failures) {
      inspectorFindings.appendChild(
        finding(
          `${check.name}: expected ${JSON.stringify(check.expected)}, got ${JSON.stringify(check.actual)}`,
          true
        )
      );
    }
  }

  if (result.judge?.reasoning) {
    inspectorFindings.appendChild(finding(`Judge: ${result.judge.reasoning}`));
  }
  if (result.error) {
    inspectorFindings.appendChild(finding(`Execution error: ${result.error}`, true));
  }

  inspectorAnswerBlock.hidden = !result.answer;
  inspectorAnswer.textContent = result.answer || "";
  caseInspector.hidden = false;
  renderCases();
}

function renderRunLedger(history) {
  runList.replaceChildren();

  for (const item of history.slice(0, 12)) {
    const row = document.createElement("div");
    row.className = "run-row";

    const suite = document.createElement("strong");
    suite.textContent = item.suite_type;
    const timestamp = document.createElement("span");
    timestamp.textContent = dateTime(item.generated_at);
    const cases = document.createElement("span");
    cases.textContent = `${item.passed_cases}/${item.total_cases} cases`;
    const rate = document.createElement("span");
    rate.textContent = percent(item.case_pass_rate);
    const model = document.createElement("span");
    model.textContent = item.judge_model || item.suite || "--";

    row.append(suite, timestamp, cases, rate, model);
    runList.appendChild(row);
  }

  if (!history.length) {
    const empty = document.createElement("p");
    empty.className = "empty-state";
    empty.style.position = "static";
    empty.style.minHeight = "90px";
    empty.textContent = "No evaluation runs found";
    runList.appendChild(empty);
  }
}

function renderErrors(errors) {
  errorBanner.hidden = !errors.length;
  errorBanner.textContent = errors.join(" ");
}

async function loadDashboard() {
  refreshButton.disabled = true;
  gateStatus.textContent = "Loading";
  gateStatus.className = "status-badge loading";

  try {
    const response = await fetch("/api/evaluations?history_limit=50");
    const data = await response.json();
    if (!response.ok) {
      throw new Error(data.detail || "Evaluation reports could not be loaded.");
    }

    latestCases = [];
    for (const suite of ["retrieval", "response"]) {
      for (const result of data.latest[suite]?.results || []) {
        latestCases.push({ suite, result });
      }
    }

    renderOverview(data);
    renderDimensions(data.latest.response);
    renderTrend(data.history);
    renderCases();
    renderRunLedger(data.history);
    renderErrors(data.errors || []);
  } catch (error) {
    gateTitle.textContent = "Evaluation dashboard unavailable";
    gateStatus.textContent = "Error";
    gateStatus.className = "status-badge fail";
    renderErrors([error.message]);
  } finally {
    refreshButton.disabled = false;
  }
}

for (const button of caseFilterButtons) {
  button.addEventListener("click", () => {
    caseFilter = button.dataset.caseFilter;
    for (const candidate of caseFilterButtons) {
      candidate.classList.toggle("active", candidate === button);
    }
    renderCases();
  });
}

closeInspector.addEventListener("click", () => {
  selectedCaseId = null;
  caseInspector.hidden = true;
  renderCases();
});

refreshButton.addEventListener("click", loadDashboard);
loadDashboard();
