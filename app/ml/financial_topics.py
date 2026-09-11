"""Auditable multi-label topics for financial narrative text."""

from __future__ import annotations

import re
from dataclasses import dataclass


TOPIC_CLASSIFIER_VERSION = "financial-keywords-v1"


@dataclass(frozen=True)
class TopicDefinition:
    """Stable topic metadata and the phrases that identify it."""

    key: str
    label: str
    terms: tuple[str, ...]


# The first topic layer is deliberately deterministic. It makes each match
# explainable, costs nothing to rerun, and provides a measurable baseline for
# a later embedding or supervised topic classifier.
TOPIC_TAXONOMY = (
    TopicDefinition(
        key="margins_profitability",
        label="Margins & Profitability",
        terms=(
            "gross margin",
            "operating margin",
            "profit margin",
            "margin expansion",
            "margin pressure",
            "margin rate",
            "margins",
            "margin",
            "profitability",
            "operating leverage",
            "deleverage",
            "gross profit",
            "operating income",
        ),
    ),
    TopicDefinition(
        key="guidance_outlook",
        label="Guidance & Outlook",
        terms=(
            "guidance",
            "outlook",
            "forecast",
            "full year",
            "next quarter",
            "expectations",
            "projected",
            "projection",
            "raise our",
            "lower our",
        ),
    ),
    TopicDefinition(
        key="revenue_growth",
        label="Revenue & Growth",
        terms=(
            "revenue growth",
            "revenues",
            "revenue",
            "net sales",
            "sales growth",
            "same store sales",
            "comparable sales",
            "organic growth",
            "top line",
        ),
    ),
    TopicDefinition(
        key="demand_consumer",
        label="Demand & Consumer",
        terms=(
            "consumer spending",
            "customer traffic",
            "store traffic",
            "unit volume",
            "market share",
            "transactions",
            "bookings",
            "backlog",
            "orders",
            "demand",
        ),
    ),
    TopicDefinition(
        key="costs_efficiency",
        label="Costs & Efficiency",
        terms=(
            "operating expense",
            "operating expenses",
            "cost reduction",
            "cost savings",
            "labor cost",
            "freight cost",
            "restructuring",
            "productivity",
            "efficiency",
            "automation",
            "headcount",
            "expenses",
        ),
    ),
    TopicDefinition(
        key="pricing_inflation",
        label="Pricing & Inflation",
        terms=(
            "price increases",
            "price increase",
            "price investment",
            "pricing",
            "inflation",
            "deflation",
            "markdowns",
            "markdown",
            "promotions",
            "promotion",
        ),
    ),
    TopicDefinition(
        key="supply_chain_inventory",
        label="Supply Chain & Inventory",
        terms=(
            "supply chain",
            "distribution center",
            "lead time",
            "inventories",
            "inventory",
            "logistics",
            "fulfillment",
            "shortages",
            "shortage",
            "suppliers",
            "supplier",
        ),
    ),
    TopicDefinition(
        key="capital_allocation",
        label="Capital Allocation",
        terms=(
            "capital allocation",
            "capital expenditures",
            "capital expenditure",
            "free cash flow",
            "share repurchase",
            "stock repurchase",
            "debt repayment",
            "buybacks",
            "buyback",
            "dividends",
            "dividend",
            "capex",
        ),
    ),
    TopicDefinition(
        key="risk_regulation",
        label="Risk & Regulation",
        terms=(
            "cyber security",
            "cybersecurity",
            "data breach",
            "foreign exchange",
            "interest rate",
            "geopolitical",
            "regulatory",
            "regulation",
            "litigation",
            "tariffs",
            "tariff",
            "risks",
            "risk",
        ),
    ),
    TopicDefinition(
        key="technology_ai",
        label="Technology & AI",
        terms=(
            "artificial intelligence",
            "generative ai",
            "machine learning",
            "digital transformation",
            "technology investment",
            "data centers",
            "data center",
            "online marketplace",
            "e commerce",
            "ecommerce",
            "cloud",
            "ai",
        ),
    ),
)


def _normalize_text(value: str) -> str:
    """Normalize punctuation while preserving exact word boundaries."""

    return " ".join(re.findall(r"[a-z0-9]+", value.lower()))


def classify_financial_topics(text: str | None) -> list[dict]:
    """Return every financial topic matched by one narrative text item."""

    normalized = _normalize_text(str(text or ""))

    if not normalized:
        return []

    padded_text = f" {normalized} "
    matches = []

    for topic in TOPIC_TAXONOMY:
        matched_terms = [
            term
            for term in topic.terms
            if f" {_normalize_text(term)} " in padded_text
        ]

        if matched_terms:
            matches.append({
                "topic_key": topic.key,
                "topic_label": topic.label,
                "matched_terms": matched_terms,
            })

    return matches


def topic_taxonomy_payload() -> dict:
    """Expose the exact classifier contract used by API responses."""

    return {
        "version": TOPIC_CLASSIFIER_VERSION,
        "method": "deterministic_multi_label_phrase_matching",
        "topics": [
            {
                "topic_key": topic.key,
                "topic_label": topic.label,
                "terms": list(topic.terms),
            }
            for topic in TOPIC_TAXONOMY
        ],
    }
