"""Tests for the LCEL event research-brief workflow."""

import unittest
from unittest.mock import patch

from fastapi.testclient import TestClient
from langchain_core.runnables import RunnableLambda

from app.main import app
from app.rag.brief_workflow import (
    EventBriefContent,
    build_brief_question,
    build_event_brief_chain,
    collect_brief_context,
    compact_transcript_sentiment,
)


def _brief(citation="[S1]"):
    return EventBriefContent(
        headline="WMT event brief",
        executive_summary=f"Management discussed margins {citation}.",
        key_developments=[f"Margins expanded {citation}."],
        topic_signals=["FinBERT margin sentiment improved."],
        market_reaction=["One-year return was measured versus SPY."],
        risks=[f"Demand remains a stated risk {citation}."],
        watch_items=["Watch gross margin next quarter."],
        limitations=[],
    )


def _context(_):
    return {
        "ticker": "WMT",
        "event_type": "earnings",
        "question": "Prepare a WMT earnings brief.",
        "document_context": "[S1]\nTicker: WMT\nContent: Margin expanded.",
        "market_context_text": "Ticker: WMT",
        "sentiment_context_text": "{}",
        "market_context": [],
        "sentiment_context": {},
        "sources": [{"source": "S1"}],
    }


class BriefWorkflowTests(unittest.TestCase):
    def test_question_builder_preserves_event_scope(self):
        question = build_brief_question(
            ticker="wmt",
            event_type="earnings",
            fiscal_period="2026Q4",
        )

        self.assertIn("WMT", question)
        self.assertIn("2026Q4 earnings call", question)
        self.assertIn("versus SPY", question)

    def test_question_builder_supports_comparison_and_focus(self):
        question = build_brief_question(
            ticker=None,
            tickers=["wmt", "cost"],
            event_type="combined",
            fiscal_period="2026Q4",
            form_type="10-K",
            focus="Compare margin quality.",
        )

        self.assertIn("comparative event brief for WMT, COST", question)
        self.assertIn("2026Q4 earnings call", question)
        self.assertIn("latest 10-K SEC filing", question)
        self.assertIn("Compare margin quality", question)

    def test_transcript_signal_selects_requested_period(self):
        timeline = {
            "ticker": "WMT",
            "calls": [
                {
                    "ticker": "WMT",
                    "fiscal_period": "2026Q3",
                    "call_date": "2026-05-20",
                    "overall": {"score": 0.1, "coverage": 1.0},
                    "score_change": None,
                    "topics": [],
                },
                {
                    "ticker": "WMT",
                    "fiscal_period": "2026Q4",
                    "call_date": "2026-08-20",
                    "overall": {"score": 0.4, "coverage": 1.0},
                    "score_change": 0.3,
                    "topics": [],
                },
            ],
        }

        result = compact_transcript_sentiment(
            timeline,
            fiscal_period="2026Q3",
        )

        self.assertEqual(result["fiscal_period"], "2026Q3")
        self.assertEqual(result["overall"]["score"], 0.1)

    def test_comparison_context_keeps_company_signals_separate(self):
        evidence = {
            "retrieved_results": [],
            "market_context": [],
        }

        with patch(
            "app.rag.brief_workflow.collect_evidence",
            return_value=evidence,
        ) as collect, patch(
            "app.rag.brief_workflow.collect_brief_sentiment",
            side_effect=lambda ticker, **_: {"ticker": ticker},
        ):
            result = collect_brief_context({
                "tickers": ["wmt", "cost"],
                "event_type": "combined",
                "focus": "Compare margins.",
                "top_k": 3,
            })

        self.assertEqual(result["tickers"], ["WMT", "COST"])
        self.assertTrue(result["sentiment_context"]["comparison"])
        self.assertEqual(
            result["sentiment_context"]["companies"],
            {
                "WMT": {"ticker": "WMT"},
                "COST": {"ticker": "COST"},
            },
        )
        self.assertEqual(
            collect.call_args.kwargs["tickers"],
            ["WMT", "COST"],
        )

    def test_lcel_chain_packages_structured_output(self):
        chain = build_event_brief_chain(
            context_loader=_context,
            generation_runnable=RunnableLambda(lambda _: _brief()),
        )

        result = chain.invoke({"ticker": "WMT"})

        self.assertEqual(result["ticker"], "WMT")
        self.assertEqual(
            result["chain_version"],
            "alphalens-event-brief-lcel-v1",
        )
        self.assertEqual(result["brief"]["headline"], "WMT event brief")
        self.assertEqual(result["sources"], [{"source": "S1"}])

    def test_lcel_chain_normalizes_brief_punctuation(self):
        generated = _brief()
        generated.executive_summary = "Guidance was 3.5%\u20134.5% [S1]."
        chain = build_event_brief_chain(
            context_loader=_context,
            generation_runnable=RunnableLambda(lambda _: generated),
        )

        result = chain.invoke({"ticker": "WMT"})

        self.assertEqual(
            result["brief"]["executive_summary"],
            "Guidance was 3.5%-4.5% [S1].",
        )

    def test_lcel_chain_rejects_unknown_citation(self):
        chain = build_event_brief_chain(
            context_loader=_context,
            generation_runnable=RunnableLambda(lambda _: _brief("[S9]")),
        )

        with self.assertRaisesRegex(RuntimeError, "unknown citations: S9"):
            chain.invoke({"ticker": "WMT"})

    def test_brief_api_returns_structured_result(self):
        result = {
            "ticker": "WMT",
            "event_type": "earnings",
            "question": "Prepare a WMT earnings brief.",
            "chain_version": "alphalens-event-brief-lcel-v1",
            "model_name": "gpt-5-mini",
            "brief": _brief("").model_dump(),
            "market_context": [],
            "sentiment_context": {},
            "sources": [],
        }

        with patch(
            "app.api.research.generate_event_brief",
            return_value=result,
        ) as generate:
            response = TestClient(app).post(
                "/api/briefs/generate",
                json={"ticker": "wmt", "event_type": "earnings"},
            )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["brief"]["headline"], "WMT event brief")
        generate.assert_called_once_with(
            ticker="wmt",
            tickers=[],
            event_type="earnings",
            fiscal_period=None,
            form_type=None,
            top_k=6,
            focus=None,
        )

    def test_brief_api_accepts_multiple_tickers(self):
        result = {
            "ticker": "WMT",
            "tickers": ["WMT", "COST"],
            "event_type": "combined",
            "question": "Compare WMT and COST.",
            "chain_version": "alphalens-event-brief-lcel-v1",
            "model_name": "gpt-5-mini",
            "brief": _brief("").model_dump(),
            "market_context": [],
            "sentiment_context": {},
            "sources": [],
        }

        with patch(
            "app.api.research.generate_event_brief",
            return_value=result,
        ) as generate:
            response = TestClient(app).post(
                "/api/briefs/generate",
                json={
                    "tickers": ["wmt", "cost"],
                    "event_type": "combined",
                    "focus": "Compare margins.",
                },
            )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["tickers"], ["WMT", "COST"])
        generate.assert_called_once_with(
            ticker=None,
            tickers=["wmt", "cost"],
            event_type="combined",
            fiscal_period=None,
            form_type=None,
            top_k=6,
            focus="Compare margins.",
        )


if __name__ == "__main__":
    unittest.main()
