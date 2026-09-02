"""Regression coverage for private, structured execution traces."""

from __future__ import annotations

import unittest

from agent.trace import TurnTraceRecorder, public_turn_trace, trace_item_detail
from agent.tools.base import ToolResult


class TurnTraceRecorderTests(unittest.TestCase):
    """Verify trace structure, explicit payload access, and credential redaction."""

    def test_public_trace_hides_bodies_and_detail_redacts_credentials(self) -> None:
        saved: list[dict[str, object]] = []
        trace = TurnTraceRecorder(
            conversation_id="conversation-1",
            turn_id=1,
            on_change=saved.append,
        )
        model_id = trace.model_started(
            step=1,
            request={"model": "test", "api_key": "sk-abcdefghijklmnopqrstuvwxyz"},
        )
        trace.model_finished(
            model_id,
            response={"response_id": "response-1", "tool_calls": [], "usage": None},
            duration_ms=12,
        )
        trace.finish(status="completed", message="done", steps=0)

        private = trace.snapshot()
        public = public_turn_trace(private)
        detail = trace_item_detail(private, model_id)

        self.assertTrue(saved)
        self.assertNotIn("request", public["items"][0])
        self.assertIsNotNone(detail)
        self.assertEqual(detail["request"]["api_key"], "[已脱敏]")

    def test_context_selection_is_visible_only_in_opened_model_detail(self) -> None:
        """The trace explains context selection without placing request bodies in list views."""

        trace = TurnTraceRecorder(
            conversation_id="conversation-1",
            turn_id=1,
            on_change=lambda _: None,
        )
        model_id = trace.model_started(step=1, request={"model": "test"})
        trace.record_llm_payload(
            "request",
            {
                "request": {"model": "test", "input": [], "tools": []},
                "context": {"estimated_input_tokens": 120, "summary_version": 2},
            },
        )
        private = trace.snapshot()
        public = public_turn_trace(private)
        detail = trace_item_detail(private, model_id)

        self.assertNotIn("request", public["items"][0])
        self.assertEqual(detail["attributes"]["context"]["summary_version"], 2)

    def test_tool_trace_exposes_safe_sandbox_facts(self) -> None:
        """The trace identifies isolation without copying arbitrary tool metadata."""

        trace = TurnTraceRecorder(
            conversation_id="conversation-1",
            turn_id=1,
            on_change=lambda _: None,
        )
        item_id = trace.tool_started(
            step=1,
            parent_id=None,
            call=type("Call", (), {"name": "run_command", "arguments": {}})(),
        )
        trace.tool_finished(
            item_id,
            result=ToolResult.succeeded(
                "exit code: 0",
                metadata={
                    "execution_scope": "sandbox",
                    "network": "disabled",
                    "internal_note": "must not appear",
                },
            ),
            duration_ms=15,
        )

        item = public_turn_trace(trace.snapshot())["items"][0]
        self.assertEqual(item["attributes"]["execution"]["execution_scope"], "sandbox")
        self.assertEqual(item["attributes"]["execution"]["network"], "disabled")
        self.assertNotIn("internal_note", item["attributes"]["execution"])
