"""Regression coverage for local context selection, summaries, and tool artifacts."""

from __future__ import annotations

from unittest.mock import patch
import unittest

from agent.conversation_memory import ConversationMemory


class ConversationMemoryTests(unittest.TestCase):
    """Verify raw history remains durable while request context becomes bounded."""

    def test_summary_replaces_only_an_older_completed_prefix(self) -> None:
        """Recent user-led turns remain exact after an incremental local summary."""

        memory = ConversationMemory()
        history = _five_completed_turns()

        with patch("agent.conversation_memory.COMPACTION_TRIGGER_TOKENS", 1):
            plan = memory.compaction_plan(history)

        self.assertIsNotNone(plan)
        assert plan is not None
        self.assertEqual(plan.source_start, 0)
        self.assertEqual(plan.source_end, 2)
        result = memory.apply_summary(
            plan,
            '{"current_goal":"继续完成项目","completed":["完成第一轮"],"open_issues":["验证后续改动"]}',
        )
        selection = memory.build_input(history)

        self.assertEqual(result.summary_version, 1)
        self.assertEqual(result.covered_history_items, 2)
        self.assertEqual(history[0]["content"], "任务 1")
        self.assertEqual(selection.input_items[0]["role"], "developer")
        self.assertEqual(selection.input_items[1], history[2])
        self.assertEqual(selection.covered_history_items, 2)

    def test_large_tool_output_is_referenced_and_read_in_bounded_chunks(self) -> None:
        """The exact output stays local while later requests receive a compact reference."""

        output = "a" * 3_500 + "END" + "b" * 3_500
        history = [
            {"role": "user", "content": "检查日志"},
            {"type": "function_call", "call_id": "call-1", "name": "run_command", "arguments": "{}"},
            {"type": "function_call_output", "call_id": "call-1", "output": output},
        ]
        memory = ConversationMemory()

        selection = memory.build_input(history)
        artifact_id = next(
            item["artifact_id"]
            for item in memory.export_state()["artifacts"]
            if isinstance(item, dict)
        )
        visible_output = selection.input_items[-1]["output"]
        content, metadata = memory.read_artifact(
            history,
            artifact_id=str(artifact_id),
            offset=3_496,
            max_chars=12,
        )

        self.assertIn("read_session_artifact", visible_output)
        self.assertEqual(content, "aaaaENDbbbbb")
        self.assertEqual(metadata["total_chars"], len(output))
        self.assertTrue(metadata["has_more"])
        self.assertEqual(history[-1]["output"], output)

    def test_usage_calibration_is_persisted_with_local_context_state(self) -> None:
        """Provider usage improves later estimates without adding a provider-specific tokenizer."""

        memory = ConversationMemory()
        memory.build_input([{"role": "user", "content": "hello"}])
        memory.record_usage(100)
        state = memory.export_state()
        restored = ConversationMemory()
        restored.restore_state(state)

        self.assertGreater(state["token_multiplier"], 1.0)
        self.assertEqual(restored.export_state()["token_multiplier"], state["token_multiplier"])


def _five_completed_turns() -> list[dict[str, object]]:
    """Build a small transcript with five completed user/assistant turn pairs."""

    history: list[dict[str, object]] = []
    for index in range(1, 6):
        history.append({"role": "user", "content": f"任务 {index}"})
        history.append(
            {
                "type": "message",
                "role": "assistant",
                "content": [{"type": "output_text", "text": f"结果 {index}"}],
            }
        )
    return history
