"""Adapter round-trip tests against the mock judge server."""

import unittest

from judge_auditor.adapters import (AnthropicAdapter, CustomHTTPAdapter,
                                    JudgeAdapter, OllamaAdapter,
                                    OpenAIAdapter)
from tests.mock_judge import MockJudgeServer

SYSTEM = "You are an impartial judge."
USER = """Question: What is 2+2?
Answer A:
4

Answer B:
5

Which answer is better? Reply with exactly one letter: A or B."""


class AdapterTestBase(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.server = MockJudgeServer()
        cls.server.start()
        cls.addClassCleanup(cls.server.stop)


class TestOpenAIAdapter(AdapterTestBase):
    def test_round_trip(self):
        adapter = OpenAIAdapter(base_url=self.server.base_url + "/v1",
                                model="mock")
        self.assertIsInstance(adapter, JudgeAdapter)
        out = adapter.judge(SYSTEM, USER)
        self.assertIn(out.strip(), ("A", "B"))

    def test_behavior_selection(self):
        from judge_auditor.adapters import CustomHTTPAdapter
        adapter = CustomHTTPAdapter(
            url=self.server.base_url + "/v1/chat/completions",
            headers={"X-Mock-Behavior": "always_first"})
        self.assertEqual(adapter.judge(SYSTEM, USER).strip(), "A")


class TestAnthropicAdapter(AdapterTestBase):
    def test_round_trip(self):
        adapter = AnthropicAdapter(base_url=self.server.base_url,
                                   model="mock")
        out = adapter.judge(SYSTEM, USER)
        self.assertIn(out.strip(), ("A", "B"))


class TestOllamaAdapter(AdapterTestBase):
    def test_is_openai_compatible(self):
        adapter = OllamaAdapter(base_url=self.server.base_url + "/v1",
                                model="mock")
        self.assertIsInstance(adapter, OpenAIAdapter)
        out = adapter.judge(SYSTEM, USER)
        self.assertIn(out.strip(), ("A", "B"))


class TestCustomHTTPAdapter(AdapterTestBase):
    def test_dot_path_extraction(self):
        adapter = CustomHTTPAdapter(
            url=self.server.base_url + "/v1/chat/completions",
            response_path="choices.0.message.content")
        out = adapter.judge(SYSTEM, USER)
        self.assertIn(out.strip(), ("A", "B"))

    def test_custom_template(self):
        adapter = CustomHTTPAdapter(
            url=self.server.base_url + "/v1/chat/completions",
            request_template=('{"model": "m", "messages": '
                              '[{"role": "user", "content": "{prompt}"}]}'))
        out = adapter.judge(SYSTEM, USER)
        self.assertIn(out.strip(), ("A", "B"))


class TestMockBehaviors(AdapterTestBase):
    def _verdict(self, behavior, user=USER, extra_headers=None):
        headers = {"X-Mock-Behavior": behavior}
        headers.update(extra_headers or {})
        adapter = CustomHTTPAdapter(
            url=self.server.base_url + "/v1/chat/completions",
            headers=headers)
        return adapter.judge(SYSTEM, user).strip()

    def test_always_long(self):
        # "4" vs "5" is a tie on length -> tie-break goes to A.
        self.assertEqual(self._verdict("always_long"), "A")
        long_b = USER.replace("Answer B:\n5", "Answer B:\n5\n5\n5\n5")
        self.assertEqual(self._verdict("always_long", user=long_b), "B")

    def test_always_first(self):
        self.assertEqual(self._verdict("always_first"), "A")

    def test_neutral_alternating(self):
        # Order-independent: over an even run it must be exactly balanced
        # and must strictly alternate.
        seen = [self._verdict("neutral_alternating") for _ in range(10)]
        self.assertEqual(seen.count("A"), 5)
        self.assertEqual(seen.count("B"), 5)
        for first, second in zip(seen, seen[1:]):
            self.assertNotEqual(first, second)

    def test_sycophantic(self):
        user = USER.replace("Answer A:\n4", "Answer A:\nYou're absolutely right, 4")
        self.assertEqual(self._verdict("sycophantic", user=user), "A")

    def test_prefer_bullets(self):
        user = USER.replace("Answer B:\n5", "Answer B:\n- five\n- 5")
        self.assertEqual(self._verdict("prefer_bullets", user=user), "B")

    def test_always_self(self):
        user = ("Note: Answer A was written by JudgeModel. "
                "Answer B was written by RivalModel.\n" + USER)
        self.assertEqual(self._verdict("always_self", user=user), "A")
        self.assertEqual(
            self._verdict("always_self", user=user,
                          extra_headers={"X-Mock-Self-Name": "RivalModel"}),
            "B")


if __name__ == "__main__":
    unittest.main()
