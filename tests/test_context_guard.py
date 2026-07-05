import os
import unittest
from unittest.mock import patch

from rag_rtl import context_guard
from rag_rtl.context_guard import ContextLengthError, clamp_max_tokens

BASE_URL = "http://guard-test/v1"


def _seed_caches(limit, tokenize_supported=False):
    context_guard._SERVER_LIMITS[BASE_URL] = limit
    context_guard._TOKENIZE_SUPPORTED[BASE_URL] = tokenize_supported


class ContextGuardTests(unittest.TestCase):
    def setUp(self):
        context_guard._SERVER_LIMITS.clear()
        context_guard._TOKENIZE_SUPPORTED.clear()

    def test_clamps_max_tokens_to_remaining_window(self):
        _seed_caches(limit=1000)
        payload = {
            "model": "m",
            "messages": [{"role": "user", "content": "a" * 400}],  # ~100 heuristic tokens + 8 overhead
            "max_tokens": 5000,
        }
        clamp_max_tokens(payload, base_url=BASE_URL, min_completion_tokens=10, margin_tokens=0)
        self.assertEqual(payload["max_tokens"], 1000 - (400 // 4 + 8))

    def test_small_request_is_untouched(self):
        _seed_caches(limit=1000)
        payload = {"model": "m", "messages": [{"role": "user", "content": "hi"}], "max_tokens": 64}
        clamp_max_tokens(payload, base_url=BASE_URL, min_completion_tokens=10, margin_tokens=0)
        self.assertEqual(payload["max_tokens"], 64)

    def test_oversized_prompt_raises_before_sending(self):
        _seed_caches(limit=100)
        payload = {
            "model": "m",
            "messages": [{"role": "user", "content": "a" * 4000}],
            "max_tokens": 64,
        }
        with self.assertRaises(ContextLengthError):
            clamp_max_tokens(payload, base_url=BASE_URL, min_completion_tokens=10, margin_tokens=0)

    def test_unknown_window_is_a_no_op(self):
        _seed_caches(limit=None)
        payload = {"model": "m", "messages": [{"role": "user", "content": "a" * 4000}], "max_tokens": 5000}
        clamp_max_tokens(payload, base_url=BASE_URL)
        self.assertEqual(payload["max_tokens"], 5000)

    def test_guard_can_be_disabled_via_env(self):
        _seed_caches(limit=100)
        payload = {"model": "m", "messages": [{"role": "user", "content": "a" * 4000}], "max_tokens": 5000}
        with patch.dict(os.environ, {"VLLM_CONTEXT_GUARD": "0"}):
            clamp_max_tokens(payload, base_url=BASE_URL)
        self.assertEqual(payload["max_tokens"], 5000)

    def test_explicit_max_model_len_wins_over_probe(self):
        # No cache seeded: an explicit window must not trigger a server probe.
        context_guard._TOKENIZE_SUPPORTED[BASE_URL] = False
        payload = {"model": "m", "messages": [{"role": "user", "content": "a" * 400}], "max_tokens": 5000}
        clamp_max_tokens(
            payload, base_url=BASE_URL, max_model_len=1000, min_completion_tokens=10, margin_tokens=0
        )
        self.assertEqual(payload["max_tokens"], 1000 - (400 // 4 + 8))

    def test_context_length_error_is_runtime_error(self):
        self.assertTrue(issubclass(ContextLengthError, RuntimeError))


if __name__ == "__main__":
    unittest.main()
