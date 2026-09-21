import unittest

from benchmarks.alfworld.skilldag_runtime import yunwu_payload_model


class YunwuModelRoutingTests(unittest.TestCase):
    def test_strips_yunwu_prefix(self):
        self.assertEqual(yunwu_payload_model("yunwu/MiniMax-M2.7"), "MiniMax-M2.7")

    def test_strips_litellm_openai_prefix(self):
        self.assertEqual(
            yunwu_payload_model("openai/gpt-5.2-codex"),
            "gpt-5.2-codex",
        )

    def test_strips_responses_routing_prefix_for_chat_fallback(self):
        self.assertEqual(
            yunwu_payload_model("openai-responses/gpt-5.2-codex"),
            "gpt-5.2-codex",
        )

    def test_preserves_downstream_model_id_and_casing(self):
        self.assertEqual(yunwu_payload_model("MiniMax-M2.7"), "MiniMax-M2.7")


if __name__ == "__main__":
    unittest.main()
