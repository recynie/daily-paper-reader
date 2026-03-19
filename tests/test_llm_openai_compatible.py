import unittest

from src.llm import (
    BltClient,
    ClientFactory,
    OpenAICompatibleClient,
    build_chat_client,
)


class LlmOpenAICompatibleTest(unittest.TestCase):
    def test_client_factory_supports_openai_provider(self):
        client = ClientFactory.from_parts(
            "openai",
            "gpt-4o-mini",
            api_key="test-key",
            base_url="https://api.openai.com/v1",
        )
        self.assertIsInstance(client, OpenAICompatibleClient)
        self.assertEqual(client.model, "gpt-4o-mini")
        self.assertEqual(client.base_url, "https://api.openai.com/v1")

    def test_build_chat_client_infers_prefixed_provider(self):
        client = build_chat_client(
            api_key="test-key",
            model="openai/gpt-5-chat",
            base_url="https://gateway.example.com/v1",
            default_provider="blt",
        )
        self.assertIsInstance(client, OpenAICompatibleClient)
        self.assertEqual(client.model, "gpt-5-chat")
        self.assertEqual(client.base_url, "https://gateway.example.com/v1")

    def test_build_chat_client_can_still_create_blt_client(self):
        client = build_chat_client(
            api_key="test-key",
            model="gemini-3-flash-preview",
            base_url="https://api.bltcy.ai/v1",
            default_provider="blt",
        )
        self.assertIsInstance(client, BltClient)
        self.assertEqual(client.model, "gemini-3-flash-preview")


if __name__ == "__main__":
    unittest.main()
