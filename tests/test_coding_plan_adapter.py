import json
import os
import sys
import unittest
from pathlib import Path
from unittest.mock import MagicMock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from coding_plan_adapter import (  # noqa: E402
    CodingPlanAdapter,
    build_request_descriptors,
    extract_content_from_response,
    infer_provider_name,
    normalize_provider_config,
)


class CodingPlanAdapterTest(unittest.TestCase):
    def test_infer_provider_from_model_and_base(self):
        self.assertEqual(infer_provider_name({"model": "claude-3-7-sonnet"}), "anthropic")
        self.assertEqual(infer_provider_name({"model": "gemini-2.5-flash"}), "gemini")
        self.assertEqual(infer_provider_name({"base_url": "https://api.deepseek.com/v1"}), "deepseek")
        self.assertEqual(infer_provider_name({"base_url": "https://api.openai.com/v1"}), "openai")

    def test_normalize_provider_config_uses_env_fallback(self):
        original = dict(os_environ=os.environ)
        try:
            os.environ["OPENAI_API_KEY"] = "test-openai-key"
            config = normalize_provider_config(
                {
                    "provider": "openai",
                    "model": "gpt-5-mini",
                }
            )
            self.assertEqual(config.api_key, "test-openai-key")
            self.assertEqual(config.base_url, "https://api.openai.com/v1")
        finally:
            os.environ.clear()
            os.environ.update(original["os_environ"])

    def test_build_request_descriptors_for_openai_family(self):
        descriptors = build_request_descriptors(
            {
                "provider": "openai",
                "model": "gpt-5-mini",
                "api_key": "k",
            },
            prompt="hello",
        )
        self.assertGreaterEqual(len(descriptors), 2)
        self.assertTrue(descriptors[0].url.endswith("/chat/completions"))
        self.assertEqual(descriptors[0].headers["Authorization"], "Bearer k")

    def test_extract_content_from_anthropic_response(self):
        text = extract_content_from_response(
            "anthropic",
            {
                "content": [
                    {"type": "text", "text": '{"tag":"SR"}'},
                ]
            },
        )
        self.assertEqual(text, '{"tag":"SR"}')

    def test_generate_openai_compatible_plan(self):
        response = MagicMock()
        response.status_code = 200
        response.text = json.dumps(
            {
                "choices": [
                    {
                        "message": {
                            "content": '{"tag":"SR","description":"desc","keywords":[{"keyword":"symbolic regression","query":"symbolic regression"}],"intent_queries":[{"query":"equation discovery"}]}'
                        }
                    }
                ]
            }
        )
        response.json.return_value = json.loads(response.text)

        adapter = CodingPlanAdapter(
            {
                "provider": "openai",
                "model": "gpt-5-mini",
                "api_key": "k",
            }
        )
        result = adapter.generate("test prompt", request_func=lambda *args, **kwargs: response)
        self.assertEqual(result["provider"], "openai")
        self.assertEqual(result["parsed"]["tag"], "SR")
        self.assertEqual(result["parsed"]["intent_queries"][0]["query"], "equation discovery")


if __name__ == "__main__":
    unittest.main()
