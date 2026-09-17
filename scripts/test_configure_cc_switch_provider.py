import importlib.util
import pathlib
import unittest


MODULE_PATH = pathlib.Path(__file__).with_name("configure_cc_switch_provider.py")
SPEC = importlib.util.spec_from_file_location("configure_cc_switch_provider", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class ConfigureProviderTests(unittest.TestCase):
    def test_model_role_inference_uses_complete_tokens(self):
        models = [
            "gpt-5.6-solar",
            "gpt-5.6-terrestrial",
            "claude-opuscular-9",
            "claude-sonnet-5",
            "gpt-5.6-sol",
        ]

        defaults = MODULE.infer_codex_to_claude_role_defaults(models)

        self.assertEqual(defaults["opus"], "gpt-5.6-sol")
        self.assertEqual(defaults["sonnet"], "claude-sonnet-5")
        self.assertEqual(defaults["fable"], "")

    def test_redirects_are_rejected_before_credentials_can_be_forwarded(self):
        handler = MODULE.RejectRedirects()
        request = MODULE.Request(
            "https://relay.example/v1/models",
            headers={"Authorization": "Bearer secret"},
        )

        redirected = handler.redirect_request(
            request,
            None,
            302,
            "Found",
            {},
            "http://other.example/v1/models",
        )

        self.assertIsNone(redirected)


if __name__ == "__main__":
    unittest.main()
