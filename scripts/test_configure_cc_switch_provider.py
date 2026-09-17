import importlib.util
import os
import pathlib
import unittest
from unittest.mock import patch


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

    def test_model_opener_uses_dedicated_provider_proxy(self):
        proxy_url = "http://127.0.0.1:17890"
        with patch.dict(
            os.environ,
            {
                "CC_SWITCH_PROVIDER_PROXY": proxy_url,
                "HTTP_PROXY": "http://127.0.0.1:7890",
                "HTTPS_PROXY": "http://127.0.0.1:7890",
                "http_proxy": "http://127.0.0.1:7890",
                "https_proxy": "http://127.0.0.1:7890",
            },
            clear=True,
        ):
            opener = MODULE.build_model_opener()

        proxy_handlers = [
            handler
            for handler in opener.handlers
            if isinstance(handler, MODULE.ProxyHandler)
        ]
        self.assertEqual(len(proxy_handlers), 1)
        self.assertEqual(proxy_handlers[0].proxies["http"], proxy_url)
        self.assertEqual(proxy_handlers[0].proxies["https"], proxy_url)

    def test_dedicated_provider_proxy_ignores_no_proxy(self):
        proxy_url = "http://127.0.0.1:17890"
        handler = MODULE.ForcedProxyHandler({"http": proxy_url})
        request = MODULE.Request("http://relay.example/v1/models")

        with patch.dict(os.environ, {"NO_PROXY": "*", "no_proxy": "*"}):
            handler.proxy_open(request, proxy_url, "http")
            self.assertEqual(os.environ["NO_PROXY"], "*")
            self.assertEqual(os.environ["no_proxy"], "*")

        self.assertEqual(request.host, "127.0.0.1:17890")

    def test_dedicated_provider_proxy_restores_absent_bypass_after_error(self):
        handler = MODULE.ForcedProxyHandler(
            {"http": "http://127.0.0.1:17890"}
        )
        request = MODULE.Request("http://relay.example/v1/models")

        with patch.dict(os.environ, {}, clear=True):
            with patch.object(
                MODULE.ProxyHandler,
                "proxy_open",
                side_effect=RuntimeError("test failure"),
            ):
                with self.assertRaisesRegex(RuntimeError, "test failure"):
                    handler.proxy_open(
                        request, "http://127.0.0.1:17890", "http"
                    )
            self.assertNotIn("NO_PROXY", os.environ)
            self.assertNotIn("no_proxy", os.environ)

    def test_model_opener_without_dedicated_proxy_uses_environment(self):
        ambient_proxy = "http://127.0.0.1:7890"
        with patch.dict(
            os.environ,
            {"https_proxy": ambient_proxy},
            clear=True,
        ):
            opener = MODULE.build_model_opener()

        proxy_handlers = [
            handler
            for handler in opener.handlers
            if isinstance(handler, MODULE.ProxyHandler)
        ]
        self.assertEqual(len(proxy_handlers), 1)
        self.assertEqual(proxy_handlers[0].proxies["https"], ambient_proxy)

    def test_model_opener_wires_redirect_rejection(self):
        opener = MODULE.build_model_opener()
        self.assertTrue(
            any(
                isinstance(handler, MODULE.RejectRedirects)
                for handler in opener.handlers
            )
        )


if __name__ == "__main__":
    unittest.main()
