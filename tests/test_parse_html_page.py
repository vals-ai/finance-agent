import logging
import unittest
from unittest.mock import patch

from aiohttp import web
from aiohttp.test_utils import TestServer

from finance_agent import tools
from finance_agent.tools import ParseHtmlPage

_real_is_public = tools.is_public_address


def _allow_test_server(host: str) -> bool:
    return host == "127.0.0.1" or _real_is_public(host)


class ParseHtmlPageAddressTests(unittest.IsolatedAsyncioTestCase):
    async def _run(self, url: str):
        return await ParseHtmlPage().execute(
            {"url": url, "key": "page"}, {}, logging.getLogger(__name__)
        )

    async def test_internal_addresses_are_refused(self) -> None:
        for url in [
            "http://169.254.169.254/latest/meta-data/",
            "http://localhost/",
            "http://127.0.0.1:8080/",
            "http://[::1]/",
            "http://[::ffff:10.0.0.1]/",
            "http://10.0.0.1/",
            "http://192.168.1.1/",
            "file:///etc/passwd",
        ]:
            with self.subTest(url=url):
                result = await self._run(url)
                self.assertIsNotNone(result.error)
                self.assertTrue(
                    "public internet address" in result.error
                    or "Only http" in result.error,
                    result.error,
                )

    async def test_redirects_to_internal_addresses_are_refused(self) -> None:
        for target in [
            "http://169.254.169.254/latest/",
            "http://127.0.0.2:9/",
            "http://[::1]/",
        ]:
            with self.subTest(target=target):

                async def redirect(request: web.Request) -> web.Response:
                    raise web.HTTPFound(target)

                app = web.Application()
                app.router.add_get("/go", redirect)
                with patch.object(tools, "is_public_address", _allow_test_server):
                    async with TestServer(app, host="127.0.0.1") as server:
                        result = await self._run(str(server.make_url("/go")))
                self.assertIn("is not a public internet address.", result.error or "")

    async def test_redirects_to_allowed_addresses_are_followed(self) -> None:
        async def redirect(request: web.Request) -> web.Response:
            raise web.HTTPFound("/doc")

        async def doc(request: web.Request) -> web.Response:
            return web.Response(text="<p>landed</p>", content_type="text/html")

        app = web.Application()
        app.router.add_get("/go", redirect)
        app.router.add_get("/doc", doc)
        state: dict[str, str] = {}
        with patch.object(tools, "is_public_address", _allow_test_server):
            async with TestServer(app, host="127.0.0.1") as server:
                result = await ParseHtmlPage().execute(
                    {"url": str(server.make_url("/go")), "key": "page"},
                    state,
                    logging.getLogger(__name__),
                )
        self.assertIsNone(result.error)
        self.assertEqual(state["page"], "landed")


if __name__ == "__main__":
    unittest.main()
