import json
import logging
import os
import tempfile
import unittest
from unittest.mock import AsyncMock, patch

from finance_agent.tools import SEARCH_COST_PATH_ENV, TavilyWebSearch


class TavilyCostReportTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        self.report = os.path.join(
            self.enterContext(tempfile.TemporaryDirectory()), "search-cost.jsonl"
        )
        self.enterContext(patch.dict(os.environ, {SEARCH_COST_PATH_ENV: self.report}))
        self.tool = TavilyWebSearch(tavily_api_key="test-key")

    async def search(self, response: dict) -> None:
        self.search_mock = AsyncMock(return_value=response)
        self.tool.client.search = self.search_mock
        await self.tool.execute({"search_query": "q"}, {}, logging.getLogger("test"))

    def charges(self) -> list[dict[str, str]]:
        with open(self.report, encoding="utf-8") as report:
            return [json.loads(line) for line in report]

    async def test_search_reports_its_credits_under_the_request_id(self):
        await self.search(
            {
                "request_id": "req-1",
                "usage": {"credits": 2},
                "results": [{"url": "https://a.com"}],
            }
        )

        self.assertIs(self.search_mock.call_args.kwargs["include_usage"], True)
        self.assertEqual(
            self.charges(),
            [{"provider": "tavily", "amount_usd": "0.016", "source_id": "req-1"}],
        )

    async def test_search_without_usage_is_not_reported_as_free(self):
        await self.search({"request_id": "req-1", "results": []})
        await self.search(
            {"request_id": "req-2", "usage": {"credits": "2"}, "results": []}
        )

        self.assertFalse(os.path.exists(self.report))

    async def test_unwritable_report_does_not_fail_the_search(self):
        with patch.dict(
            os.environ,
            {SEARCH_COST_PATH_ENV: os.path.join(self.report, "missing", "x")},
        ):
            await self.search(
                {
                    "request_id": "req-1",
                    "usage": {"credits": 1},
                    "results": [{"url": "https://a.com"}],
                }
            )

        self.assertFalse(os.path.exists(self.report))

    async def test_no_report_outside_valkyrie(self):
        with patch.dict(os.environ, {SEARCH_COST_PATH_ENV: ""}):
            await self.search(
                {"request_id": "req-1", "usage": {"credits": 1}, "results": []}
            )

        self.assertFalse(os.path.exists(self.report))
