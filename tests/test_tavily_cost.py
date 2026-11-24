import json
import logging
import os
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import AsyncMock, patch

from tavily import AsyncTavilyClient

from finance_agent.tools import TavilyWebSearch


class TavilyCostTest(unittest.IsolatedAsyncioTestCase):
    async def test_search_reports_cost_when_requested(self) -> None:
        usage: object
        for usage, amount in [
            ({"credits": 5}, "0.0215"),
            ({}, "0.0043"),
            ({"credits": True}, "0.0043"),
            ({"credits": -2}, "0.0043"),
            ({"credits": float("nan")}, "0.0043"),
            ({"credits": float("inf")}, "0.0043"),
            ({"credits": "3"}, "0.0043"),
            ("not a dict", "0.0043"),
            ({"credits": 7}, None),
        ]:
            with (
                self.subTest(usage=usage, amount=amount),
                TemporaryDirectory() as directory,
            ):
                path = Path(directory) / "search-cost.jsonl"
                result = {
                    "url": "https://example.com",
                    "title": "Example",
                    "content": "Result",
                }
                response = {
                    "request_id": "finance-5",
                    "usage": usage,
                    "results": [result],
                }
                with (
                    patch.dict(os.environ),
                    patch.object(
                        AsyncTavilyClient,
                        "search",
                        new=AsyncMock(return_value=response),
                    ) as search,
                ):
                    if amount is None:
                        os.environ.pop("VALKYRIE_SEARCH_COST_PATH", None)
                    else:
                        os.environ["VALKYRIE_SEARCH_COST_PATH"] = str(path)
                    output = await TavilyWebSearch(tavily_api_key="test").execute(
                        {"search_query": "example"}, {}, logging.getLogger(__name__)
                    )

                self.assertEqual(json.loads(output.output), [result])
                self.assertEqual(search.call_args.kwargs["search_depth"], "fast")
                if amount is None:
                    self.assertNotIn("include_usage", search.call_args.kwargs)
                    self.assertFalse(path.exists())
                else:
                    self.assertIs(search.call_args.kwargs["include_usage"], True)
                    self.assertEqual(
                        [json.loads(line) for line in path.read_text().splitlines()],
                        [
                            {
                                "provider": "tavily",
                                "amount_usd": amount,
                                "source_id": "finance-5",
                            }
                        ],
                    )

    async def test_each_search_appends_its_own_charge(self) -> None:
        with TemporaryDirectory() as directory:
            path = Path(directory) / "search-cost.jsonl"
            responses = [
                {"request_id": "finance-1", "usage": {"credits": 1}, "results": []},
                {"request_id": "finance-2", "usage": {"credits": 3}, "results": []},
            ]
            with (
                patch.dict(os.environ, {"VALKYRIE_SEARCH_COST_PATH": str(path)}),
                patch.object(
                    AsyncTavilyClient, "search", new=AsyncMock(side_effect=responses)
                ),
            ):
                for _ in responses:
                    await TavilyWebSearch(tavily_api_key="test").execute(
                        {"search_query": "example"}, {}, logging.getLogger(__name__)
                    )

            self.assertEqual(
                [json.loads(line) for line in path.read_text().splitlines()],
                [
                    {
                        "provider": "tavily",
                        "amount_usd": "0.0043",
                        "source_id": "finance-1",
                    },
                    {
                        "provider": "tavily",
                        "amount_usd": "0.0129",
                        "source_id": "finance-2",
                    },
                ],
            )

    async def test_search_returns_results_when_cost_report_write_fails(self) -> None:
        with TemporaryDirectory() as directory:
            path = Path(directory) / "search-cost.jsonl"
            path.mkdir()
            result = {
                "url": "https://example.com",
                "title": "Example",
                "content": "Result",
            }
            response = {
                "request_id": "finance-5",
                "usage": {"credits": 1},
                "results": [result],
            }
            with (
                patch.dict(os.environ, {"VALKYRIE_SEARCH_COST_PATH": str(path)}),
                patch.object(
                    AsyncTavilyClient, "search", new=AsyncMock(return_value=response)
                ),
                self.assertLogs("finance_agent.tools", logging.WARNING) as logs,
            ):
                output = await TavilyWebSearch(tavily_api_key="test").execute(
                    {"search_query": "example"}, {}, logging.getLogger(__name__)
                )

            self.assertEqual(json.loads(output.output), [result])
            self.assertIsNone(output.error)
            self.assertEqual(
                [record.getMessage() for record in logs.records],
                [f"Could not write search cost report {path}"],
            )
