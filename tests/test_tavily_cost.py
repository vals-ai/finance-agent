import json
import logging
import os
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import AsyncMock, patch

from tavily import AsyncTavilyClient

from finance_agent.tools import TavilyWebSearch

RESULT = {"url": "https://example.com", "title": "Example", "content": "Result"}


class TavilyCostTest(unittest.IsolatedAsyncioTestCase):
    async def _search(
        self, response: dict[str, object], path: Path | None
    ) -> tuple[str, AsyncMock]:
        with (
            patch.dict(os.environ),
            patch.object(
                AsyncTavilyClient, "search", new=AsyncMock(return_value=response)
            ) as search,
        ):
            if path is None:
                os.environ.pop("VALKYRIE_SEARCH_COST_PATH", None)
            else:
                os.environ["VALKYRIE_SEARCH_COST_PATH"] = str(path)
            output = await TavilyWebSearch(tavily_api_key="test").execute(
                {"search_query": "example"}, {}, logging.getLogger(__name__)
            )
        return output.output, search

    async def test_search_reports_reported_credits(self) -> None:
        with TemporaryDirectory() as directory:
            path = Path(directory) / "search-cost.jsonl"
            response = {
                "request_id": "finance-5",
                "usage": {"credits": 5},
                "results": [RESULT],
            }
            output, search = await self._search(response, path)

            self.assertEqual(json.loads(output), [RESULT])
            self.assertIs(search.call_args.kwargs["include_usage"], True)
            self.assertEqual(
                [json.loads(line) for line in path.read_text().splitlines()],
                [
                    {
                        "provider": "tavily",
                        "amount_usd": "0.0215",
                        "source_id": "finance-5",
                    }
                ],
            )

    async def test_unusable_credits_charge_one_credit_with_a_warning(self) -> None:
        usage: object
        for usage in [
            {},
            {"credits": True},
            {"credits": -2},
            {"credits": 2.5},
            {"credits": "3"},
            "not a dict",
        ]:
            with self.subTest(usage=usage), TemporaryDirectory() as directory:
                path = Path(directory) / "search-cost.jsonl"
                response = {
                    "request_id": "finance-5",
                    "usage": usage,
                    "results": [RESULT],
                }
                with self.assertLogs("finance_agent.tools", logging.WARNING) as logs:
                    output, _ = await self._search(response, path)

                self.assertEqual(json.loads(output), [RESULT])
                self.assertIn("no usable credit count", logs.output[0])
                self.assertEqual(
                    [json.loads(line) for line in path.read_text().splitlines()],
                    [
                        {
                            "provider": "tavily",
                            "amount_usd": "0.0043",
                            "source_id": "finance-5",
                        }
                    ],
                )

    async def test_source_id_is_written_only_for_a_non_empty_string(self) -> None:
        for request_id in [None, "", 7]:
            with self.subTest(request_id=request_id), TemporaryDirectory() as directory:
                path = Path(directory) / "search-cost.jsonl"
                response = {
                    "request_id": request_id,
                    "usage": {"credits": 1},
                    "results": [],
                }
                await self._search(response, path)

                self.assertEqual(
                    [json.loads(line) for line in path.read_text().splitlines()],
                    [{"provider": "tavily", "amount_usd": "0.0043"}],
                )

    async def test_search_without_cost_path_requests_no_usage(self) -> None:
        response = {
            "request_id": "finance-5",
            "usage": {"credits": 7},
            "results": [RESULT],
        }
        output, search = await self._search(response, None)

        self.assertEqual(json.loads(output), [RESULT])
        self.assertEqual(search.call_args.kwargs["search_depth"], "fast")
        self.assertNotIn("include_usage", search.call_args.kwargs)

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
