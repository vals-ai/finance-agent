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
        for usage, amount in [
            ({"credits": 5}, "0.040"),
            ({}, "0.008"),
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
