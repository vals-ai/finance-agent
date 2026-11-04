import asyncio
import logging
import unittest
from unittest.mock import AsyncMock, Mock

from finance_agent.tools import RetrieveInformation


class RetrieveInformationTests(unittest.TestCase):
    def test_prompt_braces_pass_through_unchanged(self) -> None:
        llm = Mock()
        llm.query = AsyncMock(return_value=Mock(output_text_str="ok", metadata=None))
        tool = RetrieveInformation(llm)

        result = asyncio.run(
            tool.execute(
                {"prompt": 'Return {"answer": "..."} using {{report}}'},
                {"report": "DOC"},
                logging.getLogger(__name__),
            )
        )

        self.assertIsNone(result.error)
        llm.query.assert_awaited_once_with('Return {"answer": "..."} using DOC')

    def test_character_range_applies_to_inserted_document(self) -> None:
        tool = RetrieveInformation(Mock())

        prompt = tool._format_prompt("{x} {{doc}}", {"doc": (1, 3)}, {"doc": "abcdef"})

        self.assertEqual(prompt, "{x} bc")


if __name__ == "__main__":
    unittest.main()
