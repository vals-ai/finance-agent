from finance_agent.tools import RetrieveInformation


def test_retrieve_information_prompt_preserves_json_braces() -> None:
    tool = RetrieveInformation(llm=None)  # type: ignore[arg-type]
    state = {
        "filing_excerpt": "Revenue was $383.29 billion in fiscal 2023.",
    }
    prompt = (
        'Return JSON like {"answer": "...", "sources": []} using this filing: '
        "{{filing_excerpt}}"
    )

    formatted = tool._format_prompt(prompt, {}, state)

    assert formatted == (
        'Return JSON like {"answer": "...", "sources": []} using this filing: '
        "Revenue was $383.29 billion in fiscal 2023."
    )


def test_retrieve_information_prompt_applies_character_range() -> None:
    tool = RetrieveInformation(llm=None)  # type: ignore[arg-type]
    state = {
        "filing_excerpt": "0123456789",
    }

    formatted = tool._format_prompt("Use {{filing_excerpt}}", {"filing_excerpt": (2, 6)}, state)

    assert formatted == "Use 2345"
