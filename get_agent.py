import traceback

from .agent import Agent
from .llm import GeneralLLM
from .tool import (
    RetrieveInformation,
    ParseHtmlPage,
    EDGARSearch,
    GoogleWebSearch,
)


async def get_agent(model_name: str, parameters: dict, *args, **kwargs):
    max_turns = 50
    tools = {
        "google_web_search": GoogleWebSearch(),
        "retrieve_information": RetrieveInformation(),
        "parse_html_page": ParseHtmlPage(),
        "edgar_search": EDGARSearch(),
    }
    provider, model_key = model_name.split("/", 1)
    llm = GeneralLLM(
        provider=provider,
        model_name=model_key,
        max_tokens=parameters.get("max_output_tokens", 16384),
        temperature=parameters.get("temperature", 0.0),
    )

    agent = Agent(llm=llm, tools=tools, max_turns=max_turns)
    
    return agent
