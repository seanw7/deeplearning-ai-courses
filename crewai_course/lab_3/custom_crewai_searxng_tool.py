import os
from typing import Type

import requests
from crewai.tools import BaseTool
from pydantic import BaseModel, Field


class SearXNGSearchToolInput(BaseModel):
    """Input schema for SearXNGSearchTool."""

    search_query: str = Field(..., description="The search query to perform.")


class SearXNGSearchTool(BaseTool):
    name: str = "SearXNG Search"
    description: str = "This tool allows CrewAI Agents to perform search queries \
        using the SearXNG server and view the results."
    args_schema: Type[BaseModel] = SearXNGSearchToolInput

    def _run(self, search_query: str) -> str:
        print("Function Inputs:", locals())
        if "SEARXNG_BASE_URL" in os.environ:
            base_url = os.environ["SEARXNG_BASE_URL"]
        else:
            base_url = "http://localhost:8080/search"

        params = {"q": search_query, "format": "json"}

        try:
            response = requests.get(base_url, params=params)
            response.raise_for_status()
            # results = response.content
            # print(results)
            results = response.json().get("results", [])

            if not results:
                return "No results found for the query."

            formatted_results = "\n".join(
                [f"{result['title']} - {result['url']}" for result in results]
            )
            return f"Search Results:\n{formatted_results}"

        except requests.exceptions.RequestException as e:
            return f"An error occurred: {e}"


# print(SearXNGSearchTool._run(query="foobar"))
