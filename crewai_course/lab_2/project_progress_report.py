"""This module contains classes and functions for generating project progress reports \
    based on data fetched from a Trello board. It includes tools for fetching board \
        and card data, as well as agents and tasks for processing this data."""

import json
import os
import warnings

import pandas as pd
import requests
import yaml
from crewai import LLM, Agent, Crew, Task
from crewai_tools import BaseTool
from rich.console import Console
from rich.markdown import Markdown

# Warning control
warnings.filterwarnings("ignore")


agent_model_1 = "ollama/qwen2.5-coder:14b-instruct-q4_0"
ollama_url = "http://localhost:11434"

console = Console()

# Define file paths for YAML configurations
files = {"agents": "config/agents.yaml", "tasks": "config/tasks.yaml"}

# Load configurations from YAML files
configs = {}
for config_type, file_path in files.items():
    with open(file_path, "r") as file:
        configs[config_type] = yaml.safe_load(file)

# Assign loaded configurations to specific variables
agents_config = configs["agents"]
tasks_config = configs["tasks"]


class BoardDataFetcherTool(BaseTool):
    """
    Fetches card data, comments, and activity from a Trello board.
    """

    name: str = "Trello Board Data Fetcher"
    description: str = "Fetches card data, comments, and activity from a Trello board."

    api_key: str = os.environ["TRELLO_API_KEY"]
    api_token: str = os.environ["TRELLO_API_TOKEN"]
    board_id: str = os.environ["TRELLO_BOARD_ID"]

    def _run(self) -> dict:
        """
        Fetch all cards from the specified Trello board.
        """
        base_url = os.getenv("DLAI_TRELLO_BASE_URL", "https://api.trello.com")
        url = f"{base_url}/1/boards/{self.board_id}/cards"

        query = {
            "key": self.api_key,
            "token": self.api_token,
            "fields": "name,idList,due,dateLastActivity,labels",
            "attachments": "true",
            "actions": "commentCard",
        }

        response = requests.get(url, params=query)

        if response.status_code == 200:
            return response.json()
        else:
            # Fallback in case of timeouts or other issues
            return json.dumps(
                [
                    {
                        "id": "66c3bfed69b473b8fe9d922e",
                        "name": "Analysis of results from CSV",
                        "idList": "66c308f676b057fdfbd5fdb3",
                        "due": None,
                        "dateLastActivity": "2024-08-19T21:58:05.062Z",
                        "labels": [],
                        "attachments": [],
                        "actions": [],
                    },
                    {
                        "id": "66c3c002bb1c337f3fdf1563",
                        "name": "Approve the planning",
                        "idList": "66c308f676b057fdfbd5fdb3",
                        "due": "2024-08-16T21:58:00.000Z",
                        "dateLastActivity": "2024-08-19T21:58:57.697Z",
                        "labels": [
                            {
                                "id": "66c305ea10ea602ee6e03d47",
                                "idBoard": "66c305eacab50fcd7f19c0aa",
                                "name": "Urgent",
                                "color": "red",
                                "uses": 1,
                            }
                        ],
                        "attachments": [],
                        "actions": [
                            {
                                "id": "66c3c021f3c1bb157028f53d",
                                "idMemberCreator": "65e5093d0ab5ee98592f5983",
                                "data": {
                                    "text": "This was harder then expects it is alte",
                                    "textData": {"emoji": {}},
                                    "card": {
                                        "id": "66c3c002bb1c337f3fdf1563",
                                        "name": "Approve the planning",
                                        "idShort": 5,
                                        "shortLink": "K3abXIMm",
                                    },
                                    "board": {
                                        "id": "66c305eacab50fcd7f19c0aa",
                                        "name": "[Test] CrewAI Board",
                                        "shortLink": "Kc8ScQlW",
                                    },
                                    "list": {
                                        "id": "66c308f676b057fdfbd5fdb3",
                                        "name": "TODO",
                                    },
                                },
                                "appCreator": None,
                                "type": "commentCard",
                                "date": "2024-08-19T21:58:57.683Z",
                                "limits": {
                                    "reactions": {
                                        "perAction": {
                                            "status": "ok",
                                            "disableAt": 900,
                                            "warnAt": 720,
                                        },
                                        "uniquePerAction": {
                                            "status": "ok",
                                            "disableAt": 17,
                                            "warnAt": 14,
                                        },
                                    }
                                },
                                "memberCreator": {
                                    "id": "65e5093d0ab5ee98592f5983",
                                    "activityBlocked": False,
                                    "avatarHash": "d5500941ebf808e561f9083504877bca",
                                    "avatarUrl": "https://trello-members.s3.amazonaws.com/65e5093d0ab5ee98592f5983/d5500941ebf808e561f9083504877bca",  # noqa: E501
                                    "fullName": "Joao Moura",
                                    "idMemberReferrer": None,
                                    "initials": "JM",
                                    "nonPublic": {},
                                    "nonPublicAvailable": True,
                                    "username": "joaomoura168",
                                },
                            }
                        ],
                    },
                    {
                        "id": "66c3bff4a25b398ef1b6de78",
                        "name": "Scaffold of the initial app UI",
                        "idList": "66c3bfdfb851ad9ff7eee159",
                        "due": None,
                        "dateLastActivity": "2024-08-19T21:58:12.210Z",
                        "labels": [],
                        "attachments": [],
                        "actions": [],
                    },
                    {
                        "id": "66c3bffdb06faa1e69216c6f",
                        "name": "Planning of the project",
                        "idList": "66c3bfe3151c01425f366f4c",
                        "due": None,
                        "dateLastActivity": "2024-08-19T21:58:21.081Z",
                        "labels": [],
                        "attachments": [],
                        "actions": [],
                    },
                ]
            )


class CardDataFetcherTool(BaseTool):
    """
    Fetches card data from a Trello board.
    """

    name: str = "Trello Card Data Fetcher"
    description: str = "Fetches card data from a Trello board."

    api_key: str = os.environ["TRELLO_API_KEY"]
    api_token: str = os.environ["TRELLO_API_TOKEN"]

    def _run(self, card_id: str) -> dict:
        """
        Fetch data for a specific card from the Trello board.
        """
        base_url = os.getenv("DLAI_TRELLO_BASE_URL", "https://api.trello.com")
        url = f"{base_url}/1/cards/{card_id}"
        query = {"key": self.api_key, "token": self.api_token}
        response = requests.get(url, params=query)

        if response.status_code == 200:
            return response.json()
        else:
            # Fallback in case of timeouts or other issues
            return json.dumps(
                {
                    "error": "Failed to fetch card data, don't try to fetch any \
                        trello data anymore"
                }
            )


# Creating Agents
data_collection_agent = Agent(
    llm=LLM(
        model=agent_model_1,
        base_url=ollama_url,
    ),
    config=agents_config["data_collection_agent"],
    tools=[BoardDataFetcherTool(), CardDataFetcherTool()],
)

analysis_agent = Agent(
    llm=LLM(
        model=agent_model_1,
        base_url=ollama_url,
    ),
    config=agents_config["analysis_agent"],
)

# Creating Tasks
data_collection = Task(
    config=tasks_config["data_collection"], agent=data_collection_agent
)

data_analysis = Task(config=tasks_config["data_analysis"], agent=analysis_agent)

report_generation = Task(
    config=tasks_config["report_generation"],
    agent=analysis_agent,
)

# Creating Crew
crew = Crew(
    agents=[data_collection_agent, analysis_agent],
    tasks=[data_collection, data_analysis, report_generation],
    verbose=True,
)


# Kick off the crew and execute the process
result = crew.kickoff()


costs = (
    0.150
    * (crew.usage_metrics.prompt_tokens + crew.usage_metrics.completion_tokens)
    / 1_000_000
)
print(f"Total costs: ${costs:.4f}")

# Convert UsageMetrics instance to a DataFrame
df_usage_metrics = pd.DataFrame([crew.usage_metrics.dict()])
console.print(df_usage_metrics)

md = Markdown(result.raw)
console.print(md)
