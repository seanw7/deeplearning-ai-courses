"""This module contains classes and functions for project planning in a crew-based \
    system. It includes the definition of task estimates, milestones, and the \
        overall project plan."""

import time
import warnings
from typing import List

import pandas as pd
import yaml
from crewai import LLM, Agent, Crew, Task
from dotenv import find_dotenv, load_dotenv
from pydantic import BaseModel, Field

# Warning control
warnings.filterwarnings("ignore")


def load_env():
    """Load environment variables from a .env file."""
    _ = load_dotenv(find_dotenv())


load_env()


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


class TaskEstimate(BaseModel):
    """
    Represents an estimate for a task, including its name, estimated time in hours,
    and required resources.
    """

    task_name: str = Field(..., description="Name of the task")
    estimated_time_hours: float = Field(
        ..., description="Estimated time to complete the task in hours"
    )
    required_resources: List[str] = Field(
        ..., description="List of resources required to complete the task"
    )


class Milestone(BaseModel):
    """
    Represents a milestone in the project, including its name and associated tasks.
    """

    milestone_name: str = Field(..., description="Name of the milestone")
    tasks: List[str] = Field(
        ..., description="List of task IDs associated with this milestone"
    )


class ProjectPlan(BaseModel):
    """
    Represents the overall project plan, including a list of tasks and milestones.
    """

    tasks: List[TaskEstimate] = Field(
        ..., description="List of tasks with their estimates"
    )
    milestones: List[Milestone] = Field(..., description="List of project milestones")


agent_model_1 = "ollama/qwen2.5-coder:14b-instruct-q4_0"
ollama_url = "http://localhost:11434"

project_planning_agent = Agent(
    llm=LLM(
        model=agent_model_1,
        base_url=ollama_url,
    ),
    config=agents_config["project_planning_agent"],
)

# # Creating Agents
# project_planning_agent = Agent(
#   config=agents_config['project_planning_agent']
# )


estimation_agent = Agent(
    llm=LLM(model=agent_model_1, base_url=ollama_url),
    config=agents_config["estimation_agent"],
)

# estimation_agent = Agent(
#   config=agents_config['estimation_agent']
# )

resource_allocation_agent = Agent(
    llm=LLM(model=agent_model_1, base_url=ollama_url),
    config=agents_config["resource_allocation_agent"],
)

# resource_allocation_agent = Agent(
#   config=agents_config['resource_allocation_agent']
# )

# Creating Tasks
task_breakdown = Task(
    config=tasks_config["task_breakdown"], agent=project_planning_agent
)

time_resource_estimation = Task(
    config=tasks_config["time_resource_estimation"], agent=estimation_agent
)

resource_allocation = Task(
    config=tasks_config["resource_allocation"],
    agent=resource_allocation_agent,
    output_pydantic=ProjectPlan,  # This is the structured output we want
)

# Creating Crew
crew = Crew(
    agents=[project_planning_agent, estimation_agent, resource_allocation_agent],
    tasks=[task_breakdown, time_resource_estimation, resource_allocation],
    verbose=True,
)

# Crew's inputs
project = "Website"
industry = "Technology"
project_objectives = "Create a website for a small business"
team_members = (
    "- John Doe (Project Manager)\n"
    "- Jane Doe (Software Engineer)\n"
    "- Bob Smith (Designer)\n"
    "- Alice Johnson (QA Engineer)\n"
    "- Tom Brown (QA Engineer)\n"
)
project_requirements = (
    "- Create a responsive design that works well on desktop and mobile devices\n"
    "- Implement a modern, visually appealing user interface with a clean look\n"
    "- Develop a user-friendly navigation system with intuitive menu structure\n"
    '- Include an "About Us" page highlighting the company\'s history and values\n'
    '- Design a "Services" page showcasing the business\'s offerings with \
        descriptions\n'
    '- Create a "Contact Us" page with a form and integrated map for communication\n'
    "- Implement a blog section for sharing industry news and company updates\n"
    "- Ensure fast loading times and optimize for search engines (SEO)\n"
    "- Integrate social media links and sharing capabilities\n"
    "- Include a testimonials section to showcase customer feedback and build trust\n"
)

# Format the dictionary as Markdown for a better display in Jupyter Lab
formatted_output = (
    f"**Project Type:** {project}\n\n"
    f"**Project Objectives:** {project_objectives}\n\n"
    f"**Industry:** {industry}\n\n"
    f"**Team Members:**\n{team_members}"
    f"**Project Requirements:**\n{project_requirements}"
)
# Display the formatted output as Markdown
# print(formatted_output)

# Kick off the crew
# The given Python dictionary
inputs = {
    "project_type": project,
    "project_objectives": project_objectives,
    "industry": industry,
    "team_members": team_members,
    "project_requirements": project_requirements,
}

start_time = time.time()

# Run the crew
result = crew.kickoff(inputs=inputs)

end_time = time.time()
execution_time_seconds = end_time - start_time


# Convert UsageMetrics instance to a DataFrame
# Usage and cost metrics

df_usage_metrics = pd.DataFrame([crew.usage_metrics.dict()])
print(df_usage_metrics)

total_tokens = df_usage_metrics["total_tokens"].sum()

# Calculate the number of tokens per second
if execution_time_seconds > 0:
    tokens_per_second = total_tokens / execution_time_seconds
else:
    tokens_per_second = float(
        "inf"
    )  # Handle division by zero if execution time is negligible

print(f"Total execution time: {execution_time_seconds:.2f} seconds")
print(f"Total tokens processed: {total_tokens}")
print(f"Tokens per second: {tokens_per_second:.2f} tokens/s")


costs = (
    0.150
    * (crew.usage_metrics.prompt_tokens + crew.usage_metrics.completion_tokens)
    / 1_000_000
)
print(f"Total costs: ${costs:.4f}")


result.pydantic.dict()


tasks = result.pydantic.dict()["tasks"]
df_tasks = pd.DataFrame(tasks)

# Inspect task details
# Display the DataFrame as an HTML table
df_tasks.style.set_table_attributes('border="1"').set_caption(
    "Task Details"
).set_table_styles([{"selector": "th, td", "props": [("font-size", "120%")]}])


# Inspect milestones
milestones = result.pydantic.dict()["milestones"]
df_milestones = pd.DataFrame(milestones)

# Display the DataFrame as an HTML table
df_milestones.style.set_table_attributes('border="1"').set_caption(
    "Task Details"
).set_table_styles([{"selector": "th, td", "props": [("font-size", "120%")]}])
