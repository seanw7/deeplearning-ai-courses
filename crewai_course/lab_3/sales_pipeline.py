import warnings
from typing import List, Optional

import pandas as pd
import yaml
from crewai import Agent, Crew, Flow, Task
from crewai.flow.flow import listen, start
from crewai_tools import ScrapeWebsiteTool, SerperDevTool
from pydantic import BaseModel, Field

# Warning control
warnings.filterwarnings("ignore")

# Define file paths for YAML configurations
files = {
    "lead_agents": "config/lead_qualification_agents.yaml",
    "lead_tasks": "config/lead_qualification_tasks.yaml",
    "email_agents": "config/email_engagement_agents.yaml",
    "email_tasks": "config/email_engagement_tasks.yaml",
}

# Load configurations from YAML files
configs = {}
for config_type, file_path in files.items():
    with open(file_path, "r") as file:
        configs[config_type] = yaml.safe_load(file)

# Assign loaded configurations to specific variables
lead_agents_config = configs["lead_agents"]
lead_tasks_config = configs["lead_tasks"]
email_agents_config = configs["email_agents"]
email_tasks_config = configs["email_tasks"]


class LeadPersonalInfo(BaseModel):
    name: str = Field(..., description="The full name of the lead.")
    job_title: str = Field(..., description="The job title of the lead.")
    role_relevance: int = Field(
        ...,
        ge=0,
        le=10,
        description="A score representing how relevant the lead's role is to the decision-making process (0-10).",
    )
    professional_background: Optional[str] = Field(
        ..., description="A brief description of the lead's professional background."
    )


class CompanyInfo(BaseModel):
    company_name: str = Field(
        ..., description="The name of the company the lead works for."
    )
    industry: str = Field(
        ..., description="The industry in which the company operates."
    )
    company_size: int = Field(
        ..., description="The size of the company in terms of employee count."
    )
    revenue: Optional[float] = Field(
        None, description="The annual revenue of the company, if available."
    )
    market_presence: int = Field(
        ...,
        ge=0,
        le=10,
        description="A score representing the company's market presence (0-10).",
    )


class LeadScore(BaseModel):
    score: int = Field(
        ..., ge=0, le=100, description="The final score assigned to the lead (0-100)."
    )
    scoring_criteria: List[str] = Field(
        ..., description="The criteria used to determine the lead's score."
    )
    validation_notes: Optional[str] = Field(
        None, description="Any notes regarding the validation of the lead score."
    )


class LeadScoringResult(BaseModel):
    personal_info: LeadPersonalInfo = Field(
        ..., description="Personal information about the lead."
    )
    company_info: CompanyInfo = Field(
        ..., description="Information about the lead's company."
    )
    lead_score: LeadScore = Field(
        ..., description="The calculated score and related information for the lead."
    )


# Creating Agents
lead_data_agent = Agent(
    config=lead_agents_config["lead_data_agent"],
    tools=[SerperDevTool(), ScrapeWebsiteTool()],
)

cultural_fit_agent = Agent(
    config=lead_agents_config["cultural_fit_agent"],
    tools=[SerperDevTool(), ScrapeWebsiteTool()],
)

scoring_validation_agent = Agent(
    config=lead_agents_config["scoring_validation_agent"],
    tools=[SerperDevTool(), ScrapeWebsiteTool()],
)

# Creating Tasks
lead_data_task = Task(
    config=lead_tasks_config["lead_data_collection"], agent=lead_data_agent
)

cultural_fit_task = Task(
    config=lead_tasks_config["cultural_fit_analysis"], agent=cultural_fit_agent
)

scoring_validation_task = Task(
    config=lead_tasks_config["lead_scoring_and_validation"],
    agent=scoring_validation_agent,
    context=[lead_data_task, cultural_fit_task],
    output_pydantic=LeadScoringResult,
)

# Creating Crew
lead_scoring_crew = Crew(
    agents=[lead_data_agent, cultural_fit_agent, scoring_validation_agent],
    tasks=[lead_data_task, cultural_fit_task, scoring_validation_task],
    verbose=True,
)


# Creating Agents
email_content_specialist = Agent(config=email_agents_config["email_content_specialist"])

engagement_strategist = Agent(config=email_agents_config["engagement_strategist"])

# Creating Tasks
email_drafting = Task(
    config=email_tasks_config["email_drafting"], agent=email_content_specialist
)

engagement_optimization = Task(
    config=email_tasks_config["engagement_optimization"], agent=engagement_strategist
)

# Creating Crew
email_writing_crew = Crew(
    agents=[email_content_specialist, engagement_strategist],
    tasks=[email_drafting, engagement_optimization],
    verbose=True,
)


class SalesPipeline(Flow):
    @start()
    def fetch_leads(self):
        # Pull our leads from the database
        leads = [
            {
                "lead_data": {
                    "name": "João Moura",
                    "job_title": "Director of Engineering",
                    "company": "Clearbit",
                    "email": "joao@clearbit.com",
                    "use_case": "Using AI Agent to do better data enrichment.",
                },
            },
        ]
        return leads

    @listen(fetch_leads)
    def score_leads(self, leads):
        scores = lead_scoring_crew.kickoff_for_each(leads)
        self.state["score_crews_results"] = scores
        return scores

    @listen(score_leads)
    def store_leads_score(self, scores):
        # Here we would store the scores in the database
        return scores

    @listen(score_leads)
    def filter_leads(self, scores):
        return [score for score in scores if score["lead_score"].score > 70]

    @listen(filter_leads)
    def write_email(self, leads):
        scored_leads = [lead.to_dict() for lead in leads]
        emails = email_writing_crew.kickoff_for_each(scored_leads)
        return emails

    @listen(write_email)
    def send_email(self, emails):
        # Here we would send the emails to the leads
        return emails


flow = SalesPipeline()

flow.plot()


emails = flow.kickoff()


# Convert UsageMetrics instance to a DataFrame
df_usage_metrics = pd.DataFrame(
    [flow.state["score_crews_results"][0].token_usage.dict()]
)

# Calculate total costs
costs = 0.150 * df_usage_metrics["total_tokens"].sum() / 1_000_000
print(f"Total costs: ${costs:.4f}")

# Display the DataFrame
print(df_usage_metrics)


# Convert UsageMetrics instance to a DataFrame
df_usage_metrics = pd.DataFrame([emails[0].token_usage.dict()])

# Calculate total costs
costs = 0.150 * df_usage_metrics["total_tokens"].sum() / 1_000_000
print(f"Total costs: ${costs:.4f}")

# Display the DataFrame
print(df_usage_metrics)
