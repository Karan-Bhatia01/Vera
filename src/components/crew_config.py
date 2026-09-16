from typing import Any
import threading
from pathlib import Path
import yaml
from crewai import Agent, Task, Crew, Process
from pydantic import BaseModel, Field

from src.core.connections import get_gemini_client

class QualityFlag(BaseModel):
    column: str
    severity: str
    issue: str
    detail: str

class ColumnInsight(BaseModel):
    column: str
    insight: str

class NextStep(BaseModel):
    title: str
    detail: str

class InsightsSchema(BaseModel):
    summary: str
    quality_flags: list[QualityFlag] = Field(default_factory=list)
    column_insights: list[ColumnInsight] = Field(default_factory=list)
    next_steps: list[NextStep] = Field(default_factory=list)
    uncertainty_notes: str = "Standard statistical limitations apply."

PROMPTS_DIR = Path(__file__).resolve().parent.parent / "prompts"
CREW_CONFIG_PATH = PROMPTS_DIR / "crew.yaml"

def _load_crew_config() -> dict:
    with open(CREW_CONFIG_PATH, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

# Singleton agents
_agents_lock = threading.Lock()
_auditor: Agent | None = None
_analyst: Agent | None = None
_fact_checker: Agent | None = None
_writer: Agent | None = None

def get_agents():
    global _auditor, _analyst, _fact_checker, _writer
    with _agents_lock:
        if _auditor is None:
            config = _load_crew_config().get("agents", {})
            llm = get_gemini_client()
            
            _auditor = Agent(
                role=config.get("auditor", {}).get("role", "Data Quality Auditor"),
                goal=config.get("auditor", {}).get("goal", ""),
                backstory=config.get("auditor", {}).get("backstory", ""),
                llm=llm,
                verbose=False,
            )

            _analyst = Agent(
                role=config.get("analyst", {}).get("role", "Statistical Analyst"),
                goal=config.get("analyst", {}).get("goal", ""),
                backstory=config.get("analyst", {}).get("backstory", ""),
                llm=llm,
                verbose=False,
            )

            _fact_checker = Agent(
                role=config.get("fact_checker", {}).get("role", "Fact Checker"),
                goal=config.get("fact_checker", {}).get("goal", ""),
                backstory=config.get("fact_checker", {}).get("backstory", ""),
                llm=llm,
                verbose=False,
            )

            _writer = Agent(
                role=config.get("writer", {}).get("role", "Report Writer"),
                goal=config.get("writer", {}).get("goal", ""),
                backstory=config.get("writer", {}).get("backstory", ""),
                llm=llm,
                verbose=False,
            )
            
    return _auditor, _analyst, _fact_checker, _writer


def build_analysis_crew(prompt_data: dict[str, Any]) -> Crew:
    """Assemble the 4-agent sequential crew for this analysis run."""
    context_str = str(prompt_data)
    
    auditor, analyst, fact_checker, writer = get_agents()
    
    config = _load_crew_config().get("tasks", {})

    audit_task = Task(
        description=config.get("audit_task", {}).get("description", "").format(context_str=context_str),
        expected_output=config.get("audit_task", {}).get("expected_output", ""),
        agent=auditor,
    )

    analysis_task = Task(
        description=config.get("analysis_task", {}).get("description", "").format(context_str=context_str),
        expected_output=config.get("analysis_task", {}).get("expected_output", ""),
        agent=analyst,
    )

    check_task = Task(
        description=config.get("check_task", {}).get("description", ""),
        expected_output=config.get("check_task", {}).get("expected_output", ""),
        agent=fact_checker,
        context=[audit_task, analysis_task],
    )

    write_task = Task(
        description=config.get("write_task", {}).get("description", ""),
        expected_output=config.get("write_task", {}).get("expected_output", ""),
        agent=writer,
        context=[check_task],
        output_pydantic=InsightsSchema,
    )

    return Crew(
        agents=[auditor, analyst, fact_checker, writer],
        tasks=[audit_task, analysis_task, check_task, write_task],
        process=Process.sequential,
        verbose=False,
    )
