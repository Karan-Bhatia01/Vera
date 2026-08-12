from typing import Any
import threading
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
            llm = get_gemini_client()
            
            _auditor = Agent(
                role="Data Quality Auditor",
                goal="Identify data quality issues (nulls, duplicates, dtype problems) strictly from the provided stats.",
                backstory="A meticulous data engineer who never flags an issue without a number to back it.",
                llm=llm,
                verbose=False,
            )

            _analyst = Agent(
                role="Statistical Analyst",
                goal="Interpret column-level statistics and write a plain-language summary and per-column insights.",
                backstory="A data scientist who explains statistics clearly without overreaching beyond what the numbers show.",
                llm=llm,
                verbose=False,
            )

            _fact_checker = Agent(
                role="Fact Checker",
                goal="Verify every claim made by the Auditor and Analyst against the raw stats, and remove anything unsupported.",
                backstory="A skeptical reviewer whose only job is to catch hallucinated or unsupported claims before they ship.",
                llm=llm,
                verbose=False,
            )

            _writer = Agent(
                role="Report Writer",
                goal="Compile the verified findings into the exact required JSON schema, including actionable next steps.",
                backstory="A technical writer who produces clean, schema-compliant structured output.",
                llm=llm,
                verbose=False,
            )
            
    return _auditor, _analyst, _fact_checker, _writer


def build_analysis_crew(prompt_data: dict[str, Any]) -> Crew:
    """Assemble the 4-agent sequential crew for this analysis run."""
    context_str = str(prompt_data)
    
    auditor, analyst, fact_checker, writer = get_agents()

    audit_task = Task(
        description=(
            f"Dataset stats:\n{context_str}\n\n"
            "List concrete quality issues (nulls %, duplicates, dtype mismatches). "
            "Only flag what the numbers support."
        ),
        expected_output="A list of quality issues, each with column, severity, issue, and detail.",
        agent=auditor,
    )

    analysis_task = Task(
        description=(
            f"Dataset stats:\n{context_str}\n\n"
            "Write a 2-3 sentence summary and per-column insights based on the describe_sample stats."
        ),
        expected_output="A summary string and a list of column insights.",
        agent=analyst,
    )

    check_task = Task(
        description=(
            "Review the Auditor's quality flags and the Analyst's insights. "
            "Remove or correct any claim not directly supported by the original stats. "
            "Do not invent new findings."
        ),
        expected_output="A cleaned, verified list of quality flags and column insights.",
        agent=fact_checker,
        context=[audit_task, analysis_task],
    )

    write_task = Task(
        description=(
            "Using the verified findings, produce the final report: summary, quality_flags, "
            "column_insights, next_steps (2-4 concrete actions), and uncertainty_notes "
            "(what the analysis can't tell you)."
        ),
        expected_output="A JSON object matching the InsightsSchema exactly.",
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
