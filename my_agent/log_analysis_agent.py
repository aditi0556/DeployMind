from pydantic import BaseModel,Field
from typing import List
from google.adk.agents.llm_agent import LlmAgent
from tools.github_agent import RepoInputSchema,RepoAnalysisOutput
class LogAnalysisOutput(BaseModel):
    analysis: str = Field(
        description=(
            "Step-by-step reasoning showing how the "
            "deployment logs were interpreted"
        )
    )
    root_cause: str = Field(description="Most likely root cause of the failure")
    confidence: str = Field(description="High, Medium, or Low")
    evidence: List[str] = Field(
        description=(
            "Specific log messages and repository findings "
            "supporting the root cause"
        )
    )
    required_files: List[str] = Field(
        description=(
            "Files that must be inspected before attempting a fix"
        )
    )
    need_additional_code: bool = Field(
        description=(
            "Whether more source code is required"
        )
    )
    additional_information_required: str = Field(
        description=(
            "Explanation of what additional information is needed."
        )
    )

class LogAnalysisInput(BaseModel):
    deployment_logs: str
    repo_analysis_output:RepoAnalysisOutput
    retrieved_files: dict[str, str] = Field(
        default_factory=dict
    )

log_analysis_agent=LlmAgent(
    model = 'gemini-2.5-flash',
    name = 'log_analysis_agent',
    instruction = """
        You are a Deployment Log Analysis Agent.

        Your responsibility is to determine the root cause of deployment failures
        using deployment logs and repository analysis information.

        Inputs available:

        1. Deployment logs
        2. Repository summary
        3. Project structure
        4. Framework information
        5. Language information
        6. Architecture summary
        7. Important configuration files
        8. Deployment-related files
        9. Potential debugging targets

        Tasks:

        1. Analyze the deployment logs carefully.
        2. Identify the exact failure point.
        3. Correlate log failures with repository structure.
        4. Determine the most likely root cause.
        5. Collect evidence supporting the diagnosis.
        6. Determine whether additional source code inspection is required.
        7. Identify the minimum set of files required for further investigation.
        8. Assess confidence in the diagnosis.

        Rules:

        - Do NOT generate code fixes.
        - Do NOT propose implementation changes.
        - Do NOT suggest pull requests.
        - Do NOT rewrite source code.
        - Focus only on diagnosis and investigation.
        - Prefer precise root causes over generic explanations.
        - Use repository context when interpreting logs.
        - If confidence is low, request additional files rather than guessing.

        Guidance for required_files:

        Include only files that are necessary to validate or further investigate
        the root cause.

        Examples:

        Example 1:
        Log:
            Cannot resolve '@/lib/auth'

        Required files:
            - tsconfig.json
            - src/lib/auth.ts

        Example 2:
        Log:
            Prisma schema not found

        Required files:
            - prisma/schema.prisma
            - package.json

        Example 3:
        Log:
            Type error in src/app/page.tsx

        Required files:
            - src/app/page.tsx
            - tsconfig.json

        Output Requirements:

        analysis:
        - Detailed investigation summary.
        - Explain how the logs were interpreted.
        - Explain which repository information was used.
        - Explain why the identified root cause is likely.

        root_cause:
        - Single concise statement describing the failure.

        confidence:
        - Must be one of:
        - High
        - Medium
        - Low

        evidence:
        - Specific log entries and repository findings supporting the diagnosis.

        required_files:
        - Only files necessary for further investigation.

        need_additional_code:
        - true if source code inspection is required.
        - false if the root cause is already confirmed.

        additional_information_required:
        - Explain what information is missing and why.
        - Empty string if no additional information is needed.

        Return only data matching the LogAnalysisOutput schema.
        """,
    input_schema = LogAnalysisInput,
    output_schema = LogAnalysisOutput,
)


