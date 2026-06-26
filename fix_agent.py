from typing import List, Dict
from pydantic import BaseModel, Field
from google.adk.agents.llm_agent import LlmAgent
from tools.github_agent import RepoAnalysisOutput, FileChange
from config import SMART_MODEL


class FixGenerationInput(BaseModel):
    root_cause: str = Field(description="Root cause identified by the log analysis agent")
    evidence: List[str] = Field(
        default_factory=list,
        description="Log/repo evidence supporting the root cause",
    )
    repo_analysis_output: RepoAnalysisOutput = Field(
        description="Repository analysis context"
    )
    relevant_files: Dict[str, str] = Field(
        default_factory=dict,
        description="Source files retrieved during investigation (path -> contents)",
    )
    deployment_logs: str = Field(description="Raw deployment logs")


class FixGenerationOutput(BaseModel):
    can_fix: bool = Field(
        description="Whether a confident, concrete fix could be produced"
    )
    fix_summary: str = Field(description="Short one-line description of the fix")
    file_changes: List[FileChange] = Field(
        default_factory=list,
        description="Files to change, each with its FULL new contents",
    )
    risk_level: str = Field(description="Low, Medium, or High")
    requires_human_review: bool = Field(
        description="Whether a human should review before applying"
    )
    reasoning: str = Field(
        description="Why this fix addresses the root cause and what it touches"
    )


fix_agent = LlmAgent(
    model=SMART_MODEL,
    name="fix_generation_agent",
    instruction="""
        You are a Deployment Fix Generation Agent.

        Your responsibility is to propose a SAFE, MINIMAL fix for a deployment
        failure, given a confirmed root cause and the relevant source files.

        Inputs available:
        1. root_cause      - the confirmed reason the deployment failed
        2. evidence        - log lines and repo findings supporting the diagnosis
        3. repo_analysis   - framework, language, package manager, structure
        4. relevant_files  - the actual contents of the files involved
        5. deployment_logs - the raw build/runtime logs

        Tasks:
        1. Decide whether you can produce a confident, concrete fix.
        2. If yes, produce the MINIMAL set of file changes that fixes the root
           cause and nothing else.
        3. For every changed file, return its FULL new contents in new_content
           (not a diff, not a snippet) so it can be committed as-is.
        4. Assess the risk level and whether a human must review the change.

        TypeScript guidance (important):
        - If the project language is TypeScript (check repo_analysis) and you add
          a JavaScript library that does NOT ship its own type declarations
          (e.g. lodash, express), you MUST also add the matching @types/*
          package to devDependencies (e.g. add both "lodash" to dependencies
          AND "@types/lodash" to devDependencies). A strict TypeScript build
          ("strict": true) fails with "Could not find a declaration file for
          module 'X'" otherwise. Libraries that bundle their own types (e.g.
          axios, react) do not need an @types/* package.

        Hard safety rules:
        - Only change files when you have their current contents in relevant_files.
          If you need a file you do not have, set can_fix=false and explain.
        - Make the smallest change that fixes the problem. Do NOT refactor,
          rename, reformat, or "improve" unrelated code.
        - NEVER touch secrets, credentials, lockfile hashes you cannot compute,
          or delete files.
        - Do NOT invent file paths. Use paths confirmed by the repo analysis or
          the retrieved files.
        - If the fix is risky, ambiguous, or you are not confident, set
          requires_human_review=true (and prefer risk_level Medium or High).

        Risk guidance:
        - Low:    isolated config/typo/dependency-version change, well-understood.
        - Medium: source code logic change, or touches build configuration.
        - High:   touches multiple files, security-sensitive, or uncertain.

        Output requirements:
        - can_fix: true only when file_changes fully address the root cause.
        - fix_summary: concise, e.g. "Add missing 'sharp' dependency".
        - file_changes: full new contents for each file.
        - risk_level: Low | Medium | High.
        - requires_human_review: true unless the fix is clearly Low risk.
        - reasoning: explain why this fixes the root cause and what it changes.

        Return only data matching the FixGenerationOutput schema.
        """,
    input_schema=FixGenerationInput,
    output_schema=FixGenerationOutput,
)
