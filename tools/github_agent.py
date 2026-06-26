# agent.py (modify get_tools_async and other parts as needed)
# ./adk_agent_samples/mcp_agent/agent.py
import os
import asyncio
from dotenv import load_dotenv
from google.genai import types
from google.adk.agents.llm_agent import LlmAgent
from google.adk.tools.mcp_tool import McpToolset
from google.adk.tools.mcp_tool.mcp_session_manager import StdioConnectionParams
from mcp import StdioServerParameters
from pydantic import BaseModel,Field
from typing import Optional,List,Dict
from config import FAST_MODEL, SMART_MODEL
# Load environment variables from .env file in the parent directory
# Place this near the top, before using env vars like API keys
# load_dotenv('../.env')
class FileRetrievalInput(BaseModel):
    github_owner: str = Field(description="GitHub username or organization")
    repository_name: str = Field(description="Repository name" )
    required_files: List[str] = Field(
        description=(
          "file paths that are needed from the repository"
        )
    )

class FileRetrievalOutput(BaseModel):
    files: Dict[str, str] = Field(
        description=(
            "Dictionary mapping file path to file contents"
        )
    )

class RepoInputSchema(BaseModel):
    github_owner: str = Field(description="GitHub username or organization")
    repository_name: str = Field(description="Repository name" )

class RepoAnalysisOutput(BaseModel):

    repo_summary: str = Field(
        description="High level summary of the repository"
    )
    project_structure: str = Field(
        description="Directory tree and major modules"
    )
    framework: Optional[str] = Field(
        default=None,
        description="Detected framework"
    )
    language: Optional[str] = Field(
        default=None,
        description="Primary programming language"
    )
    package_manager: Optional[str] = Field(
        default=None,
        description="npm, pnpm, yarn, pip, cargo, etc."
    )
    important_config_files: List[str] = Field(
        description="Configuration files critical for builds and deployment"
    )
    deployment_related_files: List[str] = Field(
        description="Files affecting deployment and CI/CD"
    )

    architecture_summary: str = Field(
        description="Overview of repository architecture"
    )

    potential_debugging_targets: List[str] = Field(
        description=(
            "Files likely to be relevant when investigating "
            "build or runtime failures"
        )
)

async def create_github_agent(GITHUB_TOKEN:str):
    """Creates an MCP Toolset configured to connect to the GitHub MCP server."""
    toolset = McpToolset(
        # Use StdioConnectionParams for local process communication
        connection_params=StdioConnectionParams(
            server_params = StdioServerParameters(
                command='npx',
                args=["-y", "@modelcontextprotocol/server-github"],
                # The server reads the token from this env var (NOT a CLI arg).
                env={**os.environ, "GITHUB_PERSONAL_ACCESS_TOKEN": GITHUB_TOKEN},
            ),
        ),
    )

    repo_agent = LlmAgent(
        model = FAST_MODEL,
        name = 'repo_analysis_agent',
        instruction="""
            You are a Repository Analysis Agent.
            Use GitHub MCP tools to:

                1. Retrieve repository structure.
                2. Read important configuration files.
                3. Inspect deployment-related files.
                4. Gather information needed for repository analysis.

                Allowed operations:
                - Read repository structure
                - Read file contents
                - Search repository code

                Forbidden operations:
                - Create pull requests
                - Create branches
                - Create commits
                - Create issues
                - Modify files
                - Delete files
                        
            Only analyze the repository and return RepoAnalysisOutput.
            Your responsibility is to understand the repository before
            other debugging agents begin investigation.

            Tasks:

            1. Inspect repository structure.
            2. Identify framework(s) used.
            3. Identify programming language(s).
            4. Identify package manager.
            5. Detect deployment platform configuration.
            6. Detect CI/CD configuration.
            7. Detect environment variable usage.
            8. Identify important configuration files.
            9. Summarize architecture and application entry points.
            10. Highlight files commonly involved in deployment failures.

            Focus especially on:

            - package.json
            - pnpm-lock.yaml
            - yarn.lock
            - package-lock.json
            - next.config.*
            - vite.config.*
            - tsconfig.json
            - vercel.json
            - Dockerfile
            - docker-compose.yml
            - .github/workflows/*
            - prisma/schema.prisma
            - requirements.txt
            - pyproject.toml
            - Cargo.toml
            - go.mod

            Do NOT generate fixes.

            Return only information matching RepoAnalysisOutput.
    """,
        input_schema = RepoInputSchema,
        output_schema = RepoAnalysisOutput,
        tools = [toolset],
    )
    return toolset,repo_agent

async def create_file_retrieval_agent(github_token: str):
    toolset = McpToolset(
        connection_params=StdioConnectionParams(
            server_params=StdioServerParameters(
                command="npx",
                args=["-y", "@modelcontextprotocol/server-github"],
                # The server reads the token from this env var (NOT a CLI arg).
                env={**os.environ, "GITHUB_PERSONAL_ACCESS_TOKEN": github_token},
            ),
        ),
    )
    file_agent = LlmAgent(
        model=FAST_MODEL,
        name="github_file_retrieval_agent",
        instruction="""
            You are a GitHub File Retrieval Agent.
            Your responsibility is to fetch source files from a repository.
            Input contains:
            - github_owner
            - repository_name
            - files
            Tasks:
            1. Locate each requested file.
            2. Retrieve the file contents.
            3. Return contents exactly as stored.
            Rules:
            - Do not analyze code.
            - Do not summarize code.
            - Do not diagnose issues.
            - Do not generate fixes.
            - Do not retrieve unrelated files.
            - If a file cannot be found, omit it from the response.

            Return only FileRetrievalOutput.
            """,
        input_schema=FileRetrievalInput,
        output_schema=FileRetrievalOutput,
        tools=[toolset],
    )

    return toolset, file_agent


# ---------------------------------------------------------------------------
# Session 8: applying a fix back to the repository (write mode)
# ---------------------------------------------------------------------------

class FileChange(BaseModel):
    file_path: str = Field(description="Repository-relative path of the file to change")
    new_content: str = Field(
        description="Full new content of the file after the fix is applied"
    )
    change_summary: str = Field(
        description="One-line explanation of what changed in this file and why"
    )


class FixApplyInput(BaseModel):
    github_owner: str = Field(description="GitHub username or organization")
    repository_name: str = Field(description="Repository name")
    base_branch: str = Field(default="main", description="Branch to branch off from")
    new_branch: str = Field(description="Name of the new branch to create for the fix")
    commit_message: str = Field(description="Commit message for the fix")
    pr_title: str = Field(description="Title for the pull request")
    pr_body: str = Field(description="Body/description for the pull request")
    file_changes: List[FileChange] = Field(
        description="Files to create or update with their full new contents"
    )


class FixApplyOutput(BaseModel):
    success: bool = Field(description="Whether the fix was pushed successfully")
    branch: Optional[str] = Field(default=None, description="Branch the fix was pushed to")
    pull_request_url: Optional[str] = Field(
        default=None, description="URL of the opened pull request, if any"
    )
    applied_files: List[str] = Field(
        default_factory=list, description="File paths that were created or updated"
    )
    message: str = Field(description="Human-readable summary of the result")


async def create_github_fix_agent(github_token: str):
    """Creates a WRITE-ENABLED GitHub agent that pushes an approved fix.

    Unlike the read-only repo/file agents, this one is permitted to create a
    branch, commit the changed files, and open a pull request. It must NEVER
    push directly to the base branch and must NEVER delete files.
    """
    toolset = McpToolset(
        connection_params=StdioConnectionParams(
            server_params=StdioServerParameters(
                command="npx",
                args=["-y", "@modelcontextprotocol/server-github"],
                # The server reads the token from this env var (NOT a CLI arg).
                env={**os.environ, "GITHUB_PERSONAL_ACCESS_TOKEN": github_token},
            ),
        ),
    )
    fix_apply_agent = LlmAgent(
        model=SMART_MODEL,
        name="github_fix_apply_agent",
        instruction="""
            You are a GitHub Fix Apply Agent.
            Your responsibility is to safely push an APPROVED fix to a repository.

            Input contains:
            - github_owner, repository_name
            - base_branch (the branch to start from)
            - new_branch (the branch you must create for the fix)
            - commit_message, pr_title, pr_body
            - file_changes: a list of files with their FULL new contents

            Tasks (in this exact order):
            1. Create the new branch `new_branch` from `base_branch`.
            2. For each entry in file_changes, create or update that file on the
               new branch with the EXACT contents provided in new_content.
            3. Open a pull request from `new_branch` into `base_branch` using
               pr_title and pr_body.

            Hard safety rules:
            - NEVER commit directly to the base branch.
            - NEVER delete files.
            - NEVER modify files that are not listed in file_changes.
            - Write file contents EXACTLY as provided. Do not reformat or "improve".
            - If any step fails, set success=false and explain in `message`.

            Return only data matching FixApplyOutput, including the pull request
            URL when one is created.
            """,
        input_schema=FixApplyInput,
        output_schema=FixApplyOutput,
        tools=[toolset],
    )
    return toolset, fix_apply_agent