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
                command='npx', # Command to run the server
                args=["-y",    # Arguments for the command
                    "@modelcontextprotocol/server-github",
                    GITHUB_TOKEN],
            ),
        ),
    )

    repo_agent = LlmAgent(
        model = 'gemini-2.5-flash',
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
                args=[
                    "-y",
                    "@modelcontextprotocol/server-github",
                    github_token,
                ],
            ),
        ),
    )
    file_agent = LlmAgent(
        model="gemini-2.5-flash",
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