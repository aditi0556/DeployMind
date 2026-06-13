# agent.py (modify get_tools_async and other parts as needed)
# ./adk_agent_samples/mcp_agent/agent.py
import os
import asyncio
from dotenv import load_dotenv
from google.genai import types
from google.adk.agents.llm_agent import LlmAgent
from google.adk.agents.context import Context
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.adk.artifacts.in_memory_artifact_service import InMemoryArtifactService # Optional
from google.adk.tools.mcp_tool import McpToolset
from google.adk.tools.mcp_tool.mcp_session_manager import StdioConnectionParams
from mcp import StdioServerParameters
from tools.vercel_agent import get_deployment_logs
from tools.github_agent import create_github_agent
from tools.github_agent import RepoInputSchema,RepoAnalysisOutput
from log_analysis_agent import LogAnalysisOutput,LogAnalysisInput
from google.adk import Workflow
from google.adk.workflow import node
from pydantic import BaseModel,Field
from typing import Optional
# Load environment variables from .env file in the parent directory
# Place this near the top, before using env vars like API keys
load_dotenv('.env')
GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")
VERCEL_TOKEN = os.getenv("VERCEL_TOKEN")
APP_NAME = "deployMind"

class DeploymentContext(BaseModel):
    deployment_id:str=Field(description="Vercel deployment ID to investigate deployment logs")
    github_owner:str=Field(description="Github usernmae or organisation")
    github_repo:str=Field(description="Github repository name")
    framework:Optional[str]=Field(default=None,description="Primary language/framework used")
    additional_context:Optional[str]=Field(default=None,description="Any project-specific instruction")

class RetrievedFilesOutput(BaseModel):
    files: dict[str, str]

@node(rerun_on_resume=True)
async def repo_analysis_node(ctx:Context) -> RepoAnalysisOutput:
    toolset,github_agent = await create_github_agent(ctx.state["github_token"])
    deployment_context = DeploymentContext.model_validate(
        ctx.state["deployment_context"]
    )
    repo_input = RepoInputSchema(
        github_owner= deployment_context.github_owner,
        repository_name=deployment_context.github_repo,
    )
    result:RepoAnalysisOutput = await ctx.run_node(github_agent,node_input=repo_input)
    ctx.state.update(
        {
            "deployment_id":deployment_context.deployment_id,
            "github_owner":deployment_context.github_owner,
            "repository_name":deployment_context.github_repo,
            "framework":result.framework,
            "project_structure":result.project_structure,
            "language":result.language,
            "package_manager":result.package_manager,
            "deployment_related_files":result.deployment_related_files,
            "important_config_files":result.important_config_files,
            "architecture_summary":result.architecture_summary,
            "potential_debugging_targets": result.potential_debugging_targets,
        }
    )
    # Cleanup is handled automatically by the agent framework
    # But you can also manually close if needed:
    print("Closing MCP server connection...")
    await toolset.close()
    print("Cleanup complete.")
    print("Result from the repo analysis node",result)
    return result

@node(rerun_on_resume=True)
async def log_analysis_node(ctx:Context,repo_analysis:RepoAnalysisOutput):
    logs = await get_deployment_logs(ctx.state["deployment_id"],ctx.state["vercel_token"])
    retrieved_files = {}
    MAX_INVESTIGATION_ROUNDS = 3
    
    for round_num in range(MAX_INVESTIGATION_ROUNDS):
        log_input = LogAnalysisInput(
            deployment_logs=logs.logs,
            repo_analysis=repo_analysis,
            retrieved_files=retrieved_files,
        )
        analysis: LogAnalysisOutput = await ctx.run_node(
            log_analysis_agent,
            node_input=log_input,
        )
        # Root cause determined
        if not analysis.need_additional_code:
            return analysis
        # No files requested, stop investigation
        if not analysis.required_files:
            return analysis
        # Fetch files from GitHub
        _, github_file_agent = await create_file_retrieval_agent(
            ctx.state["github_token"]
        )
        files_result = await ctx.run_node(
            github_file_agent,
            node_input=FileRetrievalInput(
                github_owner=ctx.state["github_owner"],
                repository_name=ctx.state["repository_name"],
                required_files=analysis.required_files,
            )
        )
        retrieved_files.update(
            files_result.files
        )
    print("Analysis from the log_analysis_agent is ",analysis)
    return analysis


root_agent = Workflow(
    name="root_agent",
    edges=[("START",repo_analysis_node),(repo_analysis_node,log_analysis_node)],
)

async def async_main():
    session_service = InMemorySessionService()
    query = DeploymentContext(
        deployment_id="dpl_J7K9YYZjMkzbFm2cjFdub2NKaxnw",
        github_owner="aditi0556",
        github_repo="DeployMind",
    )
    session = await session_service.create_session(
        app_name=APP_NAME, user_id='user123',
        state={
            "github_token": GITHUB_TOKEN,
            "vercel_token": VERCEL_TOKEN,
            "deployment_context": query.model_dump(),
        }
    )
    print(f"User Qeury:'{query}'")
    content = types.Content(role="user",parts=[types.Part(text="Start deployment investigation")])   
    runner = Runner(
        app_name=APP_NAME,
        agent=root_agent,
        session_service=session_service,
    )

    print("Running agent...")
    events_async = runner.run_async(
        session_id=session.id, user_id=session.user_id, new_message=content
    )

    async for event in events_async:
        print(f"Event received: {event.content}")
        print(f"Event output: {event.output}")
        print(f"Event user : {event.author}")

if __name__ == '__main__':
    try:
        asyncio.run(async_main())
    except Exception as e:
        print(f"An error occurred: {e}")
