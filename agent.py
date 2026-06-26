# agent.py - DeployMind orchestrator (Session 8: full self-healing pipeline)
import os
import time
import asyncio
from typing import Optional

from dotenv import load_dotenv
from google.genai import types
from google.adk.agents.context import Context
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.adk import Workflow
from google.adk.workflow import node
from pydantic import BaseModel, Field

from tools.vercel_agent import (
    get_deployment_logs,
    get_latest_failed_deployment,
    trigger_redeploy,
)
from tools.github_agent import (
    create_github_agent,
    create_file_retrieval_agent,
    create_github_fix_agent,
    RepoInputSchema,
    RepoAnalysisOutput,
    FileRetrievalInput,
    FileRetrievalOutput,
    FixApplyInput,
    FixApplyOutput,
)
from log_analysis_agent import (
    LogAnalysisInput,
    LogAnalysisOutput,
    log_analysis_agent,
)
from fix_agent import FixGenerationInput, FixGenerationOutput, fix_agent
from validation_agent import FixValidationInput, FixValidationOutput, validation_agent

# Load environment variables from .env before reading any tokens.
load_dotenv(".env")
GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")
VERCEL_TOKEN = os.getenv("VERCEL_TOKEN")

# Stable project config lives in .env. The deployment ID changes every run,
# so it's supplied at run time (CLI arg or prompt), not hardcoded in .env.
GITHUB_OWNER = os.getenv("GITHUB_OWNER")
GITHUB_REPO = os.getenv("GITHUB_REPO")
BASE_BRANCH = os.getenv("BASE_BRANCH", "main")
DEPLOYMENT_ID_FALLBACK = os.getenv("DEPLOYMENT_ID")  # optional explicit override
# Force the human-approval gate for EVERY fix, even low-risk ones (great for demos).
REQUIRE_APPROVAL = os.getenv("REQUIRE_APPROVAL", "false").lower() in ("1", "true", "yes")
# Vercel project name (defaults to the repo name) used for auto-discovery.
VERCEL_PROJECT = os.getenv("VERCEL_PROJECT") or GITHUB_REPO


async def resolve_deployment_id() -> str:
    """Get the failed deployment ID to investigate.

    Priority:
      1. command-line arg            (explicit:  python agent.py dpl_xxx)
      2. DEPLOYMENT_ID in .env       (explicit override)
      3. auto-discover the latest FAILED deployment for the Vercel project
      4. interactive prompt          (last resort)
    """
    import sys

    if len(sys.argv) > 1 and sys.argv[1].strip():
        return sys.argv[1].strip()
    if DEPLOYMENT_ID_FALLBACK:
        return DEPLOYMENT_ID_FALLBACK

    if VERCEL_PROJECT and VERCEL_TOKEN:
        print(f"Looking up the latest FAILED deployment for '{VERCEL_PROJECT}'...")
        found = await get_latest_failed_deployment(VERCEL_PROJECT, VERCEL_TOKEN)
        if found:
            print(f"Found failed deployment: {found}")
            return found
        print("No failed deployment found via the Vercel API.")

    return input("Enter the failed Vercel deployment ID: ").strip()

APP_NAME = "deployMind"

MAX_INVESTIGATION_ROUNDS = 3


def as_model(value, model):
    """ctx.run_node() returns a plain dict in this ADK version; coerce it (or a
    dict passed between nodes) into the expected Pydantic model."""
    if isinstance(value, model):
        return value
    return model.model_validate(value)


class DeploymentContext(BaseModel):
    deployment_id: str = Field(description="Vercel deployment ID to investigate")
    github_owner: str = Field(description="GitHub username or organisation")
    github_repo: str = Field(description="GitHub repository name")
    base_branch: str = Field(default="main", description="Production/base branch")
    framework: Optional[str] = Field(default=None, description="Primary language/framework")
    additional_context: Optional[str] = Field(
        default=None, description="Any project-specific instruction"
    )


# ---------------------------------------------------------------------------
# Node 1: Repository analysis
# ---------------------------------------------------------------------------
@node(rerun_on_resume=True)
async def repo_analysis_node(ctx: Context) -> RepoAnalysisOutput:
    toolset, github_agent = await create_github_agent(ctx.state["github_token"])
    deployment_context = DeploymentContext.model_validate(ctx.state["deployment_context"])
    repo_input = RepoInputSchema(
        github_owner=deployment_context.github_owner,
        repository_name=deployment_context.github_repo,
    )
    result = as_model(
        await ctx.run_node(github_agent, node_input=repo_input), RepoAnalysisOutput
    )
    ctx.state.update(
        {
            "deployment_id": deployment_context.deployment_id,
            "github_owner": deployment_context.github_owner,
            "repository_name": deployment_context.github_repo,
            "base_branch": deployment_context.base_branch,
            # Keep the full analysis object so downstream nodes can reuse it.
            "repo_analysis_output": result.model_dump(),
        }
    )
    print("Closing MCP server connection...")
    await toolset.close()
    print("Cleanup complete.")
    return result


# ---------------------------------------------------------------------------
# Node 2: Log analysis (with bounded file-investigation loop)
# ---------------------------------------------------------------------------
@node(rerun_on_resume=True)
async def log_analysis_node(ctx: Context) -> LogAnalysisOutput:
    repo_analysis = RepoAnalysisOutput.model_validate(ctx.state["repo_analysis_output"])
    logs = await get_deployment_logs(ctx.state["deployment_id"], ctx.state["vercel_token"])
    ctx.state["deployment_logs"] = logs

    retrieved_files: dict[str, str] = {}
    analysis: Optional[LogAnalysisOutput] = None

    for round_num in range(MAX_INVESTIGATION_ROUNDS):
        log_input = LogAnalysisInput(
            deployment_logs=logs,
            repo_analysis_output=repo_analysis,
            retrieved_files=retrieved_files,
        )
        analysis = as_model(
            await ctx.run_node(log_analysis_agent, node_input=log_input),
            LogAnalysisOutput,
        )

        # Root cause confirmed, or no more files requested -> stop investigating.
        if not analysis.need_additional_code or not analysis.required_files:
            break

        toolset, github_file_agent = await create_file_retrieval_agent(ctx.state["github_token"])
        files_result = as_model(
            await ctx.run_node(
                github_file_agent,
                node_input=FileRetrievalInput(
                    github_owner=ctx.state["github_owner"],
                    repository_name=ctx.state["repository_name"],
                    required_files=analysis.required_files,
                ),
            ),
            FileRetrievalOutput,
        )
        await toolset.close()
        retrieved_files.update(files_result.files)
        print(f"[round {round_num + 1}] retrieved {len(files_result.files)} file(s)")

    ctx.state["retrieved_files"] = retrieved_files
    ctx.state["root_cause"] = analysis.root_cause if analysis else ""
    ctx.state["log_analysis"] = analysis.model_dump() if analysis else {}
    print("Root cause:", ctx.state["root_cause"])
    return analysis


# ---------------------------------------------------------------------------
# Node 3: Fix generation
# ---------------------------------------------------------------------------
@node(rerun_on_resume=True)
async def fix_node(ctx: Context) -> FixGenerationOutput:
    log_analysis = LogAnalysisOutput.model_validate(ctx.state["log_analysis"])
    repo_analysis = RepoAnalysisOutput.model_validate(ctx.state["repo_analysis_output"])
    fix_input = FixGenerationInput(
        root_cause=log_analysis.root_cause,
        evidence=log_analysis.evidence,
        repo_analysis_output=repo_analysis,
        relevant_files=ctx.state.get("retrieved_files", {}),
        deployment_logs=ctx.state.get("deployment_logs", ""),
    )
    fix = as_model(
        await ctx.run_node(fix_agent, node_input=fix_input), FixGenerationOutput
    )
    ctx.state["proposed_fix"] = fix.model_dump()
    print("Proposed fix:", fix.fix_summary, f"(risk={fix.risk_level})")
    return fix


# ---------------------------------------------------------------------------
# Node 4: Validation
# ---------------------------------------------------------------------------
@node(rerun_on_resume=True)
async def validation_node(ctx: Context) -> FixValidationOutput:
    fix = FixGenerationOutput.model_validate(ctx.state["proposed_fix"])
    repo_analysis = RepoAnalysisOutput.model_validate(ctx.state["repo_analysis_output"])
    validation = as_model(
        await ctx.run_node(
            validation_agent,
            node_input=FixValidationInput(
                root_cause=ctx.state.get("root_cause", ""),
                deployment_logs=ctx.state.get("deployment_logs", ""),
                proposed_fix=fix,
                repo_analysis_output=repo_analysis,
            ),
        ),
        FixValidationOutput,
    )
    ctx.state["validation"] = validation.model_dump()
    print(
        "Validation:",
        f"valid={validation.is_valid}",
        f"needs_human_approval={validation.needs_human_approval}",
    )
    return validation


# ---------------------------------------------------------------------------
# Human-in-the-loop approval (console gate)
# ---------------------------------------------------------------------------
async def request_human_approval(
    fix: FixGenerationOutput, validation: FixValidationOutput
) -> bool:
    print("\n" + "=" * 70)
    print("HUMAN APPROVAL REQUIRED")
    print("=" * 70)
    print(f"Fix:        {fix.fix_summary}")
    print(f"Risk:       {fix.risk_level}")
    print(f"Reasoning:  {fix.reasoning}")
    if validation.issues_found:
        print("Issues flagged by validation agent:")
        for issue in validation.issues_found:
            print(f"  - {issue}")
    print("Files to change:")
    for change in fix.file_changes:
        print(f"  - {change.file_path}: {change.change_summary}")
    print("=" * 70)

    loop = asyncio.get_event_loop()
    answer = await loop.run_in_executor(None, input, "Approve and redeploy this fix? [y/N]: ")
    return answer.strip().lower() in ("y", "yes")


# ---------------------------------------------------------------------------
# Node 5: Apply fix (push + redeploy), gated by validation + human approval
# ---------------------------------------------------------------------------
@node(rerun_on_resume=True)
async def apply_fix_node(ctx: Context):
    validation = FixValidationOutput.model_validate(ctx.state["validation"])
    fix = FixGenerationOutput.model_validate(ctx.state["proposed_fix"])

    if not fix.can_fix or not fix.file_changes:
        print("No applicable fix was generated. Stopping without changes.")
        return {"status": "no_fix"}

    if not validation.is_valid:
        print("Validation rejected the fix. Stopping without changes.")
        return {"status": "invalid_fix", "issues": validation.issues_found}

    # Diagram branch: validation -> "if yes" -> human gate ; "if no" -> push directly.
    # REQUIRE_APPROVAL forces the gate on for every fix (demo-friendly override).
    approved = True
    if validation.needs_human_approval or REQUIRE_APPROVAL:
        approved = await request_human_approval(fix, validation)

    if not approved:
        print("Fix rejected by human reviewer. Aborting redeploy.")
        return {"status": "rejected_by_human"}

    # Push the fix via the write-enabled GitHub agent.
    # Timestamp suffix keeps the branch unique so demos can be re-run without
    # hitting "reference already exists".
    new_branch = f"deploymind/fix-{ctx.state['deployment_id'][:8]}-{int(time.time())}"
    toolset, github_fix_agent = await create_github_fix_agent(ctx.state["github_token"])
    raw_result = await ctx.run_node(
        github_fix_agent,
        node_input=FixApplyInput(
            github_owner=ctx.state["github_owner"],
            repository_name=ctx.state["repository_name"],
            base_branch=ctx.state.get("base_branch", "main"),
            new_branch=new_branch,
            commit_message=f"fix: {fix.fix_summary}",
            pr_title=f"[DeployMind] {fix.fix_summary}",
            pr_body=fix.reasoning,
            file_changes=fix.file_changes,
        ),
    )
    await toolset.close()

    # The write agent is non-deterministic: it can finish without returning a
    # structured result (e.g. a transient MCP error swallowed by graceful
    # handling). Degrade gracefully instead of crashing the whole pipeline.
    if raw_result is None:
        print(
            "GitHub fix agent returned no result. It may have partially applied "
            f"the fix on branch '{new_branch}'. Check the repo and re-run if needed."
        )
        return {"status": "apply_incomplete", "branch": new_branch}

    apply_result = as_model(raw_result, FixApplyOutput)
    print("Apply result:", apply_result)
    ctx.state["apply_result"] = apply_result.model_dump()

    if not apply_result.success:
        print("GitHub fix agent reported failure; skipping redeploy.")
        return apply_result

    # Trigger redeploy (pushing a connected repo usually auto-deploys; this is the explicit path).
    try:
        redeploy = await trigger_redeploy(
            project_name=ctx.state["repository_name"],
            github_owner=ctx.state["github_owner"],
            github_repo=ctx.state["repository_name"],
            vercel_token=ctx.state["vercel_token"],
            git_ref=new_branch,
        )
        print("Redeploy triggered:", redeploy.get("url") or redeploy)
    except Exception as exc:  # noqa: BLE001 - non-fatal; auto-deploy may already cover it
        print(f"Redeploy note (auto-deploy may handle this): {exc}")

    return apply_result


# ---------------------------------------------------------------------------
# The self-healing pipeline
# ---------------------------------------------------------------------------
root_agent = Workflow(
    name="root_agent",
    edges=[
        ("START", repo_analysis_node),
        (repo_analysis_node, log_analysis_node),
        (log_analysis_node, fix_node),
        (fix_node, validation_node),
        (validation_node, apply_fix_node),
    ],
)


async def async_main():
    session_service = InMemorySessionService()
    missing = [
        name
        for name, value in {
            "GITHUB_OWNER": GITHUB_OWNER,
            "GITHUB_REPO": GITHUB_REPO,
        }.items()
        if not value
    ]
    if missing:
        raise SystemExit(
            "Missing required .env values: "
            + ", ".join(missing)
            + ". Copy .env.example to .env and fill them in."
        )
    deployment_id = await resolve_deployment_id()
    if not deployment_id:
        raise SystemExit("No deployment ID provided.")
    query = DeploymentContext(
        deployment_id=deployment_id,
        github_owner=GITHUB_OWNER,
        github_repo=GITHUB_REPO,
        base_branch=BASE_BRANCH,
    )
    session = await session_service.create_session(
        app_name=APP_NAME,
        user_id="user123",
        state={
            "github_token": GITHUB_TOKEN,
            "vercel_token": VERCEL_TOKEN,
            "deployment_context": query.model_dump(),
        },
    )
    print(f"User Query: '{query}'")
    content = types.Content(
        role="user", parts=[types.Part(text="Start deployment investigation")]
    )
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
        print(f"Event author: {event.author}")


if __name__ == "__main__":
    try:
        asyncio.run(async_main())
    except Exception as e:
        print(f"An error occurred: {e}")
