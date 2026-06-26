# DeployMind

**Turning deployment failures into automated fixes.**

DeployMind is a semi-autonomous, multi-agent system that watches a failed cloud
deployment, figures out *why* it broke, proposes a safe fix, asks a human to
approve it, and pushes the fix back to GitHub to trigger a new deploy.

Built with **Google ADK** (Agent Development Kit) + **Gemini**, wired to
**GitHub** (via MCP) and **Vercel** APIs.

## Pipeline

```
START
  → repo_analysis_node     read-only GitHub agent: understand the repo
  → log_analysis_node      pull Vercel logs + bounded file-fetch loop (max 3 rounds)
  → fix_node               Fix Generation Agent: minimal, full-file changes
  → validation_node        Validation Agent: is it safe? does it need a human?
  → apply_fix_node         human gate → write-mode GitHub agent (branch + PR) → redeploy
END
```

Data flows between nodes through shared `ctx.state` (ADK binds node parameters
from state). The validation agent decides whether a fix needs human approval;
risky/multi-file/source-code changes always do.

## Agents

| Agent | File | Model tier | Role |
|-------|------|------------|------|
| Repo analysis | `tools/github_agent.py` | FAST | Detect framework, language, config files |
| File retrieval | `tools/github_agent.py` | FAST | Fetch specific files on request |
| Log analysis | `log_analysis_agent.py` | SMART | Diagnose root cause from logs + repo |
| Fix generation | `fix_agent.py` | SMART | Produce a safe, minimal fix |
| Validation | `validation_agent.py` | SMART | Safety gate; decide if a human must approve |
| GitHub fix-apply | `tools/github_agent.py` | SMART | Branch + commit + open PR (write mode) |

Model tiers are configured in `config.py` (`FAST_MODEL`, `SMART_MODEL`),
overridable via `.env`.

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env        # then fill in the secrets
```

Requires **Python 3.10+** and **Node.js** on PATH (the GitHub MCP server runs
via `npx`).

## Configuration (`.env`)

| Variable | Required | Purpose |
|----------|----------|---------|
| `GITHUB_TOKEN` | yes | GitHub PAT (needs write: branch + PR) |
| `VERCEL_TOKEN` | yes | Read deployment logs, trigger redeploys |
| `GOOGLE_API_KEY` | yes | Gemini access for every agent |
| `GITHUB_OWNER` / `GITHUB_REPO` | yes | The repo to debug and open PRs against |
| `BASE_BRANCH` | no | Branch to base fixes on (default `main`) |
| `VERCEL_PROJECT` | no | Vercel project name (defaults to repo name) |
| `DEPLOYMENT_ID` | no | Pin a specific deployment (else auto-discovered) |
| `REQUIRE_APPROVAL` | no | Force the human gate for every fix (default false) |
| `FAST_MODEL` / `SMART_MODEL` | no | Override model tiers |

## Run

```bash
source .venv/bin/activate

python agent.py                 # auto-discovers the latest FAILED deployment
python agent.py dpl_xxxxxxxx    # or target a specific deployment
```

The deployment ID resolves in this order: **CLI arg → `DEPLOYMENT_ID` → Vercel
auto-discovery → interactive prompt**.

## Safe-automation design

- **Structured outputs** — every agent returns a validated Pydantic schema.
- **Least privilege** — analysis agents are read-only; only the fix-apply agent
  can write, and only via branch + PR (never pushes to `main`, never deletes).
- **Loop prevention** — `MAX_INVESTIGATION_ROUNDS = 3`.
- **Human-in-the-loop** — risky fixes require explicit `y/N` approval.
- **Graceful degradation** — partial/failed agent runs are reported, not crashed.

## Testing

A ready-made broken Next.js app lives at `~/Desktop/deploymind-test-app`
(intentional cross-file TypeScript error). Push it to GitHub, import into Vercel,
let the build fail, then point DeployMind at it.
