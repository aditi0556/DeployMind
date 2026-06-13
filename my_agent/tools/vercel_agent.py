import httpx
import asyncio
from google.adk.agents.llm_agent import LlmAgent
from google.adk.tools import FunctionTool

async def get_deployment_logs(deployment_id: str, vercel_token: str):
    headers = {
        "Authorization": f"Bearer {vercel_token}"
    }

    async with httpx.AsyncClient() as client:
        response = await client.get(
            f"https://api.vercel.com/v3/deployments/{deployment_id}/events",
            headers=headers,
            params={
                "limit": -1,
                "builds": 1,
            }
        )
        print("Log fomr VERCEL ",response.json())
        response.raise_for_status()
        return response.json()