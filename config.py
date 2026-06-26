"""Central model configuration for all DeployMind agents.

Two tiers, both overridable via .env:
  - FAST_MODEL  : cheap/fast model for mechanical agents (repo + file reading).
  - SMART_MODEL : stronger model for reasoning-heavy agents (diagnosis, fixing,
                  validation, and multi-step writes).

Override either in .env, e.g.:
  FAST_MODEL=gemini-2.5-flash
  SMART_MODEL=gemini-2.5-pro
"""
import os
from dotenv import load_dotenv

load_dotenv(".env")

FAST_MODEL = os.getenv("FAST_MODEL", "gemini-2.5-flash")
SMART_MODEL = os.getenv("SMART_MODEL", "gemini-2.5-pro")
