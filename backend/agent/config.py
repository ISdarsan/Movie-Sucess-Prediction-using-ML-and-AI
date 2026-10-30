import os
from pathlib import Path

from ..app.config import ROOT

ENABLED = os.getenv("FRAME_AGENT_ENABLED", "1") not in ("0", "false", "False", "")
OLLAMA_URL = os.getenv("FRAME_OLLAMA_URL", "http://localhost:11434").rstrip("/")
OLLAMA_MODEL = os.getenv("FRAME_OLLAMA_MODEL", "llama3.1:8b")
OLLAMA_NUM_CTX = int(os.getenv("FRAME_OLLAMA_NUM_CTX", "4096"))
OLLAMA_KEEP_ALIVE = os.getenv("FRAME_OLLAMA_KEEP_ALIVE", "10m")

LLM_TIMEOUT_S = float(os.getenv("FRAME_AGENT_LLM_TIMEOUT", "120"))   # measured: reports take 28-48 s, cold load adds ~10 s
RUN_DEADLINE_S = float(os.getenv("FRAME_AGENT_RUN_DEADLINE", "240"))  # whole run, after which the template report is used
MAX_TOOL_CALLS = int(os.getenv("FRAME_AGENT_MAX_TOOL_CALLS", "6"))

DB_PATH = Path(os.getenv("FRAME_AGENT_DB", ROOT / ".agent_data" / "runs.sqlite"))  # inside the repo folder on D:, git-ignored

# ---- risk-rule thresholds (used by risks.py). Centralised here; override with env vars only for experiments.
# COHORT_THIN_N: a cohort of fewer films than this is called thin. 30 is the conventional rule-of-thumb minimum for reading class
#   proportions at all, and it is the same cut-off the existing /api/cohort endpoint already uses for its `thin` flag
#   (insight.cohort: 0 < n < 30). A tested equivalence keeps the two in step. It is a convention, not a statistical guarantee.
# MIN_COMPARABLES: the Analyst requests up to 8 similar earlier films (tools.ComparablesArgs.limit max). Fewer than 5 means more than a
#   third of the requested set is missing, so the "similar films" are too few to say much. A design heuristic, not a statistical one.
COHORT_THIN_N = int(os.getenv("FRAME_AGENT_COHORT_THIN_N", "30"))
MIN_COMPARABLES = int(os.getenv("FRAME_AGENT_MIN_COMPARABLES", "5"))

# ---- Wikipedia research (research.py). Read-only, one allowed host, strictly bounded.
RESEARCH_ENABLED = os.getenv("FRAME_AGENT_RESEARCH_ENABLED", "1") not in ("0", "false", "False", "")

RESEARCH_HOSTS = ("en.wikipedia.org",)            # exact-match allow-list: every request URL and every redirect target must be on it
RESEARCH_TIMEOUT_S = float(os.getenv("FRAME_AGENT_RESEARCH_TIMEOUT", "5"))       # per request
RESEARCH_TOTAL_S = float(os.getenv("FRAME_AGENT_RESEARCH_TOTAL", "20"))          # whole research step
RESEARCH_MAX_REQUESTS = int(os.getenv("FRAME_AGENT_RESEARCH_MAX_REQUESTS", "8")) # every HTTP request counts, redirects included
RESEARCH_MAX_SOURCES = int(os.getenv("FRAME_AGENT_RESEARCH_MAX_SOURCES", "4"))
# Wikimedia's robot policy rejects clients that do not identify themselves with contact information (verified live: a bare UA gets HTTP 403).
# The default points at the project's public repository; a deployer should set their own.
RESEARCH_USER_AGENT = os.getenv("FRAME_AGENT_RESEARCH_UA", "FRAME-capstone-research/1.0 (https://github.com/ISdarsan/Movie-Sucess-Prediction-using-ML-and-AI; read-only student project)")
RESEARCH_MAX_GENRES = 2
RESEARCH_MAX_REDIRECTS = 1
RESEARCH_MAX_BYTES = int(os.getenv("FRAME_AGENT_RESEARCH_MAX_BYTES", "300000"))  # per response, measured after decompression
RESEARCH_EXCERPT_CHARS = 400
RESEARCH_SECTION_CHARS = 700
RESEARCH_SECTION_BUDGET_CHARS = 1800
