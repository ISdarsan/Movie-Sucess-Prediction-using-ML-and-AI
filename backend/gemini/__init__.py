"""Gemini chat assistant (backend only).

Sits beside backend/app and backend/agent and imports neither the prediction model nor the agent orchestrator: in this phase the
assistant answers general questions only and is sent no project or movie data. The API key is read on the server, never returned,
logged or placed in any response."""
