"""Agentic Movie Intelligence layer.

Sits beside backend/app and only READS from it: the saved Random Forest (via the existing predict route), the dataset and the
insight helpers. Nothing here retrains, edits or replaces a model artifact. The LLM writes prose from evidence the code gathered;
it never produces or alters the prediction.
"""
