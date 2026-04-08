# server/app.py — entry point for OpenEnv multi-mode deployment
# This re-exports the FastAPI app from the root app.py
import sys
import os

# Add parent directory to path so root app.py is importable
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app import app  # noqa: F401 — re-export for openenv runner

__all__ = ["app"]
