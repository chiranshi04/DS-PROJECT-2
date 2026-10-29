"""Minimal Phase 1 web surface; it does not retain uploads or make predictions."""

from fastapi import FastAPI

app = FastAPI(title="InDivoice", version="0.1.0")


@app.get("/")
def root() -> dict[str, str]:
    """Describe the current research-demo state."""
    return {
        "name": "InDivoice",
        "status": "scaffold",
        "message": "No model is loaded in Phase 1; synthetic speech is not evidence of fraud.",
    }
