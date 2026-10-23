"""HTTP inference service for reaction CNN (port 8100)."""

from __future__ import annotations

import os
import time
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel, Field

from reaction.infer_core import load_model, model_info, predict_image_b64

SERVE_KEY = os.environ.get("REACTION_SERVE_KEY", "")


@asynccontextmanager
async def lifespan(_app: FastAPI):
    load_model()
    yield


app = FastAPI(title="Homelab Reaction Vision", version="1.0.0", lifespan=lifespan)


def _require_key(x_key: str | None) -> None:
    if SERVE_KEY and x_key != SERVE_KEY:
        raise HTTPException(status_code=401, detail="Invalid or missing reaction serve key")


class ReactionRequest(BaseModel):
    image_b64: str = Field(..., min_length=32)
    threshold: float | None = Field(default=None, ge=0.0, le=1.0)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "service": "reaction-vision"}


@app.get("/info")
def info(x_reaction_key: str | None = Header(default=None, alias="X-Reaction-Key")) -> dict[str, Any]:
    _require_key(x_reaction_key)
    return model_info()


@app.post("/infer/reaction")
def infer_reaction(
    body: ReactionRequest,
    x_reaction_key: str | None = Header(default=None, alias="X-Reaction-Key"),
) -> dict[str, Any]:
    _require_key(x_reaction_key)
    started = time.perf_counter()
    try:
        out = predict_image_b64(body.image_b64, body.threshold)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    out["duration_ms"] = int((time.perf_counter() - started) * 1000)
    return out
