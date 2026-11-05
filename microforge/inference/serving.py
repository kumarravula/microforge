# Copyright 2024 MicroForge Contributors
# SPDX-License-Identifier: Apache-2.0
"""
Lightweight REST API serving for MicroForge using FastAPI.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

app = FastAPI(title="MicroForge Edge Serving", version="0.1.0")

_engine = None


class PredictRequest(BaseModel):
    query: str
    tools: Optional[List[Dict[str, Any]]] = None
    temperature: float = 0.0
    use_grammar: bool = True


class PredictResponse(BaseModel):
    query: str
    function_calls: List[Dict[str, Any]]
    confidence: float
    latency_ms: float
    tokens_generated: int


@app.get("/health")
def health() -> Dict[str, str]:
    return {"status": "ok", "engine_loaded": str(_engine is not None)}


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest) -> Dict[str, Any]:
    global _engine
    if _engine is None:
        raise HTTPException(status_code=503, detail="Model engine not loaded")

    return _engine.run(
        query=req.query,
        tools=req.tools,
        temperature=req.temperature,
        use_grammar=req.use_grammar,
    )
