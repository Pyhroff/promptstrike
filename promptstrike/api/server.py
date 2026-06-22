"""
PromptStrike live dashboard — FastAPI + WebSocket backend.
Start with: promptstrike serve
Then open:  http://127.0.0.1:8080
"""

import asyncio
import random
import uuid
from pathlib import Path
from typing import Any

import yaml
from fastapi import BackgroundTasks, FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from promptstrike.adapters.groq import GroqAdapter
from promptstrike.adapters.openai import OpenAIAdapter
from promptstrike.adapters.ollama import OllamaAdapter
from promptstrike.config import settings
from promptstrike.core.pair import run_pair, AttackStatus
from promptstrike.core.tap import run_tap
from promptstrike.storage.db import Database
from promptstrike.report.generator import generate_html_report

app = FastAPI(title="PromptStrike", version="0.1.0")

_STATIC = Path(__file__).parent / "static"
_BEHAVIORS = Path("behaviors.yaml")

# scan_id → asyncio.Queue of event dicts
_active_scans: dict[str, asyncio.Queue] = {}


# ── Request / response models ──────────────────────────────────────────────────

class ScanRequest(BaseModel):
    target: str = "groq/llama-3.3-70b-versatile"
    algorithm: str = "pair"          # pair | tap
    goals: int | None = None
    category: str | None = None
    max_iter: int = 20
    budget: int = 200
    campaign_name: str = ""


# ── Routes ─────────────────────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
async def dashboard() -> HTMLResponse:
    html = (_STATIC / "dashboard.html").read_text(encoding="utf-8")
    return HTMLResponse(html)


@app.get("/api/campaigns")
async def list_campaigns() -> list[dict]:
    db = Database()
    return [dict(r) for r in db.list_campaigns()]


@app.get("/api/campaigns/{campaign_id}")
async def get_campaign(campaign_id: int) -> dict:
    db = Database()
    row = db.get_campaign(campaign_id)
    if not row:
        raise HTTPException(status_code=404, detail="Campaign not found")
    runs = db.get_campaign_runs(campaign_id)
    return {"campaign": dict(row), "runs": [dict(r) for r in runs]}


@app.get("/api/campaigns/{campaign_id}/report")
async def get_report(campaign_id: int) -> HTMLResponse:
    db = Database()
    row = db.get_campaign(campaign_id)
    if not row:
        raise HTTPException(status_code=404, detail="Campaign not found")
    runs = db.get_campaign_runs(campaign_id)
    html = generate_html_report(dict(row), [dict(r) for r in runs])
    return HTMLResponse(html)


@app.post("/api/scan")
async def start_scan(req: ScanRequest, background_tasks: BackgroundTasks) -> dict:
    if not settings.groq_api_key:
        raise HTTPException(status_code=400, detail="GROQ_API_KEY not configured in .env")

    scan_id = str(uuid.uuid4())[:8]
    queue: asyncio.Queue = asyncio.Queue()
    _active_scans[scan_id] = queue

    background_tasks.add_task(_run_scan, scan_id, req, queue)
    return {"scan_id": scan_id}


@app.websocket("/ws/{scan_id}")
async def websocket_stream(ws: WebSocket, scan_id: str) -> None:
    await ws.accept()
    queue = _active_scans.get(scan_id)
    if not queue:
        await ws.send_json({"type": "error", "message": f"Scan {scan_id} not found"})
        await ws.close()
        return

    try:
        while True:
            try:
                event = await asyncio.wait_for(queue.get(), timeout=180)
            except asyncio.TimeoutError:
                await ws.send_json({"type": "ping"})
                continue

            await ws.send_json(event)
            if event.get("type") == "scan_complete":
                break
    except WebSocketDisconnect:
        pass
    finally:
        _active_scans.pop(scan_id, None)


# ── Background scan runner ─────────────────────────────────────────────────────

async def _run_scan(scan_id: str, req: ScanRequest, queue: asyncio.Queue) -> None:
    db = Database()

    try:
        # Load behaviors
        with open(_BEHAVIORS) as f:
            raw = yaml.safe_load(f)
        behaviors = raw["behaviors"]
        if req.category:
            behaviors = [b for b in behaviors if b["category"] == req.category]
        if req.goals:
            behaviors = random.sample(behaviors, min(req.goals, len(behaviors)))

        # Init adapters
        provider = req.target.split("/")[0]
        model_name = req.target.split("/", 1)[-1]

        if provider == "groq":
            target_adapter = GroqAdapter(api_key=settings.groq_api_key, model=model_name)
        elif provider == "openai":
            target_adapter = OpenAIAdapter(api_key=settings.openai_api_key, model=model_name)
        elif provider == "ollama":
            target_adapter = OllamaAdapter(model=model_name, base_url=settings.ollama_url)
        else:
            target_adapter = GroqAdapter(api_key=settings.groq_api_key, model=model_name)

        attacker = GroqAdapter(api_key=settings.groq_api_key, model=settings.attacker_model)
        judge    = GroqAdapter(api_key=settings.groq_api_key, model=settings.judge_model)

        from datetime import datetime
        name = req.campaign_name or f"scan_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        campaign_id = db.create_campaign(name=name, target=req.target, algorithm=req.algorithm)

        await queue.put({
            "type": "scan_start",
            "scan_id": scan_id,
            "campaign_id": campaign_id,
            "campaign_name": name,
            "target": req.target,
            "algorithm": req.algorithm,
            "total_goals": len(behaviors),
        })

        total_calls = 0
        succeeded = 0

        for idx, beh in enumerate(behaviors):
            if total_calls >= req.budget:
                break

            await queue.put({
                "type": "goal_start",
                "goal_idx": idx + 1,
                "total_goals": len(behaviors),
                "behavior_id": beh["id"],
                "goal": beh["goal"],
                "category": beh.get("category", ""),
                "owasp": beh.get("owasp", ""),
            })

            async def emit(event: dict) -> None:
                event["goal_idx"] = idx + 1
                await queue.put(event)

            if req.algorithm == "tap":
                result_tap = await run_tap(
                    behavior_id=beh["id"],
                    goal=beh["goal"],
                    target=target_adapter,
                    attacker=attacker,
                    judge=judge,
                    branching_factor=settings.tap_branching_factor,
                    depth=settings.tap_depth,
                    pruning_threshold=settings.tap_pruning_threshold,
                    call_budget=req.budget - total_calls,
                    judge_threshold=settings.judge_threshold,
                    on_event=emit,
                )
                total_calls += result_tap.calls_used
                if result_tap.status == AttackStatus.SUCCESS:
                    succeeded += 1

                # Convert to pair-compatible for storage
                from promptstrike.core.pair import PairResult, Iteration
                from promptstrike.core.judge import JudgeResult
                compat = PairResult(
                    behavior_id=beh["id"],
                    goal=beh["goal"],
                    target_model=result_tap.target_model,
                    status=result_tap.status,
                    winning_prompt=result_tap.winning_prompt,
                    final_score=result_tap.final_score,
                    calls_used=result_tap.calls_used,
                )
                db.save_run(campaign_id, compat, beh.get("category",""), beh.get("owasp",""))
                status = result_tap.status
                final_score = result_tap.final_score
            else:
                result_pair = await run_pair(
                    behavior_id=beh["id"],
                    goal=beh["goal"],
                    target=target_adapter,
                    attacker=attacker,
                    judge=judge,
                    max_iterations=req.max_iter,
                    call_budget=req.budget - total_calls,
                    judge_threshold=settings.judge_threshold,
                    on_event=emit,
                )
                total_calls += result_pair.calls_used
                if result_pair.status == AttackStatus.SUCCESS:
                    succeeded += 1
                db.save_run(campaign_id, result_pair, beh.get("category",""), beh.get("owasp",""))
                status = result_pair.status
                final_score = result_pair.final_score

            await queue.put({
                "type": "goal_end",
                "goal_idx": idx + 1,
                "behavior_id": beh["id"],
                "goal": beh["goal"],
                "status": status.value,
                "final_score": final_score,
                "calls_used": total_calls,
            })

        db.finalize_campaign(campaign_id)
        asr = succeeded / max(1, len(behaviors)) * 100

        await queue.put({
            "type": "scan_complete",
            "campaign_id": campaign_id,
            "succeeded": succeeded,
            "total": len(behaviors),
            "asr": round(asr, 1),
            "total_calls": total_calls,
        })

    except Exception as exc:  # noqa: BLE001
        await queue.put({"type": "error", "message": str(exc)})
    finally:
        _active_scans.pop(scan_id, None)
