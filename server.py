"""
FastAPI Server — v2.2.0
All REST endpoints + WebSocket stream + static UI serving.

UI PATH RESOLUTION (supports all execution contexts):
  1. PyInstaller one-file exe  → sys._MEIPASS/ui/index.html
  2. Next to exe               → <exe_dir>/ui/index.html
  3. Running from source       → <project_root>/ui/index.html
  4. Inline fallback           → embedded HTML string (never blank)
"""

import asyncio
import json
import sys
import os
import time
import logging
import platform as _platform
from pathlib import Path
from typing import Optional

import psutil

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from config.settings import settings
from agent.scheduler import MetricScheduler
from ai.diagnostic import AIDiagnosticEngine
from alerts.manager import AlertManager
from alerts.state_manager import alert_state_manager
from remediation.actions import RemediationEngine

logger = logging.getLogger(__name__)


# ── Robust UI path resolution ──────────────────────────────────────────────
def _resolve_ui_html() -> str:
    """
    Find and return the content of ui/index.html regardless of how the
    application was launched (source, PyInstaller one-file, installed exe).

    Search order:
      1. sys._MEIPASS/ui/           — PyInstaller extraction temp dir
      2. <executable_dir>/ui/       — next to the .exe file
      3. <this_file>/../ui/         — source tree (api/ → project root → ui/)
      4. <cwd>/ui/                  — current working directory fallback
    """
    candidates = []

    # 1. PyInstaller bundle
    if getattr(sys, "frozen", False) and hasattr(sys, "_MEIPASS"):
        candidates.append(Path(sys._MEIPASS) / "ui" / "index.html")

    # 2. Next to the running executable
    if getattr(sys, "frozen", False):
        candidates.append(Path(sys.executable).parent / "ui" / "index.html")

    # 3. Relative to this source file  (api/server.py → ../ → ui/)
    candidates.append(Path(__file__).resolve().parent.parent / "ui" / "index.html")

    # 4. CWD fallback
    candidates.append(Path(os.getcwd()) / "ui" / "index.html")

    for path in candidates:
        try:
            if path.exists():
                logger.info(f"UI found: {path}")
                return path.read_text(encoding="utf-8")
        except Exception as exc:
            logger.debug(f"UI candidate failed {path}: {exc}")

    # Last resort: inline minimal fallback page so users see something useful
    logger.error("ui/index.html not found in any search path. Serving inline fallback.")
    logger.error(f"Searched: {[str(c) for c in candidates]}")
    return _inline_fallback_html()


def _inline_fallback_html() -> str:
    """
    Minimal HTML page shown when ui/index.html cannot be located.
    Gives the user enough info to diagnose the problem.
    """
    searched = []
    if getattr(sys, "frozen", False) and hasattr(sys, "_MEIPASS"):
        searched.append(str(Path(sys._MEIPASS) / "ui"))
    if getattr(sys, "frozen", False):
        searched.append(str(Path(sys.executable).parent / "ui"))
    searched.append(str(Path(__file__).resolve().parent.parent / "ui"))
    searched.append(str(Path(os.getcwd()) / "ui"))

    paths_html = "".join(f"<li><code>{p}</code></li>" for p in searched)

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8"/>
<meta name="viewport" content="width=device-width,initial-scale=1.0"/>
<title>SysAssist — Loading</title>
<style>
  body {{ background:#080a0e; color:#e8eaf0; font-family:'Segoe UI',sans-serif;
          display:flex; align-items:center; justify-content:center; height:100vh;
          margin:0; }}
  .card {{ background:#111418; border:1px solid rgba(255,255,255,.08); border-radius:12px;
           padding:40px 48px; max-width:600px; text-align:center; }}
  h1 {{ color:#00d4a0; font-size:20px; margin-bottom:12px; }}
  p  {{ color:#7a8094; font-size:13px; line-height:1.7; margin:8px 0; }}
  ul {{ text-align:left; color:#7a8094; font-size:12px; margin:12px 0; padding-left:24px; }}
  code {{ background:rgba(255,255,255,.06); padding:2px 6px; border-radius:4px;
           color:#38bdf8; font-family:'Consolas',monospace; font-size:12px; }}
  .ok {{ color:#00d4a0; font-weight:600; }}
  .btn {{ display:inline-block; margin-top:20px; padding:10px 24px;
           background:#00d4a0; color:#080a0e; font-weight:700; border-radius:8px;
           cursor:pointer; font-size:13px; border:none; }}
  .btn:hover {{ background:#00eab2; }}
</style>
</head>
<body>
<div class="card">
  <h1>⚡ SysAssist AI</h1>
  <p class="ok">✓ Backend is running on port {settings.port}</p>
  <p>The UI file <code>ui/index.html</code> could not be found.<br/>
     This usually means the build did not bundle the UI correctly.</p>
  <p>Searched locations:</p>
  <ul>{paths_html}</ul>
  <p><strong>Fix:</strong> Ensure <code>ui/index.html</code> exists next to the executable,<br/>
     or rebuild with <code>build-electron.bat</code></p>
  <p>API is fully operational:<br/>
     <a href="/api/metrics" style="color:#38bdf8">/api/metrics</a> &nbsp;
     <a href="/api/processes" style="color:#38bdf8">/api/processes</a> &nbsp;
     <a href="/api/ai/status" style="color:#38bdf8">/api/ai/status</a></p>
  <button class="btn" onclick="location.reload()">↻ Retry</button>
</div>
</body>
</html>"""


# ── Models ─────────────────────────────────────────────────────────────────

class ChatMessage(BaseModel):
    message: str
    history: Optional[list] = None

class KillProcessReq(BaseModel):
    pid: int
    force: bool = False

class RestartServiceReq(BaseModel):
    service_name: str

class ThresholdUpdate(BaseModel):
    metric: str
    level:  str    # "warning" or "critical"
    value:  float


# ── WebSocket manager ──────────────────────────────────────────────────────

class ConnectionManager:
    def __init__(self):
        self.active: list[WebSocket] = []

    async def connect(self, ws: WebSocket):
        await ws.accept()
        self.active.append(ws)

    def disconnect(self, ws: WebSocket):
        if ws in self.active:
            self.active.remove(ws)

    async def broadcast(self, data: dict):
        dead = []
        for ws in self.active:
            try:
                await ws.send_json(data)
            except Exception:
                dead.append(ws)
        for ws in dead:
            self.disconnect(ws)


# ── App factory ────────────────────────────────────────────────────────────

def create_app(scheduler: MetricScheduler, alert_manager: AlertManager) -> FastAPI:

    app = FastAPI(title="SysAssist", version="2.3.0", docs_url=None, redoc_url=None)

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"], allow_methods=["*"], allow_headers=["*"],
    )

    ws_mgr      = ConnectionManager()
    ai_engine   = AIDiagnosticEngine()
    remediation = RemediationEngine()

    # ── Bind live state + tools into the agentic engine ───────────────────
    try:
        from agent.battery  import BatteryCollector
        from agent.security import SecurityCollector
        _battery  = BatteryCollector()
        _security = SecurityCollector()
    except Exception:
        _battery  = None
        _security = None

    # WebSocket broadcast so agents can push proposals to the UI
    async def _agent_broadcast(data: dict):
        await ws_mgr.broadcast(data)

    ai_engine.bind_tools(
        state              = scheduler.state,
        remediation_engine = remediation,
        battery_collector  = _battery,
        security_collector = _security,
        broadcast_fn       = _agent_broadcast,
    )

    # Wire alert callbacks
    def _on_violations(context, critical_violations):
        if not critical_violations:
            return
        alert_manager.process_violations(critical_violations, diagnosis="")

    scheduler.state.alert_callbacks.append(_on_violations)

    # ── UI serving ─────────────────────────────────────────────────────────
    _ui_html: dict = {"content": None}

    def _get_ui() -> str:
        if _ui_html["content"] is None:
            _ui_html["content"] = _resolve_ui_html()
        return _ui_html["content"]

    @app.get("/", response_class=HTMLResponse)
    async def serve_ui():
        return HTMLResponse(_get_ui())

    @app.get("/chat", response_class=HTMLResponse)
    async def serve_chat():
        return HTMLResponse(_get_ui())

    # Hot-reload UI in dev mode (re-reads file on each request if needed)
    @app.post("/api/ui/reload")
    async def reload_ui():
        _ui_html["content"] = None   # clear cache → next request re-reads disk
        return {"success": True, "message": "UI cache cleared"}

    # ── Metrics ────────────────────────────────────────────────────────────
    @app.get("/api/metrics")
    async def get_metrics():
        m = scheduler.state.latest_metrics
        h = scheduler.state.latest_health
        if m is None:
            return JSONResponse({"error": "Metrics not yet available"}, status_code=503)

        return {
            "metrics": {
                "cpu_percent":        round(m.cpu_percent, 1),
                "cpu_freq_mhz":       round(m.cpu_freq_mhz, 0),
                "cpu_core_count":     m.cpu_core_count,
                "memory_percent":     round(m.memory_percent, 1),
                "memory_used_gb":     round(m.memory_used_gb, 2),
                "memory_total_gb":    round(m.memory_total_gb, 2),
                "disk_percent":       round(m.disk_percent, 1),
                "disk_used_gb":       round(m.disk_used_gb, 2),
                "disk_total_gb":      round(m.disk_total_gb, 2),
                "disk_read_mbps":     round(m.disk_read_mbps, 2),
                "disk_write_mbps":    round(m.disk_write_mbps, 2),
                "gpu_percent":        round(m.gpu_percent, 1),
                "gpu_memory_percent": round(m.gpu_memory_percent, 1),
                "gpu_name":           m.gpu_name,
                "network_sent_mbps":  round(m.network_sent_mbps, 2),
                "network_recv_mbps":  round(m.network_recv_mbps, 2),
                "network_latency_ms": round(m.network_latency_ms, 1),
            },
            "health": {
                "score":     h.score     if h else 0,
                "grade":     h.grade     if h else "?",
                "label":     h.label     if h else "Loading",
                "color":     h.color     if h else "#888",
                "summary":   h.summary   if h else "",
                "breakdown": h.breakdown if h else {},
            },
            "history":    scheduler.state.metric_history[-30:],
            "violations": [
                {"metric": v.metric, "severity": v.severity.value,
                 "message": v.message, "current_value": v.current_value}
                for v in scheduler.state.active_violations
            ],
            "paused": scheduler.state.paused,
        }

    @app.get("/api/processes")
    async def get_processes():
        procs = scheduler.state.latest_processes
        total_cpu = sum(p.cpu_percent for p in procs)
        total_mem = sum(p.memory_mb   for p in procs)
        return {
            "processes": [
                {"pid": p.pid, "name": p.name, "cpu_percent": p.cpu_percent,
                 "memory_mb": p.memory_mb, "status": p.status,
                 "username": p.username, "threads": p.threads,
                 "cpu_bar_color": p.cpu_bar_color}
                for p in procs
            ],
            "summary": {
                "count":           len(procs),
                "total_cpu":       round(total_cpu, 1),
                "total_memory_gb": round(total_mem / 1024, 2),
            },
        }

    @app.get("/api/system")
    async def get_system():
        return scheduler.state.system_info

    # ── Alerts ─────────────────────────────────────────────────────────────
    @app.get("/api/alerts")
    async def get_alerts():
        alerts = alert_manager.get_all_alerts(50)
        return {
            "alerts": [
                {"id": a.id, "metric": a.metric, "severity": a.severity,
                 "message": a.message, "timestamp": a.timestamp,
                 "acknowledged": a.acknowledged, "resolved": a.resolved,
                 "diagnosis": a.diagnosis}
                for a in alerts
            ],
            "summary":       alert_manager.get_alert_summary(),
            "active_states": alert_state_manager.get_active_states(),
        }

    @app.post("/api/alerts/{alert_id}/acknowledge")
    async def ack_alert(alert_id: str):
        return {"success": alert_manager.acknowledge(alert_id)}

    @app.post("/api/alerts/{alert_id}/resolve")
    async def resolve_alert(alert_id: str):
        return {"success": alert_manager.resolve(alert_id)}

    @app.get("/api/alerts/recent")
    async def get_recent_alerts(hours: float = 1.0):
        """Return all alerts from the last N hours, sorted by severity then timestamp."""
        import time as _time
        since = _time.time() - hours * 3600
        all_alerts = alert_manager.get_all_alerts(500)
        recent = [a for a in all_alerts if a.timestamp >= since]
        sev_order = {"critical": 0, "warning": 1, "info": 2}
        recent.sort(key=lambda a: (sev_order.get(a.severity, 9), -a.timestamp))
        import re as _re3
        def _source(a):
            m = _re3.search(r"'([^']+)'", a.message)
            return m.group(1) if m else a.metric.capitalize()
        return {
            "alerts": [
                {
                    "id":           a.id,
                    "type":         a.metric,
                    "severity":     a.severity,
                    "message":      a.message,
                    "timestamp":    a.timestamp,
                    "timestamp_fmt": _time.strftime("%H:%M:%S", _time.localtime(a.timestamp)),
                    "date_fmt":     _time.strftime("%Y-%m-%d %H:%M", _time.localtime(a.timestamp)),
                    "acknowledged": a.acknowledged,
                    "resolved":     a.resolved,
                    "source":       _source(a),
                    "diagnosis":    (a.diagnosis or "")[:150],
                }
                for a in recent
            ],
            "count":    len(recent),
            "period_h": hours,
        }

    @app.post("/api/alerts/chat-opened")
    async def chat_opened():
        alert_state_manager.on_chat_opened()
        return {"success": True}

    @app.post("/api/alerts/chat-closed")
    async def chat_closed():
        alert_state_manager.on_chat_closed()
        return {"success": True}

    @app.get("/api/alerts/{alert_id}/diagnosis")
    async def get_diagnosis(alert_id: str):
        alerts = alert_manager.get_all_alerts(100)
        for a in alerts:
            if a.id == alert_id:
                diag = a.diagnosis
                if not diag:
                    ctx = scheduler.state.latest_context
                    if ctx:
                        try:
                            diag = await ai_engine.analyze_context_async(ctx)
                        except Exception:
                            diag = "AI diagnosis unavailable."
                return {
                    "alert_id": alert_id, "metric": a.metric,
                    "severity": a.severity, "message": a.message,
                    "diagnosis": diag, "timestamp": a.timestamp,
                }
        return JSONResponse({"error": "Alert not found"}, status_code=404)

    # ── Monitoring control ─────────────────────────────────────────────────
    @app.post("/api/monitoring/pause")
    async def pause_monitoring():
        scheduler.state.paused = True
        return {"success": True, "paused": True}

    @app.post("/api/monitoring/resume")
    async def resume_monitoring():
        scheduler.state.paused = False
        return {"success": True, "paused": False}

    @app.get("/api/monitoring/status")
    async def monitoring_status():
        return {
            "paused":     scheduler.state.paused,
            "uptime_s":   round(time.time() - scheduler.state.start_time),
            "violations": len(scheduler.state.active_violations),
        }

    # ── Settings ───────────────────────────────────────────────────────────
    @app.get("/api/settings")
    async def get_settings():
        return settings.to_dict()

    @app.post("/api/settings/threshold")
    async def update_threshold(req: ThresholdUpdate):
        try:
            settings.update("thresholds", req.metric, req.level, value=req.value)
            return {"success": True, "message": f"Updated {req.metric}.{req.level} = {req.value}"}
        except Exception as exc:
            return {"success": False, "message": str(exc)}

    # ── Remediation ────────────────────────────────────────────────────────
    @app.post("/api/remediation/kill")
    async def kill_process(req: KillProcessReq):
        r = remediation.kill_process(req.pid, req.force)
        return {"success": r.success, "message": r.message, "details": r.details}

    @app.post("/api/remediation/clear-temp")
    async def clear_temp():
        r = await asyncio.get_event_loop().run_in_executor(
            None, remediation.clear_temp_files
        )
        return {"success": r.success, "message": r.message, "details": r.details}

    @app.post("/api/remediation/reset-network")
    async def reset_network():
        r = await asyncio.get_event_loop().run_in_executor(
            None, remediation.reset_network
        )
        return {"success": r.success, "message": r.message, "details": r.details}

    @app.post("/api/remediation/optimize-memory")
    async def optimize_memory():
        r = remediation.optimize_memory()
        return {"success": r.success, "message": r.message}

    @app.post("/api/remediation/restart-service")
    async def restart_service(req: RestartServiceReq):
        r = await asyncio.get_event_loop().run_in_executor(
            None, remediation.restart_service, req.service_name
        )
        return {"success": r.success, "message": r.message, "details": r.details}

    @app.get("/api/remediation/actions")
    async def list_actions():
        return remediation.get_available_actions()

    # ── Chat / AI ──────────────────────────────────────────────────────────
    @app.post("/api/chat")
    async def chat(msg: ChatMessage):
        """
        Agentic chat. The agent calls live tools before answering.
        For fix requests: agent proposes → user approves → agent executes.
        Approval pushed via WebSocket (type: agent_proposal).
        User calls POST /api/agents/approve/{session_id} to confirm.
        """
        context = scheduler.state.latest_context
        try:
            response = await ai_engine.chat_async(msg.message, context, msg.history)
            return {"response": response, "timestamp": time.time()}
        except Exception as exc:
            logger.error(f"Chat error: {exc}")
            return {"response": f"Error: {exc}", "timestamp": time.time()}

    @app.post("/api/analyze")
    async def analyze():
        """
        Agentic full-system analysis (Analyze Now button).
        Diagnostician uses live tools to investigate — not just a context snapshot.
        Read-only: no approval required.
        """
        ctx = scheduler.state.latest_context
        if ctx is None:
            return {"analysis": "System is initialising — please wait a few seconds.",
                    "timestamp": time.time()}
        try:
            analysis = await ai_engine.analyze_context_async(ctx)
            return {"analysis": analysis, "timestamp": time.time()}
        except Exception as exc:
            return {"analysis": f"Analysis failed: {exc}", "timestamp": time.time()}

    # ── Agent approval endpoints ───────────────────────────────────────────

    @app.post("/api/agents/approve/{session_id}")
    async def approve_proposal(session_id: str):
        """
        User approves a remediation proposal.
        For monitoring-triggered proposals: the waiting pipeline resumes and executes.
        For chat proposals: sends APPROVED back into the conversation.
        Result arrives via WebSocket (type: agent_complete).
        """
        ok = ai_engine.approve_proposal(session_id)
        if ok:
            # For chat-triggered proposals, also run the execution
            proposal = ai_engine._proposals.get(session_id)
            if proposal and proposal.trigger == "chat":
                asyncio.create_task(
                    _execute_approved_plan(ai_engine, proposal, session_id)
                )
        return {"success": ok, "session_id": session_id,
                "message": "Executing remediation…" if ok else "Session not found or already resolved"}

    @app.post("/api/agents/dismiss/{session_id}")
    async def dismiss_proposal(session_id: str):
        """User dismisses a remediation proposal. No action is taken."""
        ok = ai_engine.dismiss_proposal(session_id)
        return {"success": ok, "session_id": session_id,
                "message": "Dismissed — no action taken" if ok else "Session not found"}

    @app.get("/api/agents/proposals")
    async def get_proposals():
        """All proposals currently awaiting user approval."""
        return {"proposals": ai_engine.get_pending_proposals()}


    @app.get("/api/ai/status")
    async def ai_status():
        return ai_engine.get_status()

    @app.post("/api/ai/reload")
    async def ai_reload():
        ai_engine._available  = None
        ai_engine._last_check = 0.0
        available = ai_engine.available
        return {
            "success":  available,
            "provider": ai_engine._provider,
            "model":    ai_engine.model_name,
            "message":  f"Connected: {ai_engine.model_name}" if available
                        else "AI not available — check settings",
        }

    @app.get("/api/ai/models/local")
    async def list_local_models():
        """Return all Ollama models installed on this device."""
        models = await asyncio.get_event_loop().run_in_executor(
            None, ai_engine.get_ollama_models
        )
        return {
            "ollama_running": ai_engine._ollama.is_running(),
            "models":         models,
            "current":        ai_engine._ollama_model,
        }

    class SetLocalModelReq(BaseModel):
        model: str

    @app.post("/api/ai/set/local")
    async def set_local_model(req: SetLocalModelReq):
        result = await asyncio.get_event_loop().run_in_executor(
            None, lambda: ai_engine.set_local_model(req.model)
        )
        return result

    class SetAPIKeyReq(BaseModel):
        provider: str
        api_key:  str
        model:    str = ""

    @app.post("/api/ai/set/apikey")
    async def set_api_key(req: SetAPIKeyReq):
        """
        This application is LOCAL ONLY.
        Cloud providers (Claude, OpenAI, NVIDIA) are not supported.
        Use /api/ai/set/local to configure an Ollama model instead.
        """
        return {
            "success": False,
            "message": "This is a local-only application. Cloud AI providers are disabled. "
                       "Use Ollama (Settings → Local LLM) to configure a local model.",
        }


    class SetGenOptionsReq(BaseModel):
        temperature:    Optional[float] = None
        num_predict:    Optional[int]   = None
        top_p:          Optional[float] = None
        repeat_penalty: Optional[float] = None

    @app.post("/api/ai/options")
    async def set_gen_options(req: SetGenOptionsReq):
        opts = {k: v for k, v in req.dict().items() if v is not None}
        ai_engine.set_generation_options(opts)
        return {"success": True, "options": opts}

    @app.get("/api/device/info")
    async def device_info():
        """Return device hardware details for the topbar widget."""
        import platform
        import psutil
        info = scheduler.state.system_info or {}
        # Enrich with live data
        cpu_freq = psutil.cpu_freq()
        return {
            "hostname":      info.get("hostname", platform.node()),
            "os":            info.get("os", platform.system()),
            "os_version":    info.get("os_version", platform.version())[:60],
            "cpu_name":      info.get("processor", platform.processor())[:40] or "CPU",
            "cpu_cores":     info.get("cpu_cores", psutil.cpu_count(logical=False)),
            "cpu_threads":   info.get("cpu_threads", psutil.cpu_count(logical=True)),
            "cpu_freq_mhz":  round(cpu_freq.current) if cpu_freq else 0,
            "total_ram_gb":  info.get("total_memory_gb", round(psutil.virtual_memory().total / 1e9, 1)),
            "python_version":info.get("python_version", platform.python_version()),
            "uptime_seconds":round(time.time() - scheduler.state.start_time),
        }


    # ── Application Insights ─────────────────────────────────────────────
    @app.get("/api/apps")
    async def get_apps():
        from agent.app_collector import app_collector
        procs = None
        try:
            raw = scheduler.state.latest_processes or []
            procs = raw if raw else None
        except Exception:
            pass
        await asyncio.get_event_loop().run_in_executor(None, lambda: app_collector.collect(procs))
        apps = app_collector.get_all_apps()
        sys_hist = {}
        try:
            buf = scheduler.state.buffers
            def _bl(b): return [round(v,2) for v in list(b.data)]
            sys_hist = {"cpu":_bl(buf.cpu),"memory":_bl(buf.memory),
                        "disk":_bl(buf.disk_percent),"network":_bl(buf.network_sent),"gpu":_bl(buf.gpu)}
        except Exception:
            pass
        return {"apps":apps,"device_class":app_collector.device_class,
                "thresholds":app_collector.thresholds,"sys_history":sys_hist}

    @app.get("/api/battery")
    async def get_battery():
        from agent.battery import battery_collector
        return await asyncio.get_event_loop().run_in_executor(None, battery_collector.collect)

    @app.get("/api/security")
    async def get_security():
        from agent.security import security_collector
        return await asyncio.get_event_loop().run_in_executor(None, security_collector.collect)

    @app.post("/api/analyze/events")
    async def analyze_events():
        ctx = scheduler.state.latest_context
        ctx_text = ctx.to_prompt_text() if ctx else "No diagnostic context available."
        raw_metrics = scheduler.state.latest_metrics
        raw_vios    = list(scheduler.state.active_violations or [])
        metrics_dict = {}
        if raw_metrics:
            try:
                metrics_dict = {
                    "cpu":  raw_metrics.cpu_percent,
                    "mem":  raw_metrics.memory_percent,
                    "disk": raw_metrics.disk_percent,
                }
            except Exception:
                pass
        vios_list = []
        for v in raw_vios:
            try:
                vios_list.append({
                    "metric":   getattr(v,"metric",  v.get("metric","")  if isinstance(v,dict) else ""),
                    "severity": getattr(v,"severity",v.get("severity","") if isinstance(v,dict) else ""),
                    "message":  getattr(v,"message", v.get("message","")  if isinstance(v,dict) else ""),
                })
            except Exception:
                pass
        prompt = (
            "Perform structured endpoint diagnostic analysis. "
            "Respond ONLY in valid JSON with no markdown fences.\n"
            '{"root_cause":"...","warning_level":"critical|warning|info|healthy",'
            '"warning_score":0,"impacted_components":[],"recommended_actions":[],'
            '"preventive_suggestions":[],"summary":"..."}\n\n'
            "System data:\n" + ctx_text
        )
        try:
            ctx = scheduler.state.latest_context
            if ctx:
                raw = await ai_engine.analyze_context_async(ctx)
            else:
                raw = None
            import re as _re, json as _json
            if raw:
                match = _re.search(r'\{.*\}', raw, _re.DOTALL)
                if match:
                    return _json.loads(match.group())
            raise ValueError("no json")
        except Exception:
            wl = "critical" if any(v.get("severity")=="critical" for v in vios_list) \
                 else "warning" if vios_list else "healthy"
            return {
                "root_cause":  vios_list[0].get("message","System normal") if vios_list else "All metrics normal",
                "warning_level": wl,
                "warning_score": min(100, len(vios_list)*25),
                "impacted_components": [v.get("metric","system").upper() for v in vios_list] or ["None"],
                "recommended_actions": ["Review Processes tab for high-CPU items","Clear temp files if disk >75%"],
                "preventive_suggestions": ["Enable threshold alerts","Schedule weekly maintenance"],
                "summary": f"CPU {metrics_dict.get('cpu',0)}% · MEM {metrics_dict.get('mem',0)}%",
            }

    class ServiceNowReq(BaseModel):
        instance: str; username: str; password: str; alert_id: Optional[str] = None

    @app.post("/api/servicenow/ticket")
    async def create_sn_ticket(req: ServiceNowReq):
        """Create ServiceNow incident. Returns {success, number, url} or {success:false, error, detail}."""
        from integrations.servicenow import ServiceNowClient, build_incident_fields
        alert_info = {}; ai_text = "Endpoint issue detected by Cognix EUD AI Assist."
        try:
            if req.alert_id:
                all_a = alert_manager.get_all_alerts(200)
                hit   = next((a for a in all_a if a.id == req.alert_id), None)
                if hit:
                    alert_info = {"metric": hit.metric, "severity": hit.severity, "message": hit.message}
                    if hit.diagnosis: ai_text = hit.diagnosis
        except Exception as _ex:
            logger.debug(f"SN alert lookup: {_ex}")
        device_info = {
            "hostname":     _platform.node(), "os":         _platform.system(),
            "os_version":   _platform.version()[:60],       "cpu_name": _platform.processor()[:40] or "CPU",
            "cpu_cores":    psutil.cpu_count(logical=False), "total_ram_gb": round(psutil.virtual_memory().total/1e9,1),
        }
        fields = build_incident_fields(ai_text, device_info, alert_info)
        client = ServiceNowClient(req.instance, req.username, req.password)
        try:
            result = await asyncio.get_event_loop().run_in_executor(None, lambda: client.create_incident(fields))
            return result
        except Exception as ex:
            return {"success": False, "error": str(ex),
                    "detail": "Verify: instance URL (https://xxx.service-now.com), credentials, and user has itil role."}

    # ── Event Log Reader ─────────────────────────────────────────────────
    @app.get("/api/events/logs")
    async def get_event_logs(limit: int = 50):
        """Read Windows Event Log entries (or syslog on Linux) and return prioritised issues."""
        def _read_logs():
            events = []
            system = _platform.system()
            if system == "Windows":
                try:
                    import subprocess as _sp, json as _j
                    ps_cmd = (
                        "Get-WinEvent -LogName System,Application -MaxEvents 200 "
                        "-ErrorAction SilentlyContinue | "
                        "Where-Object {$_.Level -le 3} | "  # 1=Critical 2=Error 3=Warning
                        "Select-Object TimeCreated,Id,LevelDisplayName,ProviderName,Message | "
                        "ConvertTo-Json -Compress 2>$null"
                    )
                    r = _sp.run(
                        ["powershell", "-NonInteractive", "-NoProfile", "-Command", ps_cmd],
                        capture_output=True, text=True, timeout=12
                    )
                    if r.stdout.strip():
                        raw = _j.loads(r.stdout.strip())
                        if isinstance(raw, dict): raw = [raw]
                        for ev in raw[:200]:
                            msg = (ev.get("Message") or "")[:200].replace("\r\n", " ")
                            tc  = ev.get("TimeCreated", {})
                            ts  = str(tc) if not isinstance(tc, str) else tc
                            lvl = ev.get("LevelDisplayName", "Warning")
                            events.append({
                                "time":     ts[:19] if len(str(ts)) >= 19 else str(ts),
                                "level":    lvl,
                                "source":   ev.get("ProviderName", "Unknown"),
                                "event_id": ev.get("Id", 0),
                                "message":  msg,
                                "priority": 1 if lvl == "Critical" else 2 if lvl == "Error" else 3,
                            })
                except Exception as ex:
                    # Fallback: use wevtutil
                    try:
                        import subprocess as _sp
                        r = _sp.run(
                            ["wevtutil", "qe", "System", "/c:50", "/f:text", "/rd:true"],
                            capture_output=True, text=True, timeout=10
                        )
                        for block in r.stdout.split("\n\n"):
                            if "Level:" in block and ("Error" in block or "Critical" in block or "Warning" in block):
                                lines = {l.split(":",1)[0].strip(): l.split(":",1)[1].strip()
                                         for l in block.splitlines() if ":" in l}
                                lvl = lines.get("Level","Warning")
                                events.append({
                                    "time":     lines.get("Date", ""),
                                    "level":    lvl,
                                    "source":   lines.get("Source", "System"),
                                    "event_id": lines.get("EventID", "0"),
                                    "message":  lines.get("Description","")[:200],
                                    "priority": 1 if "Critical" in lvl else 2 if "Error" in lvl else 3,
                                })
                    except Exception:
                        pass
            else:
                # Linux: read syslog / journalctl
                try:
                    import subprocess as _sp
                    r = _sp.run(
                        ["journalctl", "-p", "warning", "-n", "100", "--no-pager", "-o", "short-iso"],
                        capture_output=True, text=True, timeout=8
                    )
                    for line in r.stdout.splitlines()[-100:]:
                        if "error" in line.lower() or "warn" in line.lower() or "crit" in line.lower():
                            parts = line.split(" ", 4)
                            lvl = "Error" if "error" in line.lower() else "Warning"
                            events.append({
                                "time":     parts[0] if parts else "",
                                "level":    lvl,
                                "source":   parts[4][:40] if len(parts) > 4 else "system",
                                "event_id": 0,
                                "message":  line[:200],
                                "priority": 2 if lvl == "Error" else 3,
                            })
                except Exception:
                    pass
            # Sort: critical first, then error, then warning; most recent first within each
            events.sort(key=lambda e: (e["priority"], e.get("time","")), reverse=False)
            return events[:limit]
        events = await asyncio.get_event_loop().run_in_executor(None, _read_logs)
        return {"events": events, "count": len(events)}

    # ── Per-tab structured analysis ───────────────────────────────────────
    @app.post("/api/analyze/tab/{tab}")
    async def analyze_tab(tab: str):
        """
        Context-aware, tab-scoped AI analysis.
        Each tab receives ONLY its own domain data — zero cross-tab leakage.
        Falls back to deterministic rule-based output when AI is unavailable.
        """
        import re as _re2, json as _j2, subprocess as _sp2

        def _sv(obj, attr, default=None):
            if obj is None: return default
            return getattr(obj, attr, default) if not isinstance(obj, dict) else obj.get(attr, default)

        raw_metrics = scheduler.state.latest_metrics
        raw_vios    = list(scheduler.state.active_violations or [])

        def _v(o): return {"metric":_sv(o,"metric","?"),"severity":_sv(o,"severity","?"),"message":_sv(o,"message","?")}
        vios_list = [_v(v) for v in raw_vios]
        metrics_dict = {}
        if raw_metrics:
            try: metrics_dict = {"cpu": raw_metrics.cpu_percent,"mem": raw_metrics.memory_percent,"disk": raw_metrics.disk_percent}
            except Exception: pass

        # ── Tab-scoped data gathering — strictly domain-isolated ──────────
        tab_context = ""
        system_prompt = ""
        tab_name = tab.title()

        if tab == "processes":
            procs = scheduler.state.latest_processes or []
            lines = []
            for p in procs[:15]:
                n   = _sv(p,"name","?"); cpu = _sv(p,"cpu_percent",0)
                mem = _sv(p,"memory_mb",0); sta = _sv(p,"status","?"); thr = _sv(p,"threads",1)
                lines.append(f"  {n:<24} CPU={cpu:5.1f}%  MEM={mem:7.0f}MB  status={sta}  threads={thr}")
            tab_context = "TOP PROCESSES (by CPU):\n" + "\n".join(lines)
            proc_vios = [v for v in vios_list if v["metric"] in ("cpu","memory","gpu")]
            if proc_vios:
                tab_context += "\n\nACTIVE THRESHOLD VIOLATIONS:\n" + "\n".join(
                    f"  [{v['severity'].upper()}] {v['message']}" for v in proc_vios[:5])
            system_prompt = (
                "You are a Windows endpoint performance engineer specialising in process analysis. "
                "Analyse ONLY the process data provided. Identify: CPU/memory hogs, "
                "hung/zombie processes (sleeping with high CPU), suspicious executable names, "
                "unusual thread counts, and which processes to terminate. "
                "Be specific — name each process. Do NOT discuss generic system health."
            )

        elif tab == "apps":
            from agent.app_collector import app_collector
            try:
                raw_apps = app_collector.get_all_apps()
                lines = []
                for a in sorted(raw_apps, key=lambda x: x.get("cpu",0), reverse=True)[:12]:
                    lines.append(
                        f"  {a.get('name','?'):<28} behavior={a.get('behavior','?'):<14} "
                        f"health={a.get('health_score',100):3d}/100  "
                        f"procs={a.get('proc_count',0):2d}")
                tab_context = "APPLICATION BEHAVIORAL ANALYSIS (no system metrics):\n" + "\n".join(lines)
            except Exception as ex:
                tab_context = f"App data unavailable: {ex}"
            system_prompt = (
                "You are an application performance specialist. "
                "Analyse ONLY the application behavioral data provided. "
                "Identify: apps with health <60 (likely crashed/hung), "
                "abnormal behavioral states (High Load, Elevated unexpectedly), "
                "usage pattern anomalies, app-level recommendations. "
                "Do NOT mention CPU%, Memory%, Disk%, Network — focus ONLY on "
                "application behaviour, crashes, hangs, and usage patterns."
            )

        elif tab == "battery":
            from agent.battery import battery_collector
            bat = await asyncio.get_event_loop().run_in_executor(None, battery_collector.collect)
            if not bat.get("available"):
                tab_context = f"Battery not available: {bat.get('reason','no battery detected')}"
            else:
                drainers = ", ".join(a.get("name","?") for a in bat.get("top_draining_apps",[])[:4]) or "none identified"
                tab_context = (
                    f"BATTERY TELEMETRY:\n"
                    f"  Level:        {bat.get('percent','?')}%\n"
                    f"  Status:       {'Plugged in/charging' if bat.get('plugged') else 'On battery power (discharging)'}\n"
                    f"  Health score: {bat.get('health_score','?')}/100 — {bat.get('health_label','?')}\n"
                    f"  Drain rate:   {bat.get('drain_rate_per_hr','?')}%/hr\n"
                    f"  Est. runtime: {bat.get('hours_remaining','?')} hrs remaining\n"
                    f"  Top drainers: {drainers}\n"
                    f"  Current tips: {'; '.join(bat.get('suggestions',[])[:2])}"
                )
            system_prompt = (
                "You are a battery health and power management specialist. "
                "Analyse ONLY the battery telemetry provided. "
                "Identify: abnormal drain rates, battery wear indicators, "
                "charging recommendations, specific apps causing drain, "
                "actionable Windows power settings to change. "
                "Be concrete — specify exact settings paths or commands."
            )

        elif tab == "security":
            from agent.security import security_collector
            sec = await asyncio.get_event_loop().run_in_executor(None, security_collector.collect)
            av  = sec.get("antivirus",{}); bl = sec.get("bitlocker",{}); fw = sec.get("firewall",{})
            sp  = sec.get("suspicious_processes",[]); na = sec.get("network_anomalies",[])
            tab_context = (
                f"SECURITY COMPLIANCE DATA:\n"
                f"  Score:      {sec.get('compliance_score','?')}/100  risk={sec.get('risk_level','?')}\n"
                f"  Antivirus:  {av.get('status','?')} — {av.get('details','')}\n"
                f"  BitLocker:  {bl.get('status','?')} — {bl.get('details','')}\n"
                f"  Firewall:   {fw.get('status','?')} — {fw.get('details','')}\n"
                f"  Suspicious: {len(sp)} procs — {', '.join(p.get('name','?') for p in sp[:5]) or 'none'}\n"
                f"  Net anomalies: {len(na)} — {', '.join(a.get('type','?') for a in na[:3]) or 'none'}\n"
                f"  Active issues: {', '.join(sec.get('issues',[]) or ['none'])}"
            )
            system_prompt = (
                "You are a Windows security compliance officer (ISO 27001 / NIST aware). "
                "Analyse ONLY the security compliance data provided. "
                "For each disabled control: explain the specific risk, give the exact "
                "PowerShell or Windows Settings path to fix it. "
                "Assess suspicious processes by name. Analyse network anomalies. "
                "Be actionable and compliance-regulation-aware."
            )

        elif tab == "events":
            try:
                if _platform.system() == "Windows":
                    ps_cmd = (
                        "Get-WinEvent -LogName System,Application -MaxEvents 80 "
                        "-ErrorAction SilentlyContinue | Where-Object {$_.Level -le 3} | "
                        "Select-Object TimeCreated,LevelDisplayName,ProviderName,Id,Message | "
                        "ConvertTo-Json -Compress 2>$null"
                    )
                    r2 = _sp2.run(["powershell","-NonInteractive","-NoProfile","-Command",ps_cmd],
                                  capture_output=True, text=True, timeout=12)
                    if r2.stdout.strip():
                        evts = _j2.loads(r2.stdout.strip())
                        if isinstance(evts, dict): evts = [evts]
                        lines2 = []
                        for ev in evts[:25]:
                            lvl = ev.get("LevelDisplayName","?"); src = ev.get("ProviderName","?")
                            eid = ev.get("Id","?")
                            msg = (ev.get("Message") or "")[:120].replace("\r\n"," ").replace("\n"," ")
                            lines2.append(f"  [{lvl}] ID={eid} {src}: {msg}")
                        tab_context = "WINDOWS EVENT LOG (Critical/Error/Warning, newest first):\n" + "\n".join(lines2)
                    else:
                        tab_context = "Windows Event Log: no errors or warnings in recent history."
                else:
                    tab_context = "Event log analysis: Linux/macOS (journalctl not supported in this version)."
            except Exception as ex:
                tab_context = f"Event log read error: {ex}"
            system_prompt = (
                "You are a Windows systems administrator specialising in event log forensics. "
                "Analyse ONLY the Windows Event Log entries provided. "
                "Identify: correlated error patterns, root causes behind repeated events, "
                "distinguish real issues from benign noise, map event IDs to known problems. "
                "Prioritise Critical > Error > Warning. Reference event IDs specifically."
            )

        else:  # sysov / fallback
            ctx = scheduler.state.latest_context
            tab_context = ctx.to_prompt_text() if ctx else "No context available."
            system_prompt = (
                "You are a senior Windows endpoint performance engineer. "
                "Provide a holistic system health assessment. "
                "Correlate violations with probable causes. Give a prioritised remediation list."
            )

        # ── AI prompt — system_prompt keeps context clean ─────────────────
        prompt = (
            f"{system_prompt}\n\n"
            "Respond ONLY in valid JSON with no markdown fences:\n"
            '{"root_cause":"...","warning_level":"critical|warning|info|healthy",'
            '"warning_score":0,"impacted_components":[],'
            '"recommended_actions":[],"preventive_suggestions":[],"summary":"..."}\n\n'
            f"=== {tab_name.upper()} DATA ===\n{tab_context}"
        )

        try:
            # Use agentic analyze for this tab's context
            ctx = scheduler.state.latest_context
            if ctx and ai_engine._model_client:
                raw = await ai_engine.analyze_context_async(ctx)
            elif ctx:
                raw = await asyncio.get_event_loop().run_in_executor(
                    None, lambda: ai_engine.analyze_context(ctx)
                )
            else:
                raw = None
            if raw:
                # Try to parse as JSON first; if not, return as plain analysis
                match = _re2.search(r"\{.*\}", raw, _re2.DOTALL)
                if match:
                    result = _j2.loads(match.group())
                    result["tab"] = tab
                    return result
                # Plain text analysis — wrap into expected shape
                return {
                    "tab":                   tab,
                    "root_cause":            raw,
                    "warning_level":         "info",
                    "warning_score":         0,
                    "impacted_components":   [],
                    "recommended_actions":   [],
                    "preventive_suggestions":[],
                    "summary":               raw[:200],
                }
        except Exception:
            pass

        # ── Deterministic rule-based fallback ─────────────────────────────
        crits = [v for v in vios_list if v.get("severity")=="critical"]
        warns = [v for v in vios_list if v.get("severity")=="warning"]
        wl  = "critical" if crits else "warning" if warns else "healthy"
        wsc = min(100, len(crits)*35 + len(warns)*15)
        return {
            "tab":                  tab,
            "root_cause":           crits[0]["message"] if crits else warns[0]["message"] if warns
                                    else f"{tab_name} operating within normal parameters",
            "warning_level":        wl,
            "warning_score":        wsc,
            "impacted_components":  list({v["metric"].upper() for v in crits+warns}) or [tab_name.upper()],
            "recommended_actions":  [
                f"Review {tab_name} panel for anomalies",
                "Connect an AI engine (Settings → AI Engine) for deeper analysis",
            ],
            "preventive_suggestions": ["Enable threshold alerts","Schedule regular maintenance"],
            "summary":              f"{tab_name} — {wl.title()} · "
                                    f"CPU {metrics_dict.get('cpu',0):.0f}% · "
                                    f"MEM {metrics_dict.get('mem',0):.0f}%",
        }

        # ── WebSocket ──────────────────────────────────────────────────────────
    @app.websocket("/ws")
    async def websocket_endpoint(ws: WebSocket):
        await ws_mgr.connect(ws)
        try:
            await _push_update(ws, scheduler, alert_manager)
            while True:
                try:
                    text    = await asyncio.wait_for(ws.receive_text(), timeout=5.0)
                    payload = json.loads(text)
                    if payload.get("type") == "ping":
                        await ws.send_json({"type": "pong"})
                    elif payload.get("type") == "chat_opened":
                        alert_state_manager.on_chat_opened(payload.get("alert_id", ""))
                except asyncio.TimeoutError:
                    await _push_update(ws, scheduler, alert_manager)
                except WebSocketDisconnect:
                    break
        except Exception:
            pass
        finally:
            ws_mgr.disconnect(ws)

    # ── Approved plan execution (chat-triggered proposals) ─────────────────
    # This runs as a task after the user approves a chat proposal.
    # It sends APPROVED back into a fresh Remediator agent to execute the plan.
    async def _execute_approved_plan(engine, proposal, session_id: str):
        try:
            if not engine._model_client:
                return
            from autogen_agentchat.agents import AssistantAgent
            from autogen_agentchat.teams import RoundRobinGroupChat
            from autogen_agentchat.conditions import TextMentionTermination, MaxMessageTermination
            from ai.diagnostic import _REM_PROMPT

            exec_agent = AssistantAgent(
                name           = "Remediator",
                model_client   = engine._model_client,
                system_message = _REM_PROMPT,
                tools          = engine._read_tools + engine._action_tools,
            )
            term = (TextMentionTermination("REMEDIATION_COMPLETE") |
                    MaxMessageTermination(14))
            team = RoundRobinGroupChat([exec_agent], termination_condition=term)
            result = await team.run(
                task=(
                    f"Previous plan:\n{proposal.plan_text}\n\n"
                    "APPROVED — execute the plan now using your action tools. "
                    "Report each result and end with REMEDIATION_COMPLETE."
                )
            )
            # Extract result text
            result_text = ""
            try:
                msgs = result.messages if hasattr(result, "messages") else []
                for msg in reversed(msgs):
                    c = getattr(msg, "content", None) or (msg.get("content") if isinstance(msg, dict) else None)
                    if isinstance(c, str) and len(c.strip()) > 15:
                        result_text = c.strip()
                        break
            except Exception:
                result_text = "Execution complete."

            await engine._push({
                "type":        "agent_complete",
                "session_id":  session_id,
                "result":      result_text,
            })
            engine._proposals.pop(session_id, None)
        except Exception as e:
            logger.error(f"[server] execute_approved_plan error: {e}")
            await engine._push({
                "type": "agent_error",
                "session_id": session_id,
                "message": str(e)[:300],
            })

    return app



async def _push_update(ws: WebSocket, scheduler: MetricScheduler, alert_manager: AlertManager):
    m = scheduler.state.latest_metrics
    h = scheduler.state.latest_health
    if m is None:
        await ws.send_json({"type": "waiting"})
        return

    await ws.send_json({
        "type":          "metrics",
        "cpu":           round(m.cpu_percent, 1),
        "memory":        round(m.memory_percent, 1),
        "disk":          round(m.disk_percent, 1),
        "gpu":           round(m.gpu_percent, 1),
        "net_sent":      round(m.network_sent_mbps, 2),
        "net_recv":      round(m.network_recv_mbps, 2),
        "net_latency":   round(m.network_latency_ms, 1),
        "disk_read":     round(m.disk_read_mbps, 2),
        "disk_write":    round(m.disk_write_mbps, 2),
        "health_score":  h.score if h else 0,
        "health_label":  h.label if h else "Loading",
        "health_color":  h.color if h else "#888",
        "violations":    len(scheduler.state.active_violations),
        "alert_summary": alert_manager.get_alert_summary(),
        "active_states": alert_state_manager.get_active_states(),
        "history":       scheduler.state.metric_history[-20:],
        "paused":        scheduler.state.paused,
        "timestamp":     m.timestamp,
    })
