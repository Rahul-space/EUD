"""
Cognix EUD AI Assist — Agentic AI Engine  v2.3.0
=================================================
ONE FILE. Fully agentic backend using AutoGen v0.4 (autogen-agentchat).
Local-only: Ollama via OpenAI-compatible endpoint.

What makes this AGENTIC (not Gen AI):
  Gen AI  → pack a context blob into a prompt → get text back. One shot.
  Agentic → agents DECIDE what tools to call, in what order, based on
            what they find. The LLM controls the investigation flow.

  Example: Agent calls get_system_metrics() → sees CPU 91% → decides to
  call get_top_processes() → finds chrome.exe → proposes kill_process().
  That chain of decisions based on real data = agency.

APPROVAL GATE — remediations NEVER execute without user approval:
  Agent builds plan → UI shows proposal → user clicks Approve/Dismiss
  Only on Approve does the agent call any action tool.

Install:
  pip install autogen-agentchat autogen-ext[openai]

Ollama:
  ollama pull llama3.2   # or phi3, mistral, etc.
"""

import asyncio
import json
import logging
import socket
import time
import threading
import urllib.request
import uuid
from typing import Optional, Callable

from agent.context_builder import DiagnosticContext

logger = logging.getLogger(__name__)

# ── AutoGen v0.4 ──────────────────────────────────────────────────────────
try:
    from autogen_agentchat.agents import AssistantAgent
    from autogen_agentchat.teams import RoundRobinGroupChat
    from autogen_agentchat.conditions import TextMentionTermination, MaxMessageTermination
    from autogen_ext.models.openai import OpenAIChatCompletionClient
    AUTOGEN_OK = True
    logger.info("[AgenticAI] autogen-agentchat loaded ✓")
except ImportError as _e:
    AUTOGEN_OK = False
    logger.warning(f"[AgenticAI] autogen-agentchat not installed: {_e}")
    logger.warning("  Run: pip install autogen-agentchat autogen-ext[openai]")

# ─────────────────────────────────────────────────────────────────
#  SYSTEM PROMPTS
# ─────────────────────────────────────────────────────────────────

_DIAG_PROMPT = """\
You are the Cognix Diagnostician — a Windows endpoint performance expert.
You have READ-ONLY tools. Investigate the live system and report findings.

Protocol (always in this order):
  1. get_system_metrics()        → baseline, always call this first
  2. get_active_violations()     → what thresholds are breached?
  3. CPU > 70%  → get_top_processes(10)
  4. MEM > 65%  → get_top_processes(10) + get_metric_trend('memory')
  5. DISK > 80% → check disk values from metrics
  6. get_predictions()           → is this getting worse?
  7. get_metric_history(12)      → spike or sustained problem?

Output format (required):
## Findings
**Current State:** [key metrics with real values from tools]
**Issues Detected:** [named processes, exact %s — or "None" if healthy]
**Root Cause:** [1-2 sentence explanation]
**Severity:** CRITICAL | WARNING | INFO | HEALTHY
**Urgency:** immediate / monitor / none

Be specific. Quote actual numbers. Name actual processes.
Do NOT suggest fixes — the Remediator handles that."""

_REM_PROMPT = """\
You are the Cognix Remediator. You receive Diagnostician findings and
build a safe remediation plan. You NEVER execute tools until the user
approves by typing APPROVED.

Available actions:
  kill_process(pid, reason)   — kill a process (get PID from get_top_processes)
  clear_temp_files()          — delete temp files, safe, recovers 1-10 GB
  reset_network()             — flush DNS + reset Winsock (~5s interruption)
  optimize_memory()           — empty Windows standby cache, always safe

PHASE 1 — after receiving findings, output ONLY this plan format:
---REMEDIATION PLAN---
Issue: [what was found]
Actions:
  1. [action_name(args)] — [reason] — Risk: LOW/MEDIUM
  2. ...
Waiting for your approval. Reply APPROVED to execute, or DISMISSED to cancel.
---END PLAN---

PHASE 2 — only after seeing APPROVED in the conversation:
  Call each tool in your plan. Report the result of each.
  Then output: REMEDIATION_COMPLETE

Safety rules — NEVER violate:
  ✗ Do NOT call action tools before seeing APPROVED
  ✗ Do NOT kill: lsass.exe, csrss.exe, winlogon.exe, smss.exe, services.exe
  ✗ Do NOT reset_network if latency < 100ms
  ✓ clear_temp_files and optimize_memory are always safe choices"""

_CHAT_PROMPT = """\
You are Cognix AI Assistant — a helpful Windows device expert embedded on this machine.
You have access to live system tools to answer questions accurately.

For ANY question about device health, performance, or status:
  → Call the relevant tools to get REAL current data before answering
  → Never guess or use cached context — always call tools for fresh data
  → Quote actual values: "CPU is at 67%" not "CPU seems high"

For fix/remediation requests:
  → Investigate with tools first
  → Tell the user what you found and what you propose to do
  → Ask: "Shall I proceed? Reply YES to execute."
  → Only execute action tools after user says YES/yes/approve/ok

Keep answers concise and specific. Use **bold** for important values.
Use `backticks` for process names and commands."""


# ─────────────────────────────────────────────────────────────────
#  PROTECTED PROCESSES — never kill these
# ─────────────────────────────────────────────────────────────────
_PROTECTED = {
    "lsass.exe", "csrss.exe", "winlogon.exe", "smss.exe",
    "services.exe", "wininit.exe", "system", "registry",
}


# ─────────────────────────────────────────────────────────────────
#  TOOL FACTORY  — closures over live state
# ─────────────────────────────────────────────────────────────────

def _make_tools(state, remediation, battery_col=None, security_col=None):
    """
    Returns (read_tools, action_tools) — plain Python closures over
    live AgentState and RemediationEngine.
    AutoGen v0.4 accepts plain functions directly as tools.
    """

    # ── READ TOOLS ───────────────────────────────────────────────

    def get_system_metrics() -> dict:
        """Get live CPU%, memory%, disk%, network latency ms, GPU%, and health score/label."""
        if not state or not state.latest_metrics:
            return {"error": "Metrics not yet collected, retry in a few seconds"}
        m, h = state.latest_metrics, state.latest_health
        return {
            "cpu_percent":        round(m.cpu_percent, 1),
            "cpu_cores":          m.cpu_core_count,
            "memory_percent":     round(m.memory_percent, 1),
            "memory_used_gb":     round(m.memory_used_gb, 1),
            "memory_total_gb":    round(m.memory_total_gb, 1),
            "disk_percent":       round(m.disk_percent, 1),
            "disk_used_gb":       round(m.disk_used_gb, 1),
            "disk_total_gb":      round(m.disk_total_gb, 1),
            "disk_read_mbps":     round(m.disk_read_mbps, 2),
            "disk_write_mbps":    round(m.disk_write_mbps, 2),
            "network_latency_ms": round(m.network_latency_ms),
            "network_sent_mbps":  round(m.network_sent_mbps, 2),
            "network_recv_mbps":  round(m.network_recv_mbps, 2),
            "gpu_percent":        round(m.gpu_percent, 1),
            "gpu_name":           m.gpu_name,
            "health_score":       h.score if h else 0,
            "health_label":       h.label if h else "Unknown",
            "as_of":              time.strftime("%H:%M:%S", time.localtime(m.timestamp)),
        }

    def get_top_processes(limit: int = 10) -> list:
        """Get top N processes by CPU%. limit: 1-20. Each entry has pid, name,
        cpu_percent, memory_mb, status. Use pid values for kill_process()."""
        if not state:
            return []
        return [
            {"pid": p.pid, "name": p.name,
             "cpu_percent": round(p.cpu_percent, 1),
             "memory_mb": round(p.memory_mb), "status": p.status}
            for p in state.latest_processes[:min(int(limit), 20)]
        ]

    def get_active_violations() -> list:
        """Get current threshold violations. Empty list means system is healthy.
        Each has: metric, severity (warning|critical), current_value, threshold,
        message, sustained_seconds."""
        if not state:
            return []
        return [
            {"metric": v.metric, "severity": v.severity.value,
             "current_value": round(v.current_value, 1),
             "threshold": v.threshold, "message": v.message,
             "sustained_seconds": round(v.sustained_seconds)}
            for v in state.active_violations
        ]

    def get_metric_trend(metric: str) -> dict:
        """Trend for a metric: 'cpu'|'memory'|'disk'|'network_latency'.
        Returns direction (rising/stable/falling), slope, change_rate.
        Use to tell if a high reading is a spike or a growing problem."""
        valid = ("cpu", "memory", "disk", "network_latency")
        if metric not in valid:
            return {"error": f"metric must be one of: {', '.join(valid)}"}
        if not state or not state.latest_context:
            return {"metric": metric, "direction": "unknown"}
        t = state.latest_context.trends.get(metric)
        if not t:
            return {"metric": metric, "direction": "stable", "note": "Not enough data yet"}
        return {"metric": metric, "direction": t.get("direction", "stable"),
                "slope": round(t.get("slope", 0), 3),
                "change_rate_pct": round(t.get("change_rate", 0), 2)}

    def get_predictions() -> list:
        """Predictive forecasts for CPU/memory/disk.
        Returns time_to_critical_hours and predicted values at 1h/24h.
        Empty list = no concerning trends."""
        if not state or not state.latest_context:
            return []
        return state.latest_context.predictions or []

    def get_metric_history(points: int = 12) -> dict:
        """Last N snapshots (~5s each, max 60 = ~5 min).
        Useful for spotting spikes vs sustained problems."""
        if not state:
            return {"error": "State unavailable"}
        pts = state.metric_history[-min(int(points), 60):]
        return {"count": len(pts), "interval_seconds": 5, "history": pts}

    def get_system_info() -> dict:
        """Static device info: hostname, OS, CPU model, total RAM, architecture."""
        return state.system_info if state else {}

    def get_battery_status() -> dict:
        """Battery level%, plugged_in, time_left_minutes, health score.
        Returns error for desktops/VMs without a battery."""
        try:
            import psutil
            b = psutil.sensors_battery()
            if not b:
                return {"error": "No battery (desktop or VM)"}
            result = {"percent": round(b.percent, 1), "plugged_in": b.power_plugged,
                      "time_left_minutes": round(b.secsleft / 60) if b.secsleft and b.secsleft > 0 else None}
            if battery_col:
                try:
                    result.update(battery_col.collect())
                except Exception:
                    pass
            return result
        except Exception as e:
            return {"error": str(e)}

    def get_security_status() -> dict:
        """AV/BitLocker/firewall compliance status, overall score, detected threats."""
        if not security_col:
            return {"error": "Security collector not bound"}
        try:
            return security_col.collect()
        except Exception as e:
            return {"error": str(e)}

    # ── ACTION TOOLS (only call after APPROVED) ──────────────────

    def kill_process(pid: int, reason: str = "") -> dict:
        """Kill a process by PID. Get pid from get_top_processes().
        ONLY call this after user has approved the remediation plan.
        Protected system processes are automatically blocked."""
        if not remediation:
            return {"success": False, "message": "Remediation engine not bound"}
        try:
            import psutil
            proc = psutil.Process(int(pid))
            if proc.name().lower() in _PROTECTED:
                return {"success": False,
                        "message": f"BLOCKED: '{proc.name()}' is a protected system process"}
        except Exception:
            pass
        logger.info(f"[Tool] kill_process pid={pid} reason={reason!r}")
        r = remediation.kill_process(int(pid))
        return {"success": r.success, "message": r.message, "details": r.details or ""}

    def clear_temp_files() -> dict:
        """Delete Windows temp files. Safe. Typically recovers 1-10 GB.
        ONLY call this after user has approved the remediation plan."""
        if not remediation:
            return {"success": False, "message": "Remediation engine not bound"}
        logger.info("[Tool] clear_temp_files")
        r = remediation.clear_temp_files()
        return {"success": r.success, "message": r.message, "details": r.details or ""}

    def reset_network() -> dict:
        """Flush DNS cache and reset Winsock (~5s network interruption).
        Use when latency > 200ms. ONLY call after user approval."""
        if not remediation:
            return {"success": False, "message": "Remediation engine not bound"}
        logger.info("[Tool] reset_network")
        r = remediation.reset_network()
        return {"success": r.success, "message": r.message, "details": r.details or ""}

    def optimize_memory() -> dict:
        """Empty Windows standby cache. Safe, Windows refills as needed.
        ONLY call after user has approved the remediation plan."""
        if not remediation:
            return {"success": False, "message": "Remediation engine not bound"}
        logger.info("[Tool] optimize_memory")
        r = remediation.optimize_memory()
        return {"success": r.success, "message": r.message, "details": r.details or ""}

    read_tools   = [get_system_metrics, get_top_processes, get_active_violations,
                    get_metric_trend, get_predictions, get_metric_history,
                    get_system_info, get_battery_status, get_security_status]
    action_tools = [kill_process, clear_temp_files, reset_network, optimize_memory]

    return read_tools, action_tools


# ─────────────────────────────────────────────────────────────────
#  OLLAMA  single-shot client (fallback when AutoGen not installed)
# ─────────────────────────────────────────────────────────────────

_SYSTEM_PROMPT_FALLBACK = """\
You are Cognix EUD AI Assist, an expert endpoint performance diagnostics assistant.
Be concise. Reference specific values. Use **bold** for key terms.
Use `backticks` for process names. Number all recommendations.
If healthy, say so in 1-2 lines. Never invent data."""

class _OllamaClient:
    def __init__(self, base_url="http://127.0.0.1:11434", timeout=120):
        self.base_url = base_url.rstrip("/")
        self.timeout  = timeout

    def is_running(self) -> bool:
        try:
            with socket.create_connection(("127.0.0.1", 11434), timeout=2):
                return True
        except OSError:
            return False

    def list_models(self) -> list:
        try:
            with urllib.request.urlopen(
                urllib.request.Request(f"{self.base_url}/api/tags"), timeout=5
            ) as r:
                return [m.get("name", "") for m in json.loads(r.read()).get("models", [])]
        except Exception:
            return []

    def chat(self, messages: list, model: str, options: dict) -> str:
        payload = json.dumps({"model": model, "messages": messages,
                              "stream": False, "options": options}).encode()
        req = urllib.request.Request(
            f"{self.base_url}/api/chat", data=payload,
            headers={"Content-Type": "application/json"}, method="POST")
        with urllib.request.urlopen(req, timeout=self.timeout) as r:
            return json.loads(r.read()).get("message", {}).get("content", "").strip()


# ─────────────────────────────────────────────────────────────────
#  PENDING PROPOSAL  — approval gate state
# ─────────────────────────────────────────────────────────────────

class _Proposal:
    """A remediation plan waiting for user approval."""
    def __init__(self, session_id: str, findings: str, plan_text: str, trigger: str):
        self.session_id  = session_id
        self.findings    = findings
        self.plan_text   = plan_text
        self.trigger     = trigger          # 'analyze' | 'chat' | 'monitor'
        self.created_at  = time.time()
        self.approved    = False
        self.dismissed   = False
        self._event      = asyncio.Event()  # set when user decides

    def to_dict(self) -> dict:
        return {"session_id": self.session_id, "findings": self.findings,
                "plan_text": self.plan_text, "trigger": self.trigger,
                "created_at": self.created_at}


# ─────────────────────────────────────────────────────────────────
#  MAIN ENGINE  (public facade — api/server.py talks only to this)
# ─────────────────────────────────────────────────────────────────

class AIDiagnosticEngine:
    """
    Agentic AI engine backed by AutoGen v0.4 + local Ollama.

    Same public interface as the original engine — api/server.py unchanged
    except for the three new approval endpoints added in server.py.

    Architecture:
      analyze_context() → Diagnostician agent (read tools only, no approval needed)
      chat()            → ChatAssistant agent (all tools, asks before acting)
      Approval gate     → approve(session_id) / dismiss(session_id)

    Fallback when AutoGen not installed:
      Single-shot Ollama → rule-based text responses
    """

    GENERATION_DEFAULTS = {
        "temperature": 0.3, "top_p": 0.9, "top_k": 40,
        "num_predict": 1024, "num_ctx": 4096, "repeat_penalty": 1.1,
    }

    def __init__(self):
        self._ollama        = _OllamaClient()
        self._ollama_model: Optional[str] = None
        self._gen_opts      = dict(self.GENERATION_DEFAULTS)
        self._available     = None
        self._last_check    = 0.0
        self._provider      = "local"

        # AutoGen model client (set by set_local_model)
        self._model_client  = None

        # Live tool references (set by bind_tools)
        self._read_tools    = []
        self._action_tools  = []

        # WebSocket broadcast (set by bind_tools)
        self._broadcast: Optional[Callable] = None

        # Pending proposals waiting for user decision
        self._proposals: dict[str, _Proposal] = {}

        logger.info(
            f"[AgenticAI] AutoGen available: {AUTOGEN_OK} — "
            f"{'agentic mode' if AUTOGEN_OK else 'single-shot fallback mode'}"
        )

    # ──────────────────────────────────────────────────────────────
    #  SETUP  (called from server.py create_app)
    # ──────────────────────────────────────────────────────────────

    def bind_tools(self, state, remediation_engine,
                   battery_collector=None, security_collector=None,
                   broadcast_fn: Optional[Callable] = None):
        """
        Wire live AgentState + RemediationEngine into the tool closures.
        Must be called after the scheduler starts. Called from server.py.
        """
        self._read_tools, self._action_tools = _make_tools(
            state, remediation_engine, battery_collector, security_collector
        )
        self._broadcast = broadcast_fn
        logger.info(f"[AgenticAI] Tools bound — "
                    f"{len(self._read_tools)} read, {len(self._action_tools)} action")

    def set_local_model(self, model_name: str) -> dict:
        """Set the Ollama model. Builds the AutoGen model client."""
        self._ollama_model = model_name
        self._available    = None
        self._last_check   = 0.0

        if not self._ollama.is_running():
            return {"provider": "local", "model": model_name,
                    "available": False, "message": "Ollama is not running"}

        if not AUTOGEN_OK:
            return {
                "provider": "local", "model": model_name,
                "available": True, "agentic": False,
                "message": (
                    f"Ollama connected ({model_name}) — single-shot mode. "
                    "Run: pip install autogen-agentchat autogen-ext[openai] for agentic mode."
                ),
            }

        try:
            base = "http://127.0.0.1:11434/v1"
            self._model_client = OpenAIChatCompletionClient(
                model    = model_name,
                base_url = base,
                api_key  = "ollama",
                model_capabilities={
                    "vision": False,
                    "function_calling": True,
                    "json_output": False,
                },
            )
            logger.info(f"[AgenticAI] Model client ready: {model_name}")
            return {
                "provider": "local", "model": model_name,
                "available": True, "agentic": True,
                "message": f"Agentic AI ready — {model_name} via Ollama",
            }
        except Exception as e:
            self._model_client = None
            logger.error(f"[AgenticAI] set_model error: {e}")
            return {"provider": "local", "model": model_name,
                    "available": False, "message": str(e)[:200]}

    # ──────────────────────────────────────────────────────────────
    #  APPROVAL GATE
    # ──────────────────────────────────────────────────────────────

    def approve_proposal(self, session_id: str) -> bool:
        p = self._proposals.get(session_id)
        if not p or p._event.is_set():
            return False
        p.approved = True
        p._event.set()
        logger.info(f"[AgenticAI] Proposal approved: {session_id}")
        return True

    def dismiss_proposal(self, session_id: str) -> bool:
        p = self._proposals.get(session_id)
        if not p or p._event.is_set():
            return False
        p.dismissed = True
        p._event.set()
        logger.info(f"[AgenticAI] Proposal dismissed: {session_id}")
        return True

    def get_pending_proposals(self) -> list:
        now = time.time()
        return [p.to_dict() for p in self._proposals.values()
                if not p._event.is_set() and (now - p.created_at) < 300]

    # ──────────────────────────────────────────────────────────────
    #  MAIN PUBLIC INTERFACE  (async — called directly from server.py)
    # ──────────────────────────────────────────────────────────────

    async def analyze_context_async(self, context: DiagnosticContext) -> str:
        """
        Agentic tab analysis — called by Analyze Now button on every tab.
        Diagnostician uses live tools to investigate. Read-only, no approval needed.
        Returns formatted markdown text for the UI.
        """
        if not self._model_client:
            return self._fallback_analyze(context)

        try:
            diag = AssistantAgent(
                name           = "Diagnostician",
                model_client   = self._model_client,
                system_message = _DIAG_PROMPT,
                tools          = self._read_tools,
            )
            term  = TextMentionTermination("## Findings") | MaxMessageTermination(12)
            team  = RoundRobinGroupChat([diag], termination_condition=term)
            ctx   = self._format_context(context)
            result = await team.run(
                task=f"System snapshot (use tools for live data):\n{ctx}\n\n"
                     "Investigate now and produce your ## Findings report."
            )
            text = self._last_message(result)
            return text if text else "System appears healthy — no issues detected."

        except Exception as e:
            logger.error(f"[AgenticAI] analyze error: {e}", exc_info=True)
            return self._fallback_analyze(context)

    async def chat_async(self, user_message: str,
                         context: Optional[DiagnosticContext] = None,
                         history: Optional[list] = None) -> str:
        """
        Agentic chat — agent fetches live data with tools before answering.
        For remediation requests: agent proposes → user approves → agent executes.
        """
        if not self._model_client:
            return self._fallback_chat(user_message, context)

        try:
            ctx_text = self._format_context(context) if context else \
                       "No context. Use get_system_metrics() for current data."

            hist_text = ""
            if history:
                hist_text = "\n[Recent conversation]\n" + "\n".join(
                    f"{t.get('role','?').upper()}: {(t.get('content') or '')[:300]}"
                    for t in history[-6:]
                ) + "\n"

            agent = AssistantAgent(
                name           = "CognixAssistant",
                model_client   = self._model_client,
                system_message = _CHAT_PROMPT,
                tools          = self._read_tools + self._action_tools,
            )
            term   = MaxMessageTermination(8)
            team   = RoundRobinGroupChat([agent], termination_condition=term)
            result = await team.run(
                task=f"System context:\n{ctx_text}\n{hist_text}\nUser: {user_message}"
            )
            response = self._last_message(result)

            # If agent is asking for approval to execute, register a proposal
            if response and any(kw in response.lower() for kw in
               ("shall i proceed", "reply yes", "want me to", "approve",
                "should i kill", "want me to clear", "type yes", "say yes")):
                sid = uuid.uuid4().hex[:10]
                prop = _Proposal(sid, ctx_text, response, trigger="chat")
                self._proposals[sid] = prop
                await self._push({
                    "type": "agent_proposal",
                    "session_id": sid,
                    "trigger": "chat",
                    "plan_text": response,
                })

            return response or "All clear — system appears healthy."

        except Exception as e:
            logger.error(f"[AgenticAI] chat error: {e}", exc_info=True)
            return self._fallback_chat(user_message, context)

    # ──────────────────────────────────────────────────────────────
    #  SYNC WRAPPERS  (kept for backward-compat with any sync callers)
    # ──────────────────────────────────────────────────────────────

    def analyze_context(self, context: DiagnosticContext) -> str:
        """Sync wrapper — only used from the alert callback (non-async context)."""
        return self._fallback_analyze(context)

    def chat(self, user_message: str, context=None, history=None) -> str:
        """Sync wrapper — only used when AutoGen is unavailable."""
        return self._fallback_chat(user_message, context)

    # ──────────────────────────────────────────────────────────────
    #  STATUS & CONFIG
    # ──────────────────────────────────────────────────────────────

    @property
    def available(self) -> bool:
        now = time.time()
        if self._available is not None and (now - self._last_check) < 30:
            return self._available
        self._last_check = now
        self._available  = self._ollama.is_running()
        return self._available

    @property
    def model_name(self) -> str:
        return self._ollama_model or "not configured"

    def get_status(self) -> dict:
        running = self._ollama.is_running()
        models  = self._ollama.list_models() if running else []
        return {
            "provider":          "local",
            "model":             self.model_name,
            "available":         running,
            "ollama_running":    running,
            "ollama_models":     models,
            "agentic_enabled":   AUTOGEN_OK,
            "agentic_active":    self._model_client is not None,
            "autogen_installed": AUTOGEN_OK,
            "pending_proposals": len(self.get_pending_proposals()),
        }

    def get_ollama_models(self) -> list:
        return self._ollama.list_models()

    def set_generation_options(self, opts: dict):
        self._gen_opts.update(opts)

    # ──────────────────────────────────────────────────────────────
    #  INTERNAL HELPERS
    # ──────────────────────────────────────────────────────────────

    def _last_message(self, result) -> str:
        """Extract last meaningful text from AutoGen v0.4 TaskResult."""
        try:
            msgs = result.messages if hasattr(result, "messages") else []
            for msg in reversed(msgs):
                content = getattr(msg, "content", None) or \
                          (msg.get("content") if isinstance(msg, dict) else None)
                if isinstance(content, str):
                    content = content.strip()
                    if content and len(content) > 15:
                        return content
        except Exception:
            pass
        return ""

    def _format_context(self, ctx) -> str:
        """Compact snapshot for agent prompts."""
        if not ctx:
            return "No snapshot. Call get_system_metrics()."
        m, h = ctx.metrics, ctx.health
        lines = [
            f"Health: {h.get('score','?')}/100 — {h.get('label','?')}",
            f"CPU: {m.get('cpu_percent',0):.1f}%  "
            f"MEM: {m.get('memory_percent',0):.1f}%  "
            f"DISK: {m.get('disk_percent',0):.1f}%  "
            f"NET: {m.get('network_latency_ms',0):.0f}ms",
        ]
        if ctx.violations:
            for v in ctx.violations[:3]:
                lines.append(f"  [{v.get('severity','').upper()}] {v.get('message','')}")
        lines.append("→ Use tools to get fresh live data.")
        return "\n".join(lines)

    async def _push(self, data: dict):
        """Broadcast to all WebSocket clients."""
        if self._broadcast:
            try:
                await self._broadcast(data)
            except Exception as e:
                logger.debug(f"[AgenticAI] broadcast error: {e}")

    # ── Fallbacks when AutoGen / Ollama unavailable ────────────────

    def _fallback_analyze(self, context) -> str:
        """Single-shot Ollama fallback for analyze_context."""
        if self._ollama.is_running() and self._ollama_model:
            prompt = (
                "Analyze the following real-time system diagnostic data. "
                "Identify issues, explain root causes, give numbered steps:\n\n"
                f"{context.to_prompt_text()}"
            )
            try:
                msgs = [{"role": "system", "content": _SYSTEM_PROMPT_FALLBACK},
                        {"role": "user",   "content": prompt}]
                return self._ollama.chat(msgs, self._ollama_model, self._gen_opts)
            except Exception as e:
                logger.error(f"[AgenticAI] fallback analyze error: {e}")
        return self._rule_based("analyze", context)

    def _fallback_chat(self, user_message: str, context) -> str:
        """Single-shot Ollama fallback for chat."""
        if self._ollama.is_running() and self._ollama_model:
            ctx_text = self._format_context(context) if context else ""
            msgs = [{"role": "system",    "content": _SYSTEM_PROMPT_FALLBACK},
                    {"role": "user",      "content": f"[System]\n{ctx_text}"},
                    {"role": "assistant", "content": "Understood."},
                    {"role": "user",      "content": user_message}]
            try:
                return self._ollama.chat(msgs, self._ollama_model, self._gen_opts)
            except Exception as e:
                logger.error(f"[AgenticAI] fallback chat error: {e}")
        return self._rule_based(user_message, context)

    def _rule_based(self, prompt: str, context) -> str:
        """Keyword-based response when nothing else is available."""
        p = prompt.lower()
        hdr = ""
        if context:
            m, h = context.metrics, context.health
            sc   = h.get("score", 0)
            ic   = "✅" if sc >= 85 else "⚠️" if sc >= 55 else "🔴"
            hdr  = (f"{ic} **Health {sc}/100 — {h.get('label','?')}** | "
                    f"CPU `{m.get('cpu_percent',0):.1f}%` "
                    f"MEM `{m.get('memory_percent',0):.1f}%` "
                    f"DISK `{m.get('disk_percent',0):.1f}%`\n\n")
            if context.violations:
                v = context.violations[0]
                return (hdr + f"⚠ **{v.get('severity','').upper()}** — {v.get('message','')}\n\n"
                        "> Start Ollama and select a model in **AI Settings** for agentic analysis.")

        if any(k in p for k in ("cpu", "processor")):
            return hdr + "**CPU high** — check Processes tab, kill top consumer."
        if any(k in p for k in ("memory", "ram")):
            return hdr + "**Memory high** — restart browsers/IDEs or use Optimize Memory."
        if any(k in p for k in ("disk", "storage", "space")):
            return hdr + "**Disk** — run Clear Temp Files from the Remediation panel."
        if any(k in p for k in ("network", "latency", "dns")):
            return hdr + "**Network** — use Reset Network to flush DNS and Winsock."
        if context and any(k in p for k in ("health", "status", "ok", "analyze")):
            sc = context.health.get("score", 0)
            return hdr + f"Health score: {sc}/100. " + context.health.get("summary", "")

        return (hdr or "") + (
            "⚙ **AI not configured** — Ollama not running or no model selected.\n\n"
            "Go to **AI Settings → Local LLM**, pick a model, click Connect.\n"
            "Then agents can investigate and fix issues autonomously."
        )
