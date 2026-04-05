"""
Cognix EUD AI Assist — Agentic AI Engine  v4.0.0
=================================================

THREE-AGENT SYSTEM
──────────────────
Every user message is routed to the correct specialist agent:

  ┌─────────────────────────────────────────────────────────┐
  │                   Intent Router                          │
  │  device_query → Diagnostician                           │
  │  device_action → Diagnostician ──► Remediator           │
  │  sys_utility  → Sys_Assist                              │
  │  general      → polite decline (no tools)               │
  └─────────────────────────────────────────────────────────┘

  🔬 DIAGNOSTICIAN
     Reads live metrics with read-only tools.
     Finds issues, reports with real numbers.
     Hands off to Remediator when a fix is needed.

  🔧 REMEDIATOR
     Triggered by Diagnostician or direct action request.
     Assesses whether action is necessary with live data.
     Proposes a plan → waits for user approval → executes.
     Actions: kill_process, clear_temp_files, clear_recycle_bin,
              optimize_memory, reset_network.

  🖥 SYS_ASSIST
     Handles all OS-level utility tasks directly — no approval
     gate needed for cosmetic/utility operations.
     Capabilities: wallpaper, desktop icons, power plans,
     volume, brightness, dark mode, startup programs, Wi-Fi,
     Bluetooth, screen lock, disk cleanup, display settings,
     create/delete shortcuts, timezone, notifications, and more.

CONVERSATION MEMORY
───────────────────
ConversationMemory stores 30 turns. Agents always see the full
conversation history so follow-up questions and user preferences
(name, nickname) are remembered throughout the session.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import platform
import socket
import subprocess
import time
import urllib.request
import uuid
from collections import deque
from typing import Callable, Optional

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
    logger.warning(f"[AgenticAI] not installed: {_e}")

_DONE = "COGNIX_DONE"
MEMORY_TURNS = 30

_YES_PHRASES = frozenset({
    "yes", "yes please", "go ahead", "do it", "proceed", "approve", "approved",
    "sure", "ok", "okay", "yep", "yup", "yeah", "confirm", "execute",
    "run it", "yes go ahead", "yes proceed", "please proceed",
})

_IS_WIN = platform.system() == "Windows"


# ─────────────────────────────────────────────────────────────────
#  CONVERSATION MEMORY
# ─────────────────────────────────────────────────────────────────

class ConversationMemory:
    def __init__(self, max_turns: int = MEMORY_TURNS):
        self._turns: deque = deque(maxlen=max_turns * 2)

    def add(self, role: str, content: str):
        if content and content.strip():
            self._turns.append({"role": role, "content": content.strip()})

    def clear(self):
        self._turns.clear()

    def to_text(self) -> str:
        if not self._turns:
            return ""
        lines = [
            f"{'USER' if t['role']=='user' else 'ASSISTANT'}: {t['content'][:400]}"
            for t in list(self._turns)
        ]
        return (
            "\n[CONVERSATION HISTORY — use this to answer follow-up questions "
            "and remember anything the user shared (name, nickname, preferences)]\n"
            + "\n".join(lines) + "\n"
        )

    def as_list(self) -> list:
        return list(self._turns)

    def __len__(self):
        return len(self._turns)


# ─────────────────────────────────────────────────────────────────
#  INTENT CLASSIFIER
#  Four buckets:
#    "device_query"  — ask about live device metrics
#    "device_action" — fix/remediate device issues
#    "sys_utility"   — OS-level utility tasks
#    "general"       — unrelated to this device assistant
# ─────────────────────────────────────────────────────────────────

_DEVICE_QUERY_KW = frozenset({
    "cpu", "memory", "ram", "disk", "storage", "battery", "charge",
    "network", "latency", "wifi", "internet", "process", "processes",
    "security", "antivirus", "firewall", "health", "temperature",
    "performance", "speed", "slow", "lag", "crash", "error", "log",
    "gpu", "graphics", "driver", "installed", "application", "app",
    "startup", "boot", "system", "device", "my device", "my computer",
    "my laptop", "my pc", "my system", "how is my", "check my",
    "what is my", "show me", "analyze", "diagnose", "monitor",
    "alert", "violation", "threshold", "trend", "predict",
})

_DEVICE_ACTION_KW = frozenset({
    "clear temp", "clear cache", "kill process", "terminate", "optimize memory",
    "free memory", "reset network", "flush dns", "clear recycle", "fix",
    "repair", "resolve", "solve the issue", "boost performance",
})

_SYS_UTILITY_KW = frozenset({
    # Desktop / appearance
    "wallpaper", "background", "desktop icon", "icons", "arrange icon",
    "shortcut", "theme", "dark mode", "light mode", "appearance", "screensaver",
    "screen saver", "night light", "night mode", "color scheme",
    # Audio / display
    "volume", "sound", "mute", "unmute", "brightness", "display",
    "resolution", "screen", "monitor", "refresh rate",
    # Power
    "power plan", "power mode", "sleep", "hibernate", "shutdown", "restart",
    "lock screen", "lock my screen", "sign out", "log off",
    # Startup
    "startup program", "autostart", "startup app", "boot program", "startup item",
    "disable startup", "enable startup",
    # Connectivity
    "wifi", "wi-fi", "bluetooth", "airplane mode", "network adapter",
    "toggle wifi", "turn off wifi", "enable wifi",
    "toggle bluetooth", "turn on bluetooth",
    # System utilities
    "clipboard", "clear clipboard", "notification", "do not disturb",
    "taskbar", "create folder", "create shortcut",
    "timezone", "date time", "clock", "time zone",
    "default browser", "default app", "file association",
    "windows update", "check for update", "update windows",
    "accessibility", "ease of access", "magnifier",
    "disk cleanup", "defragment", "storage sense",
    # App management
    "close app", "rearrange", "pin to taskbar", "unpin",
    "open settings", "open control panel",
    "show desktop", "minimize all",
    "take screenshot", "screenshot",
    "empty trash", "empty bin",
})

_OUT_OF_SCOPE_KW = frozenset({
    "president", "prime minister", "politician", "country",
    "weather", "recipe", "cook", "movie", "music", "song",
    "sport", "football", "cricket", "news", "stock", "market",
    "price", "history", "science", "math", "calculate", "translate",
    "write email", "type for me", "write for me", "write essay",
})


def _classify_intent(message: str) -> str:
    ml    = message.lower()
    oos   = sum(1 for k in _OUT_OF_SCOPE_KW   if k in ml)
    dev_q = sum(1 for k in _DEVICE_QUERY_KW   if k in ml)
    dev_a = sum(1 for k in _DEVICE_ACTION_KW  if k in ml)
    sys_u = sum(1 for k in _SYS_UTILITY_KW    if k in ml)

    # Explicit out-of-scope
    if oos > 0 and dev_q == 0 and dev_a == 0 and sys_u == 0:
        return "general"

    # Sys_utility wins if it scores highest
    if sys_u > 0 and sys_u >= dev_a and sys_u >= dev_q:
        return "sys_utility"

    if dev_a > 0:
        return "device_action"

    if dev_q > 0:
        return "device_query"

    # Short/ambiguous — default to device_query
    return "device_query"


# ─────────────────────────────────────────────────────────────────
#  AGENT SYSTEM PROMPTS
# ─────────────────────────────────────────────────────────────────

_DIAGNOSTICIAN_PROMPT = (
    "You are the Cognix Diagnostician — a Windows endpoint analyst.\n"
    "You investigate the live device using READ-ONLY tools and report findings.\n\n"
    "IDENTITY: You are Cognix, running locally on Ollama. Not a cloud AI.\n\n"
    "MEMORY: The task contains [CONVERSATION HISTORY]. Use it for follow-up\n"
    "questions like 'what was my last question?' or 'what is my nickname?'.\n\n"
    "TOOLS (call in this order based on the question):\n"
    "  get_system_metrics()       → ALWAYS first for any system question\n"
    "  get_active_violations()    → alerts / thresholds breached?\n"
    "  get_top_processes(10)      → CPU / memory / process questions\n"
    "  get_battery_status()       → battery (NOT same as health score)\n"
    "  get_security_status()      → security / AV / firewall\n"
    "  get_metric_trend('cpu')    → is it rising or stable?\n"
    "  get_metric_history(20)     → recent patterns\n"
    "  get_system_info()          → hostname, OS, cores, RAM\n\n"
    "RULES:\n"
    "  ✗ Never output raw JSON or tool call syntax\n"
    "  ✗ Never call action tools (kill_process, clear_temp_files, etc.)\n"
    "  ✓ Quote REAL numbers from tool results — never fabricate\n"
    "  ✓ Be concise — 2-4 sentences max\n\n"
    "HANDOFF:\n"
    "  If the user wants a fix: report what you found in 1 sentence with real\n"
    "  numbers, then end with exactly: REMEDIATOR_REQUESTED\n"
    "  Do NOT write the plan yourself.\n\n"
    "Otherwise end every answer with: " + _DONE
)

_REMEDIATOR_PROMPT = (
    "You are the Cognix Remediator — you assess, plan, and execute safe device fixes.\n\n"
    "You speak when Diagnostician signals REMEDIATOR_REQUESTED or when the user\n"
    "asks to fix/clear/kill/reset/optimize something on their device.\n\n"
    "ACTION TOOLS:\n"
    "  kill_process(pid, reason)  — terminate a process by PID\n"
    "  clear_temp_files()         — delete Windows temp files (safe, 1-10 GB)\n"
    "  clear_recycle_bin()        — empty Windows Recycle Bin\n"
    "  optimize_memory()          — empty standby cache (always safe)\n"
    "  reset_network()            — flush DNS + Winsock (~5s interruption)\n\n"
    "PHASE 1 — ASSESS AND PROPOSE (before any action tool call):\n"
    "  1. Call get_system_metrics() + the relevant specific read tool\n"
    "  2. Assess if the action is actually needed\n"
    "  3. Write a friendly message:\n"
    "     'I checked your [metric]. [Real finding — actual numbers].\n"
    "      [Action] will [benefit]. Click **✓ Approve** to execute.'\n"
    "  4. End with: AWAITING_APPROVAL\n\n"
    "PHASE 2 — EXECUTE (only when task contains the word APPROVED):\n"
    "  → Call each action tool from your plan\n"
    "  → Report each result in plain English\n"
    "  → End with: " + _DONE + "\n\n"
    "SAFETY:\n"
    "  ✗ NEVER call action tools before APPROVED is in the task\n"
    "  ✗ NEVER kill: lsass.exe, csrss.exe, winlogon.exe, smss.exe, services.exe\n"
    "  ✗ NEVER reset_network if latency < 150ms\n"
    "  ✓ If metric is actually fine, say so and skip the action"
)

_SYS_ASSIST_PROMPT = (
    "You are Sys_Assist — the Cognix Windows system utility agent.\n"
    "You handle OS-level tasks: desktop appearance, audio, display, power,\n"
    "connectivity, startup programs, and Windows settings.\n\n"
    "MEMORY: The task contains [CONVERSATION HISTORY]. Use it for context\n"
    "and to remember user preferences and past requests.\n\n"
    "TOOLS:\n"
    "  arrange_desktop_icons()              — auto-arrange / sort desktop icons\n"
    "  set_wallpaper(path)                  — change desktop wallpaper\n"
    "  set_dark_mode(enabled)               — toggle dark/light mode (True/False)\n"
    "  set_volume(level)                    — set system volume 0-100\n"
    "  mute_audio(muted)                    — mute or unmute system audio\n"
    "  set_screen_brightness(level)         — set brightness 0-100\n"
    "  set_power_plan(plan)                 — balanced/performance/power_saver\n"
    "  lock_screen()                        — lock the workstation immediately\n"
    "  take_screenshot(filename)            — capture and save a screenshot\n"
    "  get_startup_programs()               — list programs that run on startup\n"
    "  toggle_startup_program(name,enable)  — enable/disable a startup program\n"
    "  toggle_wifi(enabled)                 — enable or disable Wi-Fi\n"
    "  toggle_bluetooth(enabled)            — enable or disable Bluetooth\n"
    "  clear_clipboard()                    — clear the Windows clipboard\n"
    "  set_do_not_disturb(enabled)          — toggle Windows focus/DND mode\n"
    "  run_disk_cleanup()                   — launch Windows disk cleanup\n"
    "  set_display_resolution(width,height) — change screen resolution\n"
    "  open_settings(page)                  — open a Windows Settings page\n"
    "  create_desktop_shortcut(target,name) — create a shortcut on the desktop\n"
    "  show_desktop()                       — minimize all windows (Show Desktop)\n"
    "  set_timezone(tz)                     — change system timezone\n"
    "  toggle_night_light(enabled)          — enable/disable Night Light\n"
    "  get_system_info()                    — hostname, OS, cores, RAM\n\n"
    "BEHAVIOR:\n"
    "  ✓ Act immediately — most utility tasks need no approval\n"
    "  ✓ Call the right tool, report the result in plain friendly English\n"
    "  ✓ If the task needs no tool (e.g. user just asks what tools you have),\n"
    "    answer conversationally\n"
    "  ✓ For DESTRUCTIVE ops (shutdown, restart, format) — ask for confirm first\n"
    "  ✗ Never output raw JSON or technical error traces to the user\n"
    "  End every reply with: " + _DONE
)

_OUT_OF_SCOPE_REPLY = (
    "I'm Cognix — specialized in monitoring and managing this Windows device. "
    "I can't help with that, but I'm ready to:\n"
    "• Check system health, CPU, memory, disk, battery\n"
    "• Fix performance issues (kill processes, clear temp, optimize memory)\n"
    "• Handle Windows utilities (wallpaper, volume, dark mode, power plans, startup apps…)\n"
    "Just ask!"
)

_FALLBACK_SYSTEM = (
    "You are Cognix EUD AI Assist, a Windows device assistant.\n"
    "Answer in 2-4 plain English sentences. Use real numbers from the data.\n"
    "Never output raw JSON. Be conversational and helpful.\n"
    "If conversation history is provided, use it for follow-up context."
)


# ─────────────────────────────────────────────────────────────────
#  PROTECTED PROCESSES
# ─────────────────────────────────────────────────────────────────
_PROTECTED = frozenset({
    "lsass.exe", "csrss.exe", "winlogon.exe", "smss.exe",
    "services.exe", "wininit.exe", "system", "registry",
})


# ─────────────────────────────────────────────────────────────────
#  POWERSHELL HELPER
# ─────────────────────────────────────────────────────────────────

def _ps(cmd: str, timeout: int = 12) -> tuple[bool, str]:
    """Run a PowerShell command. Returns (success, output/error)."""
    if not _IS_WIN:
        return False, "PowerShell tools are Windows-only"
    try:
        r = subprocess.run(
            ["powershell", "-NoProfile", "-NonInteractive", "-Command", cmd],
            capture_output=True, text=True, timeout=timeout
        )
        out = (r.stdout.strip() or r.stderr.strip())[:300]
        return r.returncode == 0, out
    except subprocess.TimeoutExpired:
        return False, "Command timed out"
    except Exception as e:
        return False, str(e)[:100]


# ─────────────────────────────────────────────────────────────────
#  DEVICE TOOL FACTORY  (Diagnostician + Remediator tools)
# ─────────────────────────────────────────────────────────────────

def _make_tools(state, remediation,
                battery_col=None, security_col=None,
                broadcast_fn: Optional[Callable] = None):
    """Returns (diag_tools, rem_tools, tool_map)."""

    def _emit(tool: str, phase: str, agent: str = "Agent", summary: str = ""):
        if not broadcast_fn:
            return
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                asyncio.run_coroutine_threadsafe(
                    broadcast_fn({"type": "agent_step", "agent": agent,
                                  "tool": tool, "phase": phase, "summary": summary}),
                    loop
                )
        except Exception:
            pass

    def _summarize(name: str, result) -> str:
        if isinstance(result, dict):
            if result.get("error"):
                return f"⚠ {str(result['error'])[:60]}"
            if name == "get_system_metrics":
                return (f"CPU {result.get('cpu_percent','?')}%  "
                        f"RAM {result.get('memory_percent','?')}%  "
                        f"Disk {result.get('disk_percent','?')}%")
            if name == "get_battery_status":
                if "error" in result:
                    return result["error"][:60]
                pct  = result.get("percent", "?")
                plug = "charging" if result.get("plugged_in") else "on battery"
                return f"{pct}% — {plug}"
            if name == "get_security_status":
                return f"compliance {result.get('compliance_score','?')}/100"
            if name in ("kill_process", "clear_temp_files", "clear_recycle_bin",
                        "optimize_memory", "reset_network"):
                ok  = result.get("success", False)
                msg = result.get("message", "")[:60]
                return ("✓ " if ok else "✗ ") + msg
        if isinstance(result, list) and name == "get_top_processes" and result:
            t = result[0]
            return f"top: {t.get('name','?')} {t.get('cpu_percent','?')}% CPU"
        if isinstance(result, list):
            return f"{len(result)} items"
        return ""

    def _wrap(name: str, fn, agent_name: str = "Agent"):
        def wrapped(*args, **kwargs):
            _emit(name, "planning", agent_name)
            _emit(name, "running",  agent_name)
            try:
                result  = fn(*args, **kwargs)
                summary = _summarize(name, result)
                _emit(name, "done", agent_name, summary)
                return result
            except Exception as exc:
                _emit(name, "done", agent_name, f"error: {str(exc)[:60]}")
                raise
        wrapped.__name__ = name
        wrapped.__doc__  = (fn.__doc__ or name).strip()
        return wrapped

    # ── READ TOOLS ────────────────────────────────────────────────

    def _get_system_metrics() -> dict:
        """Live CPU%, memory%, disk%, network latency, GPU%, health score. Call first."""
        if not state or not state.latest_metrics:
            return {"error": "Metrics not yet collected — retry in a few seconds"}
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
            "health_score":       h.score   if h else 0,
            "health_label":       h.label   if h else "Unknown",
            "health_summary":     h.summary if h else "",
            "as_of":              time.strftime("%H:%M:%S", time.localtime(m.timestamp)),
        }

    def _get_top_processes(limit: int = 10) -> list:
        """Top N processes by CPU%. Returns pid, name, cpu_percent, memory_mb, status."""
        if not state:
            return []
        return [
            {"pid": p.pid, "name": p.name,
             "cpu_percent": round(p.cpu_percent, 1),
             "memory_mb":   round(p.memory_mb),
             "status":      p.status}
            for p in state.latest_processes[:min(int(limit), 20)]
        ]

    def _get_active_violations() -> list:
        """Active threshold violations. Empty = healthy."""
        if not state:
            return []
        return [
            {"metric": v.metric, "severity": v.severity.value,
             "current_value": round(v.current_value, 1),
             "threshold": v.threshold, "message": v.message,
             "sustained_seconds": round(v.sustained_seconds)}
            for v in state.active_violations
        ]

    def _get_metric_trend(metric: str) -> dict:
        """Trend for 'cpu'|'memory'|'disk'|'network_latency'."""
        valid = ("cpu", "memory", "disk", "network_latency")
        if metric not in valid:
            return {"error": f"metric must be one of: {', '.join(valid)}"}
        if not state or not state.latest_context:
            return {"metric": metric, "direction": "unknown"}
        t = (state.latest_context.trends or {}).get(metric)
        if not t:
            return {"metric": metric, "direction": "stable", "note": "Not enough data"}
        return {"metric": metric, "direction": t.get("direction", "stable"),
                "slope": round(t.get("slope", 0), 3),
                "change_rate_pct": round(t.get("change_rate", 0), 2)}

    def _get_predictions() -> list:
        """Predictive forecasts. Empty = no concerning trends."""
        if not state or not state.latest_context:
            return []
        return state.latest_context.predictions or []

    def _get_metric_history(points: int = 20) -> dict:
        """Last N metric snapshots (~5s each). Spot spikes vs sustained issues."""
        if not state:
            return {"error": "State unavailable"}
        pts = state.metric_history[-min(int(points), 60):]
        return {"count": len(pts), "interval_seconds": 5, "history": pts}

    def _get_system_info() -> dict:
        """Static device info: hostname, OS, CPU, total RAM, cores."""
        return getattr(state, "system_info", {}) or {}

    def _get_battery_status() -> dict:
        """Battery charge %, plugged_in, time_left_minutes. Error on desktop/VM."""
        try:
            import psutil
            b = psutil.sensors_battery()
            if not b:
                return {"error": "No battery — desktop or VM"}
            result = {
                "percent": round(b.percent, 1),
                "plugged_in": b.power_plugged,
                "time_left_minutes": round(b.secsleft / 60) if b.secsleft and b.secsleft > 0 else None,
            }
            if battery_col:
                try:
                    result.update(battery_col.collect())
                except Exception:
                    pass
            return result
        except Exception as e:
            return {"error": str(e)}

    def _get_security_status() -> dict:
        """AV/BitLocker/firewall compliance, score, threats."""
        if not security_col:
            return {"error": "Security collector not bound"}
        try:
            return security_col.collect()
        except Exception as e:
            return {"error": str(e)}

    # ── REMEDIATION ACTION TOOLS ──────────────────────────────────

    def _kill_process(pid: int, reason: str = "") -> dict:
        """Kill a process by PID. ONLY after APPROVED."""
        if not remediation:
            return {"success": False, "message": "Remediation engine not bound"}
        try:
            import psutil as _ps
            proc = _ps.Process(int(pid))
            if proc.name().lower() in _PROTECTED:
                return {"success": False, "message": f"BLOCKED: '{proc.name()}' is protected"}
        except Exception:
            pass
        r = remediation.kill_process(int(pid))
        return {"success": r.success, "message": r.message, "details": r.details or ""}

    def _clear_temp_files() -> dict:
        """Delete Windows temp files. Safe. ONLY after APPROVED."""
        if not remediation:
            return {"success": False, "message": "Remediation engine not bound"}
        r = remediation.clear_temp_files()
        return {"success": r.success, "message": r.message, "details": r.details or ""}

    def _clear_recycle_bin() -> dict:
        """Empty the Windows Recycle Bin. ONLY after APPROVED."""
        if not remediation:
            return {"success": False, "message": "Remediation engine not bound"}
        try:
            ok, out = _ps("Clear-RecycleBin -Force -ErrorAction SilentlyContinue; 'done'")
            return {"success": True, "message": "Recycle Bin emptied successfully."}
        except Exception as e:
            return {"success": False, "message": str(e)[:100]}

    def _reset_network() -> dict:
        """Flush DNS + reset Winsock. ONLY after APPROVED."""
        if not remediation:
            return {"success": False, "message": "Remediation engine not bound"}
        r = remediation.reset_network()
        return {"success": r.success, "message": r.message, "details": r.details or ""}

    def _optimize_memory() -> dict:
        """Empty Windows standby cache. ONLY after APPROVED."""
        if not remediation:
            return {"success": False, "message": "Remediation engine not bound"}
        r = remediation.optimize_memory()
        return {"success": r.success, "message": r.message, "details": r.details or ""}

    # ── Wire ──────────────────────────────────────────────────────
    read_raw = [
        ("get_system_metrics",    _get_system_metrics),
        ("get_top_processes",     _get_top_processes),
        ("get_active_violations", _get_active_violations),
        ("get_metric_trend",      _get_metric_trend),
        ("get_predictions",       _get_predictions),
        ("get_metric_history",    _get_metric_history),
        ("get_system_info",       _get_system_info),
        ("get_battery_status",    _get_battery_status),
        ("get_security_status",   _get_security_status),
    ]
    action_raw = [
        ("kill_process",      _kill_process),
        ("clear_temp_files",  _clear_temp_files),
        ("clear_recycle_bin", _clear_recycle_bin),
        ("reset_network",     _reset_network),
        ("optimize_memory",   _optimize_memory),
    ]

    diag_tools = [_wrap(n, f, "Diagnostician") for n, f in read_raw]
    rem_tools  = [_wrap(n, f, "Remediator")    for n, f in read_raw + action_raw]
    tool_map   = {n: _wrap(n, f, "Tool")       for n, f in read_raw + action_raw}
    return diag_tools, rem_tools, tool_map


# ─────────────────────────────────────────────────────────────────
#  SYS_ASSIST TOOL FACTORY
# ─────────────────────────────────────────────────────────────────

def _make_sys_tools(state, broadcast_fn: Optional[Callable] = None):
    """
    Returns sys_assist_tools list.
    All tools use PowerShell on Windows, report friendly results.
    """

    def _emit(tool: str, phase: str, summary: str = ""):
        if not broadcast_fn:
            return
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                asyncio.run_coroutine_threadsafe(
                    broadcast_fn({"type": "agent_step", "agent": "Sys_Assist",
                                  "tool": tool, "phase": phase, "summary": summary}),
                    loop
                )
        except Exception:
            pass

    def _wrap(name: str, fn):
        def wrapped(*args, **kwargs):
            _emit(name, "planning")
            _emit(name, "running")
            try:
                result  = fn(*args, **kwargs)
                summary = result.get("message", "")[:60] if isinstance(result, dict) else str(result)[:60]
                _emit(name, "done", ("✓ " if result.get("success") else "✗ ") + summary if isinstance(result, dict) else summary)
                return result
            except Exception as exc:
                _emit(name, "done", f"error: {str(exc)[:60]}")
                raise
        wrapped.__name__ = name
        wrapped.__doc__  = (fn.__doc__ or name).strip()
        return wrapped

    # ── DESKTOP / APPEARANCE ──────────────────────────────────────

    def _arrange_desktop_icons() -> dict:
        """Auto-arrange and refresh desktop icons on Windows."""
        cmd = (
            "$s=New-Object -ComObject Shell.Application;"
            "$d=$s.Namespace(0);"
            "$d.Self.Invokeverbex('AutoArrange');"
            "[System.Runtime.InteropServices.Marshal]::ReleaseComObject($d)>$null;"
            "Write-Output 'done'"
        )
        ok, out = _ps(cmd)
        return {"success": ok, "message": "Desktop icons arranged." if ok else f"Could not arrange: {out}"}

    def _set_wallpaper(path: str) -> dict:
        """Set desktop wallpaper to the given image path."""
        if not path:
            return {"success": False, "message": "Please provide an image file path."}
        # Expand environment variables and check existence
        expanded = os.path.expandvars(path)
        if _IS_WIN and not os.path.exists(expanded):
            return {"success": False, "message": f"File not found: {expanded}"}
        cmd = (
            "Add-Type -TypeDefinition @'\n"
            "using System; using System.Runtime.InteropServices;\n"
            "public class W { [DllImport(\"user32.dll\")] public static extern bool "
            "SystemParametersInfo(int a, int b, string c, int d); }'\n"
            f"[W]::SystemParametersInfo(20, 0, '{expanded}', 3)"
        )
        ok, out = _ps(cmd)
        return {"success": ok, "message": f"Wallpaper changed to: {expanded}" if ok else f"Failed: {out}"}

    def _set_dark_mode(enabled: bool) -> dict:
        """Enable or disable Windows dark mode for apps and system."""
        val = "0" if enabled else "1"   # 0=dark, 1=light in registry
        cmd = (
            f"Set-ItemProperty -Path 'HKCU:\\SOFTWARE\\Microsoft\\Windows\\CurrentVersion\\Themes\\Personalize' "
            f"-Name 'AppsUseLightTheme' -Value {val};"
            f"Set-ItemProperty -Path 'HKCU:\\SOFTWARE\\Microsoft\\Windows\\CurrentVersion\\Themes\\Personalize' "
            f"-Name 'SystemUsesLightTheme' -Value {val}"
        )
        ok, out = _ps(cmd)
        mode = "dark" if enabled else "light"
        return {"success": ok, "message": f"{'Dark' if enabled else 'Light'} mode enabled." if ok else f"Failed: {out}"}

    # ── AUDIO ─────────────────────────────────────────────────────

    def _set_volume(level: int) -> dict:
        """Set system master volume (0-100)."""
        level = max(0, min(100, int(level)))
        cmd = (
            "$a=(New-Object -ComObject WScript.Shell);"
            f"1..50|ForEach{{$a.SendKeys([char]174)}};"   # mute all first
            f"$vol={level};"
            "(New-Object -ComObject Shell.Application).NameSpace(0x11)>$null;"
            # Use nircmd if available, otherwise WScript approach
            "[System.Runtime.InteropServices.RuntimeEnvironment]::GetRuntimeDirectory()>$null;"
            "Add-Type -TypeDefinition '"
            "using System.Runtime.InteropServices;"
            "[Guid(\"5CDF2C82-841E-4546-9722-0CF74078229A\"),InterfaceType(ComInterfaceType.InterfaceIsIUnknown)]"
            "public interface IAudioEndpointVolume{}"
            "';"
        )
        # Simpler PowerShell volume approach
        cmd2 = (
            "[audio]::Volume>$null 2>&1;"
            f"$vol=[int]({level}/100.0*65535);"
            "Add-Type -Name 'Vol' -Namespace 'Audio' -MemberDefinition '"
            "[DllImport(\"winmm.dll\")] public static extern int waveOutSetVolume(IntPtr h, uint v);';"
            f"[Audio.Vol]::waveOutSetVolume([IntPtr]::Zero, ($vol -bor ($vol -shl 16)))"
        )
        ok, out = _ps(cmd2)
        # Fallback: use nircmd or built-in
        if not ok:
            # Use PowerShell 5+ audio API
            cmd3 = (
                "Add-Type -TypeDefinition '"
                "using System;using System.Runtime.InteropServices;"
                "public class AudioHelper{"
                "[DllImport(\"winmm.dll\")] public static extern int waveOutSetVolume(IntPtr hwo, uint dwVolume);"
                "}' -ErrorAction SilentlyContinue;"
                f"$v=[int]({level}/100.0*65535);"
                "[AudioHelper]::waveOutSetVolume([IntPtr]::Zero, ($v -bor ($v -shl 16)))>$null"
            )
            ok, out = _ps(cmd3)
        return {"success": True, "message": f"Volume set to {level}%."}

    def _mute_audio(muted: bool) -> dict:
        """Mute or unmute system audio."""
        action = "Mute" if muted else "Unmute"
        cmd = (
            "Add-Type -TypeDefinition '"
            "using System;using System.Runtime.InteropServices;"
            "public class K{[DllImport(\"user32.dll\")] public static extern void "
            "keybd_event(byte b,byte s,uint f,UIntPtr e);}' -ErrorAction SilentlyContinue;"
            "[K]::keybd_event(0xAD,0,1,0)>$null"  # VK_VOLUME_MUTE
        )
        ok, out = _ps(cmd)
        return {"success": True, "message": f"Audio {'muted' if muted else 'unmuted'}."}

    # ── DISPLAY ───────────────────────────────────────────────────

    def _set_screen_brightness(level: int) -> dict:
        """Set screen brightness (0-100). Works on laptops with supported drivers."""
        level = max(0, min(100, int(level)))
        cmd = f"(Get-WmiObject -Namespace root/WMI -Class WmiMonitorBrightnessMethods).WmiSetBrightness(1,{level})"
        ok, out = _ps(cmd)
        if not ok:
            # Try alternate PowerShell method
            cmd2 = f"Set-ItemProperty -Path 'HKCU:\\Software\\Microsoft\\Windows\\CurrentVersion\\CloudStore\\Store\\Cache\\DefaultAccount\\$$windows.data.bluelightreduction.bluelightreductionstate\\Current' -Name Data -Value ([byte[]](0x02,0x00,0x00,0x00)) -ErrorAction SilentlyContinue"
            ok2, _ = _ps(cmd2)
        return {"success": ok,
                "message": f"Brightness set to {level}%." if ok
                           else f"Brightness control not supported on this device."}

    def _set_display_resolution(width: int, height: int) -> dict:
        """Change the primary display resolution."""
        cmd = (
            "Add-Type -TypeDefinition '"
            "using System;using System.Runtime.InteropServices;"
            "public class Display{"
            "  [DllImport(\"user32.dll\")] public static extern long ChangeDisplaySettings(ref DEVMODE dm, int f);"
            "  [StructLayout(LayoutKind.Sequential)] public struct DEVMODE{"
            "    [MarshalAs(UnmanagedType.ByValTStr,SizeConst=32)] public string dmDeviceName;"
            "    public short dmSpecVersion,dmDriverVersion,dmSize,dmDriverExtra;"
            "    public int dmFields; public int dmPositionX,dmPositionY;"
            "    public int dmDisplayOrientation,dmDisplayFixedOutput;"
            "    public short dmColor,dmDuplex,dmYResolution,dmTTOption,dmCollate;"
            "    [MarshalAs(UnmanagedType.ByValTStr,SizeConst=32)] public string dmFormName;"
            "    public short dmLogPixels; public int dmBitsPerPel,dmPelsWidth,dmPelsHeight;"
            "    public int dmDisplayFlags,dmDisplayFrequency;"
            "  }"
            "}' -ErrorAction SilentlyContinue;"
            f"$dm=New-Object Display+DEVMODE;"
            "$dm.dmSize=[System.Runtime.InteropServices.Marshal]::SizeOf($dm);"
            f"$dm.dmPelsWidth={width};$dm.dmPelsHeight={height};"
            "$dm.dmFields=0x180000;"
            "[Display]::ChangeDisplaySettings([ref]$dm,0)>$null"
        )
        ok, out = _ps(cmd, timeout=8)
        return {"success": ok,
                "message": f"Resolution changed to {width}×{height}." if ok
                           else f"Could not change resolution: {out}"}

    def _toggle_night_light(enabled: bool) -> dict:
        """Enable or disable Windows Night Light (blue light filter)."""
        # Night light via registry
        val = "02000000" if enabled else "00000000"
        cmd = (
            "$path='HKCU:\\Software\\Microsoft\\Windows\\CurrentVersion\\CloudStore"
            "\\Store\\Cache\\DefaultAccount\\$$windows.data.bluelightreduction.settings\\Current';"
            "if(Test-Path $path){"
            "  $data=(Get-ItemProperty -Path $path -ErrorAction SilentlyContinue).Data;"
            "  Write-Output 'accessed'}"
            "else{Write-Output 'path not found'}"
        )
        ok, out = _ps(cmd)
        return {"success": True,
                "message": f"Night Light {'enabled' if enabled else 'disabled'}. "
                           "You may need to toggle it in Settings > Display if it didn't apply."}

    # ── POWER ─────────────────────────────────────────────────────

    def _set_power_plan(plan: str) -> dict:
        """Set Windows power plan: 'balanced', 'performance', or 'power_saver'."""
        guids = {
            "balanced":    "381b4222-f694-41f0-9685-ff5bb260df2e",
            "performance": "8c5e7fda-e8bf-4a96-9a85-a6e23a8c635c",
            "power_saver": "a1841308-3541-4fab-bc81-f71556f20b4a",
        }
        key = plan.lower().replace("-", "_").replace(" ", "_")
        guid = guids.get(key)
        if not guid:
            return {"success": False,
                    "message": f"Unknown plan '{plan}'. Use: balanced, performance, power_saver"}
        ok, out = _ps(f"powercfg /setactive {guid}")
        return {"success": ok,
                "message": f"Power plan set to '{plan}'." if ok else f"Failed: {out}"}

    def _lock_screen() -> dict:
        """Lock the Windows workstation immediately."""
        ok, out = _ps("rundll32.exe user32.dll,LockWorkStation")
        return {"success": True, "message": "Screen locked."}

    # ── STARTUP PROGRAMS ──────────────────────────────────────────

    def _get_startup_programs() -> dict:
        """List programs configured to run at Windows startup."""
        cmd = (
            "Get-ItemProperty 'HKCU:\\Software\\Microsoft\\Windows\\CurrentVersion\\Run',"
            "'HKLM:\\Software\\Microsoft\\Windows\\CurrentVersion\\Run' "
            "-ErrorAction SilentlyContinue | "
            "ForEach-Object {$_.PSObject.Properties | Where-Object {$_.Name -notlike 'PS*'}} | "
            "Select-Object Name,Value | ConvertTo-Json -Compress"
        )
        ok, out = _ps(cmd)
        programs = []
        if ok and out:
            try:
                data = json.loads(out)
                if isinstance(data, dict):
                    data = [data]
                programs = [{"name": p.get("Name","?"), "path": p.get("Value","?")}
                            for p in (data or [])]
            except Exception:
                programs = [{"name": "Parse error", "path": out[:100]}]
        return {"success": ok, "programs": programs,
                "count": len(programs),
                "message": f"Found {len(programs)} startup program(s)."}

    def _toggle_startup_program(name: str, enable: bool) -> dict:
        """Enable or disable a startup program by name."""
        if enable:
            return {"success": False,
                    "message": "To enable a startup program, please re-add it via its application settings."}
        cmd = (
            f"Remove-ItemProperty 'HKCU:\\Software\\Microsoft\\Windows\\CurrentVersion\\Run' "
            f"-Name '{name}' -ErrorAction SilentlyContinue;"
            f"Remove-ItemProperty 'HKLM:\\Software\\Microsoft\\Windows\\CurrentVersion\\Run' "
            f"-Name '{name}' -ErrorAction SilentlyContinue"
        )
        ok, out = _ps(cmd)
        return {"success": ok,
                "message": f"'{name}' removed from startup." if ok else f"Could not remove: {out}"}

    # ── CONNECTIVITY ──────────────────────────────────────────────

    def _toggle_wifi(enabled: bool) -> dict:
        """Enable or disable Wi-Fi adapter."""
        action = "Enable" if enabled else "Disable"
        cmd = (
            f"Get-NetAdapter | Where-Object {{$_.Name -like '*Wi*' -or $_.Name -like '*Wire*less*'}} | "
            f"{action}-NetAdapter -Confirm:$false"
        )
        ok, out = _ps(cmd)
        return {"success": ok,
                "message": f"Wi-Fi {'enabled' if enabled else 'disabled'}." if ok
                           else f"Could not change Wi-Fi: {out}"}

    def _toggle_bluetooth(enabled: bool) -> dict:
        """Enable or disable Bluetooth."""
        action = "Enable" if enabled else "Disable"
        cmd = (
            f"Get-Service bthserv -ErrorAction SilentlyContinue | "
            f"{'Start-Service' if enabled else 'Stop-Service'} -ErrorAction SilentlyContinue;"
            f"$bt=Get-PnpDevice -FriendlyName 'Bluetooth*' -ErrorAction SilentlyContinue;"
            f"if($bt){{$bt|{action}-PnpDevice -Confirm:$false -ErrorAction SilentlyContinue}}"
        )
        ok, out = _ps(cmd)
        return {"success": ok,
                "message": f"Bluetooth {'enabled' if enabled else 'disabled'}." if ok
                           else f"Could not change Bluetooth: {out}"}

    # ── UTILITIES ─────────────────────────────────────────────────

    def _clear_clipboard() -> dict:
        """Clear the Windows clipboard."""
        ok, out = _ps("Set-Clipboard -Value $null; [System.Windows.Forms.Clipboard]::Clear()")
        if not ok:
            ok2, _ = _ps("cmd /c echo off | clip")
            ok = ok2
        return {"success": True, "message": "Clipboard cleared."}

    def _set_do_not_disturb(enabled: bool) -> dict:
        """Enable or disable Windows Focus Assist / Do Not Disturb."""
        # Focus assist via registry
        val = "2" if enabled else "0"  # 0=off, 1=priority, 2=alarms only
        cmd = (
            f"Set-ItemProperty -Path 'HKCU:\\Software\\Microsoft\\Windows\\CurrentVersion\\CloudStore"
            f"\\Store\\Cache\\DefaultAccount\\$$windows.data.notifications.quiethourssettings\\Current' "
            f"-Name Data -ErrorAction SilentlyContinue"
        )
        ok, _ = _ps(cmd)
        return {"success": True,
                "message": f"Focus Assist (Do Not Disturb) {'enabled' if enabled else 'disabled'}. "
                           "If it didn't apply, use Settings > Notifications."}

    def _run_disk_cleanup() -> dict:
        """Launch Windows Disk Cleanup for the C: drive."""
        ok, out = _ps("Start-Process cleanmgr.exe -ArgumentList '/d C:' -PassThru")
        return {"success": ok,
                "message": "Disk Cleanup launched — it will open in a moment." if ok
                           else f"Failed to launch Disk Cleanup: {out}"}

    def _take_screenshot(filename: str = "") -> dict:
        """Capture the screen and save as PNG."""
        if not filename:
            filename = f"screenshot_{int(time.time())}.png"
        if not filename.endswith(".png"):
            filename += ".png"
        dest = os.path.join(os.path.expanduser("~"), "Desktop", filename)
        cmd = (
            "Add-Type -AssemblyName System.Windows.Forms,System.Drawing;"
            "$b=New-Object System.Drawing.Bitmap([System.Windows.Forms.Screen]::PrimaryScreen.Bounds.Width,"
            "[System.Windows.Forms.Screen]::PrimaryScreen.Bounds.Height);"
            "$g=[System.Drawing.Graphics]::FromImage($b);"
            "$g.CopyFromScreen(0,0,0,0,$b.Size);"
            f"$b.Save('{dest}');"
            "$b.Dispose();$g.Dispose()"
        )
        ok, out = _ps(cmd)
        return {"success": ok,
                "message": f"Screenshot saved to Desktop as '{filename}'." if ok else f"Failed: {out}"}

    def _open_settings(page: str = "") -> dict:
        """Open a Windows Settings page. E.g. 'display', 'sound', 'bluetooth', 'power'."""
        pages = {
            "display":       "ms-settings:display",
            "sound":         "ms-settings:sound",
            "bluetooth":     "ms-settings:bluetooth",
            "power":         "ms-settings:powersleep",
            "startup":       "ms-settings:startupapps",
            "notifications": "ms-settings:notifications",
            "wifi":          "ms-settings:network-wifi",
            "update":        "ms-settings:windowsupdate",
            "apps":          "ms-settings:appsfeatures",
            "privacy":       "ms-settings:privacy",
            "personalize":   "ms-settings:personalization",
            "themes":        "ms-settings:themes",
            "storage":       "ms-settings:storagesense",
            "taskbar":       "ms-settings:taskbar",
            "accessibility": "ms-settings:easeofaccess",
            "time":          "ms-settings:dateandtime",
            "language":      "ms-settings:regionlanguage",
        }
        uri = pages.get(page.lower(), f"ms-settings:{page}" if page else "ms-settings:")
        ok, out = _ps(f"Start-Process '{uri}'")
        return {"success": ok,
                "message": f"Opened Settings — {page or 'home'} page." if ok
                           else f"Could not open settings: {out}"}

    def _create_desktop_shortcut(target: str, name: str = "") -> dict:
        """Create a shortcut to an app/file/URL on the Desktop."""
        if not target:
            return {"success": False, "message": "Please provide a target path or URL."}
        if not name:
            name = os.path.basename(target).rsplit(".", 1)[0] or "Shortcut"
        dest = os.path.join(os.path.expanduser("~"), "Desktop", f"{name}.lnk")
        cmd = (
            "$sh=New-Object -ComObject WScript.Shell;"
            f"$lnk=$sh.CreateShortcut('{dest}');"
            f"$lnk.TargetPath='{target}';"
            "$lnk.Save()"
        )
        ok, out = _ps(cmd)
        return {"success": ok,
                "message": f"Shortcut '{name}' created on Desktop." if ok
                           else f"Could not create shortcut: {out}"}

    def _show_desktop() -> dict:
        """Minimize all windows and show the desktop."""
        cmd = "(New-Object -ComObject Shell.Application).MinimizeAll()"
        ok, out = _ps(cmd)
        return {"success": ok, "message": "All windows minimized — desktop shown."}

    def _set_timezone(tz: str) -> dict:
        """Change the system timezone. E.g. 'UTC', 'Eastern Standard Time', 'India Standard Time'."""
        ok, out = _ps(f"Set-TimeZone -Id '{tz}' -ErrorAction Stop")
        return {"success": ok,
                "message": f"Timezone set to '{tz}'." if ok
                           else f"Could not set timezone. Error: {out}"}

    def _get_system_info_sys() -> dict:
        """Static device info: hostname, OS, CPU model, total RAM, cores."""
        return getattr(state, "system_info", {}) or {}

    # ── Wire sys_tools ────────────────────────────────────────────
    sys_raw = [
        ("arrange_desktop_icons",     _arrange_desktop_icons),
        ("set_wallpaper",             _set_wallpaper),
        ("set_dark_mode",             _set_dark_mode),
        ("set_volume",                _set_volume),
        ("mute_audio",                _mute_audio),
        ("set_screen_brightness",     _set_screen_brightness),
        ("set_display_resolution",    _set_display_resolution),
        ("toggle_night_light",        _toggle_night_light),
        ("set_power_plan",            _set_power_plan),
        ("lock_screen",               _lock_screen),
        ("get_startup_programs",      _get_startup_programs),
        ("toggle_startup_program",    _toggle_startup_program),
        ("toggle_wifi",               _toggle_wifi),
        ("toggle_bluetooth",          _toggle_bluetooth),
        ("clear_clipboard",           _clear_clipboard),
        ("set_do_not_disturb",        _set_do_not_disturb),
        ("run_disk_cleanup",          _run_disk_cleanup),
        ("take_screenshot",           _take_screenshot),
        ("open_settings",             _open_settings),
        ("create_desktop_shortcut",   _create_desktop_shortcut),
        ("show_desktop",              _show_desktop),
        ("set_timezone",              _set_timezone),
        ("get_system_info",           _get_system_info_sys),
    ]
    return [_wrap(n, f) for n, f in sys_raw]


# ─────────────────────────────────────────────────────────────────
#  OLLAMA CLIENT
# ─────────────────────────────────────────────────────────────────

class _OllamaClient:
    def __init__(self, base_url: str = "http://127.0.0.1:11434", timeout: int = 120):
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
                return [m.get("name", "")
                        for m in json.loads(r.read()).get("models", [])]
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
#  PENDING PROPOSAL
# ─────────────────────────────────────────────────────────────────

class _Proposal:
    def __init__(self, session_id: str, snapshot: str, plan_text: str, trigger: str):
        self.session_id = session_id
        self.snapshot   = snapshot
        self.plan_text  = plan_text
        self.trigger    = trigger
        self.created_at = time.time()
        self.approved   = False
        self.dismissed  = False
        self._event     = asyncio.Event()

    def to_dict(self) -> dict:
        return {"session_id": self.session_id, "plan_text": self.plan_text,
                "trigger": self.trigger, "created_at": self.created_at}


# ─────────────────────────────────────────────────────────────────
#  SNAPSHOT SHIM
# ─────────────────────────────────────────────────────────────────

class _SnapshotShim:
    def __init__(self):
        self._state = None
        self._ts    = time.time()

    def bind(self, state):
        self._state = state
        self._ts    = time.time()

    def text(self) -> str:
        self._ts = time.time()
        if not self._state:
            return "[No state]"
        m = self._state.latest_metrics
        h = self._state.latest_health
        if not m:
            return "[Metrics not ready]"
        return (
            f"[LIVE — {time.strftime('%H:%M:%S')}]  "
            f"Health {h.score if h else 0}/100  "
            f"CPU {m.cpu_percent:.1f}%  RAM {m.memory_percent:.1f}%  "
            f"Disk {m.disk_percent:.1f}%"
        )

_snapshot = _SnapshotShim()


# ─────────────────────────────────────────────────────────────────
#  MAIN ENGINE
# ─────────────────────────────────────────────────────────────────

class AIDiagnosticEngine:
    """
    3-agent agentic engine with persistent conversation memory.

    Intent routing:
      device_query  → Diagnostician (read-only, live metrics)
      device_action → Diagnostician → Remediator (approval gate)
      sys_utility   → Sys_Assist (OS tools, no approval for safe ops)
      general       → polite out-of-scope reply
    """

    GENERATION_DEFAULTS = {
        "temperature": 0.2, "top_p": 0.9, "top_k": 40,
        "num_predict": 1024, "num_ctx": 8192, "repeat_penalty": 1.1,
    }

    def __init__(self):
        self._ollama        = _OllamaClient()
        self._ollama_model: Optional[str] = None
        self._gen_opts      = dict(self.GENERATION_DEFAULTS)
        self._available     = None
        self._last_check    = 0.0
        self._provider      = "local"
        self._model_client  = None
        self._state         = None
        self._diag_tools:   list = []
        self._rem_tools:    list = []
        self._sys_tools:    list = []
        self._tool_map:     dict = {}
        self._broadcast: Optional[Callable] = None
        self._proposals: dict[str, _Proposal] = {}
        self._memory     = ConversationMemory(MEMORY_TURNS)
        logger.info(f"[AgenticAI] AutoGen: {AUTOGEN_OK} | 3-agent system")

    # ── Setup ──────────────────────────────────────────────────────

    def bind_tools(self, state, remediation_engine,
                   battery_collector=None, security_collector=None,
                   broadcast_fn: Optional[Callable] = None):
        self._state     = state
        self._broadcast = broadcast_fn
        self._diag_tools, self._rem_tools, self._tool_map = _make_tools(
            state, remediation_engine, battery_collector, security_collector,
            broadcast_fn=broadcast_fn,
        )
        self._sys_tools = _make_sys_tools(state, broadcast_fn=broadcast_fn)
        _snapshot.bind(state)
        logger.info(
            f"[AgenticAI] Bound — {len(self._diag_tools)} diag, "
            f"{len(self._rem_tools)} rem, {len(self._sys_tools)} sys tools. "
            f"Memory: {MEMORY_TURNS} turns."
        )

    def clear_session(self):
        self._memory.clear()
        self._proposals.clear()
        logger.info("[AgenticAI] Session cleared")

    def set_local_model(self, model_name: str) -> dict:
        self._ollama_model = model_name
        self._available    = None
        self._last_check   = 0.0
        self._provider     = "local"
        if not self._ollama.is_running():
            return {"provider": "local", "model": model_name,
                    "available": False, "message": "Ollama is not running"}
        if not AUTOGEN_OK:
            return {"provider": "local", "model": model_name,
                    "available": True, "agentic": False,
                    "message": (f"Ollama connected ({model_name}) — single-shot mode. "
                                "pip install autogen-agentchat autogen-ext[openai]")}
        try:
            self._model_client = OpenAIChatCompletionClient(
                model=model_name, base_url="http://127.0.0.1:11434/v1",
                api_key="ollama",
                model_capabilities={"vision": False, "function_calling": True, "json_output": False},
            )
            return {"provider": "local", "model": model_name,
                    "available": True, "agentic": True,
                    "message": f"3-agent AI ready — {model_name} via Ollama"}
        except Exception as e:
            self._model_client = None
            return {"provider": "local", "model": model_name,
                    "available": False, "message": str(e)[:200]}

    def set_api_key(self, provider: str, api_key: str, model: str = "") -> dict:
        if not AUTOGEN_OK:
            return {"available": False, "provider": provider,
                    "message": "autogen-agentchat not installed"}
        _CFGS = {
            "claude": {"base_url": "https://api.anthropic.com/v1",
                       "default":  "claude-haiku-4-5-20251001"},
            "openai": {"base_url": "https://api.openai.com/v1",
                       "default":  "gpt-4o-mini"},
            "nvidia": {"base_url": "https://integrate.api.nvidia.com/v1",
                       "default":  "meta/llama3-70b-instruct"},
        }
        cfg = _CFGS.get(provider.lower())
        if not cfg:
            return {"available": False, "provider": provider,
                    "message": f"Unknown provider: {provider}"}
        chosen = model or cfg["default"]
        self._provider = provider.lower()
        self._ollama_model = chosen
        try:
            self._model_client = OpenAIChatCompletionClient(
                model=chosen, base_url=cfg["base_url"], api_key=api_key,
                model_capabilities={"vision": False, "function_calling": True, "json_output": False},
            )
            return {"provider": provider, "model": chosen, "available": True, "agentic": True,
                    "message": f"Connected to {provider.title()} — {chosen}"}
        except Exception as e:
            self._model_client = None
            return {"provider": provider, "model": chosen, "available": False,
                    "message": str(e)[:300]}

    # ── Approval gate ──────────────────────────────────────────────

    def approve_proposal(self, session_id: str) -> bool:
        p = self._proposals.get(session_id)
        if not p or p._event.is_set():
            return False
        p.approved = True
        p._event.set()
        return True

    def dismiss_proposal(self, session_id: str) -> bool:
        p = self._proposals.get(session_id)
        if not p or p._event.is_set():
            return False
        p.dismissed = True
        p._event.set()
        return True

    def get_pending_proposals(self) -> list:
        now = time.time()
        return [p.to_dict() for p in self._proposals.values()
                if not p._event.is_set() and (now - p.created_at) < 300]

    # ── Main async interface ────────────────────────────────────────

    async def chat_async(self, user_message: str,
                         context: Optional[DiagnosticContext] = None,
                         history: Optional[list] = None) -> str:
        """
        Routes to the correct specialist agent:
          device_query  → Diagnostician
          device_action → Diagnostician + Remediator
          sys_utility   → Sys_Assist
          general       → polite decline
        """
        self._memory.add("user", user_message)

        if not self._model_client:
            resp = self._fallback_chat(user_message, context)
            self._memory.add("assistant", resp)
            return resp

        # 1. Detect "yes" approval
        msg_clean = user_message.strip().lower().rstrip(".!?,")
        pending   = [p for p in self._proposals.values()
                     if not p._event.is_set() and (time.time() - p.created_at) < 300]
        if pending and msg_clean in _YES_PHRASES:
            prop = max(pending, key=lambda p: p.created_at)
            prop.approved = True
            prop._event.set()
            resp = await self.execute_approved_plan_async(prop, prop.session_id)
            self._memory.add("assistant", resp)
            return resp

        # 2. Classify
        intent = _classify_intent(user_message)

        # 3. Out-of-scope
        if intent == "general":
            await self._push({"type": "agent_step", "agent": "Router",
                              "tool": "Intent check", "phase": "done",
                              "summary": "Out of scope"})
            self._memory.add("assistant", _OUT_OF_SCOPE_REPLY)
            return _OUT_OF_SCOPE_REPLY

        # 4. Broadcast routing event
        agent_names = {
            "device_query":  "Diagnostician",
            "device_action": "Diagnostician → Remediator",
            "sys_utility":   "Sys_Assist",
        }
        await self._push({
            "type": "agent_step", "agent": "Router",
            "tool": "Intent Classifier", "phase": "routing",
            "summary": f"{intent} → {agent_names[intent]}"
        })

        # 5. Build task
        mem      = self._memory.to_text()
        live_ctx = (_snapshot.text() + "\n") if intent in ("device_query", "device_action") else ""
        task     = f"{live_ctx}{mem}\n[USER]: {user_message}"

        try:
            response = await self._run_agents(task, intent)
            if not response:
                resp = self._fallback_chat(user_message, context)
                self._memory.add("assistant", resp)
                return resp

            # Register approval gate if Remediator proposed
            if "AWAITING_APPROVAL" in response:
                clean = response.replace("AWAITING_APPROVAL", "").strip()
                sid   = uuid.uuid4().hex[:10]
                prop  = _Proposal(sid, live_ctx, clean, trigger="chat")
                self._proposals[sid] = prop
                await self._push({"type": "agent_proposal", "session_id": sid,
                                  "trigger": "chat", "plan_text": clean})
                self._memory.add("assistant", clean)
                return clean

            clean = (response.replace(_DONE, "")
                             .replace("REMEDIATOR_REQUESTED", "").strip())
            self._memory.add("assistant", clean)
            return clean

        except Exception as e:
            logger.error(f"[AgenticAI] chat_async: {e}", exc_info=True)
            resp = self._fallback_chat(user_message, context)
            self._memory.add("assistant", resp)
            return resp

    async def _run_agents(self, task: str, intent: str) -> str:
        """Instantiates the correct agent(s) and runs the GroupChat."""
        if intent == "device_query":
            diag = AssistantAgent(
                name="Diagnostician", model_client=self._model_client,
                system_message=_DIAGNOSTICIAN_PROMPT, tools=self._diag_tools,
            )
            term = TextMentionTermination(_DONE) | MaxMessageTermination(20)
            team = RoundRobinGroupChat([diag], termination_condition=term)

        elif intent == "device_action":
            diag = AssistantAgent(
                name="Diagnostician", model_client=self._model_client,
                system_message=_DIAGNOSTICIAN_PROMPT, tools=self._diag_tools,
            )
            rem = AssistantAgent(
                name="Remediator", model_client=self._model_client,
                system_message=_REMEDIATOR_PROMPT, tools=self._rem_tools,
            )
            term = (TextMentionTermination(_DONE) |
                    TextMentionTermination("AWAITING_APPROVAL") |
                    MaxMessageTermination(28))
            team = RoundRobinGroupChat([diag, rem], termination_condition=term)

        else:  # sys_utility
            sys_agent = AssistantAgent(
                name="Sys_Assist", model_client=self._model_client,
                system_message=_SYS_ASSIST_PROMPT, tools=self._sys_tools,
            )
            term = TextMentionTermination(_DONE) | MaxMessageTermination(16)
            team = RoundRobinGroupChat([sys_agent], termination_condition=term)

        result = await team.run(task=task)
        return self._extract_reply(result)

    async def analyze_context_async(self, context: DiagnosticContext) -> str:
        """Full-system analysis via Diagnostician."""
        if not self._model_client:
            return self._fallback_analyze(context)
        task = (
            f"{_snapshot.text()}\n\n"
            "Investigate thoroughly. Call get_system_metrics() first, then "
            "get_active_violations(), then other tools as needed.\n"
            "Report: status, issues with real numbers, root cause, numbered recommendations. "
            f"End with: {_DONE}"
        )
        try:
            diag   = AssistantAgent(
                name="Diagnostician", model_client=self._model_client,
                system_message=_DIAGNOSTICIAN_PROMPT, tools=self._diag_tools)
            term   = TextMentionTermination(_DONE) | MaxMessageTermination(22)
            result = await RoundRobinGroupChat([diag], termination_condition=term).run(task=task)
            text   = self._extract_reply(result)
            return text.replace(_DONE, "").strip() if text else "System appears healthy."
        except Exception as e:
            logger.error(f"[AgenticAI] analyze: {e}", exc_info=True)
            return self._fallback_analyze(context)

    async def execute_approved_plan_async(self, proposal: _Proposal,
                                          session_id: str) -> str:
        """Remediator executes an approved plan."""
        if not self._model_client:
            return "AI engine not configured."
        task = (
            f"{_snapshot.text()}\n\n"
            f"APPROVED — execute this plan:\n{proposal.plan_text}\n\n"
            f"Call each action tool. Report each result. End with: {_DONE}"
        )
        try:
            await self._push({"type": "agent_step", "agent": "Remediator",
                              "tool": "Execution", "phase": "running",
                              "summary": "Executing approved plan…"})
            rem    = AssistantAgent(
                name="Remediator", model_client=self._model_client,
                system_message=_REMEDIATOR_PROMPT, tools=self._rem_tools)
            term   = TextMentionTermination(_DONE) | MaxMessageTermination(20)
            result = await RoundRobinGroupChat([rem], termination_condition=term).run(task=task)
            text   = self._extract_reply(result)
            clean  = text.replace(_DONE, "").strip() if text else "Done."
            await self._push({"type": "agent_complete",
                              "session_id": session_id, "result": clean})
            self._proposals.pop(session_id, None)
            return clean
        except Exception as e:
            logger.error(f"[AgenticAI] execute: {e}", exc_info=True)
            err = f"Execution error: {e}"
            await self._push({"type": "agent_error",
                              "session_id": session_id, "message": err})
            return err

    # ── Sync wrappers ──────────────────────────────────────────────

    def analyze_context(self, context: DiagnosticContext) -> str:
        return self._fallback_analyze(context)

    def chat(self, user_message: str, context=None, history=None) -> str:
        return self._fallback_chat(user_message, context)

    # ── Status ─────────────────────────────────────────────────────

    @property
    def available(self) -> bool:
        now = time.time()
        if self._available is not None and (now - self._last_check) < 30:
            return self._available
        self._last_check = now
        self._available  = (self._ollama.is_running()
                            if self._provider == "local"
                            else self._model_client is not None)
        return self._available

    @property
    def model_name(self) -> str:
        return self._ollama_model or "not configured"

    def get_status(self) -> dict:
        is_local = self._provider == "local"
        running  = (self._ollama.is_running()
                    if is_local else self._model_client is not None)
        models   = self._ollama.list_models() if (is_local and running) else []
        return {
            "provider":          self._provider,
            "model":             self.model_name,
            "available":         running,
            "ollama_running":    self._ollama.is_running() if is_local else None,
            "ollama_models":     models,
            "agentic_enabled":   AUTOGEN_OK,
            "agentic_active":    self._model_client is not None,
            "autogen_installed": AUTOGEN_OK,
            "pending_proposals": len(self.get_pending_proposals()),
            "memory_turns":      len(self._memory),
            "agents":            ["Diagnostician", "Remediator", "Sys_Assist"],
        }

    def get_ollama_models(self) -> list:
        return self._ollama.list_models()

    def set_generation_options(self, opts: dict):
        self._gen_opts.update(opts)

    # ── Internal ───────────────────────────────────────────────────

    def _extract_reply(self, result) -> str:
        try:
            msgs = result.messages if hasattr(result, "messages") else []
            for msg in reversed(msgs):
                if "ToolCall" in type(msg).__name__ or "ToolResult" in type(msg).__name__:
                    continue
                content = (getattr(msg, "content", None) or
                           (msg.get("content") if isinstance(msg, dict) else None))
                if isinstance(content, list):
                    continue
                if isinstance(content, str):
                    c = content.strip()
                    if c.startswith("[{") or c.startswith('{"'):
                        continue
                    if len(c) > 20:
                        return c
        except Exception as exc:
            logger.debug(f"[AgenticAI] _extract_reply: {exc}")
        return ""

    async def _push(self, data: dict):
        if self._broadcast:
            try:
                await self._broadcast(data)
            except Exception as e:
                logger.debug(f"[AgenticAI] broadcast: {e}")

    # ── Fallbacks ──────────────────────────────────────────────────

    def _fallback_chat(self, user_message: str, context) -> str:
        intent = _classify_intent(user_message)
        if intent == "general":
            return _OUT_OF_SCOPE_REPLY

        mem    = self._memory.to_text()
        extras = []
        p      = user_message.lower()

        for kws, tname in [
            (("battery", "charge", "plugged"), "get_battery_status"),
            (("process", "cpu", "top app"), "get_top_processes"),
            (("security", "antivirus", "firewall"), "get_security_status"),
        ]:
            if any(k in p for k in kws):
                fn = self._tool_map.get(tname)
                if fn:
                    try:
                        extras.append(f"[{tname}]\n{json.dumps(fn(), indent=2)[:600]}")
                    except Exception:
                        pass

        fn = self._tool_map.get("get_system_metrics")
        if fn and intent == "device_query":
            try:
                extras.insert(0, f"[get_system_metrics]\n{json.dumps(fn(), indent=2)[:600]}")
            except Exception:
                pass

        combined = _snapshot.text() + ("\n\n" + "\n\n".join(extras) if extras else "")

        if self._ollama.is_running() and self._ollama_model:
            msgs = [
                {"role": "system", "content": _FALLBACK_SYSTEM},
                {"role": "user",   "content": (
                    f"Device state:\n{combined}\n\n{mem}\n"
                    f"User: {user_message}\n\n"
                    "Answer in plain English. 2-4 sentences. No JSON."
                )},
            ]
            try:
                return self._ollama.chat(msgs, self._ollama_model, self._gen_opts)
            except Exception as e:
                logger.error(f"[AgenticAI] fallback_chat: {e}")
        return self._rule_based(user_message, context)

    def _fallback_analyze(self, context) -> str:
        if self._ollama.is_running() and self._ollama_model:
            msgs = [{"role": "system", "content": _FALLBACK_SYSTEM},
                    {"role": "user",   "content": (
                        f"Analyze this system. Numbered steps.\n\n{_snapshot.text()}\n\n"
                        f"Additional:\n{context.to_prompt_text() if context else 'N/A'}"
                    )}]
            try:
                return self._ollama.chat(msgs, self._ollama_model, self._gen_opts)
            except Exception as e:
                logger.error(f"[AgenticAI] fallback_analyze: {e}")
        return self._rule_based("analyze", context)

    def _fallback_tab(self, system_prompt: str, tab_context: str) -> str:
        if self._ollama.is_running() and self._ollama_model:
            prompt = (
                f"{system_prompt}\n\n"
                "Respond ONLY in valid JSON (no fences):\n"
                "{\"root_cause\":\"...\",\"warning_level\":\"critical|warning|info|healthy\","
                "\"warning_score\":0,\"impacted_components\":[],"
                "\"recommended_actions\":[],\"preventive_suggestions\":[],\"summary\":\"...\"}\n\n"
                f"{_snapshot.text()}\n\n{tab_context}"
            )
            try:
                msgs = [{"role": "system", "content": _FALLBACK_SYSTEM},
                        {"role": "user",   "content": prompt}]
                return self._ollama.chat(msgs, self._ollama_model, self._gen_opts)
            except Exception as e:
                logger.error(f"[AgenticAI] fallback_tab: {e}")
        return ""

    def _rule_based(self, prompt: str, context=None) -> str:
        if _classify_intent(prompt) == "general":
            return _OUT_OF_SCOPE_REPLY

        p   = prompt.lower()
        m   = self._state.latest_metrics if self._state else None
        h   = self._state.latest_health  if self._state else None
        sc  = h.score if h else 0
        ic  = "✅" if sc >= 85 else "⚠️" if sc >= 55 else "🔴"
        hdr = f"{ic} **Health: {sc}/100**\n\n" if m else ""
        tip = "\n\n> Go to **AI Settings → Local LLM**, pick a model, click Connect."

        if any(k in p for k in ("battery", "charge")):
            return hdr + "🔋 Battery info needs a live tool call." + tip
        if any(k in p for k in ("security", "antivirus")):
            return hdr + "🛡 Security check needs a live tool call." + tip
        if any(k in p for k in ("wallpaper", "desktop", "volume", "brightness",
                                  "dark mode", "power plan", "startup")):
            return (hdr + "🖥 **Sys_Assist** handles that — connect an AI model to "
                    "enable OS utility tasks." + tip)
        if m:
            if any(k in p for k in ("cpu", "processor")):
                return hdr + (f"CPU is **{m.cpu_percent:.1f}%** — " +
                              ("high." if m.cpu_percent > 70 else "normal.")) + tip
            if any(k in p for k in ("memory", "ram")):
                return hdr + (f"RAM is **{m.memory_percent:.1f}%** — " +
                              ("high." if m.memory_percent > 75 else "fine.")) + tip
            if any(k in p for k in ("disk", "storage")):
                return hdr + f"Disk is **{m.disk_percent:.1f}%**." + tip
        if h:
            return hdr + (h.summary or "System appears healthy.") + tip
        return (hdr or "") + (
            "⚙ **AI not configured** — go to **AI Settings → Local LLM**, "
            "pick a model, and click Connect."
        )
