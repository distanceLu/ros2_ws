#!/usr/bin/env python3
"""多相机 ACT 推理控制页：填完参数后再启动 infer，避免 tmux 一打开就自动跑。"""

from __future__ import annotations

import argparse
import html
import json
import math
import os
import shutil
import subprocess
import sys
import threading
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any
from urllib.parse import parse_qs

ACT_ROOT = os.environ.get("ACT_ROOT", "/home/shugen/yanjie/act")
INFER_MODE = os.environ.get("INFER_MODE", "run")
OBSERVATION_ZMQ = os.environ.get("INFER_OBSERVATION_ZMQ", "tcp://127.0.0.1:5554")
TARGET_ZMQ = os.environ.get("INFER_TARGET_ZMQ", "tcp://127.0.0.1:5555")

DEFAULTS = {
    "task_name": os.environ.get("BRUSH_TASK_NAME", "brush_tool_pose_2026_06_15"),
    "ckpt_dir": os.environ.get(
        "BRUSH_CKPT_DIR",
        "/media/shugen/LcxDisk/checkpoint/brush_policy_3cam_2026_07_21_task_discrete_b1",
    ),
    "max_timesteps": os.environ.get("INFER_MAX_TIMESTEPS", "150"),
    "target_delta_gain": os.environ.get("INFER_TARGET_DELTA_GAIN", "1"),
    "chunk_size": os.environ.get("INFER_CHUNK_SIZE", "50"),
    "sleep_sec": os.environ.get("INFER_SLEEP_SEC", "0.18"),
    "max_amplified_step_m": os.environ.get("INFER_MAX_AMPLIFIED_STEP", "0.01"),
    "observation_timeout_sec": os.environ.get("INFER_OBSERVATION_TIMEOUT_SEC", "20"),
    "record_targets_csv": os.environ.get("INFER_RECORD_CSV", "/tmp/infer_3cam.csv"),
    "device": os.environ.get("INFER_DEVICE", "cuda"),
    "task_id": os.environ.get("INFER_TASK_ID", "0"),
    "discrete_decode": os.environ.get("INFER_DISCRETE_DECODE", "argmax"),
    "discrete_temperature": os.environ.get("INFER_DISCRETE_TEMPERATURE", "1.0"),
    # 0=不给网络喂 TCP；1=喂当前 TCP。默认 0。
    "use_qpos": os.environ.get("USE_QPOS", "0"),
}

_lock = threading.Lock()
_infer_proc: subprocess.Popen[str] | None = None
_last_command = ""
_last_error = ""


def _mode_label() -> str:
    return "dry-run（机械臂不动）" if INFER_MODE == "dry-run" else "正式执行（机械臂会动）"


def _is_running() -> bool:
    with _lock:
        return _infer_proc is not None and _infer_proc.poll() is None


def _build_infer_command(params: dict[str, str]) -> list[str]:
    cmd = [
        "bash",
        os.path.join(ACT_ROOT, "scripts/run_infer_brush_robot.sh"),
        "--task_name",
        params["task_name"],
        "--io_backend",
        "zmq",
        "--observation_zmq",
        OBSERVATION_ZMQ,
        "--target_zmq",
        TARGET_ZMQ,
        "--max_timesteps",
        params["max_timesteps"],
        "--device",
        params["device"],
        "--capture_scan",
        "--observation_timeout_sec",
        params["observation_timeout_sec"],
        "--debug_chunk",
        "--target_delta_gain",
        params["target_delta_gain"],
        "--max_amplified_step_m",
        params["max_amplified_step_m"],
        "--record_targets_csv",
        params["record_targets_csv"],
        "--task_id",
        params["task_id"],
        "--chunk_size",
        params["chunk_size"],
        "--sleep_sec",
        params["sleep_sec"],
        "--discrete_decode",
        params["discrete_decode"],
        "--discrete_temperature",
        params["discrete_temperature"],
    ]
    if str(params.get("use_qpos", "0")).strip() == "1":
        cmd.append("--use_qpos")
    else:
        cmd.append("--no_qpos")
    return cmd


def _start_infer(params: dict[str, str]) -> tuple[bool, str]:
    global _infer_proc, _last_command

    ckpt_dir = params["ckpt_dir"].strip()
    if not ckpt_dir:
        return False, "权重目录不能为空"
    if not os.path.isdir(ckpt_dir):
        return False, f"权重目录不存在: {ckpt_dir}"
    if params["discrete_decode"] not in {"argmax", "sample", "expectation"}:
        return False, f"无效离散解码方式: {params['discrete_decode']}"
    try:
        temperature = float(params["discrete_temperature"])
    except ValueError:
        return False, "离散温度必须是数字"
    if not math.isfinite(temperature) or temperature <= 0:
        return False, "离散温度必须是有限正数"

    with _lock:
        if _infer_proc is not None and _infer_proc.poll() is None:
            return False, "推理正在运行，请等当前任务结束后再启动"
        _last_error = ""

        env = os.environ.copy()
        env["BRUSH_CKPT_DIR"] = ckpt_dir
        env["BRUSH_TASK_NAME"] = params["task_name"]
        env["PAPER_ARUCO_COLLECT"] = "0"
        env["INFER_CHUNK_SIZE"] = params["chunk_size"]

        cmd = _build_infer_command(params)
        _last_command = " ".join(cmd)

        print("\n[infer_control_panel] 启动推理:")
        print(f"  CKPT_DIR={ckpt_dir}")
        print(f"  命令: {_last_command}\n", flush=True)

        _infer_proc = subprocess.Popen(
            cmd,
            cwd=ACT_ROOT,
            env=env,
        )
        threading.Thread(target=_watch_infer_proc, daemon=True).start()

    return True, "推理已启动，输出在本 infer 窗口；完成后可再次填写参数并启动"


def _watch_infer_proc() -> None:
    global _infer_proc, _last_error
    with _lock:
        proc = _infer_proc
    if proc is None:
        return
    rc = proc.wait()
    if rc != 0:
        msg = f"推理进程异常退出 (exit={rc})，请查看本 infer 窗口日志"
        with _lock:
            _last_error = msg
        print(f"[infer_control_panel] {msg}", flush=True)


def _render_page(message: str = "", error: bool = False) -> bytes:
    running = _is_running()
    status_text = "运行中" if running else "待命"
    status_class = "running" if running else "idle"
    alert = ""
    if not message and _last_error and not running:
        message = _last_error
        error = True
    if message:
        cls = "error" if error else "success"
        alert = f'<div class="alert {cls}">{html.escape(message)}</div>'

    fields = []
    labels = {
        "task_name": ("ACT 任务配置名", "text"),
        "ckpt_dir": ("权重目录", "text"),
        "task_id": ("轮廓任务 id (task_id)", "number"),
        "max_timesteps": ("推理步数", "number"),
        "chunk_size": ("chunk_size（须与训练一致）", "number"),
        "sleep_sec": ("每步间隔 sleep_sec (s)", "number"),
        "discrete_decode": ("离散解码方式 (argmax/sample/expectation)", "text"),
        "discrete_temperature": ("离散解码温度 (>0)", "number"),
        "target_delta_gain": ("位移放大倍数", "number"),
        "max_amplified_step_m": ("单步最大位移 (m)", "number"),
        "observation_timeout_sec": ("观测超时 (s)", "number"),
        "record_targets_csv": ("target 记录 CSV", "text"),
        "device": ("设备", "text"),
    }
    for key, (label, input_type) in labels.items():
        value = html.escape(DEFAULTS[key])
        step_attr = ' step="any"' if input_type == "number" else ""
        fields.append(
            f'<label><span>{label}</span>'
            f'<input name="{key}" type="{input_type}" value="{value}"{step_attr} required>'
            f"</label>"
        )
    fields_html = "\n".join(fields)

    page = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>多相机 ACT 推理控制</title>
  <style>
    :root {{
      color-scheme: light dark;
      --bg: #0f1419;
      --card: #1a2332;
      --text: #e6edf3;
      --muted: #8b949e;
      --accent: #3b82f6;
      --ok: #22c55e;
      --warn: #f59e0b;
      --err: #ef4444;
      --border: #30363d;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      min-height: 100vh;
      font-family: "Segoe UI", system-ui, sans-serif;
      background: radial-gradient(circle at top, #172033, var(--bg));
      color: var(--text);
      padding: 24px;
    }}
    .wrap {{ max-width: 720px; margin: 0 auto; }}
    .card {{
      background: var(--card);
      border: 1px solid var(--border);
      border-radius: 16px;
      padding: 24px;
      box-shadow: 0 20px 60px rgba(0,0,0,.35);
    }}
    h1 {{ margin: 0 0 8px; font-size: 1.5rem; }}
    .sub {{ color: var(--muted); margin-bottom: 20px; line-height: 1.5; }}
    .badges {{ display: flex; gap: 10px; flex-wrap: wrap; margin-bottom: 20px; }}
    .badge {{
      padding: 6px 12px;
      border-radius: 999px;
      font-size: .85rem;
      border: 1px solid var(--border);
      background: rgba(255,255,255,.03);
    }}
    .badge.mode {{ border-color: var(--warn); color: #fcd34d; }}
    .badge.idle {{ border-color: var(--ok); color: #86efac; }}
    .badge.running {{ border-color: var(--accent); color: #93c5fd; }}
    form {{ display: grid; gap: 14px; }}
    label {{ display: grid; gap: 6px; }}
    label span {{ color: var(--muted); font-size: .9rem; }}
    input {{
      width: 100%;
      padding: 10px 12px;
      border-radius: 10px;
      border: 1px solid var(--border);
      background: #0b1220;
      color: var(--text);
      font-size: 1rem;
    }}
    button {{
      margin-top: 8px;
      padding: 12px 16px;
      border: 0;
      border-radius: 10px;
      background: var(--accent);
      color: white;
      font-size: 1rem;
      font-weight: 600;
      cursor: pointer;
    }}
    button:disabled {{ opacity: .55; cursor: not-allowed; }}
    .hint {{
      margin-top: 16px;
      color: var(--muted);
      font-size: .9rem;
      line-height: 1.6;
    }}
    .alert {{
      padding: 12px 14px;
      border-radius: 10px;
      margin-bottom: 16px;
    }}
    .alert.success {{ background: rgba(34,197,94,.15); border: 1px solid rgba(34,197,94,.35); }}
    .alert.error {{ background: rgba(239,68,68,.15); border: 1px solid rgba(239,68,68,.35); }}
  </style>
</head>
<body>
  <div class="wrap">
    <div class="card">
      <h1>多相机 ACT 推理控制</h1>
      <p class="sub">先确认 robot / scan / pool / bridge / safety 已就绪，再一次性填写参数并启动。</p>
      <div class="badges">
        <span class="badge mode">{html.escape(_mode_label())}</span>
        <span class="badge {status_class}" id="status-badge">状态: {status_text}</span>
      </div>
      {alert}
      <form method="post" action="/start">
        {fields_html}
        <button type="submit" {"disabled" if running else ""}>启动推理</button>
      </form>
      <p class="hint">
        bridge: {html.escape(OBSERVATION_ZMQ)} · safety: {html.escape(TARGET_ZMQ)}<br>
        推理日志输出在 tmux 的 <code>infer</code> 窗口。完成后可刷新本页再次启动。
      </p>
    </div>
  </div>
  <script>
    async function pollStatus() {{
      try {{
        const resp = await fetch('/status');
        const data = await resp.json();
        const badge = document.getElementById('status-badge');
        const btn = document.querySelector('button[type=submit]');
        if (data.running) {{
          badge.textContent = '状态: 运行中';
          badge.className = 'badge running';
          if (btn) btn.disabled = true;
        }} else {{
          badge.textContent = '状态: 待命';
          badge.className = 'badge idle';
          if (btn) btn.disabled = false;
        }}
      }} catch (e) {{}}
    }}
    setInterval(pollStatus, 2000);
  </script>
</body>
</html>"""
    return page.encode("utf-8")


class ControlPanelHandler(BaseHTTPRequestHandler):
    server_version = "InferControlPanel/1.0"

    def log_message(self, fmt: str, *args: Any) -> None:
        sys.stderr.write("[infer_control_panel] " + (fmt % args) + "\n")

    def _send_html(self, body: bytes, code: int = 200) -> None:
        self.send_response(code)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_json(self, payload: dict[str, Any], code: int = 200) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:
        if self.path == "/status":
            payload = {"running": _is_running(), "mode": INFER_MODE}
            if _last_error and not _is_running():
                payload["last_error"] = _last_error
            self._send_json(payload)
            return
        if self.path != "/":
            self.send_error(404)
            return
        self._send_html(_render_page())

    def do_POST(self) -> None:
        if self.path != "/start":
            self.send_error(404)
            return

        length = int(self.headers.get("Content-Length", "0"))
        raw = self.rfile.read(length).decode("utf-8")
        form = parse_qs(raw, keep_blank_values=True)

        params = {key: form.get(key, [DEFAULTS[key]])[0] for key in DEFAULTS}
        ok, message = _start_infer(params)
        self._send_html(_render_page(message, error=not ok), code=200 if ok else 400)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="多相机 ACT 推理 Web 控制页")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=int(os.environ.get("INFER_PANEL_PORT", "8765")))
    parser.add_argument("--no-browser", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    url = f"http://{args.host}:{args.port}/"

    print(f"[infer_control_panel] 模式: {_mode_label()}")
    print(f"[infer_control_panel] 控制页: {url}")
    print("[infer_control_panel] 填写参数后点击「启动推理」，不会自动开始。\n", flush=True)

    server = ThreadingHTTPServer((args.host, args.port), ControlPanelHandler)
    if not args.no_browser and shutil.which("xdg-open"):
        threading.Timer(1.0, lambda: webbrowser.open(url)).start()

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n[infer_control_panel] 已停止")
    finally:
        server.server_close()
        with _lock:
            if _infer_proc is not None and _infer_proc.poll() is None:
                _infer_proc.terminate()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
