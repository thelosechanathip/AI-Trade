"""
run.py -- Single entry point.

    python run.py            # starts backend + dashboard + trading engine
    python run.py --no-ui    # starts backend + trading engine only

Services:
  Backend API + WebSocket  -> http://localhost:8000   (FastAPI, background thread)
  Next.js Dashboard        -> http://localhost:3000   (default; disable with --no-ui)
  Trading Engine           ->                         (main thread, 60s cycle)
"""

import os
import sys
import subprocess
import shutil
import threading
import webbrowser
import time
import json
import urllib.error
import urllib.request


def _start_web(port: int) -> None:
    import socket
    import uvicorn
    from web_app import app

    # Warn early if port is already taken
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            s.bind(("0.0.0.0", port))
        except OSError:
            print(f"\n  WARNING: Port {port} is already in use by another app!")
            print(f"  Dashboard will NOT be available. Change 'dashboard.port' in config.yaml.\n")
            return

    uvicorn.run(app, host="0.0.0.0", port=port, log_level="warning")


def _wait_for_web(port: int, timeout: float = 8.0) -> bool:
    """Wait until the dashboard API is reachable from this project directory."""
    deadline = time.time() + timeout
    expected_cwd = os.path.abspath(os.getcwd())
    url = f"http://127.0.0.1:{port}/api/state"

    while time.time() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=1.0) as resp:
                payload = json.loads(resp.read().decode("utf-8") or "{}")
            runtime_cwd = payload.get("runtime", {}).get("cwd")
            if not runtime_cwd:
                print(f"\n  ERROR: Port {port} is serving an old dashboard backend.")
                print("         Stop that process, then run python run.py again.")
                return False
            if os.path.abspath(runtime_cwd) != expected_cwd:
                print(f"\n  ERROR: Port {port} is serving another AI-Trade folder:")
                print(f"         {runtime_cwd}")
                print(f"         expected: {expected_cwd}")
                return False
            return True
        except (OSError, urllib.error.URLError, json.JSONDecodeError):
            time.sleep(0.25)
    return False


def _find_npm() -> str | None:
    """Return an npm executable path, including common Windows install paths."""
    commands = ("npm.cmd", "npm") if sys.platform == "win32" else ("npm",)
    for command in commands:
        npm = shutil.which(command)
        if npm:
            return npm

    if sys.platform != "win32":
        return None

    candidates = []
    for env_name in ("ProgramFiles", "ProgramFiles(x86)"):
        base_dir = os.environ.get(env_name)
        if base_dir:
            candidates.append(os.path.join(base_dir, "nodejs", "npm.cmd"))

    local_app_data = os.environ.get("LOCALAPPDATA")
    if local_app_data:
        candidates.append(
            os.path.join(local_app_data, "Programs", "nodejs", "npm.cmd")
        )

    return next((path for path in candidates if os.path.isfile(path)), None)


def _install_nextjs_dependencies(dashboard_dir: str, npm: str) -> bool:
    """Install dashboard dependencies and report whether installation succeeded."""
    print("  Installing Next.js dependencies (first run)...")
    try:
        result = subprocess.run([npm, "install"], cwd=dashboard_dir, check=False)
    except OSError as exc:
        print(f"  WARNING: Could not run npm install: {exc}")
        return False

    if result.returncode != 0:
        print(f"  WARNING: npm install failed with exit code {result.returncode}.")
        return False

    print("  Done. Starting Next.js dev server...")
    return True


def _start_nextjs(dashboard_dir: str, npm: str) -> subprocess.Popen | None:
    """Try to start Next.js dev server. Returns process or None."""
    try:
        proc = subprocess.Popen(
            [npm, "run", "dev"],
            cwd=dashboard_dir,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            creationflags=(subprocess.CREATE_NEW_PROCESS_GROUP
                           if sys.platform == "win32" else 0),
        )
        return proc
    except OSError as exc:
        print(f"  WARNING: Could not start Next.js: {exc}")
        return None


def main() -> None:
    launch_ui = "--no-ui" not in sys.argv

    # ── Load config ───────────────────────────────────────────────────
    try:
        import yaml
        with open("config.yaml", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        port = cfg.get("dashboard", {}).get("port", 8000)
    except Exception:
        port = 8000

    # ── Banner ────────────────────────────────────────────────────────
    print()
    print("=" * 58)
    print("  AI-Trade  |  starting all services")
    print("=" * 58)
    print(f"  Backend API      ->  http://localhost:{port}")
    print(f"  WebSocket feed   ->  ws://localhost:{port}/ws")
    print(f"  Next.js UI       ->  http://localhost:3000  (cd dashboard && npm run dev)")
    print("  Trading Engine   ->  60s cycle")
    print("=" * 58)
    print()

    # ── Backend (daemon thread) ───────────────────────────────────────
    web_thread = threading.Thread(
        target=_start_web,
        args=(port,),
        daemon=True,
        name="web-server",
    )
    web_thread.start()
    if not _wait_for_web(port):
        print(f"\nERROR: Dashboard backend did not start on http://localhost:{port}.")
        print("Close the old process using that port, then run python run.py again.\n")
        sys.exit(1)

    # ── Next.js dashboard (optional) ─────────────────────────────────
    nextjs_proc = None
    dashboard_dir = os.path.join(os.path.dirname(__file__), "dashboard")
    node_modules = os.path.join(dashboard_dir, "node_modules")

    if launch_ui:
        npm = _find_npm()
        if not npm:
            print("  WARNING: Node.js/npm not found; Next.js UI was skipped.")
            print("           Install Node.js LTS, then run python run.py again.")
            webbrowser.open(f"http://localhost:{port}")
        else:
            dependencies_ready = os.path.exists(node_modules)
            if not dependencies_ready:
                dependencies_ready = _install_nextjs_dependencies(dashboard_dir, npm)

            if dependencies_ready:
                nextjs_proc = _start_nextjs(dashboard_dir, npm)

            if nextjs_proc:
                print("  Next.js starting on http://localhost:3000")
                time.sleep(4)
                webbrowser.open("http://localhost:3000")
            else:
                print(f"  Opening fallback dashboard on http://localhost:{port}")
                webbrowser.open(f"http://localhost:{port}")
    else:
        webbrowser.open(f"http://localhost:{port}")

    # ── Trading engine (main thread) ──────────────────────────────────
    from main import TradingEngine
    engine = TradingEngine()
    if not engine.connect():
        print("\nERROR: Could not connect to MetaTrader 5.")
        print("Make sure MT5 is open and logged in, then try again.\n")
        if nextjs_proc:
            nextjs_proc.terminate()
        sys.exit(1)

    try:
        engine.run()
    finally:
        if nextjs_proc:
            nextjs_proc.terminate()


if __name__ == "__main__":
    main()
