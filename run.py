"""
run.py -- Single entry point.

    python run.py            # starts backend + trading engine
    python run.py --ui       # also launches Next.js dashboard (npm run dev)

Services:
  Backend API + WebSocket  -> http://localhost:8000   (FastAPI, background thread)
  Next.js Dashboard        -> http://localhost:3000   (if --ui flag or Node installed)
  Trading Engine           ->                         (main thread, 60s cycle)
"""

import os
import sys
import subprocess
import threading
import webbrowser
import time


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


def _start_nextjs(dashboard_dir: str) -> subprocess.Popen | None:
    """Try to start Next.js dev server. Returns process or None."""
    npm = "npm.cmd" if sys.platform == "win32" else "npm"
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
    except FileNotFoundError:
        return None


def main() -> None:
    launch_ui = "--ui" in sys.argv

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
    time.sleep(1.5)

    # ── Next.js dashboard (optional) ─────────────────────────────────
    nextjs_proc = None
    dashboard_dir = os.path.join(os.path.dirname(__file__), "dashboard")
    node_modules = os.path.join(dashboard_dir, "node_modules")

    if launch_ui:
        if not os.path.exists(node_modules):
            print("  Installing Next.js dependencies (first run)…")
            npm = "npm.cmd" if sys.platform == "win32" else "npm"
            subprocess.run([npm, "install"], cwd=dashboard_dir, check=False)
            print("  Done. Starting Next.js dev server…")
        nextjs_proc = _start_nextjs(dashboard_dir)
        if nextjs_proc:
            print("  Next.js starting on http://localhost:3000")
            time.sleep(4)
            webbrowser.open("http://localhost:3000")
        else:
            print("  WARNING: npm not found — open the dashboard manually")
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
