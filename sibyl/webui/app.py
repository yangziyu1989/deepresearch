"""WebUI Flask application with WebSocket support."""
from __future__ import annotations
import json
from pathlib import Path

try:
    from flask import Flask, jsonify, request, send_from_directory
    HAS_FLASK = True
except ImportError:
    HAS_FLASK = False

from sibyl.orchestration.dashboard_data import get_dashboard_data, list_all_projects


def create_webui_app(workspaces_dir: str = "workspaces") -> "Flask":
    """Create the WebUI Flask application."""
    if not HAS_FLASK:
        raise RuntimeError("Flask not installed")

    app = Flask(__name__)
    app.config["WORKSPACES_DIR"] = workspaces_dir

    @app.route("/api/projects")
    def api_projects():
        return jsonify(list_all_projects(workspaces_dir))

    @app.route("/api/project/<name>/dashboard")
    def api_project_dashboard(name: str):
        ws_path = Path(workspaces_dir) / name
        if not ws_path.exists():
            return jsonify({"error": "not found"}), 404
        return jsonify(get_dashboard_data(ws_path))

    @app.route("/api/project/<name>/files/<path:filepath>")
    def api_project_file(name: str, filepath: str):
        ws_path = Path(workspaces_dir) / name
        full_path = ws_path / filepath
        if not full_path.exists():
            return jsonify({"error": "file not found"}), 404
        if full_path.suffix in (".json", ".jsonl"):
            return jsonify(json.loads(full_path.read_text()))
        return full_path.read_text(encoding="utf-8")

    @app.route("/api/health")
    def health():
        return jsonify({"status": "ok", "service": "webui"})

    return app
