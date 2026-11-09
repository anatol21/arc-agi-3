#!/usr/bin/env python3
"""Loopback-only dashboard server. Run: python apps/cluster_dashboard/server.py --demo."""
from __future__ import annotations
import argparse
import json
import mimetypes
import os
import secrets
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse
from core import ConfigError, format_yaml, normalize_config, normalize_profile, parse_yaml
from sdk import atomic_json
from transport import Client

HERE = Path(__file__).resolve().parent

def demo_profile(root):
    return {"name": "Local simulator", "simulation": True, "remote_root": str(root.resolve()), "python": sys.executable, "dev_permission": "Simulator only", "partitions": [{"name": "normal", "role": "normal", "time_limit": "02:00:00", "concurrency_limit": 4}, {"name": "development", "role": "development", "time_limit": "00:30:00", "concurrency_limit": 1}]}

def templates():
    common = {"name": "New evaluation", "kind": "evaluation", "command": ["/absolute/path/to/environment/bin/python", "/absolute/path/to/adapter.py"], "workers": 1, "concurrency": 1, "gpus": 1, "task_count": 1, "resume_capable": True, "auto_continue": True, "completion": "All declared items have committed terminal results", "fingerprints": {}, "recipe": {}, "estimates": {}, "budget": {}}
    demo = {**common, "name": "Checkpoint and continuation smoke test", "command": ["{python}", "{run_dir}/demo_adapter.py"], "task_count": 24, "fingerprints": {"environment": "demo-v1"}, "recipe": {"chunks_per_item": 3, "chunk_seconds": 0.08}, "estimates": {"seconds_per_task": 0.24, "preparation_seconds": 0.01}}
    train = {**common, "name": "Training experiment", "kind": "train", "completion": "Target training step reached; full optimizer, scheduler, RNG and sampler state saved", "recipe": {"target_steps": 1000}}
    arc = {**common, "name": "ARC optimized direct-answer example", "gpus": 3, "task_count": 276, "completion": "138 tasks × base/fine-tuned arms have terminal responses; all query grids scored", "recipe": {"tasks": 138, "arms": ["base", "finetuned"], "allow_small_final_batches": True, "preserve_token_budgets": True}, "notes": "Example only: point command to an SDK-compatible adapter for the chosen native runner. Existing experimental identities and results remain separate."}
    return {"evaluation": {**common, "resume_capable": False, "auto_continue": False, "completion": "Command exits successfully"}, "training": {**train, "resume_capable": False, "auto_continue": False, "completion": "Training command exits successfully; review saved metrics and checkpoints"}, "arc_example": arc, "simulator": demo}

class App:
    def __init__(self, state, demo=False):
        self.state = state
        self.state.mkdir(parents=True, exist_ok=True)
        self.state.chmod(0o700)
        (self.state / "profiles").mkdir(mode=0o700, exist_ok=True)
        self.clients = {}
        self.mutex = threading.Lock()
        self.token = secrets.token_urlsafe(32)
        self.cache = {}
        if demo and not (self.state / "profiles" / "simulator.yaml").exists():
            self.save_profile("simulator", demo_profile(state / "simulated-cluster"))

    def save_profile(self, profile_id, value):
        import re
        if not re.fullmatch(r"[A-Za-z0-9_-]+", profile_id): raise ValueError("Invalid profile identifier")
        p = normalize_profile(value)
        path = self.state / "profiles" / (profile_id + ".yaml")
        temp = path.with_suffix(".tmp")
        temp.write_text(format_yaml(p))
        temp.chmod(0o600)
        temp.replace(path)
        with self.mutex:
            self.clients.pop(profile_id, None)
            self.cache.pop(profile_id, None)
        return p

    def profiles(self):
        return [{"id": p.stem, "profile": normalize_profile(parse_yaml(p.read_text()))} for p in sorted((self.state / "profiles").glob("*.yaml"))]

    def client(self, profile_id):
        with self.mutex:
            if profile_id not in self.clients:
                matches = [p for p in self.profiles() if p["id"] == profile_id]
                if not matches: raise ValueError("Choose a cluster profile")
                self.clients[profile_id] = Client(matches[0]["profile"])
            return self.clients[profile_id]

    def runs(self, profile_id, force=False):
        client = self.client(profile_id)
        with self.mutex:
            cached = self.cache.get(profile_id)
        interval = client.profile["poll_seconds"] if not client.profile["simulation"] else 2
        if cached and not force and time.time() - cached["time"] < interval: return cached["runs"]
        runs = client.call("list")
        atomic_json(self.state / ("index-" + profile_id + ".json"), runs)
        with self.mutex: self.cache[profile_id] = {"time": time.time(), "runs": runs}
        return runs

class Server(ThreadingHTTPServer):
    daemon_threads = True
    def __init__(self, address, app):
        self.app = app
        super().__init__(address, Handler)

class Handler(BaseHTTPRequestHandler):
    def log_message(self, *args): pass

    def reply(self, value, status=200):
        data = json.dumps(value, allow_nan=False).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.end_headers()
        self.wfile.write(data)

    def trusted(self):
        host = self.headers.get("Host", "")
        return host in (f"127.0.0.1:{self.server.server_port}", f"localhost:{self.server.server_port}")

    def do_GET(self):
        if not self.trusted(): return self.reply({"error": "Invalid host"}, 403)
        parsed = urlparse(self.path)
        query = {k:v[0] for k,v in parse_qs(parsed.query).items()}
        try:
            if parsed.path == "/api/bootstrap":
                return self.reply({"token": self.server.app.token, "profiles": self.server.app.profiles(), "templates": templates(), "template_yaml": {k: format_yaml(v) for k, v in templates().items()}})
            if parsed.path == "/api/runs":
                try: return self.reply({"runs": self.server.app.runs(query["profile"]), "offline": False})
                except (ConnectionError, OSError) as error:
                    path = self.server.app.state / ("index-" + query["profile"] + ".json")
                    from sdk import read_json
                    return self.reply({"runs": read_json(path, []), "offline": True, "error": str(error)})
            if parsed.path == "/api/tail":
                return self.reply(self.server.app.client(query["profile"]).call("tail", id=query["id"], file=query["file"], offset=int(query.get("offset", 0))))
            assets = {"/": "index.html", "/app.js": "app.js", "/app.css": "app.css", "/workflow.js": "workflow.js"}
            if parsed.path not in assets: return self.reply({"error": "Not found"}, 404)
            path = HERE / "static" / assets[parsed.path]
            data = path.read_bytes()
            self.send_response(200)
            self.send_header("Content-Type", mimetypes.guess_type(path.name)[0] or "application/octet-stream")
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Cache-Control", "no-cache")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("Content-Security-Policy", "default-src 'self'; script-src 'self'; style-src 'self'; object-src 'none'; frame-ancestors 'none'")
            self.end_headers()
            self.wfile.write(data)
        except (ValueError, KeyError, OSError) as error:
            self.reply({"error": str(error)}, 400)

    def do_POST(self):
        if not self.trusted() or self.headers.get("X-Dashboard-Token") != self.server.app.token:
            return self.reply({"error": "Refresh the local dashboard before making changes"}, 403)
        try:
            length = int(self.headers.get("Content-Length", 0))
            if not 0 < length <= 262144: raise ValueError("Invalid request size")
            body = json.loads(self.rfile.read(length))
            action = urlparse(self.path).path.removeprefix("/api/")
            app = self.server.app
            if action == "profile":
                result = app.save_profile(body["id"], parse_yaml(body["yaml"]))
                return self.reply({"profile": result})
            if action == "validate":
                config = normalize_config(parse_yaml(body["yaml"]))
                return self.reply({"config": config, "yaml": format_yaml(config)})
            client = app.client(body["profile"])
            if action in ("discover", "create"):
                config = normalize_config(parse_yaml(body["yaml"]))
                result = client.call(action, config=config, **({"request_id": body["request_id"]} if action == "create" and "request_id" in body else {}))
            elif action == "refresh":
                return self.reply({"runs": app.runs(body["profile"], force=True)})
            elif action in ("launch", "resume", "stop", "concurrency", "notes", "browse", "review"):
                result = client.call(action, **{k:v for k,v in body.items() if k != "profile"})
            else: raise ValueError("Unknown action")
            with app.mutex: app.cache.pop(body["profile"], None)
            self.reply({"result": result})
        except (ValueError, KeyError, TypeError, ConnectionError, OSError) as error:
            self.reply({"error": str(error)}, 400)

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8766)
    parser.add_argument("--state", type=Path, default=HERE / ".local")
    parser.add_argument("--demo", action="store_true", help="Add a CPU-only simulator profile")
    args = parser.parse_args()
    server = Server(("127.0.0.1", args.port), App(args.state.resolve(), args.demo))
    print(f"Cluster dashboard: http://127.0.0.1:{server.server_port}", flush=True)
    try: server.serve_forever()
    except KeyboardInterrupt: pass
    finally: server.server_close()

if __name__ == "__main__": main()
