"""System SSH transport. Credentials stay with ssh-agent/interactive SSH, never in the UI."""
import hashlib
import json
import shlex
import subprocess
import sys
import threading
from pathlib import Path
from core import normalize_profile

HERE = Path(__file__).resolve().parent
REMOTE_FILES = ("runtime.py", "core.py", "sdk.py", "demo_adapter.py", "training_example.py", "workspace.py", "live_generation.py")
BOOTSTRAP = """
import hashlib,json,os,pathlib,sys
p=json.load(sys.stdin); root=pathlib.Path(p['root']); root.mkdir(parents=True,exist_ok=True)
for name,source in p['files'].items():
    if name not in ('runtime.py','core.py','sdk.py','demo_adapter.py','training_example.py','workspace.py','live_generation.py'): raise ValueError('Invalid source')
    target=root/name
    if not target.exists():
        temporary=target.with_name(name+'.'+str(os.getpid())+'.tmp')
        temporary.write_text(source); temporary.replace(target)
    if hashlib.sha256(target.read_bytes()).hexdigest()!=hashlib.sha256(source.encode()).hexdigest(): raise ValueError('Runtime integrity mismatch')
print('ready')
"""

class Client:
    def __init__(self, profile):
        self.profile = normalize_profile(profile)
        self.ready = False
        self.mutex = threading.Lock()
        self.request_mutex = threading.Lock()
        self.files = {name: (HERE / name).read_text() for name in REMOTE_FILES}
        self.version = hashlib.sha256(json.dumps(self.files, sort_keys=True).encode()).hexdigest()[:20]
        self.bundle = str(Path(profile["remote_root"]) / ".cluster-dashboard" / self.version)

    def ssh(self, remote):
        p = self.profile
        return ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=10", *p["ssh_options"], p["ssh_host"], remote]

    def ensure(self):
        if self.profile["simulation"]: return
        with self.mutex:
            if self.ready: return
            argv = self.ssh(shlex.join([self.profile["python"], "-c", BOOTSTRAP]))
            try:
                result = subprocess.run(argv, input=json.dumps({"root": self.bundle, "files": self.files}), capture_output=True, text=True, timeout=30)
            except (OSError, subprocess.TimeoutExpired) as error:
                raise ConnectionError("SSH unavailable. Connect through VPN and authenticate in your terminal first. " + str(error)) from error
            if result.returncode or result.stdout.strip() != "ready":
                raise ConnectionError("SSH setup failed. Authenticate in a terminal and verify the configured workspace. " + result.stderr.strip())
            self.ready = True

    def call(self, action, **data):
        if self.profile['simulation']:
            return self._call(action, **data)
        # A shared SSH master has a finite channel limit. Browsing and polling
        # must not fan out into competing remote sessions for one profile.
        with self.request_mutex:
            return self._call(action, **data)

    def _call(self, action, **data):
        if self.profile["simulation"]:
            import runtime
            return runtime.dispatch(self.profile["remote_root"], self.profile, action, data)
        self.ensure()
        argv = self.ssh(shlex.join([self.profile["python"], self.bundle + "/runtime.py", self.profile["remote_root"], action]))
        try:
            result = subprocess.run(argv, input=json.dumps({"profile": self.profile, "data": data}), capture_output=True, text=True, timeout=150 if action in ("launch", "resume", "discover") else 45)
        except (OSError, subprocess.TimeoutExpired) as error:
            raise ConnectionError("Connection interrupted. Reconnect and refresh before repeating a launch; the cluster records submission intent. " + str(error)) from error
        try: reply = json.loads(result.stdout)
        except ValueError as error:
            raise ConnectionError("SSH command did not return dashboard data. " + result.stderr.strip()) from error
        if not reply.get("ok"): raise ValueError(reply.get("error", "Remote operation failed"))
        return reply["result"]
