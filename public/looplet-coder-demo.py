"""Run one live Looplet coding task in Docker, then verify its artifacts independently."""

from __future__ import annotations

import argparse
import difflib
import importlib.util
import io
import json
import os
from pathlib import Path
import subprocess
import tarfile
import time
import urllib.request
import zipfile


REVISION = "8384404061b9c22e9639679d1a4c61fbf9737e22"
IMAGE = "looplet-blog-coder-demo:python311"
DOCKERFILE = """FROM python:3.11-slim-bookworm
RUN apt-get update && apt-get install -y --no-install-recommends git && rm -rf /var/lib/apt/lists/*
RUN pip install --no-cache-dir openai==3.14.1 pytest==9.1.1 ruff==0.16.7
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PYTHONPATH=/opt/looplet/src
WORKDIR /work
"""
AGENT = '''import importlib.metadata
import json
import os
from pathlib import Path
import time

from looplet import ProvenanceSink, cartridge_to_preset
from looplet.backends import OpenAIBackend
from looplet.types import DefaultState

root = Path("/work")
task = json.loads(Path("/runner/task.json").read_text())
preset = cartridge_to_preset("/opt/looplet/coder.cartridge", runtime={"project_root": str(root)})
preset.config.max_steps = 40
preset.state = DefaultState(max_steps=40)
sink = ProvenanceSink(dir=root / ".run")
backend = sink.wrap_llm(OpenAIBackend(
    base_url=os.environ["OPENAI_BASE_URL"],
    api_key=os.environ.get("OPENAI_API_KEY") or "local",
    model=os.environ["OPENAI_MODEL"],
))
recorder = sink.trajectory_hook()
preset.hooks = [*preset.hooks, recorder]
started = time.monotonic()
steps = []
error = None
try:
    for step in preset.run(backend, task=task):
        steps.append(step)
        print(json.dumps({"step": len(steps), "tool": step.tool_call.tool,
                          "error": step.tool_result.error}), flush=True)
except Exception as exc:
    error = type(exc).__name__ + ": " + str(exc)
finally:
    sink.flush()
    preset.close()
termination = recorder.trajectory.termination_reason
if termination != "done":
    error = error or "Loop ended without completion: " + str(termination)
result = {
    "requested_model": os.environ["OPENAI_MODEL"],
    "elapsed_s": round(time.monotonic() - started, 2),
    "steps": len(steps),
    "error": error,
    "termination_reason": termination,
    "packages": {name: importlib.metadata.version(name) for name in ["openai", "pytest", "ruff"]},
}
(root / ".run" / "live-result.json").write_text(json.dumps(result, indent=2) + "\\n")
print(json.dumps(result), flush=True)
raise SystemExit(1 if error else 0)
'''
VERIFY = '''import ast
import importlib
import json
from pathlib import Path
import sys

sys.path.insert(0, "/work")
import app

def fresh():
    app.accounts.clear()
    app.create_account("alice")
    app.create_account("bob")

def history_and_withdrawal_undo():
    fresh()
    app.deposit("alice", 100)
    app.withdraw("alice", 30)
    history = app.history("alice")
    assert isinstance(history, list) and len(history) >= 2
    assert all(isinstance(record, dict) and "amount" in record for record in history)
    assert history[-1]["amount"] == 30
    app.undo_last("alice")
    assert app.balance("alice") == 100

def deposit_undo():
    fresh()
    app.deposit("alice", 80)
    app.undo_last("alice")
    assert app.balance("alice") == 0

def transfer_undo():
    fresh()
    app.deposit("alice", 100)
    app.transfer("alice", "bob", 40)
    assert (app.balance("alice"), app.balance("bob")) == (60, 40)
    assert app.history("alice") and app.history("bob")
    app.undo_last("bob")
    assert (app.balance("alice"), app.balance("bob")) == (100, 0)

def rejected_withdrawal():
    fresh()
    app.deposit("alice", 20)
    before = list(app.history("alice"))
    try:
        app.withdraw("alice", 21)
    except ValueError:
        pass
    else:
        raise AssertionError("insufficient funds accepted")
    assert app.balance("alice") == 20 and app.history("alice") == before

def imported_modules():
    sources = []
    for module in list(sys.modules.values()):
        filename = getattr(module, "__file__", None)
        if filename:
            path = Path(filename).resolve()
            if path.is_relative_to(Path("/work")) and path.suffix == ".py" and path.name != "app.py":
                if any(isinstance(node, (ast.FunctionDef, ast.ClassDef)) for node in ast.walk(ast.parse(path.read_text()))):
                    sources.append(str(path))
    assert len(set(sources)) >= 2, sources

results = {}
for check in [history_and_withdrawal_undo, deposit_undo, transfer_undo, rejected_withdrawal, imported_modules]:
    try:
        check()
        results[check.__name__] = {"passed": True}
    except Exception as error:
        results[check.__name__] = {"passed": False, "error": type(error).__name__ + ": " + str(error)}
print(json.dumps(results))
raise SystemExit(0 if all(result["passed"] for result in results.values()) else 1)
'''


def run_command(command, *, timeout=600, **kwargs):
    return subprocess.run(command, timeout=timeout, check=False, **kwargs)


def source_snapshot(root, repo):
    source = root / "source"
    source.mkdir()
    if repo:
        archive = subprocess.check_output([
            "git", f"--git-dir={repo / '.git'}", f"--work-tree={repo}",
            "archive", REVISION,
        ])
        with tarfile.open(fileobj=io.BytesIO(archive)) as bundle:
            bundle.extractall(source, filter="data")
    else:
        request = urllib.request.Request(
            f"https://codeload.github.com/hsaghir/looplet/zip/{REVISION}",
            headers={"User-Agent": "looplet-coder-demo"},
        )
        with urllib.request.urlopen(request, timeout=60) as response:
            archive = response.read()
        with zipfile.ZipFile(io.BytesIO(archive)) as bundle:
            bundle.extractall(source)
        source = source / f"looplet-{REVISION}"
    return source


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True, help="New output directory.")
    parser.add_argument("--repo", type=Path, help="Optional local Looplet Git repository; always exports the pinned revision.")
    parser.add_argument("--base-url", default=os.environ.get("OPENAI_BASE_URL"), help="OpenAI-compatible provider URL.")
    parser.add_argument("--model", default=os.environ.get("OPENAI_MODEL"), help="Requested model ID.")
    parser.add_argument("--skip-build", action="store_true", help="Reuse this demo's existing Docker image.")
    args = parser.parse_args()
    if not args.base_url or not args.model:
        parser.error("provide --base-url and --model (or OPENAI_BASE_URL and OPENAI_MODEL)")
    root = args.out.resolve()
    root.mkdir(parents=True, exist_ok=False)
    source = source_snapshot(root, args.repo.resolve() if args.repo else None)
    spec = importlib.util.spec_from_file_location("public_tasks", source / "benchmarks/coder_vs_agents/hard_tasks.py")
    tasks = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(tasks)
    task = next(task for task in tasks.TASKS if task["id"] == "h3_refactor")
    workspace = root / "workspace"
    runner = root / "runner"
    workspace.mkdir()
    runner.mkdir()
    for name, content in task["seed"].items():
        (workspace / name).write_text(content)
    (runner / "agent.py").write_text(AGENT)
    (runner / "task.json").write_text(json.dumps({"goal": task["prompt"]}, indent=2) + "\n")
    (runner / "Dockerfile").write_text(DOCKERFILE)
    if not args.skip_build:
        built = run_command(["docker", "build", "-t", IMAGE, str(runner)], timeout=600)
        if built.returncode:
            return built.returncode
    container = f"looplet-blog-demo-{os.getpid()}"
    mounts = [
        "-v", f"{workspace}:/work:rw",
        "-v", f"{source / 'src'}:/opt/looplet/src:ro",
        "-v", f"{source / 'examples/coder.cartridge'}:/opt/looplet/coder.cartridge:ro",
        "-v", f"{runner}:/runner:ro",
    ]
    common = ["docker", "run", "--rm", "--read-only", "--cap-drop=ALL",
              "--security-opt=no-new-privileges", "--pids-limit=128", "--memory=1g", "--cpus=2",
              "--user", f"{os.getuid()}:{os.getgid()}", "--tmpfs", "/tmp:rw,nosuid,size=256m",
              "-e", "HOME=/tmp", *mounts]
    environment = dict(os.environ, OPENAI_BASE_URL=args.base_url, OPENAI_MODEL=args.model)
    started = time.monotonic()
    try:
        with (root / "agent.log").open("w") as log:
            agent = run_command([*common, "--name", container, "--network=host",
                                 "-e", "OPENAI_BASE_URL", "-e", "OPENAI_MODEL", "-e", "OPENAI_API_KEY",
                                 IMAGE, "python", "/runner/agent.py"], env=environment, stdout=log, stderr=subprocess.STDOUT)
    except subprocess.TimeoutExpired:
        run_command(["docker", "kill", container], capture_output=True)
        agent = subprocess.CompletedProcess([], 124)
    before = task["seed"]
    changes = []
    seen = set()
    for path in sorted(workspace.rglob("*.py")):
        if any(part.startswith(".") or part == "__pycache__" for part in path.relative_to(workspace).parts):
            continue
        relative = str(path.relative_to(workspace))
        seen.add(relative)
        changes.extend(difflib.unified_diff(before.get(relative, "").splitlines(keepends=True),
                                            path.read_text().splitlines(keepends=True),
                                            fromfile="before/" + relative, tofile="after/" + relative))
    for relative in sorted(set(before) - seen):
        changes.extend(difflib.unified_diff(before[relative].splitlines(keepends=True), [],
                                            fromfile="before/" + relative, tofile="/dev/null"))
    (root / "agent.patch").write_text("".join(
        line if line.endswith("\n") else line + "\n\\ No newline at end of file\n"
        for line in changes
    ))
    original_test = workspace / "test_app.py"
    preserved = original_test.is_file() and original_test.read_text() == before["test_app.py"]
    (workspace / "test_app.py").write_text(before["test_app.py"])
    provided = run_command([*common, "--network=none", IMAGE, "python", "-m", "pytest", "-q", "test_app.py"], capture_output=True, text=True)
    hidden = run_command([*common, "--network=none", "-i", IMAGE, "python", "-"], input=VERIFY, capture_output=True, text=True)
    (root / "provided-tests.txt").write_text(provided.stdout + provided.stderr)
    (root / "independent-checks.txt").write_text(hidden.stdout + hidden.stderr)
    (root / "verify.py").write_text(VERIFY)
    result = {
        "revision": REVISION, "task": task["id"], "requested_model": args.model,
        "attempts": 1, "agent_exit": agent.returncode,
        "elapsed_s": round(time.monotonic() - started, 2),
        "original_test_file_unchanged": preserved, "provided_tests_pass": provided.returncode == 0,
        "independent_checks_pass": hidden.returncode == 0,
    }
    (root / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    print(f"Inspect agent.patch, agent.log, workspace/.run/, and verification results in {root}")
    return 0 if agent.returncode == 0 and provided.returncode == 0 and hidden.returncode == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())