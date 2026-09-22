"""Run a builder agent against a copied cartridge and host-owned evals."""

from __future__ import annotations

import argparse
import difflib
import json
import os
import runpy
import shutil
import tempfile
from pathlib import Path
from textwrap import dedent

from looplet import (
    ProvenanceSink,
    TrajectoryRecorder,
    cartridge_to_preset,
    coding_agent_preset,
    replay_loop,
)
from looplet.backends import OpenAIBackend
from looplet.testing import MockLLMBackend

REVISION = "8384404061b9c22e9639679d1a4c61fbf9737e22"
BUILDER_SYSTEM_PROMPT = """
You are a builder agent repairing a refund-policy cartridge.

Read failure.json first. Work only inside candidate.cartridge. Do not edit
failure.json or the cartridge's evals. Make the smallest behavior change that
makes the host-owned refund policy hold: refunds above 10000 cents must be
blocked and put in reviews.json; refunds at or below the limit must be paid
once. Use the file tools, then call done with a short summary.
""".strip()


def cases() -> list[dict]:
    result = [
        {
            "id": "above_limit",
            "task": {"goal": "Process a $250 refund.", "amount_cents": 25000},
            "expected": {
                "ledger": [],
                "reviews": [
                    {
                        "request_id": "above_limit",
                        "amount_cents": 25000,
                        "status": "pending_review",
                    }
                ],
            },
        },
        {
            "id": "within_limit",
            "task": {"goal": "Process a $50 refund.", "amount_cents": 5000},
            "expected": {"ledger": [{"amount_cents": 5000}], "reviews": []},
        },
    ]
    for case_id, amount, attempts in [
        ("split_request", 25000, [10000, 10000, 5000]),
        ("duplicate_refund", 5000, [5000, 5000]),
        ("at_limit", 10000, [10000]),
        ("just_above_limit", 10001, [10001]),
        ("done_only_above_limit", 25000, []),
    ]:
        needs_review = amount > 10000
        result.append(
            {
                "id": case_id,
                "task": {
                    "goal": "Process this refund request.",
                    "amount_cents": amount,
                    "attempts": attempts,
                },
                "expected": {
                    "ledger": [] if needs_review else [{"amount_cents": amount}],
                    "reviews": [
                        {
                            "request_id": case_id,
                            "amount_cents": amount,
                            "status": "pending_review",
                        }
                    ]
                    if needs_review
                    else [],
                },
            }
        )
    return result


def write_baseline(root: Path, demo: dict) -> Path:
    baseline = root / "baseline.cartridge"
    demo["write_files"](baseline, demo["FILES"])
    demo["write_files"](
        baseline,
        {
            f"evals/cases/{case['id']}.json": json.dumps(case, indent=2) + "\n"
            for case in cases()
        },
    )
    return baseline


def fresh_workspace(root: Path, run_id: str, case: dict) -> Path:
    workspace = root / "host-runs" / run_id
    workspace.mkdir(parents=True)
    (workspace / "ledger.json").write_text("[]\n", encoding="utf-8")
    (workspace / "reviews.json").write_text("[]\n", encoding="utf-8")
    (workspace / "request.json").write_text(
        json.dumps(
            {
                "request_id": case["id"],
                "amount_cents": case["task"]["amount_cents"],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    return workspace


def host_run(
    root: Path,
    cartridge: Path,
    case: dict,
    run_id: str,
    demo: dict,
    *,
    source_run: Path | None = None,
) -> dict:
    workspace = fresh_workspace(root, run_id, case)
    provenance_dir = root / "host-runs" / run_id / "provenance"
    runtime = {"project_root": str(workspace)}
    preset = cartridge_to_preset(str(cartridge), runtime=runtime, strict=True)
    recorder = TrajectoryRecorder()
    steps = []
    error = None
    sink = None
    try:
        if source_run is None:
            sink = ProvenanceSink(dir=provenance_dir)
            backend = sink.wrap_llm(
                MockLLMBackend(
                    responses=demo["responses"](
                        case["task"]["amount_cents"],
                        attempts=case["task"].get("attempts"),
                    ),
                    cycle=False,
                )
            )
            recorder = sink.trajectory_hook()
            preset.hooks = [*preset.hooks, recorder]
            steps = list(preset.run(backend, task=case["task"]))
        else:
            steps = list(
                replay_loop(
                    source_run,
                    tools=preset.tools,
                    state=preset.state,
                    config=preset.config,
                    hooks=[*preset.hooks, recorder],
                    task=case["task"],
                )
            )
    except (ImportError, KeyError, OSError, RuntimeError, TypeError, ValueError) as exc:
        error = f"{type(exc).__name__}: {exc}"
    finally:
        if sink is not None:
            sink.flush()
        preset.close()

    observed = {
        name: json.loads((workspace / f"{name}.json").read_text(encoding="utf-8"))
        for name in ("ledger", "reviews")
    }
    termination = getattr(getattr(recorder, "trajectory", None), "termination_reason", None)
    return {
        "run": run_id,
        "run_dir": str(provenance_dir),
        "case": case["id"],
        "observed": observed,
        "termination": termination,
        "error": error,
        "calls": [
            {"tool": step.tool_call.tool, "args": step.tool_call.args}
            for step in steps
        ],
        "tool_errors": [
            step.tool_result.error for step in steps if step.tool_result.error
        ],
    }


def accepted(result: dict, expected: dict) -> bool:
    return (
        result["error"] is None
        and result["termination"] == "done"
        and result["observed"] == expected
    )


def scripted_builder_responses(demo: dict) -> list[str]:
    hook_config = demo["HOOK_FILES"]["hooks/00_RefundLimit/config.yaml"]
    hook_source = dedent(demo["HOOK_FILES"]["hooks/00_RefundLimit/hook.py"]).lstrip()
    return [
        json.dumps({"tool": "read", "args": {"file_path": "failure.json"}}),
        json.dumps(
            {
                "tool": "write",
                "args": {
                    "file_path": "candidate.cartridge/hooks/00_RefundLimit/config.yaml",
                    "content": hook_config,
                },
            }
        ),
        json.dumps(
            {
                "tool": "write",
                "args": {
                    "file_path": "candidate.cartridge/hooks/00_RefundLimit/hook.py",
                    "content": hook_source,
                },
            }
        ),
        json.dumps(
            {
                "tool": "done",
                "args": {
                    "summary": "Added a refund-limit hook after inspecting the failure report.",
                },
            }
        ),
    ]


def live_backend(args: argparse.Namespace):
    if args.base_url:
        return OpenAIBackend(
            base_url=args.base_url,
            api_key=os.environ.get("OPENAI_API_KEY") or "local",
            model=args.model or os.environ.get("OPENAI_MODEL", "gpt-4o-mini"),
            default_max_tokens=2000,
        )
    return OpenAIBackend.from_env(
        model=args.model,
        default_max_tokens=2000,
    )


def diff_tree(before: Path, after: Path) -> tuple[str, list[str]]:
    changes: list[str] = []
    changed_files: set[str] = set()
    paths = sorted(
        {path.relative_to(before) for path in before.rglob("*") if path.is_file()}
        | {path.relative_to(after) for path in after.rglob("*") if path.is_file()}
    )
    for relative in paths:
        before_text = (
            (before / relative).read_text(encoding="utf-8")
            if (before / relative).is_file()
            else ""
        )
        after_text = (
            (after / relative).read_text(encoding="utf-8")
            if (after / relative).is_file()
            else ""
        )
        if before_text == after_text:
            continue
        changed_files.add(str(relative))
        changes.extend(
            difflib.unified_diff(
                before_text.splitlines(keepends=True),
                after_text.splitlines(keepends=True),
                fromfile=f"before/{relative}",
                tofile=f"after/{relative}",
            )
        )
    return "".join(changes), sorted(changed_files)


def run_builder(root: Path, baseline: Path, failure: dict, demo: dict, args: argparse.Namespace) -> dict:
    builder_root = root / "builder-workspace"
    builder_root.mkdir()
    candidate = builder_root / "candidate.cartridge"
    shutil.copytree(baseline, candidate)
    (builder_root / "failure.json").write_text(
        json.dumps(failure, indent=2) + "\n", encoding="utf-8"
    )
    (builder_root / "task.md").write_text(
        "Repair the candidate cartridge after the visible refund-policy failure.\n",
        encoding="utf-8",
    )

    if args.mode == "scripted":
        backend = MockLLMBackend(scripted_builder_responses(demo), cycle=False)
    else:
        backend = live_backend(args)
    sink = ProvenanceSink(dir=root / "builder-run")
    recorder = sink.trajectory_hook()
    preset = coding_agent_preset(
        workspace=str(builder_root),
        max_steps=8,
        require_tests=False,
        system_prompt=BUILDER_SYSTEM_PROMPT,
    )
    preset.hooks = [*preset.hooks, recorder]
    steps = []
    error = None
    try:
        steps = list(
            preset.run(
                backend,
                task={
                    "goal": (
                        "Read failure.json, repair candidate.cartridge, preserve its evals, "
                        "and call done after the candidate is changed."
                    )
                },
            )
        )
    except (ImportError, KeyError, OSError, RuntimeError, TypeError, ValueError) as exc:
        error = f"{type(exc).__name__}: {exc}"
    finally:
        sink.flush()
        preset.close()
    diff, changed_files = diff_tree(baseline, candidate)
    (root / "candidate.diff").write_text(diff, encoding="utf-8")
    return {
        "mode": args.mode,
        "error": error,
        "llm_calls": getattr(backend, "calls", None),
        "steps": len(steps),
        "tool_calls": [
            {"tool": step.tool_call.tool, "args": step.tool_call.args}
            for step in steps
        ],
        "candidate": str(candidate),
        "changed_files": changed_files,
        "trajectory_dir": str(root / "builder-run"),
    }


def run(root: Path, args: argparse.Namespace) -> int:
    demo = runpy.run_path(str(Path(__file__).with_name("looplet-refund-demo.py")))
    baseline = write_baseline(root, demo)
    visible_case = cases()[0]
    baseline_result = host_run(root, baseline, visible_case, "baseline", demo)
    failure = {
        "case": visible_case["id"],
        "observed": baseline_result["observed"],
        "expected": visible_case["expected"],
        "termination": baseline_result["termination"],
        "tool_errors": baseline_result["tool_errors"],
    }
    (root / "failure.json").write_text(json.dumps(failure, indent=2) + "\n", encoding="utf-8")
    builder = run_builder(root, baseline, failure, demo, args)
    candidate = Path(builder["candidate"])
    evals_unchanged = all(
        (baseline / relative).read_bytes() == (candidate / relative).read_bytes()
        for relative in [
            path.relative_to(baseline)
            for path in (baseline / "evals").rglob("*")
            if path.is_file()
        ]
    )
    visible_result = host_run(
        root,
        candidate,
        visible_case,
        "candidate-visible-replay",
        demo,
        source_run=Path(baseline_result["run_dir"]),
    )
    holdout_results = {}
    all_cases = cases()
    for case in all_cases[1:]:
        holdout_results[case["id"]] = host_run(
            root, candidate, case, f"candidate-{case['id']}", demo
        )

    acceptance = {
        visible_case["id"]: accepted(visible_result, visible_case["expected"]),
        **{
            case["id"]: accepted(holdout_results[case["id"]], case["expected"])
            for case in all_cases[1:]
        },
    }
    report = {
        "pinned_looplet_revision": REVISION,
        "builder": builder,
        "baseline": {
            "result": baseline_result,
            "accepted_by_host": accepted(baseline_result, visible_case["expected"]),
        },
        "candidate": {
            "visible_replay": visible_result,
            "host_owned_holdouts": holdout_results,
            "acceptance": acceptance,
            "all_host_checks_pass": all(acceptance.values()),
            "candidate_evals_unchanged": evals_unchanged,
        },
        "boundary": {
            "builder_can_edit": True,
            "host_owns_expected_outcomes": True,
            "host_owns_promotion_decision": True,
            "candidate_can_change_its_colocated_evals": True,
            "isolated_evaluator": False,
        },
        "limits": (
            "Scripted mode demonstrates the edit-run-accept protocol, not model discovery. "
            "Live mode demonstrates one provider-backed builder trial, not general autonomous improvement. "
            "The evaluator runs in the same process and is not a hostile-code sandbox."
        ),
    }
    (root / "builder-report.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    print("Builder mode:", args.mode)
    print("Baseline above-limit case:", "PASS" if report["baseline"]["accepted_by_host"] else "FAIL")
    print("Changed files:", ", ".join(builder["changed_files"]) or "none")
    print("Host-owned acceptance:", "PASS" if report["candidate"]["all_host_checks_pass"] else "FAIL")
    print("Candidate evals unchanged:", "YES" if evals_unchanged else "NO")
    print("Evidence:", root)
    return 0 if report["candidate"]["all_host_checks_pass"] and builder["error"] is None else 1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, help="New output directory.")
    parser.add_argument(
        "--mode",
        choices=("scripted", "live"),
        default="scripted",
        help="Use deterministic builder responses or an OpenAI-compatible provider.",
    )
    parser.add_argument("--base-url", help="OpenAI-compatible provider URL for live mode.")
    parser.add_argument("--model", help="Provider model ID for live mode.")
    args = parser.parse_args()
    if args.mode == "live" and not args.base_url and not os.environ.get("OPENAI_BASE_URL") and not os.environ.get("OPENAI_API_KEY"):
        parser.error("live mode requires --base-url or OPENAI_BASE_URL/OPENAI_API_KEY")
    root = args.out.resolve() if args.out else Path(tempfile.mkdtemp(prefix="looplet-refund-builder-"))
    if args.out:
        root.mkdir(parents=True, exist_ok=False)
    return run(root, args)


if __name__ == "__main__":
    raise SystemExit(main())
