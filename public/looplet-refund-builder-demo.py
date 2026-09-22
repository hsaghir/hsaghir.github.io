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
    HookDecision,
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
After failure.json, read only these relevant files if needed:
candidate.cartridge/prompts/system.md and
candidate.cartridge/tools/refund/execute.py. Then create the policy hook in
candidate.cartridge/hooks/00_RefundLimit/config.yaml and hook.py. Do not
inventory the cartridge or inspect its eval files. Make the candidate edit
before calling done.

The hook config must use this Looplet cartridge schema, not a guessed schema:
class_name: RefundLimit
kwargs:
    max_cents: 10000
    project_dir: "@project_dir"
The hook.py must define class RefundLimit with pre_dispatch and check_done.
Review records must use the exact request_id from request.json, never the
workspace directory name or an empty fallback.
The host will load the candidate before accepting done and will report any
loading error back to you.
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


class ResponsesProxyBackend:
    """OpenAI Responses backend with stateful function-call continuation."""

    def __init__(self, *, base_url: str, api_key: str, model: str) -> None:
        from openai import OpenAI

        self.client = OpenAI(base_url=base_url, api_key=api_key)
        self.model = model
        self.calls = 0
        self.previous_response_id: str | None = None
        self.pending_call_ids: list[str] = []

    @staticmethod
    def _tools(schemas: list[dict]) -> list[dict]:
        return [
            {
                "type": "function",
                "name": schema["name"],
                "description": schema.get("description", ""),
                "parameters": schema.get(
                    "input_schema", {"type": "object", "properties": {}}
                ),
            }
            for schema in schemas
        ]

    def _input(self, prompt: str) -> str | list[dict]:
        if not self.previous_response_id or not self.pending_call_ids:
            return prompt
        return [
            {
                "type": "function_call_output",
                "call_id": call_id,
                "output": prompt,
            }
            for call_id in self.pending_call_ids
        ]

    def _blocks(self, response: object) -> list[dict]:
        blocks = []
        self.pending_call_ids = []
        for item in getattr(response, "output", []) or []:
            if getattr(item, "type", None) == "function_call":
                self.pending_call_ids.append(item.call_id)
                try:
                    arguments = json.loads(item.arguments)
                except (TypeError, ValueError):
                    arguments = {"_raw_arguments": item.arguments}
                blocks.append(
                    {
                        "type": "tool_use",
                        "id": item.call_id,
                        "name": item.name,
                        "input": arguments,
                    }
                )
        return blocks

    def generate_with_tools(
        self,
        prompt: str,
        *,
        tools: list[dict],
        max_tokens: int = 2000,
        system_prompt: str = "",
        temperature: float = 0.2,
    ) -> list[dict]:
        self.calls += 1
        kwargs = {
            "model": self.model,
            "input": self._input(prompt),
            "tools": self._tools(tools),
            "max_output_tokens": max_tokens,
        }
        if self.previous_response_id:
            kwargs["previous_response_id"] = self.previous_response_id
        if system_prompt and not self.previous_response_id:
            kwargs["instructions"] = system_prompt
        response = self.client.responses.create(**kwargs)
        self.previous_response_id = response.id
        return self._blocks(response)

    def generate(
        self,
        prompt: str,
        *,
        max_tokens: int = 2000,
        system_prompt: str = "",
        temperature: float = 0.2,
    ) -> str:
        self.calls += 1
        kwargs = {
            "model": self.model,
            "input": self._input(prompt),
            "max_output_tokens": max_tokens,
        }
        if self.previous_response_id:
            kwargs["previous_response_id"] = self.previous_response_id
        if system_prompt and not self.previous_response_id:
            kwargs["instructions"] = system_prompt
        response = self.client.responses.create(**kwargs)
        self.previous_response_id = response.id
        self.pending_call_ids = []
        return getattr(response, "output_text", "") or ""


def live_backend(args: argparse.Namespace):
    if args.protocol == "responses":
        return ResponsesProxyBackend(
            base_url=args.base_url or os.environ["OPENAI_BASE_URL"],
            api_key=os.environ.get("OPENAI_API_KEY") or "local",
            model=args.model or os.environ.get("OPENAI_MODEL", "gpt-4o-mini"),
        )
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


class CandidateChangedHook:
    def __init__(self, baseline: Path, candidate: Path) -> None:
        self.baseline = baseline
        self.candidate = candidate

    def check_done(self, state, session_log, context, step_num):
        _, changed_files = diff_tree(self.baseline, self.candidate)
        if not changed_files:
            return HookDecision(
                block="The candidate has no file changes. Edit candidate.cartridge before done."
            )
        try:
            preset = cartridge_to_preset(
                str(self.candidate),
                runtime={"project_root": str(self.candidate.parent)},
                strict=True,
            )
        except (ImportError, KeyError, OSError, RuntimeError, TypeError, ValueError) as exc:
            return HookDecision(
                block=(
                    "The candidate cartridge does not load. Fix the files before done: "
                    f"{type(exc).__name__}: {exc}"
                )
            )
        else:
            preset.close()
        return None


def run_builder(
    root: Path,
    baseline: Path,
    failure: dict,
    demo: dict,
    args: argparse.Namespace,
    *,
    candidate: Path | None = None,
    iteration: int = 1,
    feedback_file: str | None = None,
) -> dict:
    if candidate is None:
        builder_root = root / "builder-workspace"
        builder_root.mkdir()
        candidate = builder_root / "candidate.cartridge"
        shutil.copytree(baseline, candidate)
        (builder_root / "failure.json").write_text(
            json.dumps(failure, indent=2) + "\n", encoding="utf-8"
        )
    else:
        builder_root = candidate.parent
    (builder_root / "task.md").write_text(
        "Repair the candidate cartridge after the visible refund-policy failure.\n",
        encoding="utf-8",
    )

    if args.mode == "scripted":
        backend = MockLLMBackend(scripted_builder_responses(demo), cycle=False)
    else:
        backend = live_backend(args)
    system_prompt = BUILDER_SYSTEM_PROMPT
    if feedback_file:
        system_prompt += f"""

This is hill-climb iteration {iteration}. Read {feedback_file} first. It contains
host-owned holdout failures from the previous candidate. Repair the existing
candidate with the smallest edit, preserve all eval files, and call done only
after the candidate loads.
"""
    sink = ProvenanceSink(dir=root / f"builder-run-iteration-{iteration}")
    recorder = sink.trajectory_hook()
    preset = coding_agent_preset(
        workspace=str(builder_root),
        max_steps=24,
        require_tests=False,
        system_prompt=system_prompt,
    )
    preset.config.use_native_tools = True
    allowed_tools = {"read", "write", "edit", "done"}
    preset.tools._tools = {
        name: spec
        for name, spec in preset.tools._tools.items()
        if name in allowed_tools
    }
    preset.hooks = [
        *preset.hooks,
        CandidateChangedHook(baseline, candidate),
        recorder,
    ]
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
        "builder_root": str(builder_root),
        "iteration": iteration,
        "feedback_file": feedback_file,
        "changed_files": changed_files,
        "trajectory_dir": str(root / f"builder-run-iteration-{iteration}"),
    }


def evaluate_candidate(
    root: Path,
    baseline: Path,
    candidate: Path,
    baseline_result: dict,
    demo: dict,
    iteration: int,
) -> dict:
    all_cases = cases()
    visible_case = all_cases[0]
    visible_result = host_run(
        root,
        candidate,
        visible_case,
        f"candidate-visible-replay-{iteration}",
        demo,
        source_run=Path(baseline_result["run_dir"]),
    )
    holdout_results = {
        case["id"]: host_run(
            root, candidate, case, f"candidate-{iteration}-{case['id']}", demo
        )
        for case in all_cases[1:]
    }
    acceptance = {
        visible_case["id"]: accepted(visible_result, visible_case["expected"]),
        **{
            case["id"]: accepted(holdout_results[case["id"]], case["expected"])
            for case in all_cases[1:]
        },
    }
    evals_unchanged = all(
        (baseline / relative).read_bytes() == (candidate / relative).read_bytes()
        for relative in [
            path.relative_to(baseline)
            for path in (baseline / "evals").rglob("*")
            if path.is_file()
        ]
    )
    return {
        "visible_replay": visible_result,
        "host_owned_holdouts": holdout_results,
        "acceptance": acceptance,
        "all_host_checks_pass": all(acceptance.values()),
        "candidate_evals_unchanged": evals_unchanged,
    }


def write_holdout_feedback(
    builder_root: Path, evaluation: dict, iteration: int
) -> str:
    expected_by_id = {case["id"]: case["expected"] for case in cases()}
    observed_by_id = {
        "above_limit": evaluation["visible_replay"],
        **evaluation["host_owned_holdouts"],
    }
    failures = []
    for case_id, passed in evaluation["acceptance"].items():
        if passed:
            continue
        result = observed_by_id[case_id]
        expected_reviews = expected_by_id[case_id].get("reviews", [])
        observed_reviews = result["observed"].get("reviews", [])
        failure = {
            "case": case_id,
            "expected": expected_by_id[case_id],
            "observed": result["observed"],
            "tool_errors": result["tool_errors"],
            "termination": result["termination"],
        }
        expected_ids = {
            review.get("request_id")
            for review in expected_reviews
            if isinstance(review, dict)
        }
        observed_ids = {
            review.get("request_id")
            for review in observed_reviews
            if isinstance(review, dict)
        }
        if expected_ids != observed_ids:
            failure["host_guidance"] = (
                "Every review record must use the exact request_id from the host "
                "request.json. Do not use the workspace directory name, a generated "
                "fallback, or an empty string."
            )
        if case_id == "split_request":
            failure["host_guidance"] = failure.get("host_guidance", "") + (
                " "
                if "host_guidance" in failure
                else ""
            ) + (
                "The original request is 25000 cents. The review record is correct, "
                "but every partial refund still reached the ledger. In Looplet, "
                "pre_dispatch must return HookDecision(permission='deny', block='...') "
                "to stop a tool call; a plain string is not a dispatch denial."
            )
        elif case_id == "done_only_above_limit":
            failure["host_guidance"] = failure.get("host_guidance", "") + (
                " "
                if "host_guidance" in failure
                else ""
            ) + (
                "check_done must create the pending review for an oversized original "
                "request even when no refund tool call occurred."
            )
        failures.append(failure)
    name = f"holdout-failures-iteration-{iteration + 1}.json"
    (builder_root / name).write_text(
        json.dumps(
            {
                "instruction": "Repair only candidate.cartridge. Do not edit eval files.",
                "source_iteration": iteration,
                "failures": failures,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return name


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
    (root / "failure.json").write_text(
        json.dumps(failure, indent=2) + "\n", encoding="utf-8"
    )

    candidate = None
    iteration_reports = []
    for iteration in range(1, args.iterations + 1):
        feedback_file = None
        if iteration > 1:
            feedback_file = write_holdout_feedback(
                candidate.parent, iteration_reports[-1]["evaluation"], iteration - 1
            )
        builder = run_builder(
            root,
            baseline,
            failure,
            demo,
            args,
            candidate=candidate,
            iteration=iteration,
            feedback_file=feedback_file,
        )
        candidate = Path(builder["candidate"])
        evaluation = evaluate_candidate(
            root, baseline, candidate, baseline_result, demo, iteration
        )
        iteration_reports.append(
            {"iteration": iteration, "builder": builder, "evaluation": evaluation}
        )
        if evaluation["all_host_checks_pass"] or builder["error"] is not None:
            break

    final_iteration = iteration_reports[-1]
    builder = final_iteration["builder"]
    evaluation = final_iteration["evaluation"]
    report = {
        "pinned_looplet_revision": REVISION,
        "iterations_requested": args.iterations,
        "iterations_completed": len(iteration_reports),
        "iterations": iteration_reports,
        "builder": builder,
        "baseline": {
            "result": baseline_result,
            "accepted_by_host": accepted(baseline_result, visible_case["expected"]),
        },
        "candidate": evaluation,
        "boundary": {
            "builder_can_edit": True,
            "host_owns_expected_outcomes": True,
            "host_owns_promotion_decision": True,
            "candidate_can_change_its_colocated_evals": True,
            "isolated_evaluator": False,
        },
        "limits": (
            "Scripted mode demonstrates the edit-run-accept protocol, not model discovery. "
            "Live mode demonstrates a bounded provider-backed hill climb, not general autonomous improvement. "
            "The evaluator runs in the same process and is not a hostile-code sandbox."
        ),
    }
    (root / "builder-report.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    print("Builder mode:", args.mode)
    print("Iterations:", report["iterations_completed"])
    print("Baseline above-limit case:", "PASS" if report["baseline"]["accepted_by_host"] else "FAIL")
    print("Changed files:", ", ".join(builder["changed_files"]) or "none")
    print("Host-owned acceptance:", "PASS" if report["candidate"]["all_host_checks_pass"] else "FAIL")
    print(
        "Candidate evals unchanged:",
        "YES" if report["candidate"]["candidate_evals_unchanged"] else "NO",
    )
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
    parser.add_argument(
        "--iterations",
        type=int,
        default=1,
        help="Maximum builder feedback iterations (default: 1).",
    )
    parser.add_argument(
        "--protocol",
        choices=("chat", "responses"),
        default="chat",
        help="Provider protocol for live mode.",
    )
    args = parser.parse_args()
    if args.iterations < 1:
        parser.error("--iterations must be positive")
    if args.mode == "live" and not args.base_url and not os.environ.get("OPENAI_BASE_URL") and not os.environ.get("OPENAI_API_KEY"):
        parser.error("live mode requires --base-url or OPENAI_BASE_URL/OPENAI_API_KEY")
    root = args.out.resolve() if args.out else Path(tempfile.mkdtemp(prefix="looplet-refund-builder-"))
    if args.out:
        root.mkdir(parents=True, exist_ok=False)
    return run(root, args)


if __name__ == "__main__":
    raise SystemExit(main())
