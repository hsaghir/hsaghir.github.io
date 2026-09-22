---
layout: ../layouts/PageLayout.astro
title: Looplet Demo Notes
description: Reproduce the Looplet cartridge demonstrations, inspect their evidence, and understand their limits.
---

Supporting material for [Agents Are Files](/blog/2026-09-11-agents-are-files/).
The refund and coding demos use public Looplet revision
[`8384404`](https://github.com/hsaghir/looplet/tree/8384404061b9c22e9639679d1a4c61fbf9737e22).
The analytics comparison below is a separate recorded experiment. These
artifacts support the article, not claims of general reliability or production
readiness.

## Refund demo: change the actual cartridge

Download the [refund demo script](/looplet-refund-demo.py). With
[uv](https://docs.astral.sh/uv/) and Git installed, run:

```bash
uv run --no-project --python 3.12 \
    --with 'git+https://github.com/hsaghir/looplet.git@8384404061b9c22e9639679d1a4c61fbf9737e22' \
    looplet-refund-demo.py --out ./refund-demo
```

Choose a new output directory. Installation may need network access; execution
uses scripted model responses, local JSON ledgers, and no payment service or
API key. Pinning source makes the version used in this experiment explicit.

The three comparison runs produce:

```text
v1  $250 request -> $250 paid, no review -> outcome evals FAIL
v2  same responses -> $0 paid, one pending review -> outcome evals PASS
v2  $50 control -> $50 paid, no review -> outcome evals PASS
```

Five additional cases cover a split request, duplicate refund, amounts exactly
at and just above the limit, and going straight to `done` without attempting a
refund. The hook reads the host-supplied original request rather than trusting
the amount in each proposed call. It uses `pre_dispatch` for refund calls and
`check_done` for terminal calls, which bypass `pre_dispatch`.

Two outcome graders compare the ledger and pending-review records with exact
expectations. A separate termination grader checks that the loop reached
`done`. That last check is not proof of task success. Disabling the queue write
fails the review grader even if the ledger stays empty.

Open `harness.diff` in the output directory. Compare `ledger.json` and
`reviews.json` under each `workspaces/` run directory. Captured responses,
trajectories, and eval results are under `runs/`.

Now copy the cartridge:

```bash
cp -R refund-demo/refund-v2.cartridge ./my-refund.cartridge
```

Inside the copy, open `config.yaml` in the `00_RefundLimit` hook directory.
Change `max_cents: 10000` to `max_cents: 30000`. Leave the evals untouched and
run the copied definition against the original captured responses:

```bash
uv run --no-project --python 3.12 \
    --with 'git+https://github.com/hsaghir/looplet.git@8384404061b9c22e9639679d1a4c61fbf9737e22' \
    looplet-refund-demo.py \
    --cartridge ./my-refund.cartridge \
    --replay-from ./refund-demo/runs/v1-above-limit \
    --out ./changed-refund
```

Expected exit status: **1**. The larger limit allows the \$250 refund and
creates no review record, so both unchanged outcome evals fail. The command
saves its results before exiting. Restore the limit and rerun into another new
output directory to get a pass. The runner does not regenerate the cartridge
when `--cartridge` is supplied.

### What this does not establish

Replay fixes the model responses, not the world. Tools execute again, with
fresh side effects. It cannot predict a live model's reaction to a denial or
measure a prompt/model improvement. Those require fresh sampled runs.

The hook was supplied by the demo author, not discovered by an autonomous
builder. The experiment verifies that an edited cartridge can run through
the same interfaces and be checked against unchanged outcomes. It does not
demonstrate autonomous agent improvement.

A pending-review record does not mean a human has reviewed or approved the
request. The demo does not grade customer-facing language, implement durable
payment idempotency, or coordinate concurrent workers. Another tool with
arbitrary filesystem access could bypass this local policy. Real payments
need service-side authorization; arbitrary candidate code needs isolation.

### Probe the failure contracts

The optional probe mode runs the eight original cases, then reproduces two
limitations of the pinned public revision:

```bash
uv run --no-project --python 3.12 \
    --with 'git+https://github.com/hsaghir/looplet.git@8384404061b9c22e9639679d1a4c61fbf9737e22' \
    python -B looplet-refund-demo.py --probe-contracts --out ./refund-probes
```

The completion probe creates a hook that records its invocation and raises
an exception. On a done-only oversized request, the loop logs the exception
and accepts `done`; the required review is absent, so the outcome check fails.
The traceback is expected. It is the behavior this probe verifies, not a
successful completion policy. The
[pinned runtime code](https://github.com/hsaghir/looplet/blob/8384404061b9c22e9639679d1a4c61fbf9737e22/src/looplet/loop.py#L3139)
treats the exception as no decision. Do not assume another version has the
same failure contract without checking it.

The second probe copies the original, incorrect agent and changes only its
graders to return true. It replays the same proposed calls and records the
same incorrect \$250 refund without a review. All candidate-supplied graders
now report a pass. Comparing the actual files with the original fixed
expectations still produces two failures.

Both observations are asserted and saved to `contract-probes.json`, alongside
the generated cartridges and run artifacts. The command exits 0 only when
the original cases and these expected observations hold. The fixed comparisons
are made by the demo runner in the same process; this does **not** implement
an isolated evaluator or establish protection against hostile code.

## Builder demo: a cartridge can improve under host tests

Download the [builder demo](/looplet-refund-builder-demo.py). Its default mode
uses scripted builder responses so the protocol is reproducible without an API
key:

```bash
uv run --no-project --python 3.12 \
    --with 'git+https://github.com/hsaghir/looplet.git@8384404061b9c22e9639679d1a4c61fbf9737e22' \
    looplet-refund-builder-demo.py --out ./refund-builder-demo
```

The builder reads the failed run, edits a copied cartridge, and calls `done`.
The host then runs the copy through the same Looplet interfaces. It replays
the failed request and checks six more cases: a valid refund, split and
duplicate attempts, the exact limit, just above the limit, and finishing
without a refund call.

The expected results are in the host script, not the candidate cartridge. The
run reports:

```text
Builder mode: scripted
Baseline above-limit case: FAIL
Changed files: hooks/00_RefundLimit/config.yaml, hooks/00_RefundLimit/hook.py
Host-owned acceptance: PASS
Candidate evals unchanged: YES
```

This demonstrates the cartridge boundary and the copy-edit-run-keep loop.
It does not demonstrate that a live model can discover the repair or that
autonomous improvement works in general. To run a provider-backed builder,
set `OPENAI_BASE_URL` and `OPENAI_MODEL` (or `OPENAI_API_KEY`) and add
`--mode live`. The evaluator still runs in the same process, so this is not
a hostile-code sandbox or a production promotion system.

### Live hill climb: the model improves the cartridge

On September 21, 2026, I ran the builder against a live `gpt-5.6-sol` model
through an OpenAI-compatible Copilot LM Proxy. The proxy's Responses API was
used so tool calls could continue across turns:

```bash
uv run --no-project --python 3.12 \
    --with openai \
    --with 'git+https://github.com/hsaghir/looplet.git@8384404061b9c22e9639679d1a4c61fbf9737e22' \
    looplet-refund-builder-demo.py \
    --mode live --protocol responses \
    --base-url http://127.0.0.1:19823/v1 \
    --model gpt-5.6-sol --iterations 3 --out ./live-refund-builder
```

The first copy fixed the visible `$250` failure but failed the split refund
and no-refund-call cases. The host sent those failures back to the builder.
The next round fixed the completion case. The final round changed the hook
to return Looplet's `HookDecision(permission="deny")` for each oversized
dispatch. All seven host tests then passed:

```text
above_limit          PASS
within_limit         PASS
split_request        PASS
duplicate_refund     PASS
at_limit              PASS
just_above_limit     PASS
done_only_above_limit PASS
```

The live builder took three rounds. It changed only
`hooks/00_RefundLimit/config.yaml` and `hooks/00_RefundLimit/hook.py`; the
nine candidate eval files were byte-identical to the baseline. This is a
single bounded experiment, not evidence that every model or task will improve.

### Runtime details behind the diagrams

The cover illustrates an editable agent definition, with a changed hook
highlighted. It is not a runtime or security diagram. Prompts, tools, memory,
context configuration, and task-specific hooks belong to the definition.
Runtime bindings, credentials, service authorization, and release policy are
separately controlled. A file layout does not enforce that separation:
arbitrary candidate code needs isolation.

The control-loop diagram shows ordinary tool calls and completion requests
with application-supplied checks. In the runtime, `pre_prompt` can add context
and `build_prompt` can replace prompt construction. For ordinary tools,
`pre_dispatch` and `check_permission` run before the tool body. Tool results
and denials become feedback for the next model call. `post_dispatch` handles
results, and `should_stop` can stop independently of the drawn completion path.

Terminal `done` takes a separate path through `check_done`. Rejection asks
for more work. Once accepted, the pinned runtime dispatches the registered
done tool and reports its step, details collapsed into the diagram's Stop
node. The diagram shows normal control flow, not a guarantee of safe error
handling: the exception probe above demonstrates the pinned completion
hook's fail-open behavior. Required checks need a tested failure contract;
detecting a bad outcome afterward does not prevent the earlier action.

A returned step is after the attempt, not an approval gate. Several calls in
a batch may execute before their steps are yielded. No combination of editable
hooks is an operating-system or service-authorization boundary.

The article's Python excerpt uses the runner's imports, prepared workspace,
eval case, and backend. It shows only loading and execution. The full runner
separately loads the evals, installs `EvalHook` and `TrajectoryRecorder`,
saves evidence with `save_eval_run`, and captures responses for replay.
The excerpt is not a standalone script or a protected release verifier;
use the full script above to reproduce the experiment.

The development-loop diagram is a design enabled by these interfaces, not
a recorded builder run or an automatic optimizer supplied by the cartridge
loader. A controller must bound the builder, arrange isolated candidate runs,
collect evidence, and apply release policy. The feedback arrow represents
development results, not held-out answers revealed after every edit. Acceptance
checks and permissions remain outside both the builder's and candidate's
control; promotion may be automatic or human-reviewed under the host's policy.

For release evaluation, run candidate code in an isolated worker, then
evaluate its outputs in a separately controlled environment. Treat those
outputs as untrusted input. A check stored in another directory is not
protected if candidate code still runs with the evaluator's permissions.

## Analytics comparison: removing blocking review

On September 18-19, 2026, Looplet Analytics compared two versions of its
data-analysis workflow on eight DAB public development cases. Both delegated
uncertainty investigation and required an explicit final commitment tied to
evidence. One added a model reviewer that could reject a commitment inconsistent
with the agent's recorded interpretation. It was a consistency check, not an
independent source of ground truth.

Each variant ran all eight cases in three fresh trials, using the same frozen
runtime (fingerprint prefix `c6fab872`), requested model `gpt-5.6-luna`, and
reasoning effort `max`. Each case had a shared 200-call allowance, 16,000 output
tokens per call, and a 3,600-second limit. Each arm used four workers. Arm order
was no-review/review, review/no-review, then no-review/review. No source changes
or selective retries were made during the comparison.

Answers accepted by the official benchmark validator:

| Trial | No Review | Review |
| --- | ---: | ---: |
| 1 | 3/8 | 1/8 |
| 2 | 2/8 | 2/8 |
| 3 | 5/8 | 2/8 |
| Total | 10/24 | 5/24 |

Without review, all 24 runs completed and used 338 model calls: 112, 110, and
116 by trial. With review, 22/24 completed and used 410 calls: 129, 158, and
123. The 30 reviewer calls are included in that total, not additional to it.
These are invocation counts, not token costs or dollar costs.

The two incomplete reviewed runs exhausted their review allowance, then hit
a transport defect: synthetic parse-error IDs were sent as tool outputs even
though the provider had issued no matching calls. Their raw records retain
37 and 30 calls. The original aggregate summaries omitted those 67 calls;
410 restores them to the reported 343. The failed runs remain in every
denominator. All 48 saved grade sets, including the incomplete cases, were
reproduced offline without altering the original artifacts.

The traces also limit the conclusion. The variants generated different
trajectories, so the score gap is not solely an effect of reviewer vetoes.
Some benchmark answers conflicted with plausible task interpretations. In
one case, review rejected a benchmark-accepted answer because it excluded
zero-transfer agents from the recorded population. Other incorrect answers
passed review because they were consistent with the chosen interpretation.

These are three repetitions of eight reused development tasks, not 24
independent tasks or a held-out estimate. The result justified removing this
blocking reviewer, not a general claim that reviewers harm accuracy. The
implementation subsequently removed the reviewer and repaired the transport
and incomplete-run accounting paths. Those repairs were not folded into the
recorded comparison, and removal was a developer decision, not autonomous
cartridge evolution.

## Live coding demo: inspect the generated work

Download the [live coding runner](/looplet-coder-demo.py) or the
[recorded evidence archive](/looplet-coder-evidence.zip).

The runner uses the public coder cartridge on the ledger-refactor task from
the [existing benchmark](https://github.com/hsaghir/looplet/blob/8384404061b9c22e9639679d1a4c61fbf9737e22/benchmarks/coder_vs_agents/hard_tasks.py).
It starts with a monolithic account ledger and four tests, asks for separate
implementation modules while preserving the public API, and requests
transaction history and undo support.

### Recorded result and caveats

One live generation requested `gpt-5.4` through a local OpenAI-compatible proxy.
It produced a public API module, an account model, and a ledger service.
The restored original tests passed, as did five independent checks:

- History and withdrawal undo.
- Deposit undo.
- Transfer undo from the recipient, restoring both balances.
- Rejected withdrawals leaving balance and history unchanged.
- At least two imported implementation modules containing functions or classes.

These checks are bounded examples, not exhaustive API or financial correctness
verification. They were supplied only after the agent process exited, and
rerun on the exact generated code in a fresh, read-only, network-disabled
container.

The run took **40 steps and about 9.6 minutes**, including argument-type errors
and response-parsing retries. An earlier setup attempt failed to reach the
provider and received no model response. There was no second live generation
or manual repair of the generated implementation.

The initial runner returned exit status 1 because its byte-identical test-file
gate rejected two tests the agent appended. An AST comparison confirmed the
original checks were preserved. The final runner retains file identity as a
diagnostic, but acceptance depends on normal completion, the restored original
tests, and independent checks. That corrected path was verified against the
same saved artifact, without another model call. The original rejection is
preserved in the archive; it has not been rewritten as an original clean pass.

Missing-final-newline markers were added to the exported patch so it can be
applied normally. Applying it reproduced the implementation byte-for-byte.
The archive includes the patch, verified source, step/error log, verifier,
test results, task, model/runtime metadata, and an audit of these corrections.
Full captured model-call evidence stays in the local run output; it is not
included in the downloadable archive.

### Run it yourself

Requirements: Linux, Docker, uv, and an OpenAI-compatible provider. Set
`OPENAI_BASE_URL`, `OPENAI_MODEL`, and, if required, `OPENAI_API_KEY` in your
shell. Do not put secrets in the script. Then run:

```bash
uv run --no-project --python 3.12 \
    looplet-coder-demo.py --out ./coder-demo
```

The provider address must be reachable **from Docker**. On Docker Desktop,
a host-local provider may need `host.docker.internal` instead of `localhost`.
The script downloads pinned Looplet source and builds a Python image with
pinned provider SDK, pytest, and Ruff versions. It has a 40-step budget and a
ten-minute timeout for the agent invocation. Image setup is separate.

This makes real model calls and can incur charges. Do not use sensitive inputs:
task and working context go to your configured provider. The container limits
filesystem access but the agent phase uses host networking. This is a disposable
demonstration environment, not a hardened sandbox for hostile code. Verification
runs in a fresh network-disabled container. The checks are not a tamper-proof
oracle against a malicious candidate.

Inspect `agent.patch`, `result.json`, and the verification logs. The generated
project and raw run evidence remain under `workspace/`. A nonzero exit or failed
check is a result to inspect, not something to discard until a run passes.

## Additional evidence and context

The older [four-task comparison](https://github.com/hsaghir/looplet/blob/8384404061b9c22e9639679d1a4c61fbf9737e22/benchmarks/coder_vs_agents/HARD_REPORT.md)
reported four hidden-suite passes each for Looplet and Copilot CLI. Both paths
requested the same model family through different serving connections. This
does not establish general parity or isolate the harness from the provider.

[Tinyloop conformance tests](https://github.com/hsaghir/looplet/blob/8384404061b9c22e9639679d1a4c61fbf9737e22/tests/conformance/test_tinyloop_cross_runtime.py)
cover five declarative fixtures and two rejection cases with a second loader.
They support a bounded portability contract, not universal runtime equivalence.

Related approaches include [Anthropic's agent guidance](https://www.anthropic.com/engineering/building-effective-agents),
[Pi](https://mariozechner.at/posts/2025-11-30-pi-coding-agent/),
[Agent Skills](https://agentskills.io/what-are-skills),
[PydanticAI testing](https://pydantic.dev/docs/ai/guides/testing/), and
[LangGraph](https://docs.langchain.com/oss/python/langgraph/overview).
Looplet does not claim to have invented simple loops, file-based instructions,
or testing. Its proposition is their combination around an owned, editable
agent harness.