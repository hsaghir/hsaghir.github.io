---
title: "Agents Are Files"
description: "Keep an agent's instructions, tools, and rules separate from the application that runs it. Then change them and test whether the agent does a better job."
date: 2026-09-11
featured: true
tags: ["agents", "engineering", "open-source", "python"]
category: "engineering"
cover: "/images/looplet/platform-ownership.png"
coverMobile: "/images/looplet/platform-ownership-mobile.png"
coverAlt: "A stack of files that define an agent, with one changed row highlighted. Edit. Run. Evaluate."
---

Suppose an agent refunds a customer \$250. Its instructions say refunds
above \$100 need review. Every tool call succeeds, but the agent has broken
the rule.

An **agent** is a program that lets a model choose some of its next steps.
Its behavior depends on instructions, tools, rules, and **context**: the
information the model sees. These parts are harder to change and test
when spread through an application.

**Keep these parts together as an agent definition, separate from the
application that runs it.** Put the definition in ordinary files so a person
or another program can read and edit it. Change the definition, run it, and
check what happened.

Another agent can use the same files to build the next version. But making
an edit is not the same as making an improvement. We need a way to compare
the results.

## Separate the agent from the application

The refund rule and investigation instructions belong to the agent
definition. The web server that receives requests does not. Nor does the
billing database.

In [Looplet](https://github.com/hsaghir/looplet), my Python toolkit for
building and evaluating agents, I call this collection a **cartridge**.
It holds the agent's instructions, tools, rules, and context settings in
one directory. The refund example has this layout:

```text
refund.cartridge/
├── cartridge.json
├── config.yaml
├── prompts/
├── tools/
├── hooks/
├── resources/
└── evals/
```

The application that loads and runs the cartridge is the **host**. It supplies
the model connection, libraries, credentials, and workspace. The billing
service still stores payment records and decides which actions are authorized.
A cartridge tool calls that service; it does not replace it.

An **eval** combines a task with checks of the result. For a large refund,
the checks might require no payment and one request awaiting review. Evals
can live beside the definition, making the expected behavior visible. The
test runner loads them separately; loading the cartridge alone does not
evaluate the agent.

An ordinary Python package can provide this separation too. Looplet adds
a standard layout and shared functions for loading, running, and evaluating
definitions. The layout alone does not make the code safe or portable.

Tool arguments, the order of checks, and the format of saved results can
change across versions. Use fixed versions of the definition, Looplet,
and its dependencies, and rerun the
relevant evals when any of them change. Even with fixed versions, a model
may make different choices from one run to the next.

## A proposal is not an action

Suppose a customer says, "I was charged twice." Resolving the complaint may
require comparing transactions, reading an invoice, or asking for missing
details. A model can help choose the next step when the cases vary too much
for a fixed workflow. If the steps are known in advance, write that workflow
directly.

The refund rule is simpler: pay refunds up to \$100, once per request;
send larger requests for review. Code can enforce this rule; the model
does not need to decide it.

<figure>
  <picture>
    <source media="(max-width: 600px)" srcset="/images/looplet/control-loop-mobile.png" width="800" height="1620" />
    <img src="/images/looplet/control-loop.png" alt="The model proposes a tool call. A check allows it or returns a reason for rejection. The model sees the result before choosing its next step. A request to finish goes through a separate check." width="1800" height="1100" loading="lazy" />
  </picture>
  <figcaption>The host supplies the checks. The model receives a tool result or a reason for rejection before choosing its next step. A separate check decides whether it may finish. Finishing does not prove the task was done correctly.</figcaption>
</figure>

The code around the model is the **harness**. It supplies information, runs
tools, applies checks, and decides when to stop. Looplet's `composable_loop`
provides this loop. You add **hooks**, methods it calls at specific points,
to change those behaviors without copying the loop.

A tool can be a short function that calls an existing service.
`lookup_invoice` might call the billing API; `run_tests` might run the
repository's test command.

Calling `done` is a request to finish, not proof of success. A separate check
decides whether to allow it. Looplet also reports why the run stopped.
It reports tool calls after attempting them. To block an action, the check
must run before the tool does.

A check that rejects a request should tell the model why. If a required
check crashes or times out, the run should stop and report that failure.
The Looplet version used in this demo has a bug here: a crashing completion
check can still allow `done`. The [failure test](/looplet-demo-notes/#probe-the-failure-contracts)
reproduces it. An eval can detect missing work afterward, but it cannot
make a broken completion check safe.

And stopping a run cannot undo a payment. The payment service must enforce
authorization and prevent duplicate payments. If a call times out, check
whether it took effect before retrying.

## Make a change you can measure

To test the refund rule, we do not need a live model. The
[runnable refund example](/looplet-demo-notes/#refund-demo-change-the-actual-cartridge)
uses two scripted model responses: request a \$250 refund, then call `done`.
The first version has the rule in its prompt but no code to enforce it.
The refund goes through, and the run finishes without an error.

The tools worked. The task failed.

Now add a hook that blocks the oversized refund and sends it for review.
It checks the original amount so the agent cannot avoid the limit by
splitting the request into smaller refunds.

A new model run might choose a different action and never test this failure.
Instead, Looplet can feed the saved responses through the changed harness.
This is **captured-response replay**: testing the new code with the same
model responses, without another model call.

<figure>
  <picture>
    <source media="(max-width: 600px)" srcset="/images/looplet/replay-comparison-mobile.png" width="800" height="1520" />
    <img src="/images/looplet/replay-comparison.png" alt="Both versions receive the same saved refund request. The original records a 250-dollar refund and no review, with no tool errors. It fails the eval. The changed version blocks the refund, records one pending review, and passes the same eval." width="1800" height="1160" loading="lazy" />
  </picture>
  <figcaption>Both versions receive the same responses and start with empty files. The result checks are unchanged. Only the hook and its configuration differ.</figcaption>
</figure>

**The successful run contains a tool error. The unsuccessful run does not.**
The error reports that the refund was blocked, which is what we wanted.
Both runs reach `done`. Neither an error-free log nor a completion message
tells us whether the task was done correctly.

The eval checks the refund and review records, not the agent's claim of
success. For the \$250 request, it requires no refund and exactly one pending review.
A separate \$50 case must still produce a refund, so blocking every request
would fail. Five more cases test split requests, duplicates, amounts at and
just above the limit, and going straight to `done`. The demo uses scripted
responses and local files; no real payments are made. It tests the harness,
not whether a live model follows instructions.

Copy the cartridge and raise its refund limit from \$100 to \$300. The
hook's configuration stores the amount in cents:

```diff
 kwargs:
-  max_cents: 10000
+  max_cents: 30000
```

Replay the same \$250 request. It now goes through. The original checks
still expect a review, so they fail and the runner exits with status 1.
A file edit changed the behavior; the unchanged evals caught it.

The host runs either definition with the same code. Here `cartridge` is the
definition's directory, `workspace` is a temporary working directory,
`backend` supplies the model responses, and `case` contains the task:

<div class="agent-wiring">

```python
runtime = dict(
    project_root=str(workspace)
)
preset = cartridge_to_preset(
    cartridge,
    runtime=runtime, strict=True
)
try:
    list(preset.run(
        backend, task=case.task
    ))
finally:
    preset.close()
```

</div>

<style>
  .prose .agent-wiring pre,
  .prose .agent-wiring pre code {
    font-size: 13px;
    line-height: 1.65;
    white-space: pre;
    overflow-wrap: normal;
    word-break: normal;
  }
  .prose .agent-wiring pre {
    padding: 16px 12px;
    overflow-x: auto;
  }
</style>

The [full runner](/looplet-refund-demo.py) prepares these inputs, adds the
evals, and saves the responses, tool calls, and results. Follow the
[reproduction steps](/looplet-demo-notes/#refund-demo-change-the-actual-cartridge)
to make the edit and run the comparison.

Replay fixes the model responses, not the tool results. Tools run again,
including their side effects, so use temporary workspaces and test services.
To test a prompt change or
the model's response to rejection, compare repeated runs with a live model.

Save the failure as an eval. Future changes can be checked against it,
even when the person or agent making the change does not know the original
bug.

## Let the results change the design

Sometimes the best change is to remove code. In Looplet Analytics, our
data-analysis agent, I added a review step that asked a model to check the
answer before the run could finish. Both versions already investigated
unclear parts of the task and required a final answer supported by evidence.
The reviewer checked whether the answer matched the agent's written
interpretation of the task. It could reject the answer and ask the agent
to try again.

I tested each version three times on the same eight tasks, for 24 runs per
version. **With review, the benchmark accepted the answer in 5 of 24 runs.
Without review, it accepted 10 of 24.** The review version made **410 model
calls, compared with 338** without review. It finished 22 of its 24 runs;
the version without review finished all 24.

I removed the blocking reviewer. That decision applies to this implementation,
not to reviewers in general. These were reused development tasks, not unseen
tests. The versions also took different steps, so the score difference
cannot be attributed only to the reviewer's rejections. Some expected answers
also conflicted with reasonable readings of the questions.
The [comparison notes](/looplet-demo-notes/#analytics-comparison-removing-blocking-review)
include each trial, the unfinished runs, and the limits of the comparison.

A reviewer can confirm that an answer follows the agent's interpretation
and still accept the wrong answer. The interpretation itself may be wrong.
In this experiment, adding review did not solve that problem. The simpler
version used fewer model calls and did better on the benchmark. Evals
helped decide what to remove, not just what to add.

Context changes also need testing.
If the agent lacks a current record, making its instructions longer may
not help. Looplet's
[memory sources](https://github.com/hsaghir/looplet/blob/8384404061b9c22e9639679d1a4c61fbf9737e22/src/looplet/memory.py)
can fetch information during a run. Context hooks manage old tool results
as the context fills up. Stored notes may be stale or wrong. A note saying
"the customer was approved" is not an
approval record.

**Delegation** means giving part of the task to another agent. Looplet's
[run_sub_loop](https://github.com/hsaghir/looplet/blob/8384404061b9c22e9639679d1a4c61fbf9737e22/src/looplet/subagent.py)
gives the child its own working state and log. That alone does not prevent
it from accessing the parent's files or services. The calling program must
choose the child's tools and checks, limit its budget, and verify its result.
Add another agent only when its contribution justifies the cost.

## Let another agent build the next version

A **builder agent** can create a cartridge from a task description or
revise one after a failed run. It uses the same files and runner as a
developer.

There are two different jobs here. The **task loop** uses tools to handle
a request. The **development loop** edits and tests the agent that will
handle future requests. Each edited version is a **candidate**. It should
not replace the deployed agent just because the builder finished writing it.

<figure>
  <picture>
    <source media="(max-width: 600px)" srcset="/images/looplet/improvement-loop-mobile.png" width="800" height="1480" />
    <img src="/images/looplet/improvement-loop.png" alt="A builder edits and tests a cartridge. Feedback from development tests guides its next edit. Separate checks and release rules decide whether to deploy the candidate. Neither the builder nor the candidate may change those checks." width="1800" height="1000" loading="lazy" />
  </picture>
  <figcaption>The builder can edit and test without waiting for a person at each step. Separate checks decide whether its candidate is ready to deploy. The builder and candidate cannot change those checks.</figcaption>
</figure>

Give the builder a goal, an editable copy, and a time or cost budget. Run
candidates in fresh test workspaces and give the builder the eval results.
Compare each candidate with the starting version on typical tasks, repeating
live model runs where needed. Keep a separate set of tasks for the final
decision. Do not feed their answers back after each edit.

The evals stored with the cartridge run with the agent's permissions.
They are useful for development, but the candidate can change them. In the
[failure test](/looplet-demo-notes/#probe-the-failure-contracts), the refund
still breaks the rule, but changing only the grading code makes the evals
report a pass.

For the final decision, run the candidate separately from the code that
judges its results. Neither the builder nor the candidate should be able to
change those checks or grant itself extra permissions. Putting the files
in another directory is not enough; permissions and isolation must enforce
the separation.

Deployment can be automatic or require a person to review the change. In
either case, the release rules decide whether a passing candidate replaces
the current version. The principle is **automate the edits; keep the final
checks and release decision outside the builder's control.**

In these experiments, a developer added the refund hook and removed the
analytics reviewer. Neither experiment demonstrates autonomous improvement.
The development loop above is a design we could build using the same
interfaces.

## Start with one agent

Start with an agent you already run. Keep its model and tools. Separate
the definition, add checks of its results, and save one real failure as an
eval. The [migration guide](https://hsaghir.com/looplet/migrate/) shows a
small first step. If your current tools already provide this separation,
you do not need to replace them.

For each run, record the definition and model versions, inputs, tool calls,
results, why it stopped, and any files it produced. Use those records to
understand failures and turn them into evals. When comparing versions, check
both the quality of the work and whether the agent broke any rules. Also
measure time, model usage, and cost.

Teams can use the same runtime and eval tools without sharing credentials
or business data. Keep access control, storage, worker isolation, and
reliable job scheduling in the surrounding infrastructure. The
[demo notes](/looplet-demo-notes/) contain commands, exact versions, and
more examples, including a recorded coding run.

A running agent answers requests. A separate agent definition gives us
something we can improve: inspect the files, make a change, run it, and
compare the results. The edit may come from a person or another agent.
Either way, the evals help us decide whether to keep it.

**Own the harness. Turn failures into evals.**