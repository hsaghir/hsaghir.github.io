---
title: "Agents Are Files"
description: "Keep an agent's behavior in editable files, separate from the application that runs it. Then let another agent edit and test new versions against tests it cannot change."
date: 2026-09-21
featured: true
tags: ["agents", "engineering", "open-source", "python"]
category: "engineering"
cover: "/images/looplet/platform-ownership.png"
coverMobile: "/images/looplet/platform-ownership-mobile.png"
coverAlt: "A stack of files that define an agent, with one changed row highlighted. Edit. Run. Evaluate."
---

Suppose an agent refunds a customer \$250. Its instructions say refunds above
\$100 need review. Every tool call succeeds, but the agent has broken the rule.

This is the hard part of building agents. Their behavior is spread across an
application: instructions in one place, tools in another, memory somewhere
else, and rules for stopping or blocking actions somewhere else again. When
the agent fails, it is hard to know what to change. When we make a change, it
is hard to know whether the change helped.

The answer is to give the agent's changeable behavior a clear home. Keep it
separate from the application that runs it. Put it in ordinary files so a
person or another agent can inspect it, edit it, version it, and test it.

That gives us a useful development loop: copy the definition, change it, run
it, and compare the result with the old version. The agent can make the edit,
but the application that runs it can keep control of the tests and the final
decision.

## Put behavior in its own unit

The refund rule and investigation instructions belong to the agent. The web
server that receives requests does not. Nor does the billing database.

In [Looplet](https://github.com/hsaghir/looplet), my Python toolkit for
building and evaluating agents, I call the agent's files a **cartridge**:

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

This boundary gives the team a small unit to work on. A person can review a
change without reviewing a copied web server. A new version can keep the same
tools and change only its instructions or policy. Another agent can edit a
copy without needing access to the whole application.

An **eval** is a task with checks of its result. For a large refund, the check
might require no payment and one request waiting for review. Evals make the
expected behavior visible. They can travel with the cartridge for development,
while the host can keep separate tests for final acceptance.

The files do not make the code safe by themselves. They are not a sandbox,
and they do not grant service authorization. The host still controls the
runtime, credentials, and release decision. Tool formats and saved results are
also contracts, so pin versions and rerun the relevant evals when they change.

## Keep the model's choice separate from the action

Suppose a customer says, "I was charged twice." Resolving the complaint may
require comparing transactions, reading an invoice, or asking for missing
details. A model can help choose the next step when the path is not known in
advance. If the steps are fixed, ordinary application code is simpler.

The refund rule is fixed: pay refunds up to \$100 once per request, and send
larger requests for review. The model may ask for a refund, but code should
enforce this rule.

<figure>
  <picture>
    <source media="(max-width: 600px)" srcset="/images/looplet/control-loop-mobile.png" width="800" height="1620" />
    <img src="/images/looplet/control-loop.png" alt="The model proposes a tool call. A check allows it or returns a reason for rejection. The model sees the result before choosing its next step. A request to finish goes through a separate check." width="1800" height="1100" loading="lazy" />
  </picture>
  <figcaption>The host supplies the checks. The model receives a tool result or a reason for rejection before choosing its next step. A separate check decides whether it may finish. Finishing does not prove the task was done correctly.</figcaption>
</figure>

The code around the model is the **harness**. It supplies information, runs
tools, applies checks, and decides when to stop. The same harness can run many
cartridges, so changing an agent does not require copying the execution loop.

Calling `done` is a request to finish, not proof of success. A separate check
decides whether to allow it. A returned record describes an action after it
has been attempted. To block an action, the check must run before the tool.

A check that rejects a request should tell the model why. If a required check
crashes or times out, the run should stop and report that failure. The Looplet
version used in this demo has a bug here: a crashing completion check can still
allow `done`. The [failure test](/looplet-demo-notes/#probe-the-failure-contracts)
reproduces it. A later eval can find missing work, but it cannot make a broken
completion check safe.

Stopping a run cannot undo a payment. The payment service must enforce
authorization and prevent duplicate payments. If a call times out, check
whether it took effect before retrying.

## Turn a failure into a test

The first useful test is not "did the model finish?" It is "did the customer
get the right result?"

The [runnable refund example](/looplet-demo-notes/#refund-demo-change-the-actual-cartridge)
uses two fixed model responses: request a \$250 refund, then call `done`. The
first version has the rule in its prompt but no code to enforce it. The refund
goes through, and the run finishes without an error.

The tools worked. The task failed.

Now add a check that blocks the oversized refund and sends it for review. It
checks the original request, so the agent cannot avoid the limit by splitting
the request into smaller refunds.

If we call the model again, it might choose a different action and never test
this failure. Instead, we feed the saved responses through the changed host.
This is **captured-response replay**: testing new code with the same model
responses and fresh tool state.

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
success. For the \$250 request, it requires no refund and one pending review.
A separate \$50 case must still produce a refund, so blocking every request
would fail. Five more cases test split requests, duplicates, the limit, and
finishing without a refund call. The demo uses local files, not real payments.
It tests the host code, not whether a live model follows instructions.

Copy the cartridge and raise its refund limit from \$100 to \$300. The
policy configuration stores the amount in cents:

```diff
 kwargs:
-  max_cents: 10000
+  max_cents: 30000
```

Replay the same \$250 request. It now goes through. The original checks
still expect a review, so they fail and the runner exits with status 1.
A file edit changed the behavior; the unchanged evals caught it.

The host runs both definitions with the same runner. The [full runner](/looplet-refund-demo.py)
prepares the workspace, adds the checks, and saves the responses, tool calls,
and results. Follow the
[reproduction steps](/looplet-demo-notes/#refund-demo-change-the-actual-cartridge)
to make the edit and run the comparison.

Replay fixes the model responses, not the tool results. Tools run again,
including their side effects, so use temporary workspaces and test services.
To test a prompt change or a model's response to rejection, run the model
again and compare several trials.

Save the failure as an eval. Future changes can be checked against it,
even when the person or agent making the change does not know the original
bug.

## Let the results change the design

Sometimes the best change is to remove a step. In Looplet Analytics, our
data-analysis agent already investigated unclear parts of a task and required
an answer tied to evidence. I added a second model that checked the answer
against the first model's interpretation and could ask it to try again.

I tested each version three times on the same eight tasks, for 24 runs per
version. **With review, the benchmark accepted the answer in 5 of 24 runs.
Without review, it accepted 10 of 24.** The review version made **410 model
calls, compared with 338** without review. It finished 22 of its 24 runs;
the version without review finished all 24.

I removed the blocking reviewer for this implementation. This was not a test
of reviewers in general. These were reused development tasks, not unseen
tests. The versions also took different steps, so the score difference cannot
be attributed only to the reviewer's rejections. Some expected answers also
conflicted with reasonable readings of the questions.
The [comparison notes](/looplet-demo-notes/#analytics-comparison-removing-blocking-review)
include each trial, the unfinished runs, and the limits of the comparison.

A reviewer can check whether an answer follows an interpretation and still
accept a wrong answer if the interpretation is wrong. In this experiment, the
simpler version used fewer calls and did better on the benchmark. Evals helped
decide what to remove, not just what to add.

The same rule applies to context and delegation. If an agent lacks a current
record, a longer instruction may not help. Give it a way to fetch current
information, and test that change. A stored note may be stale; it is not an
authorization record.

Delegation can help with a large task, but a child agent is not automatically
isolated from the parent. The host must choose the child's tools, limit its
budget, and check its result. Add another agent only when it earns its cost.

## A cartridge makes autonomous improvement possible

Once behavior is in a cartridge, another agent can improve it. Here,
**autonomous** means the model makes the edits. The host still runs the tests
and decides whether to keep the change.

The loop is simple: copy the cartridge, let the builder edit the copy, run it
against tests owned by the host, and keep it only if it passes. Repeat from
the best version. This is a **hill climb**: each round tries to find a better
version without giving the version being tested control of the tests.

The agent doing a task and the agent improving the agent have different jobs.
The task agent handles a request. The builder changes the definition that will
handle future requests. The edited copy is a candidate; it does not replace
the deployed version just because the builder finished writing it.

The live hill climb shows this working. I ran `gpt-5.6-sol` against the failed
refund case. The first copy fixed that case but failed two other tests. The
host sent those failures back to the builder. After two more rounds, the copy
passed all seven tests owned by the host.

Only two hook files changed. The nine eval files stayed byte-identical to the
baseline. The host supplied the expected results and decided whether the
candidate passed.

This shows that a live model can discover and improve a cartridge when the
host owns the tests and sends failures back to the builder. It is one task,
one model, and one bounded experiment. It is not a general result about all
autonomous agents. The evaluator also runs in the same process, so this is
not a hostile-code sandbox.

The [builder demo](/looplet-refund-builder-demo.py) also has a scripted mode.
It makes the same copy-edit-run-keep loop reproducible without a model or an
API key.

<figure>
  <picture>
    <source media="(max-width: 600px)" srcset="/images/looplet/improvement-loop-mobile.png" width="800" height="1480" />
    <img src="/images/looplet/improvement-loop.png" alt="A builder edits and tests a cartridge. Feedback from development tests guides its next edit. Separate checks and release rules decide whether to deploy the candidate. Neither the builder nor the candidate may change those checks." width="1800" height="1000" loading="lazy" />
  </picture>
  <figcaption>The builder can edit and test without waiting for a person at each step. Separate checks decide whether its candidate is ready to deploy. The builder and candidate cannot change those checks.</figcaption>
</figure>

Give the builder a goal, a copy to edit, and a time or cost budget. Run each
copy in a fresh workspace and return the test results. Keep final tests outside
the cartridge. A candidate can change tests stored inside itself; in the
[failure test](/looplet-demo-notes/#probe-the-failure-contracts), changing only
the grading code makes the same wrong refund report a pass.

For the final decision, run the candidate and the code that judges it with
separate permissions. Putting the files in another directory is not enough.
The permissions and the worker boundary must enforce the separation.

The host can promote a passing candidate automatically or ask a person to
review it. Either way, the rule is simple: **automate the edits; keep the
final tests and release decision outside the program being edited.**

The earlier refund comparison used a developer-supplied hook. The live hill
climb used a model to make the cartridge edits. The development loop above is
the general pattern supported by the same interfaces.

## Start with one agent

Start with an agent you already run. Keep its model and tools. Move its
changeable behavior into a separate definition, add one check of the result,
and save one real failure as an eval. The
[migration guide](https://hsaghir.com/looplet/migrate/) shows a small first
step. You do not need to replace tools that already provide this boundary.

For each run, record the definition and model versions, inputs, tool calls,
results, stop reason, and files produced. Use those records to understand
failures and turn them into evals. Compare quality and policy behavior, then
measure time, model use, and cost.

The runtime and eval tools can be shared without sharing credentials or
business data. Keep access control, storage, worker isolation, and job
scheduling in the surrounding infrastructure. The
[demo notes](/looplet-demo-notes/) contain commands and the evidence behind
the examples.

A running agent answers requests. A separate definition gives us something
we can improve: inspect it, change it, run it, and compare the result. The
edit may come from a person or another agent. The eval tells us whether to
keep it.

**Own the harness. Turn failures into evals.**