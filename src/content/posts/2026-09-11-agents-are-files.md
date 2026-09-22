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

Suppose an agent refunds a customer \$250. Its instructions say refunds above \$100 need review. Every tool call succeeds, but the agent has broken the rule.

An agent is a program in which a model chooses some of its next steps and uses tools to act.

This failure points to a fundamental friction in building agents: their behavior is scattered across an application. Prompts live in configuration strings, tools live inside API clients, state sits in a database, and stopping rules are buried in gateway middleware. When an agent fails, it is hard to isolate what went wrong. When you change something, it is hard to know whether you broke other behaviors.

The solution is to give the agent's behavior a clear home. Keep its instructions, tools, and rules together in ordinary files, separate from the application that runs it.

Once an agent is defined as files in a directory, three capabilities follow naturally:
1. You can inspect, diff, version, and edit the agent without touching the application that hosts it.
2. You can capture failures and turn them into automated tests that verify outcomes directly.
3. Another agent can copy the definition, edit it, and test candidate versions against tests the model cannot rewrite.

## Put behavior in its own unit

In most codebases, changing an agent requires changing backend service code. Adding a policy check means modifying a request handler. Tweaking a prompt requires redeploying a service.

In [Looplet](https://github.com/hsaghir/looplet), my Python toolkit for building and evaluating agents, I separate this behavior into a directory called a **cartridge**:

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

The cartridge holds only what defines the agent: its system prompt, tool declarations, local rules, and development tests.

The application that loads and runs the cartridge is the **host**. The host supplies the model API connection, runtime libraries, credentials, and execution workspace. Existing backend services still own business records and authorization. A cartridge tool calls the billing service; it does not replace it.

This separation gives you a concrete unit of change. An engineer can review an update to the agent's policy without reviewing web server code. A developer can create a new variant by copying the directory and changing one file. Most importantly, another program can read and edit the cartridge directly.

Beside the prompts and tools live **evals**: tasks with automated checks of the final outcome. For an oversized refund, the check requires that no money was paid and that one request was placed in the review queue. Evals make the intended behavior explicit and version-controlled alongside the code they test.

## Keep actions under host control

Once the agent definition lives in files, the next question is how to run it safely.

When a model suggests calling a tool, that request is only a proposal. In our refund example, when the model calls `refund(amount=250)`, money should not move immediately. The application code surrounding the model must inspect the call before it runs. If the amount exceeds \$100, application code must block the call and route the request to human review.

<figure>
  <picture>
    <source media="(max-width: 600px)" srcset="/images/looplet/control-loop-mobile.png" width="800" height="1620" />
    <img src="/images/looplet/control-loop.png" alt="The model proposes a tool call. A check allows it or returns a reason for rejection. The model sees the result before choosing its next step. A request to finish goes through a separate check." width="1800" height="1100" loading="lazy" />
  </picture>
  <figcaption>The host supplies the checks. The model receives a tool result or a reason for rejection before choosing its next step. A separate check decides whether it may finish. Finishing does not prove the task was done correctly.</figcaption>
</figure>

The code around the model is the **harness**. It prepares context, runs tools, enforces limits, and decides when the loop stops.

In Looplet, `done` is also treated as a proposal. When a model calls `done`, it is asking to finish, not proving that the task succeeded. A separate host check verifies that required work is complete before letting the loop stop.

A rejected action should return a clear explanation so the model can choose a better next step. If a required check crashes or times out, the run must halt with an error. In the pinned demonstration revision, a crashing completion check can fail open and allow `done`. The [failure test](/looplet-demo-notes/#probe-the-failure-contracts) reproduces this limitation. An eval can catch missing work after the fact, but it cannot substitute for a safe completion gate during execution.

Stopping a run also cannot undo an action that has already occurred. The payment service must enforce idempotency and authorization. If a network call times out, the system must check whether the payment took effect before retrying.

## Turn a failure into a test

With the host harness in place, we can test the refund policy systematically.

In the [runnable refund example](/looplet-demo-notes/#refund-demo-change-the-actual-cartridge), we start with an unguarded cartridge that has the policy only in its prompt. We feed it two scripted responses: request a \$250 refund, then call `done`. Every tool call succeeds. The ledger records a \$250 refund. The task fails.

Now we add a policy hook to the cartridge. The hook reads the original request, blocks the \$250 refund, and creates a record in `reviews.json`.

To confirm the fix, we do not need to call a live model and hope it reproduces the scenario. Instead, we take the saved responses from the failed run and pass them through the updated cartridge. We call this **captured-response replay**: re-running the exact proposals against fresh tool state to test how the new code handles them.

<figure>
  <picture>
    <source media="(max-width: 600px)" srcset="/images/looplet/replay-comparison-mobile.png" width="800" height="1520" />
    <img src="/images/looplet/replay-comparison.png" alt="Both versions receive the same saved refund request. The original records a 250-dollar refund and no review, with no tool errors. It fails the eval. The changed version blocks the refund, records one pending review, and passes the same eval." width="1800" height="1160" loading="lazy" />
  </picture>
  <figcaption>Both versions receive the same responses and start with empty files. The result checks are unchanged. Only the hook and its configuration differ.</figcaption>
</figure>

**The successful run contains a tool error. The unsuccessful run does not.**

In the guarded run, the tool call returns an expected denial error, which prevents the payment. Both runs reach `done`. Neither a clean execution log nor a completion message tells you whether the business policy was respected; only the outcome check does.

The eval inspects the ledger and review records directly. For the \$250 request, it asserts that the ledger is empty and that exactly one review record exists. A control test verifies that a valid \$50 request still issues a refund, ensuring the hook does not simply block every call. Five additional test cases cover split amounts, duplicate submissions, edge values, and immediate completion requests.

Now test the reverse. In the cartridge hook configuration, change the limit:

```diff
 kwargs:
-  max_cents: 10000
+  max_cents: 30000
```

Replay the same \$250 request. With the higher limit, the refund goes through. The unchanged outcome checks expect a review record, so they fail and the runner exits with status 1. An ordinary file edit changed the behavior, and an automated test immediately caught the regression.

The [demo runner](/looplet-refund-demo.py) executes both versions through the same interface. The [reproduction notes](/looplet-demo-notes/#refund-demo-change-the-actual-cartridge) document the full test suite.

Saving this failure as an eval provides permanent protection. Future developers, or future models, can change prompts and tools without silently reintroducing the bug.

## A cartridge makes autonomous improvement possible

Once an agent is defined in files and evaluated against outcome checks, a new capability emerges: **another agent can improve it.**

When an agent fails, a human developer typically reads the error, edits the prompt or tool logic, runs the tests, and checks whether the pass rate improved. Because the cartridge is just files in a directory, a **builder agent** can execute that exact same loop.

This process is a **hill climb**:
1. Copy the cartridge to a fresh workspace.
2. Let the builder model read the failure report and edit the candidate cartridge files.
3. Run the candidate against the test suite.
4. If all tests pass, keep the candidate; if tests fail, return the error feedback to the builder and try again.

The critical architectural requirement is that the host, not the candidate, must own the evaluation suite. If the candidate could modify its own eval checks, it could achieve a passing score by simply deleting the assertions. The [probe script](/looplet-demo-notes/#probe-the-failure-contracts) demonstrates this: modifying only the grading code produces a reported pass on the broken \$250 refund.

<figure>
  <picture>
    <source media="(max-width: 600px)" srcset="/images/looplet/improvement-loop-mobile.png" width="800" height="1480" />
    <img src="/images/looplet/improvement-loop.png" alt="A builder edits and tests a cartridge. Feedback from development tests guides its next edit. Separate checks and release rules decide whether to deploy the candidate. Neither the builder nor the candidate may change those checks." width="1800" height="1000" loading="lazy" />
  </picture>
  <figcaption>The builder can edit and test without waiting for a person at each step. Separate checks decide whether its candidate is ready to deploy. The builder and candidate cannot change those checks.</figcaption>
</figure>

This autonomous loop is not hypothetical. We ran a live hill climb using `gpt-5.6-sol` against the failed refund cartridge:
- **Round 1:** The builder model read `failure.json`, created the policy hook, and fixed the primary \$250 failure. But host-owned holdout tests caught two regressions: an applicant could bypass the limit by splitting the refund into smaller requests, and finishing without a tool call bypassed the review queue.
- **Round 2:** The host returned those failure reports to the builder. The model revised the hook to ensure oversized requests queue for review during the completion check.
- **Round 3:** Guided by the remaining split-request failure, the builder updated the hook to return an explicit denial decision.

After three iterations, the candidate passed all seven host-owned test cases. Only two hook files were modified. All nine eval files remained byte-identical to the baseline because they were maintained outside the builder workspace.

The [builder runner](/looplet-refund-builder-demo.py) includes both this live mode and a deterministic scripted mode for offline reproduction.

This experiment demonstrates that a live model can discover and refine working cartridge logic when given clear outcome feedback. It is a single bounded experiment on one task and one model, not proof of universal autonomy. But it shows how treating agents as files turns agent optimization into an automated, test-driven engineering discipline.

## Start with one agent

You do not need to replace your application architecture or adopt an all-in-one framework to use this approach.

Start with a single agent you already run in production:
1. Move its prompts, tool declarations, and local policies into a dedicated directory.
2. Keep your database clients, credentials, and business authorization inside the host application.
3. When the agent fails, do not patch a prompt string in production and hope for the best. Capture the failure, write an automated outcome check, edit the cartridge files, and verify the result.

For each run, log the cartridge version, model identifier, inputs, tool calls, results, and stop reasons. These traces become the test cases for your next eval suite.

A running agent is an active service component. A separate agent definition is an artifact you can systematically version, evaluate, and improve.

**Own the harness. Turn failures into evals.**