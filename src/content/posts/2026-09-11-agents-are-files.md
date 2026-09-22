---
title: "Agents Are Files"
description: "Keep an agent's behavior in editable files, separate from the application that runs it. Use outcome checks to compare versions, then let another agent make and test changes."
date: 2026-09-21
featured: true
tags: ["agents", "engineering", "open-source", "python"]
category: "engineering"
cover: "/images/looplet/platform-ownership.png"
coverMobile: "/images/looplet/platform-ownership-mobile.png"
coverAlt: "A stack of files that define an agent, with one changed row highlighted. Edit. Run. Evaluate."
---

Suppose an agent refunds a customer \$250. Its instructions say refunds above \$100 need review. Every tool call succeeds, but the agent has broken the rule.

An agent is a program in which a model chooses some of its next steps and uses tools to act. Here, the refund needs a check in code before it can run. You can add that check and a test inside your existing application.

As the agent changes, you will want to compare different prompts, tools, and rules. That becomes harder when these pieces are mixed with request handling, database access, and other service code. You need a way to change the agent while keeping the surrounding application fixed.

I keep the agent's definition in a directory of ordinary files. The application loads that directory to run it. Each version can be copied, edited, and tested through the same interface.

This gives an engineer a small, reviewable change. With a runner and outcome checks in place, another agent can make and test those changes too.

## Put behavior in its own unit

For the refund agent, the definition includes the instructions the model receives, the tools it can request, and the code that checks those requests. In [Looplet](https://github.com/hsaghir/looplet), my Python toolkit for building and evaluating agents, I call this directory a **cartridge**:

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

The cartridge groups the system prompt, tool definitions, and agent-specific code. A **hook** is a function called at a particular point in a run, such as before a tool executes.

The **host** is the application that loads and runs the cartridge. It supplies the model connection, runtime libraries, credentials, and workspace. Business records and permission to issue payments stay in existing services. A cartridge's refund tool calls the billing service.

To try a different refund check, copy the cartridge and edit its hook. The host can load either version without changes to its request handlers or database clients. The diff shows the change being tested, and the original version remains available for comparison.

Alongside the definition live **evals**: tasks with automated checks of the outcome. For an oversized refund, the expected result is no payment and one request in the review queue. These development tests record what a correct result looks like.

## Keep actions under host control

The part of the host that runs the agent's loop is the **harness**. It prepares each model call, handles tool requests, and decides when to stop. It also calls the hooks supplied by the cartridge.

When the model requests a \$250 refund, the harness calls the refund hook before running the tool. The hook checks the amount in the original customer request. If it exceeds \$100, the hook rejects the refund and records a request for review. The harness passes that result back to the model.

<figure>
  <picture>
    <source media="(max-width: 600px)" srcset="/images/looplet/control-loop-mobile.png" width="800" height="1620" />
    <img src="/images/looplet/control-loop.png" alt="The model proposes a tool call. A check allows it or returns a reason for rejection. The model sees the result before choosing its next step. A request to finish goes through a separate check." width="1800" height="1100" loading="lazy" />
  </picture>
  <figcaption>The harness runs the cartridge's checks before tools execute and before the run finishes. A tool result or a reason for rejection becomes input to the next model call.</figcaption>
</figure>

Looplet treats `done` as a request to finish. The harness calls a separate completion check. In this example, that check verifies that the request has either been paid or queued for review. An eval checks the result independently after the run.

Cartridge hooks are editable. For real payments, the billing service must require approval above the limit and prevent duplicate payments, even if a hook is changed or removed. If a payment call times out, the system must check whether it took effect before retrying.

## Turn a failure into a test

The [runnable refund example](/looplet-demo-notes/#refund-demo-change-the-actual-cartridge) uses local JSON files for the ledger and review queue. It makes no real payments. Its first cartridge has the rule only in its prompt. Two scripted model responses request a \$250 refund, then call `done`. Both calls succeed. The ledger records a \$250 refund, so the task fails.

Now we add the refund hook described above. It blocks the \$250 refund and creates a record in `reviews.json`.

To test the change, we pass the saved model responses from the failed run through the updated cartridge. This is **captured-response replay**: running the same proposed calls through changed code. Each run starts with fresh local files.

<figure>
  <picture>
    <source media="(max-width: 600px)" srcset="/images/looplet/replay-comparison-mobile.png" width="800" height="1520" />
    <img src="/images/looplet/replay-comparison.png" alt="Both versions receive the same saved refund request. The original records a 250-dollar refund and no review, with no tool errors. It fails the eval. The changed version blocks the refund, records one pending review, and passes the same eval." width="1800" height="1160" loading="lazy" />
  </picture>
  <figcaption>Both versions receive the same responses and start with empty files. The result checks are unchanged. Only the hook and its configuration differ.</figcaption>
</figure>

**The successful run contains a tool error. The unsuccessful run does not.**

In the guarded run, the refund call returns an expected denial error. Both runs reach `done`. The outcome check distinguishes them by reading the ledger and review records.

For the \$250 request, the eval requires an empty ledger and exactly one review record. A control test verifies that a valid \$50 request still issues a refund. Five more cases cover split amounts, duplicate submissions, amounts at and just above the limit, and a request to finish without calling the refund tool.

Now test the reverse. In the cartridge hook configuration, change the limit:

```diff
 kwargs:
-  max_cents: 10000
+  max_cents: 30000
```

Replay the same \$250 request. With the higher limit, the refund goes through. The unchanged checks expect an empty ledger and a review record, so they fail and the runner exits with status 1. This gives us a way to check future edits against the original failure.

Replay fixes the model responses, but tools execute again. It needs test doubles or a separate test environment when tools have real side effects. It also cannot tell us how a model will react to a rejection or whether a changed prompt works better. Those questions need fresh model runs against the outcome checks.

The demo has a runtime limitation too: in its pinned version, an exception in the completion check can still let `done` through. The [failure probe](/looplet-demo-notes/#probe-the-failure-contracts) reproduces this behavior. A required check should stop the run if it crashes. Finding missing work afterward cannot undo an earlier action.

The [demo runner](/looplet-refund-demo.py) and [reproduction notes](/looplet-demo-notes/#refund-demo-change-the-actual-cartridge) include the comparison, additional cases, and these limitations.

## Let another agent improve the cartridge

So far, an engineer has made the edit and a runner has checked the result. A coding agent can do the editing. I call it the **builder** to distinguish it from the refund agent being changed.

Because the host can load each version through the same interface, a controller can repeat the process:
1. Copy the cartridge. This copy is the **candidate**.
2. Give the builder the failure report and access to the candidate files.
3. Run the candidate against the outcome checks.
4. Return failures for another edit. Stop when the checks pass or the iteration limit is reached.

This makes the edit-and-test loop autonomous within a fixed budget. The files give the builder a defined place to work; the runner and checks supply feedback.

The checks used to accept a candidate need separate control. A builder may edit development tests alongside the code. The final decision must also use checks it cannot change. Otherwise, deleting an assertion can turn a wrong result into a reported pass, as the [probe script](/looplet-demo-notes/#probe-the-failure-contracts) demonstrates.

Enforcing this requires restricted permissions. Candidate code needs a test environment where it cannot modify the evaluator, and the builder must lack that access too. A separate directory alone offers no such protection.

<figure>
  <picture>
    <source media="(max-width: 600px)" srcset="/images/looplet/improvement-loop-mobile.png" width="800" height="1480" />
    <img src="/images/looplet/improvement-loop.png" alt="A proposed deployment design: a builder edits a cartridge, development tests provide feedback, and separately protected checks decide whether the candidate can be released." width="1800" height="1000" loading="lazy" />
  </picture>
  <figcaption>A design for deployment. Development results guide edits. Protected acceptance checks and the host's release policy decide what can ship. That policy can allow automatic deployment or require review.</figcaption>
</figure>

I ran this edit-and-test loop with `gpt-5.6-sol` as the builder. The refund agent received scripted calls; the live model's job was to write and revise the hook. The starting prompt supplied the hook format, and failure feedback included implementation advice about how to reject a tool call.

- **Round 1:** Three of seven cases passed. The original \$250 case, split requests, duplicate refunds, and an amount just above the limit still failed.
- **Round 2:** The builder revised the hook, but the same four cases failed.
- **Round 3:** All seven cases passed.

The builder changed two hook files. All nine eval files stayed byte-identical, although writable copies were present inside its workspace. The runner also compared results with expected outcomes defined in its own code. This was a same-process test, with no security isolation between candidate code and evaluator.

These seven cases were development feedback, so the final score measures progress on those cases. A separate, unseen suite would be needed to measure performance on new cases. The checks inspect final local files; they do not establish that an unwanted payment action never occurred earlier.

The [builder runner](/looplet-refund-builder-demo.py) includes this live mode and a scripted mode for offline reproduction. The experiment shows a model making and revising a repair on one small task, with human-written instructions and feedback rules. Each attempted repair was an ordinary file change that the same runner could load and test.

## Start with one agent

Start with an agent you already run. Put its prompts, tool code, and local checks in a directory your host can load. Keep credentials and service authorization under the application's control.

Record the definition version, model, inputs, tool calls, results, and reason for stopping. When a run fails, use that record to write a case with an explicit expected outcome. Then change the definition and run the case again.

Once this works by hand, give a builder access to a copy and let a controller run the same tests. Keep the acceptance checks protected, and choose a release policy suited to the consequences of a mistake. The same process can support a human edit or an automated one.

This is what [Looplet](https://github.com/hsaghir/looplet) provides: a framework to own and package the harness, and to turn failures into evals.

**Own and package the harness. Turn failures into evals.**