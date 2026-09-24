---
title: "Agents Are Files"
description: "Like a game console with cartridges, one host runs different agents. Edit their files, turn failures into evals, and keep release checks with the host."
date: 2026-09-21
featured: true
tags: ["agents", "engineering", "open-source", "python"]
category: "engineering"
cover: "/images/looplet/platform-ownership.png"
coverMobile: "/images/looplet/platform-ownership-mobile.png"
coverAlt: "A stack of files that define an agent, with one changed row highlighted. Edit. Run. Evaluate."
---

An agent here is a program in which a model chooses tool calls. Once it is part of an application, its instructions, tools, hooks, and configuration can become mixed with request handlers and service code. But those are different things to change. I want to run another agent, or let a builder improve this one, without handing over the rest of the application for editing.

Think of the application as a game console and each agent's editable harness as a cartridge. A cartridge groups its instructions, tools, hooks, configuration, and self-tests in a directory of files. The same host can run a refund agent or an incident-response agent by loading different cartridges, provided it supplies the capabilities each needs. An editor can share or change a copy without rewriting the host's loop; a compatible host can load it elsewhere.

I built [Looplet](https://github.com/hsaghir/looplet), a Python toolkit for running agents that call tools, to keep the loop under the application's control. The host supplies model access and service permissions and decides what can ship. An agent builder can propose a changed cartridge and learn from development tests, while the host compares versions. The files alone do not isolate candidate code or protect release checks.

Once an agent can change without rewriting its host, the next question is how to judge a new version. In the example below, a person edits a refund cartridge, then a live coding agent tries to improve a copy. The refund agent receives scripted model calls and writes to local files. The host can compare what the versions do, but these runs do not provide protected release checks.

## What a version includes

Packaging only the prompt misses changes that can alter a run. A new tool argument changes what the model can request; a **hook**, code called at a specific point, can allow or refuse the same request before it executes. If those pieces are mixed into request handlers and database clients, a prompt diff cannot tell you which behavior actually ran.

Looplet can run an agent directly from Python tools and hooks, yielding each step to the application. The files are optional. When I want to compare versions, I group the parts I expect to change in a cartridge. The refund example uses this directory:

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

The evals beside the definition give the developer quick feedback; they are not an independent release check when the candidate can edit them.

The host can load a changed copy without changing its request handlers or service clients. It decides which credentials and workspace a candidate receives. The diff identifies what the candidate proposed; the original remains available to compare. In this demo, candidate Python runs in the same process as the host, so the directory itself provides no isolation. Loading a cartridge executes its Python tool and hook modules before the first model call; a tool-call hook cannot constrain that code.

## Keep actions under host control

The host runs the agent through Looplet's **runtime loop**. It prepares model calls, handles tool requests, and decides when to stop. It calls cartridge hooks along the way, but those hooks are editable code. Looplet's optional permission engine allows unmatched tool calls by default; it is not a sandbox.

In a real refund application, a hook on the refund tool would not stop an agent with a shell and billing credentials from calling the billing API directly. The host has to restrict what the candidate can execute, and the billing service must enforce approval. A hook can reject a proposal early and tell the model why; it cannot be the only authority.

In the local example, the runtime loop calls a refund hook before running the tool. The hook checks the original customer request supplied by the host, not just the amount the model proposed. Above the \$100 limit it rejects the request and records it for review. The model receives the rejection as a tool result.

<figure>
  <picture>
    <source media="(max-width: 600px)" srcset="/images/looplet/control-loop-mobile.png" width="800" height="1620" />
    <img src="/images/looplet/control-loop.png" alt="The model proposes a tool call. A check allows it or returns a reason for rejection. The model sees the result before choosing its next step. A request to finish goes through a separate check." width="1800" height="1100" loading="lazy" />
  </picture>
  <figcaption>The runtime loop runs the cartridge's checks before tools execute and before the run finishes. A tool result or a reason for rejection becomes input to the next model call.</figcaption>
</figure>

Looplet treats `done` as a request to finish. The runtime loop calls a separate completion check. In this example, that check verifies that the request has either been paid or queued for review. An outcome eval then inspects the local records after the run.

Even this host-side path has a failure contract to test. In the pinned runtime, an exception in the completion check can still let `done` through; the [failure probe](/looplet-demo-notes/#probe-the-failure-contracts) reproduces it. For real payments, the billing service must require approval and prevent duplicates, even if a hook is changed, skipped, or crashes. After a timeout, the system must check whether a payment took effect before trying again.

## What the test actually sees

To compare an old cartridge with a proposed version, the host needs a failure to fix and a case that must still work. The files identify what changed in the agent, but they do not freeze the model or the world around it. A useful run record also names the model and definition version, inputs, tool results, and reason for stopping. Without that context, repeating a task with the same files may give a different result for reasons the diff cannot explain.

The [runnable example](/looplet-demo-notes/#refund-demo-change-the-actual-cartridge) is deliberately small: scripted model calls, a local JSON ledger and review queue, and no payment service. Its first cartridge puts a \$100 limit in the prompt alone. The model still requests a \$250 refund, then calls `done`. Both calls succeed, but the ledger shows an unauthorized refund and no review.

With the hook added, we pass the saved model responses through the new cartridge. Both runs start with empty files; the prompt, tool, and expected outcomes stay fixed. Only the hook and its configuration change. This is **captured-response replay**: it lets us compare how changed code handles the same proposed calls. The second run denies the refund and records a pending review. Its tool log now contains a denial error, while the first run had no tool error. Reading the ledger and review queue tells us which result was right.

A control case checks that a permitted \$50 refund still works. Five more cases cover split and duplicate attempts, the limit itself, just above it, and a request to finish without trying the refund tool. The [runner](/looplet-refund-demo.py) and [reproduction notes](/looplet-demo-notes/#refund-demo-change-the-actual-cartridge) include the complete cases and the changed-limit counterexample.

Replay holds the model's responses fixed, not the world. The tools run again and have new side effects. A payment or deployment tool needs a test double or an isolated environment before replay is safe. A different prompt or a rejection from the tool might change what the model does next; that needs fresh model runs and the same outcome checks.

Our local example checks final ledger and review files; it cannot establish that an unwanted action never happened earlier in the run. An incident-response agent could take a service offline and restore it before stopping. A check of the final configuration would see a healthy service, though users experienced an outage. A host testing that cartridge would need a record of when availability changed and whether requests failed.

## Let a builder propose the next version

So far, a person wrote the new hook. Once the editable behavior sits in a directory, another agent can propose a new version without changing the host. I gave a coding agent, the **builder**, a copy of the cartridge and reports from failing runs. A controller I wrote, not Looplet's core, loaded each edited copy, ran development cases, and sent failures back for another attempt. The proposal was a file change the same host could load and test.

In the live run, `gpt-5.6-sol` edited the hook while the refund agent received scripted calls. The starting prompt specified the hook format; feedback even explained how to reject a tool call. Three of seven development cases passed in round one. After a revision, the same four cases still failed in round two: the oversized request, a split request, a duplicate, and an amount just above the limit. All seven passed in round three.

The builder changed two hook files. All nine eval files remained byte-identical, though they were writable inside its workspace. The seven cases were shown to the builder repeatedly, so the final score measures progress on those cases. There was no separate unseen suite to measure performance on new ones. The [builder runner](/looplet-refund-builder-demo.py) includes both this live mode and a scripted mode that can be reproduced offline.

In a separate scripted probe, changing only the graders made the same bad refund look like a pass. The live builder did not make that change. The runner checked outcomes against expectations in its own code as well as candidate-supplied evals. But candidate code and evaluator ran in the same process, without security isolation. A candidate directory outside the evaluator's directory does not stop hostile code from reaching it. A release system would have to restrict the builder's access, isolate candidate execution, and run acceptance checks neither can change.

<figure>
  <picture>
    <source media="(max-width: 600px)" srcset="/images/looplet/improvement-loop-mobile.png" width="800" height="1480" />
    <img src="/images/looplet/improvement-loop.png" alt="A proposed deployment design: a builder edits a cartridge, development tests provide feedback, and separately protected checks decide whether the candidate can be released." width="1800" height="1000" loading="lazy" />
  </picture>
  <figcaption>A design for deployment, not the isolation provided by this demo. Development tests guide edits; separately protected checks and the host's release policy decide what can ship.</figcaption>
</figure>

## From a test result to a release decision

Seven passing development cases show what the builder learned, not how a new cartridge will behave on unseen requests. For an agent already in use, the host needs the run records and outcome checks above to compare a candidate with the current version. Replay captured calls against edited code in a disposable environment; run the model again when changing what it sees or how it responds.

Development cases can guide either a person or a builder. Before a candidate reaches a real service, acceptance checks need their own permissions and an account of the actions taken during the run, not just its final state. The service that performs the action must enforce authorization even if the agent's hook says yes.

In this experiment, the host compared versions of one refund cartridge, and a coding agent improved it on seven known cases. We did not test loading two different agents or running the cartridge in another runtime. The files made the proposed change inspectable and repeatable, but the candidate and evaluator shared a process. Deciding whether the next version can act on users requires protected checks and service controls that this experiment did not build.