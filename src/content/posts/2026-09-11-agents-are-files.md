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

Think of an agent's editable harness as a game cartridge and the application that runs it as the console. The cartridge holds its instructions, tools, hooks, and configuration; the host supplies the loop, model access, service permissions, and release checks. Swapping cartridges changes the agent without rebuilding the console, but a cartridge cannot approve its own release.

Imagine an incident-response agent taking a service offline, noticing its mistake, and restoring it before it stops. A test that reads only the final configuration sees a healthy service. Users still experienced an outage. The test missed the part that mattered.

I want to change an agent like that and know whether the next version does better. An agent here is a program in which a model chooses tool calls. Its behavior depends on instructions, the tools that carry out those calls, and code that checks what may run. Once the agent is in use, an edit must fix the failure without breaking what already worked. Another agent can propose that edit, but a pass on tests it can change is not enough to ship it.

I built [Looplet](https://github.com/hsaghir/looplet), a Python toolkit for running agents that call tools, to keep that loop under the application's control and test changes after the first working version. It can load an agent's instructions, tool definitions, and checks from a directory of files. The application that runs the agent supplies the model and controls access to real services. Each edited copy of that directory is a proposed version you can test against a recorded failure and a case that already worked; the files alone cannot protect the release checks or authorize an action.

The experiment below is smaller than an incident responder. It uses scripted calls and local files to simulate a refund; a live coding agent edits its checks. In one run the builder improves the development results. In a separate scripted probe, changing only the graders turns the same bad outcome into a reported pass. Neither run is a protected release test.

## What a version includes

Versioning only the prompt misses two changes that can alter a run. A new tool argument changes what the model can request; a **hook**, code called at a specific point, can allow or refuse the same request before it executes. If those pieces are mixed into request handlers and database clients, a prompt diff cannot tell you which behavior actually ran.

Looplet can run an agent directly from Python tools and hooks, yielding each step to the application. The files are optional. When I want to compare versions of the behavior, I group the parts we expect to change in a directory called a **cartridge**.

Think of Looplet's runtime and the host application as a game console. Each cartridge holds an agent-specific harness: its instructions, tools, hooks, and configuration. Swap a refund cartridge for an incident-response cartridge, and the same host can run a different agent without rewriting its loop, provided it supplies the capabilities each needs. An editor can propose changes to the cartridge; the host retains control of model access, service approvals, and release checks. A compatible host can load the files elsewhere, but the directory alone cannot make its tools and hooks run in every runtime. This is the cartridge used by the runnable example:

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

The **host** is the application that loads and runs the cartridge. It supplies the model connection and runtime, and decides which credentials and workspace a candidate receives. It can load a changed copy without changing its request handlers or service clients. The diff identifies what the candidate proposed; the original remains available to compare. In this demo, candidate Python runs in the same process as the host, so the directory itself provides no isolation. Loading a cartridge executes its Python tool and hook modules before the first model call; a tool-call hook cannot constrain that code.

Those files do not freeze the model or the world around it. A useful run record also names the model and definition version, inputs, tool results, and reason for stopping. Without that context, repeating a task with the same files may give a different result for reasons the diff cannot explain.

## Keep actions under host control

The host runs the agent through Looplet's **runtime loop**. It prepares model calls, handles tool requests, and decides when to stop. It calls cartridge hooks along the way, but those hooks are editable code. Looplet's optional permission engine allows unmatched tool calls by default; it is not a sandbox.

Consider an incident-response agent with a deployment tool and a shell tool. A hook that checks only deployment requests does not stop the shell from calling the same API if it has the same credentials. The host has to restrict what the candidate can execute, and the deployment service must enforce approval. A hook can reject a proposal early and tell the model why; it cannot be the only authority.

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

The [runnable example](/looplet-demo-notes/#refund-demo-change-the-actual-cartridge) is deliberately small: scripted model calls, a local JSON ledger and review queue, and no payment service. Its first cartridge puts a \$100 limit in the prompt alone. The model still requests a \$250 refund, then calls `done`. Both calls succeed, but the ledger shows an unauthorized refund and no review.

With the hook added, we pass the saved model responses through the new cartridge. Both runs start with empty files; the prompt, tool, and expected outcomes stay fixed. Only the hook and its configuration change. This is **captured-response replay**: it lets us compare how changed code handles the same proposed calls. The second run denies the refund and records a pending review. Its tool log now contains a denial error, while the first run had no tool error. Reading the ledger and review queue tells us which result was right.

A control case checks that a permitted \$50 refund still works. Five more cases cover split and duplicate attempts, the limit itself, just above it, and a request to finish without trying the refund tool. The [runner](/looplet-refund-demo.py) and [reproduction notes](/looplet-demo-notes/#refund-demo-change-the-actual-cartridge) include the complete cases and the changed-limit counterexample.

Replay holds the model's responses fixed, not the world. The tools run again and have new side effects. A payment or deployment tool needs a test double or an isolated environment before replay is safe. A different prompt or a rejection from the tool might change what the model does next; that needs fresh model runs and the same outcome checks.

The incident-response example in the opening would need a different outcome check: a record of when availability changed and whether requests failed, not just the final service configuration. Our local example checks final ledger and review files; it cannot establish that an unwanted action never happened earlier in the run.

## Let a builder propose the next version

So far, a person wrote the new hook. Once the editable behavior sits in a directory, another agent can propose a new version without changing the host. I gave a coding agent, the **builder**, a copy of the cartridge and reports from failing runs. A controller I wrote, not Looplet's core, loaded each edited copy, ran development cases, and sent failures back for another attempt. The proposal was a file change the same host could load and test.

In the live run, `gpt-5.6-sol` edited the hook while the refund agent received scripted calls. The starting prompt specified the hook format; feedback even explained how to reject a tool call. Three of seven development cases passed in round one. After a revision, the same four cases still failed in round two: the oversized request, a split request, a duplicate, and an amount just above the limit. All seven passed in round three.

The builder changed two hook files. All nine eval files remained byte-identical, though they were writable inside its workspace. The seven cases were shown to the builder repeatedly, so the final score measures progress on those cases. There was no separate unseen suite to measure performance on new ones. The [builder runner](/looplet-refund-builder-demo.py) includes both this live mode and a scripted mode that can be reproduced offline.

The grader-tampering probe in the opening was scripted, not something the live builder did. The runner checked outcomes against expectations in its own code as well as candidate-supplied evals. But candidate code and evaluator ran in the same process, without security isolation. A candidate directory outside the evaluator's directory does not stop hostile code from reaching it. A release system would have to restrict the builder's access, isolate candidate execution, and run acceptance checks neither can change.

<figure>
  <picture>
    <source media="(max-width: 600px)" srcset="/images/looplet/improvement-loop-mobile.png" width="800" height="1480" />
    <img src="/images/looplet/improvement-loop.png" alt="A proposed deployment design: a builder edits a cartridge, development tests provide feedback, and separately protected checks decide whether the candidate can be released." width="1800" height="1000" loading="lazy" />
  </picture>
  <figcaption>A design for deployment, not the isolation provided by this demo. Development tests guide edits; separately protected checks and the host's release policy decide what can ship.</figcaption>
</figure>

## From a test result to a release decision

For an agent already in use, record the version of its definition and model, the original input, proposed tool calls, tool results, actual effects, and reason for stopping. When it fails, write down the expected outcome and one control case that must keep working. Replay captured calls against edited code in a disposable environment; run the model again when changing what it sees or how it responds.

Development cases can guide either a person or a builder. Before a candidate reaches a real service, acceptance checks need their own permissions and an account of the actions taken during the run, not just its final state. The service that performs the action must enforce authorization even if the agent's hook says yes.

In this experiment, a coding agent repaired one file-defined behavior on seven known cases. The files made that change inspectable and repeatable. Deciding whether the next version can act on users requires a protected evaluator and service controls that this experiment did not build.