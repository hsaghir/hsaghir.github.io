---
title: "Close the gap by iteration, not specification"
description: "A follow-up. The under-specification problem in coding agents does not close by writing better specs upfront. Models do not stick to instructions anyway. The gap closes by building better detection: catching deviation cheaply, often, and at the right level of abstraction. The lever is the loop, not the prompt."
date: 2026-05-18
featured: true
tags: ["agents", "engineering", "unified-views"]
category: "engineering"
cover: "/images/vermeer-geographer.jpg"
coverAlt: "The Geographer by Johannes Vermeer, 1668. A scholar pauses mid-measurement, dividers in hand over a chart, a globe and wall map behind him as his references. Städel Museum, Frankfurt. Public domain."
---

In a [previous post](/blog/2026-05-02-under-specified-coding-agent/) I
argued that coding agents accumulate technical debt because what
we ask them to do is under-specified. The obvious follow-up
question is: can we fix this by writing better specs?

The honest answer is no. Or more precisely: not primarily. The
gap closes through better iteration, not through better upfront
specification.

## The thesis

**The under-specification gap closes through detection, not
declaration.** You cannot prevent the agent from deviating from
your intent. You can only catch it doing so, cheaply enough and
often enough to correct the deviation before it compounds. The
lever is the loop around the agent, not the prompt going in.

This reframes the whole problem. The work shifts from "write a
better spec" to "build a richer feedback loop." From "what should
the agent know in advance" to "what should we check after every
change." The first draft of any change is not the product. The
loop around it is.

## Why upfront specs are not the answer

It is tempting to believe the fix is more discipline at the input.
Write a longer `CLAUDE.md`. Document the architecture. List the
conventions. The agent will follow them.

Three things break this hope.

First, **models do not stick to long instructions**. The longer
the spec, the lower the per-rule adherence. The model treats the
spec as a prior, weighted against everything else in context,
including the failing test it is trying to fix right now. Under
pressure, it will violate any single rule. The longer the rule
list, the more violations per task.

Second, **the spec is never complete**. The post that preceded
this one made the case in detail: the correct specification of a
system does not exist at the moment of writing. It is discovered
through implementation and operation. Trying to write it all
upfront is trying to commit decisions you do not yet have the
information to make.

Third, **spec authoring competes with code authoring**. Time
spent writing exhaustive specs is time not spent on the work they
are meant to enable. For most tasks, full upfront specification
costs more than the code itself. The equilibrium is partial
specification, indefinitely.

The conclusion: specs are useful as priors, but they cannot be the
mechanism that closes the gap. Something else must.

## Why iteration is that something else

Detection is the only feedback that scales.

A spec tells the agent what to do. A check tells you what the
agent actually did. Only the second observation can be acted on,
because only the second one exists.

Detection has three properties that make it the right lever.

It is **post-hoc**, which matches the structure of the problem.
You discover that the agent violated your intent by looking at
what it produced. You cannot inspect intent compliance before the
output exists.

It is **iterative**, which matches the structure of engineering.
The correct spec gets discovered during implementation, not
before it. Every detected deviation is a piece of spec being
written by the loop instead of by you.

It is **automatable**, which matches the structure of agentic
work. The agent produces edits at machine speed. The only thing
that can review them at machine speed is more machinery.

A senior engineer working alone does this implicitly. They write
code, look at it, refactor, run it, see it break, fix it,
refactor again. The work is fundamentally a feedback loop. We
have been treating coding agents as one-shot translators when we
should be embedding them in the same loop.

## How to detect what the agent did wrong

This is the hard problem. Saying detection is the lever does not
say how to detect. Let me be more rigorous.

A violation is a gap between what the agent did and what was
wanted. To detect it, you need:

1. **A signal about what the agent did.** Easy. Diff, trajectory,
   tool calls, edits. All inspectable.
2. **A signal about what was wanted.** Hard. The whole point of
   the previous post is that this is incomplete.
3. **A comparison mechanism.** Also hard, because the comparison
   has to be meaningful at the right level of abstraction.

Most current tooling skips step 2 entirely and does syntactic
comparison at step 3. That is why linters catch typos and miss
architectural drift.

The way out is to find sources of "what was wanted" that exist
even when no one wrote them down.

### Signal source 1: The existing codebase

The codebase *is* a partial specification. Every line in it
expresses a choice: naming, structure, error handling,
abstraction depth. A new change either fits these patterns or
breaks them. Breaking them is sometimes correct, but breaking
them silently is almost always wrong.

So the first detection mechanism is **deviation from existing
patterns**. New code uses different naming than nearby code. New
code introduces a wrapper where neighbors call directly. New code
reaches across a boundary that no existing code crosses. New code
reimplements something that already exists. All checkable against
the codebase itself, without any written spec.

### Signal source 2: A critic model

The highest-leverage check available is a fresh model with a
sharp prompt reading the diff. The agent that wrote the code is
the worst possible reviewer of it; it is invested in its own
choice and shares the same blind spots. A separate model, given
the diff and the surrounding context and asked "what is wrong
with this change?", catches what mechanical tooling cannot:
abstractions that fight the existing design, naming that
confuses future readers, choices that close off natural
extensions.

It is the same pattern as RLAIF and constitutional AI, pointed at
code instead of chat, and barely exploited in the coding tools
shipping today.

### Signal source 3: The change itself

Some violations are visible in the shape of the diff alone.

**Defensive accretion.** Try/except blocks, optional parameters,
fallback branches added without clear reason. The agent papering
over uncertainty.

**Conditional sprawl.** An if/elif chain handling cases that
should have been a polymorphism or a table lookup. Often a sign
of patching the wrong layer.

**Inline duplication.** A code block that closely resembles
existing code. The agent reached for copy-paste because it did
not look for the canonical version.

**Comment confessions.** "TODO," "for now," "hack." The agent
literally admitted it. Surprisingly common signal that almost
nothing catches.

**Test asymmetry.** Code added but no tests, or tests added that
pin down implementation details rather than behavior.

All detectable from the diff alone with simple heuristics. None
are checked by current agentic loops.

### Signal source 4: The trajectory

Some violations only show up across many changes. A file edited
repeatedly across unrelated tasks is structurally wrong. Today's
change modifying code written yesterday in a contradictory
direction signals unstable design. Tests growing faster than
features, or complexity rising across commits, both indicate
stitching rather than building.

These are the signals that catch what Alibaba's
[SWE-CI study](https://arxiv.org/abs/2603.03823) measured: agents
that pass each individual task while degrading the codebase. No
single-step check can see this. Only trajectory tracking can.

### Signal source 5: The output behavior

Some violations only surface when the code runs against real
inputs. Production replay against recorded traffic. Generated
adversarial inputs. Latency, memory, and error-rate monitoring on
a canary. The closer the loop gets to actual workloads, the more
violations surface before they become incidents.

### Signal source 6: The agent's reasoning trace

This one is underexplored. The agent leaves a trace of how it got
to the change: prompts received, files read, tools called,
explanations generated. The trace itself is a signal.

Did the agent read the callers before editing the function? Did
it state uncertainty and then suppress it? Did it abandon a plan
after one failing test, switching to band-aid mode? Did it
suppress an error instead of investigating its root cause?

If the harness owns the loop, all of this is inspectable. If the
harness is opaque, it is lost. This is one of the practical
reasons to control the loop yourself, which I made the broader
case for [in an earlier post](/blog/2026-04-23-the-loop-is-the-product/).

### Signal source 7: Disagreement across agents

Run the same task through several different agents (different
prompts, different models, different scaffolds). Cases where they
agree are probably right. Cases where they diverge deserve
attention. Disagreement is a free signal: it does not require a
human or a critic to know the right answer, only to notice that
the right answer is unclear.

## The detection meta-principle

The signals that work share three properties:

**They use information the agent did not see.** The codebase the
agent ignored. The trajectory it does not remember. The production
behavior it cannot simulate. The other agents it is unaware of.
Information the agent already had is information it already used;
it does not catch what it missed.

**They compare at the right level of abstraction.** Syntactic
comparison catches little. Semantic comparison catches more.
Structural comparison catches the most. Higher levels are harder
to implement but find the violations that matter.

**They are cheap enough to run on every change.** Detection that
runs once a week catches problems a week late. Detection on every
edit catches them before they compound. The loop's value comes
from frequency.

So the principle is: **detect by exposing the change to
information and abstractions the agent could not access during
generation, cheaply enough to run constantly.**

## What this changes

If the lever is detection, several things follow.

**The harness becomes the product.** Not the model, not the
prompt. The loop around them, where signals get checked and fed
back. This is where the differentiation will be in the next phase
of coding tools. The model is rentable; the loop is yours to
build.

**The agent's first draft is not the deliverable.** The
deliverable is what survives the loop. Treating the first output
as the answer is the mistake that produces SWE-CI's results. The
output is a candidate, not a finished change.

**Spec gets written by the loop.** Every detected violation is a
piece of spec being articulated through correction. Over time,
this accumulates into the explicit constraints the codebase
needed all along. The spec is built by iteration, not before it.

**Human judgment moves up the stack.** Engineers stop reviewing
every diff and start curating the detection layer: which signals
to surface, which to ignore, what counts as a violation in this
codebase. The reviewer's job becomes designing the reviewer.

**Reward shaping for coding agents becomes a frontier.** Once
mechanical, structural, and critic signals can run automatically
in a sandbox, you have the ingredients for RL on a richer
objective than "tests pass." This is where the gap between
current models and lifecycle-trained models genuinely narrows.

## The conclusion

The under-specification gap does not close by writing more
specification. It closes by catching deviation faster.

Coding agents are not generators that need better instructions.
They are participants in a feedback loop that needs more sensors
and a cleaner cycle. The work of the next several years in
agentic coding is the work of building those sensors and that
cycle.

The gap is still real. Detection narrows it but does not
eliminate it. The irreducible core from the previous post
remains: you cannot specify what has not yet happened, and you
cannot detect violations of constraints no one has yet noticed.
But within those limits, the lever is detection. The agent's
first draft is not the product. The loop around it is.

> The lever was never the prompt. It was always the loop.
