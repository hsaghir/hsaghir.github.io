---
title: "Your coding agent is under-specified"
description: "Coding agents write impressive first drafts. But under the surface, corners are cut, details are missing, and technical debt accumulates with every change. The problem is not the model. It is that what we ask it to do is fundamentally under-specified."
date: 2026-05-02
featured: true
tags: ["agents", "engineering", "unified-views"]
category: "engineering"
cover: "/images/babel-tower.jpg"
coverAlt: "The Tower of Babel by Pieter Bruegel the Elder, 1563. Kunsthistorisches Museum, Vienna. Public domain."
---

Coding agents are genuinely good now. I have spent the past year
building production systems with frontier models, and the
improvement since Opus 4.5 is not incremental. These tools write
sophisticated, idiomatic code across languages and frameworks. They
handle multi-file changes, reason about types, generate tests, and
follow complex instructions. The first draft is often impressive.

But look under the hood.

Corners are cut. Error handling is optimistic. Abstractions are
introduced that duplicate ones that already exist. Naming drifts
from the codebase's conventions. Edge cases are papered over with
conditionals instead of addressed at the right layer. When a bug is
found, the fix is a band-aid. When the band-aid introduces another
bug, the next fix is another band-aid. Each change works. The sum
of changes rots.

You can feel the technical debt accumulating without reading a
single line. The errors the system produces start telling you: this
codebase is no longer coherent. The model is not building inside a
design. It is negotiating with its own previous patches.

This is not just a feeling. Alibaba's
[SWE-CI study](https://arxiv.org/abs/2603.03823) measured it: 18
AI models maintaining 100 real codebases over 233 days and 71
consecutive commits. Most agents (75%+) showed accelerating
regression rates: their changes broke previously passing tests at
increasing rates over time. The agents passed each immediate task.
The codebase degraded anyway.

The natural reaction is to blame the model. But I have seen the
same models write high-quality code when given precise, detailed
instructions. The quality ceiling is high. What varies is how much
of the actual problem reaches the model.

The problem is not the model. The problem is that what we ask
coding agents to do is fundamentally under-specified.

## The thesis

**Once model capability crosses a threshold, code quality becomes a
function of specification completeness, not model intelligence.**

A frontier model given a tight spec writes clean code. The same
model given a loose prompt writes code that runs, passes the
obvious test, and quietly degrades the architecture. The difference
is not capability. It is how much of the real engineering problem
was communicated.

This is not a temporary gap that the next release will close. It is
a structural property of the interface between natural language and
code. Four things make it structural.

## Why this happens

Four structural reasons.

### 1. Code is precise; prompts are not

Code must specify every branch, type, state transition, error path,
and side effect. Natural language typically names only the happy
path.

"Add authentication" compresses an enormous number of decisions into
two words. The implementation must choose: what identity model?
What happens on expired tokens? Role-based or resource-based
permissions? Where do checks live? What gets logged? What errors
are safe to expose? How does this interact with existing tests, the
CLI, the API schema, database migrations?

A human engineer encountering that compression slows down and asks
questions. A coding agent proceeds, filling every gap from
training-data priors.

Those gaps do not disappear. They become hidden implementation
choices, buried inside the code, undocumented, and often
inconsistent with each other and with choices the human would have
made. The code compiles. The decisions are invisible. Until
something breaks three changes later and nobody can explain why.

### 2. Humans do not prompt at the precision of code

When humans write code, the medium enforces precision. The compiler,
type checker, and test runner reject ambiguity.

When humans write prompts, the medium absorbs it. The model never
replies "this instruction is insufficient." It fills in the missing
details silently and proceeds.

This creates a strange inversion. The human moves from a strict
medium (code) to a permissive one (natural language), but the
output returns to a strict medium (code). The looseness does not
vanish. It gets converted into unexamined decisions inside the
generated implementation.

The first result feels magical. The agent guessed enough hidden
assumptions to produce something that works. But over repeated
changes, the guesses accumulate. Each one is a small undocumented
commitment baked into the codebase. Over weeks and months, the
codebase becomes a fossil record of guesses, each one locally
reasonable, collectively incoherent.

### 3. Even with a complete spec, scale defeats faithfulness

Even if a user took the time to write a precise and complete
specification, the agent still has to follow it faithfully across a
large codebase. That is genuinely hard, and current models are not
reliable at it.

The relevant context may span dozens of files. Constraints are
often implicit in existing patterns, never stated anywhere. Long
instructions compete with local code context for the model's
attention. The model can satisfy the visible part of the spec while
violating an invariant three modules away.

Production software is not a single, isolated problem. It is a
network of constraints: preserve interfaces, respect module
boundaries, avoid duplicating concepts, migrate data safely,
maintain observability, keep tests meaningful, and do not make
future changes harder. A model can be excellent at local synthesis
and still weak at global stewardship.

Models are improving fast on this axis. Context windows are growing,
retrieval is getting better, and tool use lets agents explore
codebases before editing. But the gap between "can follow a spec in
a single file" and "can honor all constraints across a 50k-line
project" remains real. Scale turns a capable model into a locally
correct but globally inconsistent editor.

### 4. Models are trained on artifacts, not the engineering process

This is the deepest point, and the one least likely to be solved by
scaling alone.

Models are trained on code snapshots: repositories, files, diffs,
documentation, Stack Overflow answers. But software engineering is
not the final artifact. It is the process that produced it.

The training data is systematically missing:

- the design that was considered and rejected,
- the refactor that happened six months later,
- the incident caused by a shortcut,
- the review comment that prevented a bad abstraction,
- the conversation where the team decided "not yet,"
- the operational pain that eventually forced an architecture
  change.

The model sees the code. It does not see the code *aging*.

So it learns what code tends to look like, but not why certain code
survives and other code becomes a liability. It can reproduce the
surface of good engineering without the lifecycle reasoning that
made each design decision appropriate in its original context.

The cleanest formulation:

> Coding agents are trained on software artifacts, but production
> engineering is an intertemporal optimization problem.

The quality of a change is not only whether it works today. It is
whether it preserves optionality tomorrow. The temporal dimension,
the cost of a decision over time, is largely absent from the
training signal.

This is why agent-written code can look clean in any single commit
and still degrade a codebase over a sequence of commits. Each
commit optimizes for "now." Nobody optimized for the trajectory.

## It is not just architecture: semantic errors too

So far I have framed this as an architectural and structural
problem. But under-specification bites at every level, including
pure logic.

A model asked to "extract entities from log lines" will write a
regex or parser that works on the examples in front of it. But the
prompt did not specify the full distribution of formats, edge
cases, or failure modes. The code passes the visible tests and
silently misses 30% of production inputs. The logic is wrong, not
because the model cannot write correct regex, but because the
prompt did not define what "correct" means across the full input
space.

I have seen this repeatedly in ML code. A model builds a neural
network that trains and converges, but quietly detaches a
regularization branch, or wires a skip connection to the wrong
layer, or initializes an embedding that never receives gradients.
The training loss goes down. The architecture diagram looks right.
The actual computation is not what was intended. The bug only
surfaces when the system touches reality at scale, and by then it
is buried under layers of subsequent changes.

These are not architectural debt. They are *semantic* errors:
cases where the code does something other than what the human
meant, in ways invisible to tests scoped to the examples the human
provided. The model wrote syntactically valid, test-passing code
that implements the wrong function. This is one of the most
dangerous failure modes, because it looks correct until it does
not.

The root cause is the same: the prompt specified the visible
behavior but not the full semantics. The model filled in the rest.

## Why tests do not save you

The obvious counter is: write tests, run them in the loop, and the
agent will converge on correct code. This is true for *behavioral*
correctness but nearly useless for *architectural* quality.

Tests verify: given this input, produce that output. They do not
verify: the code should be organized this way, these concepts
should not be duplicated, this module boundary should not be
crossed, this abstraction should not exist.

A codebase can have 100% test coverage and be unmaintainable.

Agentic loops with auto-test feedback feel productive but can
actively accelerate debt. The loop's reward is "tests pass." The
loop's blind spot is "the design got worse to make them pass." Each
iteration of "fix the failing test" can simultaneously increase
passing tests and increase structural entropy.

## Why better models do not fix this

Every frontier release writes better code than the last. But the
improvement is in *general capability*, not in *alignment to your
project's specific objectives*.

A more capable model given an under-specified prompt does not write
code better aligned to your intent. It writes more *confidently*
wrong code. The mistakes become more plausible, more idiomatic,
harder to spot. The failure mode shifts from obvious (broken
syntax) to subtle (clever-looking code that quietly fights the
architecture, or a regex that covers 90% of cases instead of 100%).

The cost of *detecting* a bad decision tends to rise with model
capability. The rate of bad decisions stays roughly constant
relative to specification completeness. Better models make the
surface shinier while leaving the structural problem untouched.
Labs could train for maintenance-awareness or project-specific
alignment, but currently they optimize for benchmark performance on
isolated tasks, not for long-term codebase health.

More precisely: let $Q$ be code quality, $C$ be model capability,
and $S$ be specification completeness. For low $C$, quality is
bottlenecked by capability and better models help. For high $C$,
quality saturates and becomes a function of $S$:

$$Q \approx f(S) \cdot \mathbb{1}[C \geq C^*]$$

Once you cross the capability threshold, the lever is no longer the
model. It is the specification. And writing a good specification is
expensive, often more expensive than writing the code it describes.

## Where the missing spec lives

The specification is always incomplete because it is distributed
across layers that a prompt rarely reaches:

1. **Behavioral**: what the program should do.
2. **Interface**: input/output shapes, invariants, error contracts.
3. **Architectural**: how this fits the rest of the system, what to
   reuse, what not to introduce.
4. **Lifecycle**: how this code will evolve, what future changes to
   anticipate, what to defer.
5. **Cultural**: the codebase's conventions, the team's taste, what
   "good" means here.

A typical prompt covers (1) and part of (2). Layers (3) through (5)
are almost never written down. The model fills them from training
priors. On every dimension you did not constrain, the model tends
to regress toward the average of its training corpus, and the
average of public code is not a well-maintained codebase.

## The harness helps, but is not a fix

Between the human and the model sits a layer that gets less
attention than it deserves: the harness. Claude Code, Cursor,
GitHub Copilot, Codex, the agent loop, the scaffold that decides
what files the model sees, what tools it can call, when to stop,
and what feedback gets fed back in.

Harnesses are the practical place where missing specification
either enters the loop or does not. A `CLAUDE.md` or `.cursorrules`
file injects convention. A repo map injects architecture. A typed
interface injects contract. A pre-commit linter injects taste. A
reviewer agent injects judgment. Each of these is a way to convert
tacit spec into something the model can see at the moment of
generation.

But it would be wrong to call the harness a fix. Everything I
described in the opening (the band-aid bug fixes, the
architectural drift, the silent semantic errors) happened to me
with Claude Code, Cursor, GitHub Copilot, and Codex actively in
the loop, with `CLAUDE.md` files, repo conventions, and typed
interfaces in place. Modern harnesses raise the floor. They do not
eliminate the gap. The intent is still ambiguous. The problem is
still under-specified. The harness can only inject what the human
took the time to write down, and most projects only ever write
down a fraction of what the agent needs to know.

What the harness does change is *who decides what the model sees*.
A thin wrapper passes the prompt through and hopes. A
well-structured loop lets you inject context, gate tool calls,
inspect every step, and decide when to stop. Most production agent
problems live between iterations, not inside any single model call,
which is why I built [looplet](https://github.com/hsaghir/looplet)
around the idea that the loop should be a `for` loop you own. (I
made the longer case for that
[in a previous post](/2026/04/23/the-loop-is-the-product/).)

Owning the loop does not write the missing spec for you. It just
makes the spec *injectable* at the point where it matters, and
makes the trajectory inspectable when something goes wrong. That is
necessary, not sufficient.

## What follows

If the bottleneck is specification rather than capability, several
things follow.

**The frontier moves to specification tooling.** Anything that
makes tacit codebase knowledge visible to the agent is high
leverage. Architecture docs the agent actually reads. Convention
files. Typed interfaces. Property tests. Semantic linters. Codebase
maps. Custom skills that encode "in this repo, we do X this way."
These are not documentation. They are specification surface area,
and most of them live inside the harness rather than the prompt.

**The engineer's job shifts.** Not from writing code to prompting.
From writing code to *authoring specification*: continuously
articulating the constraints that previously lived only in your
head. This is closer to writing a constitution for a codebase than
writing requirements for a feature.

**Agents need a maintenance loss function, not just a behavioral
one.** Today the agentic loop optimizes "make it work." A
maintenance-aware loop would also minimize structural drift, prefer
reuse over creation, prefer narrowing types over widening, and
prefer deletion over addition. That signal has to enter the loop,
whether through a reviewer, a linter, or a second model that acts
as an architectural critic.

**Prototypes and production are different workflows.** Coding agents
are extraordinary at prototypes and unreliable on production
without heavy specification scaffolding. The mistake is using the
same workflow for both and expecting the prototype's magic to
sustain a production codebase.

## The punchline

The hard part of software engineering was never typing. It was
deciding what code should exist, what should not, and what changes
the system should make easy a year from now.

Coding agents make typing dramatically cheaper. They do not reduce
the cost of judgment. By making code cheaper to produce, they make
design relatively more expensive, not less.

> AI makes code cheap. Cheap code makes design expensive.

Your coding agent is not broken. It is under-specified. Until we
close the gap between what we say and what production code actually
requires, through better harnesses, richer specification
artifacts, and a more honest division of labor between human
judgment and machine execution, the pattern will repeat: fast
first drafts, slow-burning technical debt. The bottleneck is not
the model. It is the channel between us and the model.
