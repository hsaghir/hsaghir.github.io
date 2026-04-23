---
title: "The loop is the product"
description: "Agent frameworks hide the loop behind agent.run() and a graph DSL. But the loop is where every interesting decision happens: what the model sees, whether a tool call proceeds, when to stop, what to record. What if you owned the loop and the framework just made it composable?"
date: 2026-04-23
featured: true
tags: ["agents", "python", "design", "open-source"]
category: "engineering"
cover: "/images/ouroboros.jpg"
coverAlt: "An ouroboros, the snake eating its own tail, from a 1478 drawing by Theodoros Pelecanos in the alchemical treatise Synosius. Public domain."
---

## The claim

The right abstraction for most LLM agents is not a graph, not a
state machine, and not a pipeline of nodes. It is a `for` loop that
you own and can observe, interrupt, or extend at every step.

I built [looplet](https://github.com/hsaghir/looplet) around this
idea. The entire API is one iterator:

```python
for step in composable_loop(llm=llm, tools=tools, task=task, ...):
    print(step.pretty())   # "#1 search(query='...') -> 12 items [340ms, 1.2k tok]"
```

Zero runtime dependencies. Four extension points. Works with any
OpenAI-compatible endpoint or Anthropic directly. The rest of this
post explains why the loop is the right cut, not the library itself.

## Three reasons the loop is the right abstraction

### 1. Most agents are loops, not graphs

Draw a picture of what a tool-calling agent does: build a prompt,
call the model, parse the response, run the tool, decide whether to
stop, repeat. That is a loop with a body and a termination condition.
Not a DAG, not a state machine.

Some agents genuinely are graphs. A triage node routing to research
and coding branches, each with their own state, joining at a review
node: that is a graph, and LangGraph is the right tool for it. But
most agents I have built, and most I see other people build, are
loops. One model, a bag of tools, a stopping condition. The graph
has one node and one edge.

When a framework forces you to express a loop as a graph, you pay a
conceptual tax: node types, edge types, state schemas, checkpoint
semantics, a graph compiler. All to describe a structure that `for`
already describes.

### 2. The interesting decisions are loop-body decisions

The model decides which tool to call. That part is handled. The
decisions that *you* need to make are all between iterations:

- **When to stop.** Not just "the model said done" but "we have
  spent \$0.40 and the user's budget is \$0.50" or "the last three
  calls returned the same error."
- **What the model sees.** Context management (pruning, summarising,
  injecting retrieved documents) is the difference between an agent
  that works on toy problems and one that works on real ones.
- **Whether a tool call proceeds.** Permission checks, PII
  redaction, argument rewriting, human approval gates.
- **What gets recorded.** The full prompt, not just the tool name.
  Wall-clock time per step, not just the final result. Token usage
  per step, not just the total.

When you own the loop, these are just Python:

```python
for step in composable_loop(...):
    if step.usage.total_tokens > budget:
        break
    if step.tool_call.name == "delete_file":
        if input(f"Allow {step.tool_call}? [y/n] ") != "y":
            break
    log.append(step)
```

When the framework owns the loop, each of these becomes a callback,
a hook class, a middleware layer, or a custom node type. Same
complexity, more indirection.

### 3. Debugging and evaluation become the same thing

This is the design choice I care most about.

Think about how you debug a coding agent like Claude Code. You look at
the sequence of tool calls, check what the model saw at each step,
and ask: did it have the right context? Did it call the right tool
with the right arguments? Did it stop at the right time? You are
inspecting a trajectory.

Now think about how you evaluate an agent. You run it on a set of
tasks, record the trajectories, and check: did it find the answer?
Did it use the expected tools? Did it stay within budget? You are
inspecting the same trajectory, just with assertions instead of eyes.

Debugging and evaluation are the same activity at different levels
of automation. If your framework gives you full trajectories as
first-class objects, the path from "I am staring at this in a
terminal" to "I have a regression test for this" is just wrapping
your observation in a function:

```python
# What you do while debugging
for step in composable_loop(...):
    print(step.pretty())
    # "Hmm, step 3 called search but got zero results"

# What you write as an eval
def eval_found_results(ctx: EvalContext) -> bool:
    return any(
        s.tool_call.name == "search" and len(s.tool_result.data) > 0
        for s in ctx.steps
    )
```

The `step.pretty()` trace, the `ProvenanceSink` that dumps it to
disk, and the `eval_*` helpers that read it back are all operating on
the same `Step` dataclass. No separate logging pipeline. No eval
framework. No telemetry SDK. One artifact, three uses.

If you have used Claude Code or a similar coding agent, you know the
feeling of watching the tool calls scroll by and thinking "that was
the wrong move." Now imagine you could intercept that step, rewrite
the tool arguments, inject missing context, enforce a permission
check, or just log it to a file that your test suite reads tomorrow.
That is what owning the loop gives you. Not for coding tasks
specifically, but for any task you point an agent at.

## How the extension model works

Putting everything in the loop body does not scale. If every agent
needs PII redaction, approval gates, tracing, and context compaction,
you end up with a 200-line loop body.

The fix is Python's `Protocol` pattern (PEP 544). Four structural
interfaces:

```python
class PrePrompt(Protocol):
    def pre_prompt(self, state, log, ctx, step): ...

class PreDispatch(Protocol):
    def pre_dispatch(self, state, log, tool_call, step): ...

class PostDispatch(Protocol):
    def post_dispatch(self, state, log, tool_call, result, step): ...

class CheckDone(Protocol):
    def check_done(self, state, log, step): ...
```

A hook is any object that implements one or more of these methods.
No base class, no inheritance, no registration:

```python
class RedactPII:
    def pre_prompt(self, state, log, ctx, step):
        return scrub_emails(ctx)

class ApprovalGate:
    def pre_dispatch(self, state, log, tc, step):
        if tc.tool in SENSITIVE_TOOLS:
            return Deny("needs human approval")

for step in composable_loop(..., hooks=[RedactPII(), ApprovalGate()]):
    print(step.pretty())
```

Hooks compose by stacking in a list. The loop calls them in order.

Why Protocol instead of base classes? Because structural typing
decouples the hook from the framework. `RedactPII` is just a class
with a `pre_prompt` method. It works in looplet, it works in a test
harness, it works in a different agent library that calls the same
method name. No import dependency, no vendor lock-in at the extension
layer.

Imagine you could take Claude Code's tool-calling loop and slot in
your own `pre_dispatch` hook that enforces file-write permissions, a
`post_dispatch` hook that logs every shell command to a
compliance trail, and a `check_done` hook that stops the agent if
it has been running for more than 60 seconds. That is the level of
control looplet gives you, for any agent you build.

## Sub-agents are just nested loops

Once the loop is the product, sub-agents stop being a special
feature. They are a tool that happens to run another loop.

Claude Code's `Task` tool spawns a sub-agent with its own context,
its own tools, and its own stopping condition, then returns a
summary to the parent. In looplet, that is a regular tool:

```python
def research(query: str) -> str:
    sub_task = f"Research: {query}. Return a 3-bullet summary."
    steps = list(composable_loop(
        llm=cheap_llm,
        tools=[search, fetch_url],
        task=sub_task,
        hooks=[BudgetCap(tokens=4000)],
        max_steps=10,
    ))
    return steps[-1].message.content

for step in composable_loop(
    llm=smart_llm,
    tools=[research, write_file, run_tests],
    task=user_task,
):
    print(step.pretty())
```

The parent agent sees `research` as a normal tool call. Inside,
a whole sub-loop runs with its own model (cheaper), its own tool
set (narrower), its own hooks (tighter budget), and its own
trajectory that you can inspect and evaluate independently.

This gives you the three things people actually want from
sub-agents: **context isolation** (the parent never sees the
sub-agent's 50-step browsing trajectory, just the answer), **cost
control** (run the explorer on a cheap model, the planner on a
smart one), and **specialisation** (different tools, different
prompts, different stopping rules per sub-agent). No orchestrator,
no agent registry, no message bus. Just a function that calls
`composable_loop` and returns a string.

The same `ProvenanceSink` dumps the parent and child trajectories
to the same store with a parent-child link. The same `eval_*`
helpers work on either level. Debugging a sub-agent is debugging
a loop, because that is all it is.

## What it costs

Zero runtime dependencies. `pip install looplet` pulls in nothing.
The `openai` and `anthropic` packages are optional extras, imported
lazily when you instantiate a backend.

| Framework | Cold import | PyPI deps |
|-----------|------------|-----------|
| looplet | 289 ms | 0 |
| strands-agents | 1,885 ms | 6 |
| LangGraph | 2,294 ms | 31 |
| Claude Agent SDK | 2,409 ms | 13 |
| Pydantic AI | 3,975 ms | 12 |

(Median of 9 runs, Python 3.11, Linux x86_64. Scripts in the repo.)

Agents are increasingly invoked as CLI tools, serverless functions,
and dev-loop scripts that restart on every save. A 4-second import
tax on every invocation is the difference between snappy and slow.
And dependency trees are liability trees: every transitive dependency
is a surface for breaking changes, CVEs, and version conflicts.

## When not to use it

**Use LangGraph when** your agent is genuinely a multi-node graph
with branching state. A triage node routing to separate research and
coding branches, joining at a review node. That is the shape
LangGraph is designed for.

**Use LangGraph when** you need durable checkpointing with built-in
backends (SQLite, Postgres, Redis) tied to node boundaries. looplet
has `ProvenanceSink` and you can build checkpointing as a hook, but
LangGraph gives it to you out of the box.

**Stay in LangChain when** your prompts are `ChatPromptTemplate`s,
your retrievers are LangChain retrievers, and your consumers expect
LangChain events.

## The punchline

The agent framework landscape in 2026 looks like the web framework
landscape in 2010. Large, opinionated systems competing to own your
stack. The frameworks that survived that era were the ones that bet
on the language's control flow rather than replacing it. Flask beat
most of its contemporaries not because it was more powerful, but
because a Flask app was just a Python function.

looplet is the same bet. A `for` loop is the right abstraction for
"call an LLM, run a tool, repeat." If you keep the abstraction
honest, the language gives you control flow, error handling,
concurrency, and testing for free. The framework's job is to make
the loop composable and observable, not to own it.

```bash
pip install looplet
```

[GitHub](https://github.com/hsaghir/looplet) |
[Docs](https://hsaghir.github.io/looplet/) |
[PyPI](https://pypi.org/project/looplet/)
