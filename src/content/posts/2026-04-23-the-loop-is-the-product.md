---
title: "The loop is the product"
description: "Most LLM agent frameworks give you a graph DSL and hide the loop. But the loop is where all the interesting decisions happen. What if the framework was just the loop, and you owned it?"
date: 2026-04-23
tags: ["agents", "python", "design", "open-source"]
category: "engineering"
cover: "/images/jacquard-loom.jpg"
coverAlt: "Close-up of a Jacquard loom's punch-card chain, Science Museum, London."
---

I shipped [looplet](https://github.com/hsaghir/looplet) last week.
It is a Python library for building LLM agents that call tools. The
entire API is a `for` loop:

```python
for step in composable_loop(llm=llm, tools=tools, task=task, ...):
    print(step.pretty())
```

Each `step` is a dataclass containing the prompt the model saw, the
tool call it made, the result, token usage, and wall-clock time. You
own the iterator. `break`, `continue`, `try/except`, `asyncio.sleep` --
just Python. Zero runtime dependencies.

This post is about *why* the loop is the right abstraction, not about
the library itself. The library is a consequence of the argument.

## The shape of most agents

Draw a picture of what an LLM agent does:

1. Build a prompt (system message + history + tool schemas).
2. Call the model.
3. Parse the response into a tool call.
4. Execute the tool.
5. Decide whether to stop.
6. If not, go to 1.

That is a loop. Not a DAG, not a state machine, not a pipeline of
nodes with conditional edges. A loop with a body and a termination
condition.

Some agents genuinely are graphs. A triage node that routes to
separate research and coding branches, each with their own state,
joining at a review node -- that is a graph, and LangGraph is the
right tool for it. But most agents I have built, and most agents I
see other people build, are loops. One model, a bag of tools, a
stopping condition. The graph has one node and one edge.

When a framework forces you to express a loop as a graph, two things
happen. First, you pay a conceptual tax: you learn node types, edge
types, state schemas, checkpoint semantics, and a graph compiler, all
to describe a structure that `for` already describes. Second, and
worse, the framework takes ownership of the loop. You hand it a graph
definition and call `.invoke()`. The loop runs inside the framework.
You get callbacks, not control.

## Why control of the loop matters

The interesting decisions in an agent are not "which tool to call" --
the model handles that. The interesting decisions are:

- **When to stop.** Not just "the model said done" but "we've spent
  $0.40 and the user's budget is $0.50" or "the last three tool calls
  returned the same error."
- **What to show the model.** Context management -- pruning old
  messages, summarising, injecting retrieved documents -- is the
  difference between an agent that works on toy problems and one that
  works on real ones.
- **Whether to let a tool call proceed.** Permission checks, PII
  redaction, argument rewriting, human approval gates.
- **What to record.** The full prompt the model saw, not just the
  tool call. The wall-clock time of each step, not just the final
  result. Token usage per step, not just the total.

All of these are *loop-body decisions*. They happen between iterations,
not at graph edges. When you own the loop, they are just Python:

```python
for step in composable_loop(...):
    if step.usage.total_tokens > budget:
        break
    if step.tool_call.name == "delete_file":
        approval = input(f"Allow {step.tool_call}? [y/n] ")
        if approval != "y":
            break
    log.append(step)
```

When the framework owns the loop, each of these becomes a callback
registration, a hook class, a middleware layer, or a custom node type.
The complexity is the same, but the indirection is not.

## The Protocol pattern

Putting everything in the loop body doesn't scale either. If every
agent needs PII redaction, approval gates, tracing, and context
compaction, you end up with a 200-line loop body that is just as
opaque as the framework you were trying to avoid.

The fix is Python's `Protocol` pattern (PEP 544). Define four
structural interfaces:

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

A *hook* is any object that implements one or more of these methods.
No base class, no inheritance, no registration:

```python
class RedactPII:
    def pre_prompt(self, state, log, ctx, step):
        return scrub_emails(ctx)

class ApprovalGate:
    def pre_dispatch(self, state, log, tc, step):
        if tc.tool in SENSITIVE_TOOLS:
            return Deny("needs human approval")
```

Hooks compose by stacking in a list. The loop calls them in order.
That is the entire extension model.

Why Protocol instead of base classes? Because base classes create a
coupling between the hook and the framework. If `RedactPII` inherits
from `looplet.Hook`, it cannot be used outside looplet without
carrying looplet as a dependency. With Protocol, `RedactPII` is just
a class with a `pre_prompt` method. It works in looplet, it works in
your test harness, it works in a completely different agent library
that happens to call the same method name. Structural typing is the
right abstraction for extension points.

## The eval trick

Here is a design choice I am particularly happy with: the debug trace
and the evaluation harness are the same artifact.

`step.pretty()` prints a human-readable one-liner:

```
#1 search(query='quarterly revenue') -> 12 results [340ms, 1.2k tok]
```

`ProvenanceSink` writes each step to disk as JSON:

```python
sink = ProvenanceSink(dir="traces/run_42/")
for step in composable_loop(..., hooks=[sink]):
    ...
```

The `eval_*` helpers read `ProvenanceSink` output directly:

```python
def eval_found_answer(ctx: EvalContext) -> bool:
    return any("revenue" in s.tool_result.text for s in ctx.steps)
```

There is no separate logging pipeline, no eval framework, no
telemetry SDK. The thing you print while debugging is the thing
you save to disk is the thing you evaluate against. One artifact,
three uses.

This is only possible because the loop yields `Step` objects with
full provenance. If the framework owns the loop and gives you
callbacks, you get fragments -- a tool name here, a result there --
and you have to reconstruct the full picture yourself.

## Zero dependencies

looplet's core has zero runtime dependencies. `pip install looplet`
pulls in nothing. The `openai` and `anthropic` packages are optional
extras, imported lazily when you instantiate a backend.

This is a deliberate choice, not an accident. Cold-import time for
looplet is 289 ms. For comparison:

| Framework | Cold import |
|-----------|------------|
| looplet | 289 ms |
| strands-agents | 1,885 ms (6.5x) |
| LangGraph | 2,294 ms (7.9x) |
| Claude Agent SDK | 2,409 ms (8.3x) |
| Pydantic AI | 3,975 ms (13.8x) |

(Median of 9 runs, Python 3.11, Linux x86_64. Scripts in the repo.)

Why does this matter? Agents are increasingly invoked as CLI tools,
serverless functions, and dev-loop scripts that restart on every save.
A 4-second import tax on every invocation is the difference between
"snappy" and "go get coffee." And dependency trees are liability
trees: every transitive dependency is a surface for breaking changes,
CVEs, and version conflicts.

## When not to use it

Honesty about limitations is more useful than marketing.

**Use LangGraph when** your agent is genuinely a multi-node graph --
a triage node routing to research and coding branches with separate
state, joining at a review node. That is the shape LangGraph is
designed for, and expressing it as a single loop with hooks is
fighting the abstraction.

**Use LangGraph when** you need durable checkpointing with built-in
backends (SQLite, Postgres, Redis) tied to node boundaries. looplet
has `ProvenanceSink` and you can build checkpointing as a hook, but
LangGraph gives it to you out of the box.

**Stay in LangChain when** your prompts are `ChatPromptTemplate`s,
your retrievers are LangChain retrievers, and your consumers expect
LangChain events. Bridging all of that through looplet adds friction
for no gain.

## The punchline

The agent framework landscape in 2026 looks like the web framework
landscape in 2010. Lots of large, opinionated systems competing to
own your entire stack. The history of web frameworks suggests that
the systems which survive are the ones that bet on the language's
own control flow rather than replacing it. Flask beat most of its
contemporaries not because it was more powerful, but because a Flask
app was just a Python function.

looplet is the same bet. A `for` loop is the right abstraction for
"call an LLM, run a tool, repeat." If you keep the abstraction honest,
the language gives you control flow, error handling, concurrency,
and testing for free. The framework's job is to make the loop
composable and observable, not to own it.

```bash
pip install looplet
```

[GitHub](https://github.com/hsaghir/looplet) |
[Docs](https://hsaghir.github.io/looplet/) |
[PyPI](https://pypi.org/project/looplet/)
