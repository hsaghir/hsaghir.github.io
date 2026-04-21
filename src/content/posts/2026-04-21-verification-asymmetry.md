---
title: "The verification asymmetry"
description: "Offense asks 'does a bug exist?' Defense asks 'are all bugs gone?' One is an existential claim you can check with a single example. The other is a universal claim nobody can check. This asymmetry, not model capability, is what determines where AI agents work in security."
date: 2026-04-21
tags: ["security", "agents", "unified-views"]
category: "security"
cover: "/images/dijkstra-portrait.jpg"
coverAlt: "Edsger W. Dijkstra at his desk, 2002. Portrait by Hamilton Richards."
---

The most important question about AI in security is not "how capable
is the model?" It is "can you verify the model's output?" Offense and
defense have fundamentally different answers to that question, and
the difference explains most of what is happening in the field right
now, including a non-obvious consequence: that AI is moving the cost
of software from *generation* to *security*.

## What happened

In early April 2026, Anthropic's restricted cybersecurity model
discovered a 27-year-old vulnerability in OpenBSD and a 16-year-old
bug in FFmpeg, bugs that five million automated scans had missed.
A week later, OpenAI shipped GPT-5.4-Cyber, a model fine-tuned for
defensive security. The UK AI Safety Institute independently
confirmed that these models are "exceptionally effective at
identifying security vulnerabilities."

Drew Breunig wrote the line everyone is now repeating: "Security is
reduced to a brutally simple equation: to harden a system you need
to spend more tokens discovering exploits than attackers will spend
exploiting them." Security is proof of work.

It is a clean thesis. It is also half-right. "Security" is actually
two different problems with fundamentally different logical
structures, and the difference explains far more than cybersecurity.

## The two questions

**Offense** asks: *Does a vulnerability exist in this system?*

This is an existential claim, $\exists$. You verify it by producing
a single example. Find one working exploit and you have answered the
question. The reward signal is binary, immediate, and unambiguous:
the exploit either works or it does not. This is why AI excels at it.
RL needs a clean reward, and offense has one.

**Defense** asks: *Have we caught every threat to this system?*

This is a universal claim, $\forall$. You cannot verify it by
producing examples. You would need to check every possible attack,
including attacks nobody has conceived of yet. No finite amount of
testing can prove a universal claim. This is Dijkstra's observation
about software ("testing can show the presence of bugs, but never
their absence"), applied to security.

When Breunig says "spend more tokens defending than attackers spend
attacking," what he is really describing is **offense-as-defense**:
using your own offensive agent to find bugs before adversaries do.
That works, and it is genuinely powerful. But it answers the $\exists$
question ("did we find a bug?"), not the $\forall$ question ("are we
safe?"). The two are not the same, and conflating them leads to a
false sense of security.

## Where the asymmetry bites

The gap between $\exists$ and $\forall$ is not academic. It shapes
which security problems AI can actually get better at, because it
determines where reinforcement learning works.

Modern LLMs improve through RL: the model takes an action, receives a
reward signal, and updates its weights to produce more of what gets
rewarded. The entire pipeline depends on one thing: a verifier that
can score the output. In math, you can check proofs mechanically. In
code, you can run tests. In offensive security, you can run the
exploit. The verifier is cheap, fast, and definitive. This is why RL
produces rapid capability gains in these domains. The reward signal
is clean, so the learning loop is tight.

Defense has no such verifier. Consider training a defensive security
agent with RL. The agent triages an alert and says "all clear." What
reward do you assign? A true negative (correctly identifying no
attack) is indistinguishable from a false negative (missing a real
attack) at the time the decision is made. You only discover missed
attacks after the damage, sometimes months later, when stolen data
appears on a marketplace. Without a reliable verifier, the RL loop
cannot close cleanly. The model cannot learn from outcomes it cannot
observe.

This is the same reason code-writing agents improve faster than
code-reviewing agents. Writing code produces a verifiable artifact
(tests pass or they do not). Reviewing code requires judging whether
something is *correct enough*, a $\forall$ claim that resists
automated verification.

The asymmetry compounds across three dimensions:

**Reward signals.** Offense: exploit succeeds = +1, fails = 0. Dense,
binary, immediate. Defense: the critical failure mode (missed attack)
is invisible at decision time. Any RL reward function for defense
must either use synthetic ground truth (which introduces a sim-to-real
gap) or wait for real-world outcomes (which are sparse, delayed, and
ambiguous).

**Data quality.** Offensive agents operate on clean artifacts: source
code, binaries, protocols. The ground truth is the system's behavior.
Defensive agents operate on logs, and logs are incomplete (not
everything is logged), ambiguous (is a login from an unusual country
a breach or a business trip?), noisy (signal-to-noise ratios of
1:10,000 are common), and adversarially manipulated (sophisticated
attackers suppress or forge log entries).

**Coverage.** Finding one bug requires search. Proving no bugs remain
requires exhaustive verification or formal proof. These are
fundamentally different computational problems. The former scales
with token budget; the latter may not scale at all.

The pattern is not specific to security. It appears everywhere AI
agents are deployed, and it tracks the $\exists$/$\forall$ line
precisely:

- In medicine: verifying that a drug treats *a* condition ($\exists$)
  is a clinical trial. Verifying that a diagnostic system catches
  *every* disease ($\forall$) is impossible.
- In finance: verifying that a trading strategy made money ($\exists$)
  is a backtest. Verifying that a risk model prevents *all* losses
  ($\forall$) is what banks have been failing to do for centuries.
- In code: verifying that a change passes *the* tests ($\exists$) is
  CI. Verifying that the change introduces *no* regressions ($\forall$)
  is why code review remains stubbornly hard to automate. The hard
  part of code is not writing it but verifying that nothing broke.

The verification structure of a domain, $\exists$ or $\forall$, turns
out to be a better predictor of where AI agents succeed than the raw
capability of the model. Wherever you can build a cheap verifier, RL
works and capability improves rapidly. Wherever you cannot, progress
stalls regardless of how large the model is.

## Can self-play close the gap?

There is an appealing argument that adversarial self-play solves the
defense verification problem. Put an offensive agent and a defensive
agent in the same environment and let them co-train. The offensive
agent attacks; the defensive agent detects. The offensive agent
provides the ground truth that defense otherwise lacks. Defense
becomes verifiable by extension, the same logic behind AlphaZero,
where you cannot verify "is this the best move?" but you *can* verify
"did I win the game?"

This is genuinely powerful. It closes the reward loop (the red agent
*knows* what it did, so blue's performance is measurable). It
generates infinite diverse training data with no scarcity or labeling
cost. It creates an arms race where both sides improve. And it has
precedent: GANs, game AI, centuries of military wargaming.

But it does not fully close the gap, for five reasons worth
understanding:

**1. Red bounds blue.** Blue can only learn to defend against attacks
red knows how to generate. If red does not know supply-chain attacks
or firmware implants, blue never learns to detect them. The most
damaging real-world attacks (SolarWinds, Log4Shell) were novel
techniques that no existing red team was practicing. Self-play
produces strong defense against *known* attack classes, not against
the unknown.

**2. Sim-to-real gap.** Self-play happens in a simulated environment.
Every simplification in the simulation (a missing EDR interaction,
an unmodeled network latency, a human behavior pattern not captured)
is a gap that real attackers can exploit. AISI acknowledged this
directly: their evaluation environments "lack security features that
are often present, such as active defenders and defensive tooling."

**3. The $\forall$ problem is displaced, not solved.** Even with
perfect self-play, you have verified that blue catches attacks red can
generate. The question "is my defense complete?" transforms into "is
my red agent complete?", the same universal verification problem
moved one level up.

**4. Non-stationarity.** Chess rules do not change between training
and deployment. IT environments do: new software, new configurations,
new users, new attacker tools. A defense trained via self-play against
yesterday's environment may fail against today's. GANs suffer from
exactly this problem (mode collapse); security self-play inherits it.

**5. The reward function is itself unverifiable.** Even in self-play,
you must design the reward. Does blue get credit for detecting an
attack after six months? Is there a false-positive penalty? How do
you score partial detection of a multi-stage attack? Every reward
design decision encodes an assumption about what "good defense"
means, and those assumptions may not match operational reality.

The honest conclusion: adversarial self-play makes defense
**measurably improvable**, not **verifiably complete**. That is an
enormous difference. "Measurably improvable" means you have a metric,
a training signal, and a trajectory. "Verifiably complete" means you
have proven $\forall$, which remains impossible.

This is exactly Dijkstra's point about testing, applied one level up.
Testing cannot prove correctness, but it is still enormously
valuable. Self-play cannot prove safety, but a defense trained
against a frontier-class red agent is vastly better than one built on
static signature libraries.

## Where the cost of software actually lands

Here is a consequence of the asymmetry that I think is underappreciated.

For most of software's history, the expensive part was *writing* code.
Design, implementation, debugging, testing: all of it required
skilled human time, and skilled human time was the binding constraint.
AI has collapsed that cost. A solo developer with a frontier coding
agent can build in a weekend what used to take a team months. The
"vibe coding" movement, whatever you think of its aesthetics, is a
real demonstration that software generation is rapidly approaching
commodity.

But the verification asymmetry tells us where the cost *migrates*.
It does not disappear; it moves downstream. If security hardening
scales linearly with token spend (Breunig's thesis), then every piece
of custom software you generate now carries a recurring security tax:
tokens spent continuously scanning, patching, and monitoring it. The
cost of software shifts from *writing it once* to *defending it
forever*.

This has a non-obvious consequence for the "replace SaaS with custom
code" thesis. If you vibe-code a replacement for a SaaS tool you are
paying $200/month for, you have eliminated the subscription. But you
have also taken ownership of the entire security surface. You now
need to spend tokens hardening that code, monitoring it in production,
and patching it when new vulnerability classes emerge. Nobody else is
sharing that cost with you.

Compare that to using a SaaS product, or an open-source library with
a large user base. The security tokens spent on shared infrastructure
are amortized across all users. When Anthropic's model finds a bug in
FFmpeg, every user of FFmpeg benefits. When it finds a bug in your
bespoke internal tool, only you benefit, and only you pay.

This is the same logic as the economics of shared infrastructure in
any domain. A bridge is expensive to secure, but the cost is shared
across everyone who crosses it. A private road is cheap to build but
you bear the full maintenance burden alone.

So the economics of shared code (open source, SaaS, platforms) may
actually *strengthen* in a world of cheap generation and expensive
verification, not weaken. The bottleneck is no longer writing code.
It is securing the code you wrote. And security, unlike generation,
does not have a clean stopping point: you cannot verify that you are
done.

## Three questions that follow

The verification asymmetry is structural, not a temporary limitation
but a consequence of logic. It raises questions that go beyond any
particular industry.

**How do you create partial verifiers for $\forall$ domains?** The
history of engineering is largely the history of converting universal
claims into bounded, testable ones. You cannot prove a bridge will
never fail, but you can prove it withstands a specific load under
specific conditions. You cannot prove software is bug-free, but you
can prove it passes a test suite. The gap between the bounded test
and the universal claim is your residual risk, and you can work to
narrow it. The same logic applies to defense: adversarial testing
with diverse, realistic attack scenarios is how you convert an
article of faith ("we are secure") into a bounded measurement ("we
detect these attack classes with this accuracy"). The practical
question is what it takes to build such a verifier: realistic
simulated environments, diverse red agents, deterministic replay,
and shared benchmarks so that measurements are comparable across
organizations.

**Does the gap widen or narrow?** If offense has tighter RL loops
than defense (clean reward signals, fast iteration, dense feedback),
then offensive capability will improve faster. This suggests the
asymmetry could *widen* over time, not close. Attackers get better
faster because the structure of their problem is more amenable to
learning. The counterweight is that defenders can use offensive
agents against their own systems (Breunig's thesis), converting
offense into a bounded verifier for defense. Whether the gap widens
or narrows depends on how quickly defenders adopt adversarial
self-testing and how faithfully their test environments approximate
reality.

**Is accountability a stopgap or a permanent feature?** In any domain
where outputs cannot be fully verified, someone must absorb the
residual risk that the verifier cannot eliminate. Kingsbury calls
this the "meat shield" problem. A less cynical framing: in $\forall$
domains, human accountability is not a limitation of current
technology. It is a logical consequence of deploying systems whose
correctness cannot be proven. This is as true for a physician
reviewing an AI diagnosis as for an analyst reviewing an agent's
"all clear." The human does not make the decision better. The human
makes the decision *accountable*. No amount of model improvement
changes this, because the issue is not capability but verifiability.

## The question that matters

The verification asymmetry is not a problem to be solved. It is a
structural feature of the world: $\exists$ claims are checkable,
$\forall$ claims generally are not. The question worth asking about
any AI system acting in the world is not "how capable is the model?"
but rather: *can you tell whether the model got it right?*

Wherever the answer is yes, expect rapid progress. Wherever it is
no, expect that the hardest problems remain hard, and that the
infrastructure around the model (the verifiers, the benchmarks, the
accountability structures) matters as much as the model itself.

---

*Sources: Anthropic [Project Glasswing](https://www.anthropic.com/glasswing) (April 2026), UK AISI ["Our evaluation of Claude Mythos Preview's cyber capabilities"](https://www.aisi.gov.uk/blog/our-evaluation-of-claude-mythos-previews-cyber-capabilities) (April 2026), Drew Breunig ["Cybersecurity Looks Like Proof of Work Now"](https://www.dbreunig.com/2026/04/14/cybersecurity-is-proof-of-work-now.html) (April 2026).*

*Cover image: Edsger W. Dijkstra at his desk, 2002, by [Hamilton Richards](https://commons.wikimedia.org/wiki/File:EdsgerDijkstra.jpg), [CC BY-SA 3.0](https://creativecommons.org/licenses/by-sa/3.0).*
