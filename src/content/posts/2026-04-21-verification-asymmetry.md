---
title: "The verification asymmetry"
description: "Offense asks 'does a bug exist?' Defense asks 'are all bugs gone?' The first is an existential claim you can check with one example. The second is a universal claim you cannot fully check. This difference determines where AI agents work in security more than model capability does."
date: 2026-04-21
featured: true
tags: ["security", "agents", "unified-views"]
category: "security"
cover: "/images/dijkstra-portrait.jpg"
coverAlt: "Edsger W. Dijkstra at his desk, 2002. Portrait by Hamilton Richards."
---

The first question about AI in security is "can you verify the
model's output?" Offense and defense give different answers. That
difference explains much of the field today, including why AI shifts
software costs from *generation* to *security*.

## What happened

In early April 2026, Anthropic's restricted cybersecurity model
found a 27-year-old vulnerability in OpenBSD and a 16-year-old bug
in FFmpeg. Five million automated scans had missed the bugs.
A week later, OpenAI released GPT-5.4-Cyber, a model fine-tuned for
defensive security. The UK AI Safety Institute (AISI) independently
confirmed that these models are "exceptionally effective at
identifying security vulnerabilities."

These results make it tempting to treat security as a contest over
token budgets. Defenders must spend more to find exploits than
attackers spend to use them. Security becomes a form of proof of work.

That view covers only part of security. Finding a flaw and proving
a system safe are different logical problems. This difference
extends beyond cybersecurity.

## The two questions

**Offense** asks: *Does a vulnerability exist in this system?*

This is an existential claim, $\exists$. You verify it with a
single example. Find one working exploit and you have answered the
question. The exploit either works or it does not, so the reward is
immediate and unambiguous. That is why AI excels at it.
Reinforcement learning (RL) needs a clear reward, and offense has one.

**Defense** asks: *Have we caught every threat to this system?*

This is a universal claim, $\forall$. Examples cannot verify it.
You would need to check every possible attack, including attacks
nobody has thought of yet. No finite amount of testing can prove a
universal claim. Dijkstra's observation about software applies to
security: "testing can show the presence of bugs, but never their
absence."

**Offense-as-defense** means using your own offensive agent to find
bugs before attackers do. It works, but it answers the $\exists$
question ("did we find a bug?"). It does not answer the $\forall$
question ("are we safe?"). Treating those answers as equivalent can
create a false sense of security.

## Where the asymmetry bites

The gap between $\exists$ and $\forall$ determines where RL can
improve AI security systems.

Large language models (LLMs) improve through RL: the model takes an
action and receives a reward. Training updates its weights to favor
actions that earn rewards. This process needs a verifier that can
score the output. In math, you can check proofs mechanically. In
code, you can run tests. In offensive security, you can run the
exploit.

These verifiers are cheap, fast, and definitive. They provide clear
rewards, so models can learn rapidly.

Defense has no such verifier. Consider a defensive security agent
trained with RL. The agent triages an alert and says "all clear."
What reward do you assign?

At that moment, you cannot distinguish a true negative (correctly
identifying no attack) from a false negative (missing a real attack).
You discover missed attacks after the damage, sometimes months
later, when stolen data appears on a marketplace. Without a reliable
verifier, the model cannot learn from outcomes it cannot observe.

Code-writing agents improve faster than code-reviewing agents for
the same reason. Writing code produces an artifact you can check:
tests pass or they do not. Reviewing code requires you to judge
whether something is *correct enough*, a $\forall$ claim that
resists automated verification.

The differences affect training in three ways:

**Reward signals.** An exploit that succeeds earns +1; a failed
exploit earns 0. The feedback is frequent, binary, and immediate.
In defense, the critical failure, a missed attack, is invisible at
decision time. An RL reward must use synthetic ground truth, which
creates a gap between simulation and reality. Otherwise, it must
wait for real outcomes, which are sparse, delayed, and ambiguous.

**Data quality.** Offensive agents use clean artifacts: source code,
binaries, and protocols. The system's behavior provides the ground
truth. Defensive agents use logs, which are incomplete because
systems do not log everything. A login from an unusual country
could mean a breach or a business trip. Logs are also noisy:
signal-to-noise ratios of 1:10,000 are common. Sophisticated
attackers suppress or forge log entries.

**Coverage.** Finding one bug requires search. Proving no bugs remain
requires exhaustive verification or formal proof. These are
different computational problems. Finding bugs scales with the
token budget. Proving their absence may not scale at all.

The pattern extends beyond security and follows the
$\exists$/$\forall$ distinction:

- In medicine: verifying that a drug treats *a* condition ($\exists$)
  is a clinical trial. Verifying that a diagnostic system catches
  *every* disease ($\forall$) is impossible.
- In finance: verifying that a trading strategy made money ($\exists$)
  is a backtest. Banks still cannot verify that a risk model prevents
  *all* losses ($\forall$), despite centuries of trying.
- In code: verifying that a change passes *the* tests ($\exists$) is
  continuous integration (CI). Verifying that the change introduces
  *no* regressions ($\forall$) makes code review hard to automate.
  Checking that nothing broke is the hard part.

A domain's verification structure, $\exists$ or $\forall$, predicts
where AI agents succeed better than raw model capability does.
With a cheap verifier, RL can improve capabilities rapidly. Without
one, progress stalls regardless of model size.

## Can self-play close the gap?

Adversarial self-play could supply the missing reward signal for
defense. Put an offensive agent and a defensive agent in the same
environment and let them train together. The offensive agent
attacks; the defensive agent detects the attacks. The offensive
agent provides the ground truth that defense otherwise lacks.

This makes defense verifiable in the same way as AlphaZero. You
cannot verify "is this the best move?" but you *can* verify "did I
win the game?"

This closes the reward loop: the red agent *knows* what it did, so
blue's performance is measurable. It can generate unlimited, varied
training data without a shortage of examples or a labeling cost.
Both sides improve as they compete. Generative adversarial networks
(GANs), game AI, and centuries of military wargaming offer precedents.

Five problems remain:

**1. Red bounds blue.** Blue can only learn to defend against attacks
red knows how to generate. If red cannot generate supply-chain
attacks or firmware implants, blue never learns to detect them. The most
damaging real-world attacks (SolarWinds, Log4Shell) were novel
techniques that no existing red team was practicing. Self-play
trains blue on the attack classes red knows. It cannot prepare blue
for unknown classes.

**2. Sim-to-real gap.** Self-play happens in a simulated environment.
Real attackers can exploit each simplification in the simulation.
Examples include a missing interaction with endpoint detection and
response tools, unmodeled network latency, or omitted human behavior.
AISI acknowledged this directly: their evaluation environments "lack
security features that are often present, such as active defenders
and defensive tooling."

**3. The $\forall$ problem moves up one level.** Even with perfect
self-play, you have verified that blue catches attacks red can
generate. The question "is my defense complete?" becomes "is my red
agent complete?" Both questions require the same universal
verification.

**4. Non-stationarity.** Chess rules do not change between training
and deployment. IT environments change with new software,
configurations, users, and attacker tools. A defense trained through
self-play against yesterday's environment may fail against today's.
GANs suffer from this problem (mode collapse); security self-play
inherits it.

**5. The reward function is itself unverifiable.** Even in self-play,
you must design the reward. Does blue get credit for detecting an
attack after six months? Is there a false-positive penalty? How do
you score partial detection of a multi-stage attack? Each reward
decision assumes something about "good defense," which may not
match operational reality.

Adversarial self-play makes defense **measurably improvable**.
It does not make defense **verifiably complete**. "Measurably
improvable" means you have a metric and a training signal to track
progress. "Verifiably complete" means you have proven $\forall$,
which remains impossible.

Dijkstra's point about testing applies here too. Testing remains
useful even though it cannot prove correctness. Self-play cannot
prove safety either. A defense trained against a frontier-class red
agent is still much better than one built on static signature
libraries.

## Where the cost of software actually lands

For most of software's history, *writing* code was expensive.
Design, implementation, debugging, and testing all required skilled
human time, which limited how much software people could build.
AI has cut that cost. A solo developer with a frontier coding agent
can build in a weekend what used to take a team months. Whatever you
think of "vibe coding," it shows how quickly software generation is
becoming a commodity.

Verification shifts those costs downstream. If security hardening
scales linearly with token spend, each piece of custom software
carries a recurring security cost. You must keep spending tokens to
scan, patch, and monitor it. The cost shifts from *writing it once*
to *defending it forever*.

This affects the argument for replacing software-as-a-service
(SaaS) products with custom code. If you vibe-code a replacement for
a SaaS tool that costs \$200/month, you eliminate the subscription.
You also take ownership of the entire security surface. You must
spend tokens to harden the code, monitor it in production, and patch
it when new vulnerability classes emerge. Nobody else shares that
cost with you.

Compare that with a SaaS product or an open-source library with a
large user base. Users share the cost of tokens spent on security.
When Anthropic's model finds a bug in FFmpeg, every FFmpeg user
benefits. When it finds a bug in your custom internal tool, only you
benefit, and only you pay.

Shared infrastructure works the same way. A bridge is expensive to
secure, but everyone who crosses it shares the cost. A private road
is cheap to build, but you bear its full maintenance cost alone.

Cheap generation and expensive verification may therefore
*strengthen* the economics of shared code: open source, SaaS, and
platforms. Writing code is no longer the bottleneck; securing it is.
Security has no clear stopping point because you cannot verify that
you are done.

## Three questions that follow

The verification asymmetry follows from logic. It persists as
models improve and raises questions beyond any one industry.

**How do you create partial verifiers for $\forall$ domains?**

Engineering often converts universal claims into bounded, testable
ones. You cannot prove that a bridge will never fail. You can prove
it withstands a specific load under specific conditions. You cannot
prove software is bug-free, but you can prove it passes a test
suite. The gap between the bounded test and the universal claim
is your residual risk. You can work to narrow it.

In defense, adversarial testing uses varied, realistic attack
scenarios to replace "we are secure" with "we detect these attack
classes with this accuracy." To build such a verifier, you need
realistic simulated environments, diverse red agents, and
deterministic replay. Shared benchmarks let organizations compare
their measurements.

**Does the gap widen or narrow?** If offense has tighter RL loops
than defense, offensive capability will improve faster. Clear reward
signals, fast iteration, and frequent feedback all help attackers
learn. The asymmetry could *widen* over time.

Defenders can use offensive agents against their own systems as a
bounded verifier for defense. Whether the gap widens or narrows
depends on how quickly defenders adopt adversarial self-testing.
It also depends on how closely their test environments match
reality.

**Is accountability a stopgap or a permanent feature?** When you
cannot fully verify an output, someone must absorb the residual
risk. Kingsbury calls this the "meat shield" problem.

Human accountability follows from the logic of $\forall$ domains:
people deploy systems whose correctness they cannot prove. This
applies to a physician who reviews an AI diagnosis and an analyst
who reviews an agent's "all clear." A human makes the decision
*accountable*. That does not make the decision better. Model
improvement cannot remove this need because the limit is
verifiability.

## The question that matters

The verification asymmetry is a structural feature of the world.
You can check $\exists$ claims; you generally cannot check
$\forall$ claims. For any AI system acting in the world, ask:
*can you tell whether the model got it right?*

Where the answer is yes, expect rapid progress. Where it is no,
expect the hardest problems to remain hard. The verifiers,
benchmarks, and accountability structures around the model matter as
much as the model itself.

---

*Sources: Anthropic [Project Glasswing](https://www.anthropic.com/glasswing) (April 2026), UK AISI ["Our evaluation of Claude Mythos Preview's cyber capabilities"](https://www.aisi.gov.uk/blog/our-evaluation-of-claude-mythos-previews-cyber-capabilities) (April 2026).*

*Cover image: Edsger W. Dijkstra at his desk, 2002, by [Hamilton Richards](https://commons.wikimedia.org/wiki/File:EdsgerDijkstra.jpg), [CC BY-SA 3.0](https://creativecommons.org/licenses/by-sa/3.0).*
