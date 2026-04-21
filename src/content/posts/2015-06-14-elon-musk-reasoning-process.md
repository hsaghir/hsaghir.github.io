---
title: "First-principles reasoning (a note on separating ideas from the people who held them)"
description: "An old note on Elon Musk's first-principles reasoning, updated for 2026. The politics and the personality have not aged well; the reasoning technique still has."
date: "2015-06-14"
draft: false
tags: ["philosophy"]
category: "philosophy"
cover: "/images/Elon_Musk_-_The_Summit_2013.jpg"
---

> **Note from 2026.** I originally wrote this post in 2015 and called Elon
> Musk my "real-life superhero". That framing has not aged well. I do not
> agree with much of his politics or the way he treats people, and I've
> kept a steady distance from the cult of personality that formed around
> him. But I still think the *reasoning technique* described here is worth
> preserving, and worth separating from the man. Ideas travel; you don't
> have to endorse everyone who has ever used them. This post keeps the
> technique and drops the hero-worship.

When Musk was [asked](https://www.ted.com/talks/elon_musk_the_mind_behind_tesla_spacex_solarcity?language=en) how he'd managed to take on so many hard problems in so many hard industries — PayPal, then rockets at [SpaceX](https://www.spacex.com), then electric cars at [Tesla](https://www.tesla.com), then residential energy at SolarCity, then satellite internet — he answered with something that wasn't about him at all. He described a habit, not a talent. The habit is **reasoning from first principles** instead of reasoning by analogy, and it is stealable.

## The habit

Most of the time, when we "solve" a problem, we don't actually solve it. We reach for the closest neighbour in problem-space and copy its solution, with small edits.

> You already have rockets. So you design a slightly better control system.
> You already have cars. So you add a bigger battery.
> You already have a database. So you add an index.

This is reasoning *by analogy*. It is fast, usually correct, and the main thing every organisation on Earth rewards. It is also how you end up with incremental improvements to an obviously broken baseline — and never ask whether the baseline itself is necessary.

First-principles reasoning is the opposite posture. You strip the problem down to things you are forced to believe — physics, math, identities — and build back up. The test you apply to every assumption on the way down is brutal: *is this true because the universe makes it true, or is it true because someone built it this way and I inherited their choice?*

## The SpaceX example

The canonical illustration — and the one Musk himself used — is rocket cost. Rockets are expensive. Everyone knows this. Reasoning by analogy, you accept the price tag and argue about discount curves. Reasoning from first principles, you ask a different question:

> *Why* are rockets expensive? What do they physically have to contain?
> Aluminium, titanium, carbon fiber, copper, some kerosene and liquid
> oxygen. What do those raw materials cost on the commodities market?
> And what is the markup on the finished rocket?

The gap between "raw-material cost" and "market price of a rocket" was apparently around two orders of magnitude. Which means the problem wasn't *physics*. It was *supply chain, institutional habit, and cost-plus contracting*. That is a completely different problem, with a completely different set of solutions. SpaceX drove launch cost down roughly ten-fold by attacking the second problem instead of the first.

The important claim here isn't "Musk is a genius". The important claim is: **if you keep reasoning by analogy, you will never find an order-of-magnitude improvement, because every analogy you reach for already assumes the old answer.**

## Where this applies in normal work

First-principles reasoning sounds like it only matters for rocket companies. It doesn't. Every field has inherited answers that no one can currently defend from scratch:

- *Why* does this training job take 40 GPU-hours? Because the data loader is I/O-bound, or because we are, out of habit, re-tokenising a dataset that hasn't changed since the last run?
- *Why* does this meeting recur? Because it solves a current problem, or because it was created three reorgs ago for a problem nobody remembers?
- *Why* is this codebase this size? Because the domain is that complex, or because every refactor was cheaper to skip than to do?

The technique is uncomfortable because it keeps forcing you to admit *I don't know why we are doing this, actually*. And that is the exact feeling it's trying to produce.

## How to separate the idea from the person

The reason I'm keeping this post up — rewritten, but up — is that I think one of the quietly important skills of an educated life is learning to absorb a technique from someone whose choices you would never want to imitate. History is full of important ideas held by unpleasant people, and full of unpleasant ideas held by charming ones. The job is to keep the ideas and throw out the politics, the personality cults, and the follow-me-off-a-cliff energy that often comes stapled to them.

So: keep first-principles reasoning. Keep "what problem are we actually solving, measured in atoms and dollars, before any inherited answer". Drop the rest.

---

*This is the original 2015 post, substantially rewritten in 2026 to keep
what still holds and remove what doesn't. The Khan Academy
[conversation](https://www.khanacademy.org/talks-and-interviews/khan-academy-living-room-chats/v/elon-musk)
is still a decent introduction to the technique if you want to hear it
from the source.*

Photo credit: The Summit 2013 — Dan Taylor / Heisenberg Media.
