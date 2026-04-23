---
title: "Hello again"
description: "Back after a long hiatus, what's changed and what's coming."
date: 2026-04-17
tags: [meta, writing]
category: philosophy
cover: "/images/hello-again.jpg"
coverAlt: "An open notebook with a pen resting across the page in warm natural light."
---

It has been a while. The last post on this blog is from **2019**, and the one
before that from 2017. In the years since I've written plenty, in notebooks,
in drafts, in papers, but very little of it landed here. This post is a short
note to mark the restart.

## What changed

- **The site.** The previous version of this blog was on a Jekyll stack
  that had quietly rotted, old jQuery, old build tools, a theme that
  wouldn't cleanly rebuild. Pipe-cleaning turned out to be harder than
  starting over, so I did. The new site is a modern static setup with
  Markdown content, dark mode, view transitions, and KaTeX for math.
- **The scope.** I'm still drawn to the ideas behind machine learning, but
  I'll also be writing about evaluation, agents, and the practical side of
  building research tooling. Less *tutorial*, more *notebook*.

## What's coming

There's a pile of old drafts on VAEs, GANs, attention, and reinforcement
learning that I never published. Some of them still hold up, I'll polish the
better ones and post them, dated as historical pieces so the timeline stays
honest. New writing will appear alongside them, not in place of them.

If you subscribed years ago and this arrives in your RSS reader: hi. I'm
glad you're still around.

---

*P.S., math still works, in case you need it:*

$$
\nabla_\theta \, \mathcal{L}(\theta)
= \mathbb{E}_{x \sim p_\text{data}}
\bigl[\, \nabla_\theta \log p_\theta(x) \,\bigr]
$$
