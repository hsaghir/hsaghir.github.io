# Intuitions behind the world

Personal blog of Hamidreza Saghir, rebuilt on a modern stack. Inspired by the
old [Skinny Bones Jekyll][skinny] theme (Georgia typography, minimal chrome,
sliding mobile menu) and by [Minimal Mistakes][mm].

[skinny]: https://mmistakes.github.io/skinny-bones-jekyll/
[mm]: https://mmistakes.github.io/minimal-mistakes/

## Stack

- **[Astro 4](https://astro.build)** — static site, Markdown/MDX content collections, zero JS by default
- **[MDX](https://mdxjs.com)** for rich posts
- **[Shiki](https://shiki.matsu.io/)** for syntax highlighting (light + dark themes)
- **[KaTeX](https://katex.org/)** via `remark-math` + `rehype-katex` for math
- **[@astrojs/rss](https://docs.astro.build/en/recipes/rss/)** for the feed
- **[@astrojs/sitemap](https://docs.astro.build/en/guides/integrations-guide/sitemap/)** for `sitemap-index.xml`
- **[Giscus](https://giscus.app/)** for comments (optional, GitHub Discussions-backed)
- **GitHub Actions** → GitHub Pages for deployment
- **View Transitions**, **dark mode**, **responsive images**, typed frontmatter

## Getting started

```bash
npm install
npm run dev        # http://localhost:4321
npm run build      # outputs to dist/
npm run preview    # preview the built site
```

Requires **Node 20+**.

## Project structure

```
.
├── astro.config.mjs         # Astro config (MDX, sitemap, KaTeX, Shiki)
├── public/                  # static assets (served at /)
├── scripts/
│   └── migrate-jekyll.mjs   # one-shot migration from old _posts/
├── src/
│   ├── consts.ts            # edit site title, nav, social, Giscus here
│   ├── content/
│   │   ├── config.ts        # typed frontmatter schema
│   │   └── posts/           # your posts live here
│   ├── pages/
│   │   ├── index.astro      # home (hero + featured + recent)
│   │   ├── blog/            # archive + per-post pages
│   │   ├── tags/            # tag index + per-tag pages
│   │   ├── about.md         # about page (uses PageLayout)
│   │   ├── research.md      # research page
│   │   ├── rss.xml.js       # RSS feed
│   │   └── 404.astro
│   ├── layouts/             # BaseLayout, PostLayout, PageLayout
│   ├── components/          # Header, Footer, PostCard, TOC, Giscus, ThemeToggle
│   ├── styles/              # global.css + variables.css (design tokens)
│   └── utils/               # helpers (date formatting)
└── .github/workflows/deploy.yml
```

## Writing a post

Create a file in `src/content/posts/`:

````markdown
---
title: "Your title"
description: "One-line summary for cards + OG tags."
date: 2026-04-20
tags: [machine-learning, notes]
category: data_science
featured: false
# cover: ../../images/posts/some-image.jpg   # optional
---

Your content here. Markdown works. MDX works.

```python
import torch
```

Math works too:

$$
\mathcal{L}(\theta) = -\mathbb{E}_q[\log p_\theta(x)]
$$
````

Set `draft: true` to hide it from listings, feed, and sitemap.

## Migrating from Jekyll

A helper converts your old `_posts/` tree:

```bash
node scripts/migrate-jekyll.mjs ../hsaghir.github.io/_posts
```

It preserves category folders, normalizes frontmatter, and flags posts that
reference Google Sites images or Liquid tags (which need manual fixes).

## Configuration

Everything personal lives in [`src/consts.ts`](src/consts.ts):

- Site title, description, author, avatar
- Navigation links
- Social links (shown in footer; empty strings are hidden)
- Giscus comment settings (leave `repo: ''` to disable)

Design tokens live in [`src/styles/variables.css`](src/styles/variables.css) —
typography, palette, spacing. Both light and dark themes read from the same
variables.

## Deployment

Pushes to `main` (or `master`) trigger the workflow in
`.github/workflows/deploy.yml`, which builds with Astro and deploys to GitHub
Pages. Enable Pages in the repo settings with **Source: GitHub Actions**.

For a custom domain, add a `CNAME` file to `public/`.

## License

MIT for the code. Content (posts, images) © Hamidreza Saghir — all rights reserved.
