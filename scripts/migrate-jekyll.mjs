#!/usr/bin/env node
/**
 * migrate-jekyll.mjs
 *
 * One-shot migration from the old Jekyll `_posts/` tree to the new Astro
 * content collection at `src/content/posts/`.
 *
 * Usage:
 *   node scripts/migrate-jekyll.mjs ../hsaghir.github.io/_posts
 *
 * What it does:
 *   - Recursively finds Jekyll posts (YYYY-MM-DD-*.md / *.markdown).
 *   - Skips files prefixed with '-' (your "unpublished" marker).
 *   - Extracts front matter, normalizes field names, and writes to
 *     src/content/posts/<same-filename>.md.
 *   - Infers `category` from the parent directory name.
 *   - Preserves the rest of the body verbatim.
 *   - Flags obvious issues (Google Sites image URLs, Liquid tags) as TODOs.
 *
 * Review every migrated file. The front-matter schema (src/content/config.ts)
 * will surface any problems at build time.
 */
import { readdir, readFile, writeFile, mkdir, stat } from 'node:fs/promises';
import { dirname, join, basename, relative } from 'node:path';
import { fileURLToPath } from 'node:url';

const __dirname = dirname(fileURLToPath(import.meta.url));
const ROOT = join(__dirname, '..');
const SRC = process.argv[2];
const OUT = join(ROOT, 'src', 'content', 'posts');

if (!SRC) {
  console.error('Usage: node scripts/migrate-jekyll.mjs <path-to-_posts>');
  process.exit(1);
}

const FM = /^---\n([\s\S]*?)\n---\n?/;
// Accept both published (YYYY-MM-DD-...) and unpublished (-YYYY-MM-DD-...) Jekyll posts.
const POST_NAME = /^(-?)(\d{4})-(\d{2})-(\d{2})-(.+)\.(md|markdown)$/;

async function* walk(dir) {
  for (const entry of await readdir(dir, { withFileTypes: true })) {
    // Skip Jupyter autosave folders and other irrelevant dotdirs
    if (entry.name === '.ipynb_checkpoints' || entry.name === '.git') continue;
    const p = join(dir, entry.name);
    if (entry.isDirectory()) yield* walk(p);
    else yield p;
  }
}

function parseFrontMatter(src) {
  const m = src.match(FM);
  if (!m) return { data: {}, body: src };
  const data = {};
  const lines = m[1].split('\n');
  let currentKey = null;
  for (const line of lines) {
    // Nested child: "  teaser: foo.jpg" under "image:"
    const nested = line.match(/^\s{2,}([\w-]+)\s*:\s*(.*)$/);
    if (nested && currentKey) {
      let [, k, v] = nested;
      v = v.trim().replace(/^["']|["']$/g, '');
      if (typeof data[currentKey] !== 'object' || data[currentKey] === null) data[currentKey] = {};
      data[currentKey][k] = v;
      continue;
    }
    const kv = line.match(/^([\w-]+)\s*:\s*(.*)$/);
    if (!kv) continue;
    let [, k, v] = kv;
    v = v.trim().replace(/^["']|["']$/g, '');
    currentKey = k;
    if (v === '') data[k] = null; // parent of a nested block
    else if (v === 'true') data[k] = true;
    else if (v === 'false') data[k] = false;
    else data[k] = v;
  }
  return { data, body: src.slice(m[0].length) };
}

// Turn the first paragraph-ish block of markdown into a plain-text blurb.
function extractDescription(body, maxLen = 180) {
  // Strip frontmatter-like remnants, HTML comments, Liquid tags
  const clean = body
    .replace(/<!--[\s\S]*?-->/g, '')
    .replace(/{%[\s\S]*?%}/g, '')
    .replace(/{{[\s\S]*?}}/g, '');
  // Find first non-empty paragraph that isn't an image or heading
  const blocks = clean.split(/\n\s*\n/);
  for (const block of blocks) {
    const trimmed = block.trim();
    if (!trimmed) continue;
    if (/^#{1,6}\s/.test(trimmed)) continue; // heading
    if (/^!\[/.test(trimmed)) continue; // image
    if (/^[-*]\s/.test(trimmed)) continue; // list
    if (/^```/.test(trimmed)) continue; // code fence
    if (/^\|/.test(trimmed)) continue; // table
    // Strip inline markdown: links, emphasis, code, images
    let text = trimmed
      .replace(/!?\[([^\]]*)\]\([^)]*\)/g, '$1')
      .replace(/[`*_~]/g, '')
      .replace(/\s+/g, ' ')
      .trim();
    if (text.length < 20) continue;
    if (text.length <= maxLen) return text;
    // Truncate on word boundary + ellipsis
    const cut = text.slice(0, maxLen);
    const lastSpace = cut.lastIndexOf(' ');
    return (lastSpace > maxLen * 0.6 ? cut.slice(0, lastSpace) : cut).trim() + '…';
  }
  return undefined;
}

function stringifyFrontMatter(obj) {
  const keyOrder = ['title','description','date','updated','draft','tags','category','cover','coverAlt','featured'];
  const lines = ['---'];
  for (const k of keyOrder) {
    if (obj[k] === undefined || obj[k] === null || obj[k] === '') continue;
    if (Array.isArray(obj[k])) lines.push(`${k}: [${obj[k].map((x) => JSON.stringify(x)).join(', ')}]`);
    else if (typeof obj[k] === 'boolean') lines.push(`${k}: ${obj[k]}`);
    else lines.push(`${k}: ${JSON.stringify(obj[k])}`);
  }
  lines.push('---', '');
  return lines.join('\n');
}

let migrated = 0, skipped = 0, flagged = 0;
await mkdir(OUT, { recursive: true });

for await (const file of walk(SRC)) {
  const name = basename(file);
  const isMd = /\.(md|markdown)$/i.test(name);
  if (!isMd) { skipped++; continue; }

  const dated = name.match(POST_NAME);
  const category = basename(dirname(file));

  let y, m, d, slug, isDraft;
  if (dated) {
    const [, dash, yy, mm, dd, sl] = dated;
    y = yy; m = mm; d = dd; slug = sl;
    isDraft = dash === '-';
  } else {
    // Undated markdown note (e.g. Fraud detection/*.md). Use mtime, mark draft.
    const st = await stat(file);
    const dt = st.mtime;
    y = String(dt.getFullYear());
    m = String(dt.getMonth() + 1).padStart(2, '0');
    d = String(dt.getDate()).padStart(2, '0');
    slug = name
      .replace(/\.(md|markdown)$/i, '')
      .replace(/^_+/, '')
      .replace(/\s+/g, '-')
      .replace(/[^a-zA-Z0-9-_]/g, '')
      .toLowerCase() || 'untitled';
    isDraft = true;
  }

  const raw = await readFile(file, 'utf8');
  const { data, body } = parseFrontMatter(raw);

  // Map old Jekyll fields to new schema
  const teaser = data.image && typeof data.image === 'object' ? data.image.teaser : undefined;
  // Normalize Windows backslashes (old frontmatter had "practical\foo.png")
  const cover = teaser ? `/images/${String(teaser).replace(/\\/g, '/')}` : undefined;

  const description = data.description || data.excerpt || extractDescription(body);

  // Combine explicit tags with the inferred category so the new tags page has content.
  const rawTags = data.tags ? String(data.tags).split(/[\s,]+/).filter(Boolean) : [];
  const catTag = category && category !== '_posts'
    ? category.replace(/_/g, '-').toLowerCase()
    : undefined;
  const tags = [...new Set([...rawTags, ...(catTag ? [catTag] : [])])];

  const out = {
    title: (data.title || slug.replace(/[-_]/g, ' ')).trim(),
    description,
    date: `${y}-${m}-${d}`,
    tags,
    category: category && category !== '_posts' ? category : undefined,
    cover,
    draft: isDraft,
  };

  const warnings = [];
  if (body.includes('sites.google.com')) warnings.push('Google Sites image URLs — replace with local images');
  if (/{%|{{/.test(body)) warnings.push('Contains Liquid tags — rewrite to MDX/plain markdown');
  if (warnings.length) flagged++;

  const header = warnings.length
    ? `\n{/* TODO migration:\n - ${warnings.join('\n - ')}\n*/}\n\n`
    : '\n';

  const target = join(OUT, `${y}-${m}-${d}-${slug}.md`);
  await writeFile(target, stringifyFrontMatter(out) + header + body);
  migrated++;
  const marks = [
    isDraft ? 'draft' : 'published',
    ...(warnings.length ? [`⚠ ${warnings.length} flag(s)`] : []),
  ].join(', ');
  console.log(`✓ ${relative(ROOT, target)}  [${marks}]`);
}

console.log(`\nMigrated: ${migrated}  Skipped: ${skipped}  Flagged: ${flagged}`);
console.log('Review each migrated file; the schema in src/content/config.ts will catch missing fields at build time.');
