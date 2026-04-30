import { defineCollection, z } from 'astro:content';

const posts = defineCollection({
  type: 'content',
  schema: z.object({
    title: z.string(),
    description: z.string().optional(),
    date: z.coerce.date(),
    updated: z.coerce.date().optional(),
    draft: z.boolean().default(false),
    tags: z.array(z.string()).default([]),
    category: z.string().optional(),
    // String path served from /public (e.g. "/images/foo.jpg").
    // For processed/optimized images, use a regular <Image> import inside the post body.
    cover: z.string().optional(),
    coverAlt: z.string().optional(),
    featured: z.boolean().default(false),
    // Hide from the home page (still reachable under /blog/ and tag/category
    // archives). Use for posts that are published but no longer representative
    //, e.g. deprecated tooling write-ups.
    archived: z.boolean().default(false),
  }),
});

export const collections = { posts };
