// Site-wide constants. Edit these to rebrand.
export const SITE = {
  title: 'Intuitions behind the world',
  description: 'Notes on machine learning, AI, and the non-linear nature of things, by Hamidreza Saghir.',
  author: 'Hamidreza Saghir',
  authorBio: 'A notebook on machine learning, AI, and the non-linear nature of things.',
  email: 'saghir.hr@gmail.com',
  url: 'https://hsaghir.github.io',
  locale: 'en',
  avatar: '/images/avatar.jpg',
  // Social links, leave empty string to hide
  social: {
    github: 'https://github.com/hsaghir',
    scholar: 'https://scholar.google.com/citations?user=0QH0nTcAAAAJ&hl=en',
    linkedin: '',
    twitter: 'https://twitter.com/hrsaghir',
    mastodon: '',
  },
  // Newsletter. Provider = 'mailerlite' | 'none'.
  // Uses MailerLite's Universal embed (loads assets.mailerlite.com/js/universal.js).
  // Leave accountId/formId empty to ship an RSS + Kill-the-Newsletter fallback.
  newsletter: {
    provider: 'mailerlite' as 'mailerlite' | 'none',
    accountId: '2284644',
    formId: '7ohleY',
    blurb: "Occasional notes on machine learning, AI, and security. No spam, unsubscribe any time.",
  },
  // Analytics. Privacy-respecting, no cookies, no consent banner needed.
  // To activate: claim a code at https://goatcounter.com (free for personal blogs),
  // then paste the code below. Leave empty to ship with no analytics.
  analytics: {
    provider: 'goatcounter' as 'goatcounter' | 'none',
    code: '', // e.g. 'hsaghir' once claimed
  },
  // Comments (Giscus). Leave repo empty to disable.
  giscus: {
    repo: 'hsaghir/hsaghir.github.io',
    repoId: 'MDEwOlJlcG9zaXRvcnk3MTE2MDQ2NA==',
    category: 'Announcements',
    categoryId: 'DIC_kwDOBD3SkM4C7VW2',
    mapping: 'pathname',
    reactionsEnabled: '1',
    theme: 'preferred_color_scheme',
  },
  // Navigation (top of every page)
  nav: [
    { label: 'Home', href: '/' },
    { label: 'Blog', href: '/blog/' },
    { label: 'Tags', href: '/tags/' },
    { label: 'About', href: '/about/' },
    { label: 'Research', href: '/research/' },
    { label: 'Subscribe', href: '/subscribe/' },
  ],
  postsPerPage: 20,
} as const;
