// Site-wide constants. Edit these to rebrand.

// Sister site: the focused senior-ML interview-prep companion.
export const MLMENTORSHIP_URL = 'https://mlmentorship.com';

export const SITE = {
  title: 'Hamidreza Saghir',
  description: 'Notes on applied LLMs, agents, and machine learning, by Hamidreza Saghir. Principal Applied Scientist at Microsoft. Author of Looplet.',
  author: 'Hamidreza Saghir',
  authorBio: 'Notes on applied LLMs, agents, and machine learning.',
  email: 'saghir.hr@gmail.com',
  url: 'https://hsaghir.com',
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
  // Shared with mlmentorship.com; BaseHead prefixes paths with the hostname.
  analytics: {
    provider: 'goatcounter' as 'goatcounter' | 'none',
    code: 'hsaghir',
  },
  // Comments (Giscus). Leave repo empty to disable.
  giscus: {
    repo: 'hsaghir/hsaghir.github.io',
    repoId: 'R_kgDOSKPBpg',
    category: 'Announcements',
    categoryId: 'DIC_kwDOSKPBps4C7eyZ',
    mapping: 'pathname',
    reactionsEnabled: '1',
    theme: 'preferred_color_scheme',
  },
  // Navigation (top of every page)
  nav: [
    { label: 'Home', href: '/' },
    { label: 'Blog', href: '/blog/' },
    { label: 'Research', href: '/research/' },
    { label: 'ML Interviews', href: MLMENTORSHIP_URL, external: true },
    { label: 'Work with me', href: '/work-together/' },
    { label: 'About', href: '/about/' },
  ],
  postsPerPage: 20,
} as const;
