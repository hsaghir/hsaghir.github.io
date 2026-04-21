// Site-wide constants. Edit these to rebrand.
export const SITE = {
  title: 'Intuitions behind the world',
  description: 'Notes on machine learning, AI, and the non-linear nature of things — by Hamidreza Saghir.',
  author: 'Hamidreza Saghir',
  authorBio: 'A notebook on machine learning, AI, and the non-linear nature of things.',
  email: 'saghir.hr@gmail.com',
  url: 'https://hsaghir.github.io',
  locale: 'en',
  avatar: '/images/avatar.jpg',
  // Social links — leave empty string to hide
  social: {
    github: 'https://github.com/hsaghir',
    scholar: 'https://scholar.google.com/citations?user=0QH0nTcAAAAJ&hl=en',
    linkedin: '',
    twitter: 'https://twitter.com/hrsaghir',
    mastodon: '',
  },
  // Newsletter. Provider = 'mailerlite' | 'none'.
  // Sign up at https://mailerlite.com (free up to 1,000 subscribers, 12k sends/mo).
  // Create an embedded form, then paste the form's accountId + formId below.
  // Leave them empty to ship an RSS + Kill-the-Newsletter fallback (no email capture).
  newsletter: {
    provider: 'mailerlite' as 'mailerlite' | 'none',
    accountId: '', // e.g. '1234567' from the MailerLite embed snippet
    formId: '',    // e.g. '9876543' from the MailerLite embed snippet
    blurb: "Occasional notes on machine learning, AI, and security. No spam, unsubscribe any time.",
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
