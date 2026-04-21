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
  // Newsletter. Provider = 'buttondown' | 'none'.
  // Sign up at https://buttondown.email, then set username below.
  // Leave username empty to show an RSS-only subscribe card (no email capture).
  newsletter: {
    provider: 'buttondown' as 'buttondown' | 'none',
    username: '', // e.g. 'hsaghir' once you've claimed it at buttondown.email
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
