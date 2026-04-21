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
    scholar: 'https://scholar.google.com/citations?user=N0lNjVcAAAAJ',
    linkedin: '',
    twitter: '',
    mastodon: '',
  },
  // Comments (Giscus). Leave repo empty to disable.
  giscus: {
    repo: '',
    repoId: '',
    category: 'General',
    categoryId: '',
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
  ],
  postsPerPage: 20,
} as const;
