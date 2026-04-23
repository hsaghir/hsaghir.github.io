// Lightweight date formatter, avoids pulling in a whole library.
// Format in UTC to match the date authored (avoids timezone shifts that make
// 2016-12-16 display as Dec 15 for users west of UTC).
export function formatDate(date: Date, locale = 'en-US'): string {
  return date.toLocaleDateString(locale, {
    year: 'numeric',
    month: 'long',
    day: 'numeric',
    timeZone: 'UTC',
  });
}

export function formatDateShort(date: Date, locale = 'en-US'): string {
  return date.toLocaleDateString(locale, {
    year: 'numeric',
    month: 'short',
    day: 'numeric',
    timeZone: 'UTC',
  });
}

export function yearOf(date: Date): number {
  return date.getUTCFullYear();
}
