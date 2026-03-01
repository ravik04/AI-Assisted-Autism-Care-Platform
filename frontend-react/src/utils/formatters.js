// Format a score as percentage
export function formatScore(score) {
  if (score == null || isNaN(score)) return '—';
  return `${(score * 100).toFixed(1)}%`;
}

// Format a score as percentage integer
export function formatScoreInt(score) {
  if (score == null || isNaN(score)) return '—';
  return `${Math.round(score * 100)}%`;
}

// Format date string
export function formatDate(dateStr) {
  if (!dateStr) return '—';
  return new Date(dateStr).toLocaleDateString('en-US', {
    year: 'numeric', month: 'short', day: 'numeric',
  });
}

// Format date with time
export function formatDateTime(dateStr) {
  if (!dateStr) return '—';
  return new Date(dateStr).toLocaleString('en-US', {
    year: 'numeric', month: 'short', day: 'numeric',
    hour: '2-digit', minute: '2-digit',
  });
}

// Truncate text
export function truncate(text, maxLen = 100) {
  if (!text || text.length <= maxLen) return text;
  return text.slice(0, maxLen) + '…';
}

// Get initials from name
export function getInitials(name) {
  if (!name) return '?';
  return name.split(' ').map(w => w[0]).join('').toUpperCase().slice(0, 2);
}

// Age in months to readable string
export function formatAge(months) {
  if (!months) return '—';
  const years = Math.floor(months / 12);
  const m = months % 12;
  if (years === 0) return `${m}mo`;
  if (m === 0) return `${years}y`;
  return `${years}y ${m}mo`;
}

// Capitalize first letter
export function capitalize(str) {
  if (!str) return '';
  return str.charAt(0).toUpperCase() + str.slice(1);
}

// Parent-friendly risk description
export function parentRiskDescription(score) {
  if (score < 0.3) return "Your child's screening shows a low likelihood of developmental differences. Keep up the great work supporting their growth!";
  if (score < 0.5) return "The screening suggests some areas that might benefit from a closer look. This isn't a diagnosis — it's an opportunity to learn more.";
  if (score < 0.7) return "Some signals suggest your child could benefit from a professional evaluation. Remember, early support leads to wonderful outcomes.";
  return "The screening indicates areas where professional guidance would be very helpful. Many children thrive with the right support — you're taking a great first step.";
}
