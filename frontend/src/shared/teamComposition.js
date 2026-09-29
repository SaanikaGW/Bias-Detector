/**
 * teamComposition.js — turns an optional "team composition" CSV into a real,
 * math-backed Representation Balance score (0-100), used by the Fair Hiring
 * Index to blend an org-data signal in with the language-bias signal.
 *
 * Deliberately does NOT try to infer diversity/fairness from free-text
 * documents (a PDF of "diversity stats" can't be reliably turned into a
 * number without guessing). A CSV with one categorical column is something
 * we can compute an honest, transparent statistic from -- so that's the
 * bar for "uploaded org data" actually changing the score.
 */

const CATEGORY_COLUMN_HINTS = [
  "gender", "sex", "group", "category", "demographic", "department",
  "team", "identity", "role type", "race", "ethnicity",
];

/** Minimal CSV parser: handles quoted fields with embedded commas. */
export function parseCsv(text) {
  const lines = text.split(/\r\n|\n|\r/).filter((l) => l.trim().length);
  if (!lines.length) return { headers: [], rows: [] };

  function parseLine(line) {
    const cells = [];
    let cur = "", inQuotes = false;
    for (let i = 0; i < line.length; i++) {
      const c = line[i];
      if (inQuotes) {
        if (c === '"') {
          if (line[i + 1] === '"') { cur += '"'; i++; } else inQuotes = false;
        } else cur += c;
      } else if (c === '"') inQuotes = true;
      else if (c === ",") { cells.push(cur); cur = ""; }
      else cur += c;
    }
    cells.push(cur);
    return cells.map((c) => c.trim());
  }

  const headers = parseLine(lines[0]);
  const rows = lines.slice(1).map(parseLine);
  return { headers, rows };
}

/**
 * Compute a Representation Balance score from a team-composition CSV.
 * Picks a categorical column (by name hint, else the first column),
 * counts rows per category, and scores how evenly distributed they are:
 * 100 = perfectly even split across categories, 0 = everyone in one bucket.
 *
 * Math: for k categories summing to N, ideal count per category is N/k.
 * balance = 100 * (1 - sum(|count_i - N/k|) / maxPossibleDeviation)
 * where maxPossibleDeviation = 2N(k-1)/k (everyone in a single category).
 *
 * Throws a user-facing Error if the CSV doesn't have usable data.
 */
export function analyzeTeamComposition(csvText) {
  const { headers, rows } = parseCsv(csvText);
  if (!headers.length || !rows.length) {
    throw new Error("This CSV has no data rows.");
  }

  let colIndex = headers.findIndex((h) => CATEGORY_COLUMN_HINTS.includes(h.trim().toLowerCase()));
  const guessed = colIndex < 0;
  if (colIndex < 0) colIndex = 0;
  const columnUsed = headers[colIndex];

  const counts = {};
  let total = 0;
  rows.forEach((r) => {
    const val = (r[colIndex] || "").trim();
    if (!val) return;
    counts[val] = (counts[val] || 0) + 1;
    total += 1;
  });

  const categories = Object.keys(counts);
  if (total === 0 || categories.length === 0) {
    throw new Error(`No usable values found in the "${columnUsed}" column.`);
  }

  const k = categories.length;
  const ideal = total / k;
  const sumDev = categories.reduce((s, cat) => s + Math.abs(counts[cat] - ideal), 0);
  const maxDev = k > 1 ? (2 * total * (k - 1)) / k : total;
  const rawScore = k > 1 ? 100 * (1 - sumDev / maxDev) : 0;
  const balanceScore = Math.max(0, Math.min(100, Math.round(rawScore)));

  return {
    columnUsed,
    columnGuessed: guessed,
    counts,
    total,
    categoryCount: k,
    balanceScore,
  };
}
