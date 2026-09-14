/**
 * fhiHistory.js — local run-history store for the Fair Hiring Index.
 *
 * There's no backend datastore for BIOS Career Check yet (Flask is
 * stateless), so v2.0 tracks metrics client-side in localStorage: every
 * time a user runs a batch analysis, a lightweight snapshot is saved so
 * they can see fairness trend over time on this device/browser. This is
 * intentionally simple and dependency-free -- a natural upgrade path is
 * swapping this module for a real API-backed store (e.g. /api/fhi/history)
 * without changing any of the calling UI code.
 */

const STORAGE_KEY = "bios_fhi_history_v1";
const MAX_RUNS = 100;

function readRaw() {
  try {
    const raw = localStorage.getItem(STORAGE_KEY);
    if (!raw) return [];
    const parsed = JSON.parse(raw);
    return Array.isArray(parsed) ? parsed : [];
  } catch {
    return [];
  }
}

function writeRaw(runs) {
  try {
    localStorage.setItem(STORAGE_KEY, JSON.stringify(runs.slice(-MAX_RUNS)));
  } catch {
    // localStorage unavailable (private browsing, quota) -- fail silently,
    // history just won't persist this session.
  }
}

/**
 * Save a completed FHI batch run.
 * record: { fhi, jdCount, avgBiasScore, avgInclusiveScore, categoryCounts, sources }
 */
export function saveRun(record) {
  const runs = readRaw();
  runs.push({
    id: `${Date.now()}-${Math.random().toString(36).slice(2, 8)}`,
    timestamp: new Date().toISOString(),
    ...record,
  });
  writeRaw(runs);
  return runs;
}

/** All saved runs, oldest first. */
export function getHistory() {
  return readRaw();
}

/** Delete a single run by id. */
export function deleteRun(id) {
  const runs = readRaw().filter((r) => r.id !== id);
  writeRaw(runs);
  return runs;
}

/** Wipe all history. */
export function clearHistory() {
  try {
    localStorage.removeItem(STORAGE_KEY);
  } catch {
    /* ignore */
  }
}

/** Rollup stats used by the Metrics dashboard. */
export function summarize(runs) {
  if (!runs.length) return null;
  const totalJds = runs.reduce((s, r) => s + (r.jdCount || 0), 0);
  const avgFhi = Math.round(runs.reduce((s, r) => s + r.fhi, 0) / runs.length);
  const best = runs.reduce((a, b) => (b.fhi > a.fhi ? b : a));
  const worst = runs.reduce((a, b) => (b.fhi < a.fhi ? b : a));
  const latest = runs[runs.length - 1];
  const prev = runs.length > 1 ? runs[runs.length - 2] : null;
  const trend = prev ? latest.fhi - prev.fhi : 0;

  const categoryTotals = {};
  runs.forEach((r) => {
    Object.entries(r.categoryCounts || {}).forEach(([cat, count]) => {
      categoryTotals[cat] = (categoryTotals[cat] || 0) + count;
    });
  });

  return { totalRuns: runs.length, totalJds, avgFhi, best, worst, latest, trend, categoryTotals };
}
