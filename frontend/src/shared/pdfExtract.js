/**
 * pdfExtract.js — client-side PDF -> plain text extraction, used by the
 * Fair Hiring Index page to accept batch PDF uploads of job descriptions.
 *
 * Runs entirely in the browser (pdf.js / pdfjs-dist); no file ever touches
 * the backend just to get text out of it. Only the extracted text is sent
 * to the analysis API, same as pasted text.
 */
import * as pdfjsLib from "pdfjs-dist";
// Vite-specific asset import: bundles the worker file and gives us a URL.
import pdfjsWorkerUrl from "pdfjs-dist/build/pdf.worker.min.mjs?url";

pdfjsLib.GlobalWorkerOptions.workerSrc = pdfjsWorkerUrl;

const MAX_PDF_BYTES = 15 * 1024 * 1024; // 15MB per file, generous for a JD

/**
 * Extract all text from a single PDF File/Blob.
 * Returns { text, pageCount } or throws an Error with a user-facing message.
 */
export async function extractPdfText(file) {
  if (file.size > MAX_PDF_BYTES) {
    throw new Error(`"${file.name}" is too large (max 15MB per PDF).`);
  }
  let buf;
  try {
    buf = await file.arrayBuffer();
  } catch {
    throw new Error(`Couldn't read "${file.name}".`);
  }

  let pdf;
  try {
    pdf = await pdfjsLib.getDocument({ data: buf }).promise;
  } catch {
    throw new Error(`"${file.name}" doesn't look like a valid PDF.`);
  }

  const pageTexts = [];
  for (let i = 1; i <= pdf.numPages; i++) {
    const page = await pdf.getPage(i);
    const content = await page.getTextContent();
    const pageText = content.items.map((it) => it.str).join(" ");
    pageTexts.push(pageText);
  }
  const text = pageTexts.join("\n\n").replace(/[ \t]+/g, " ").trim();

  if (!text) {
    throw new Error(
      `"${file.name}" has no extractable text (it may be a scanned image -- try pasting the text instead).`
    );
  }
  return { text, pageCount: pdf.numPages };
}

/**
 * Extract text from multiple PDF files in parallel.
 * Never throws -- returns { ok: [{file, text, pageCount}], failed: [{file, error}] }
 * so the caller can add successes and surface per-file errors.
 */
export async function extractPdfTextBatch(files) {
  const results = await Promise.allSettled(
    Array.from(files).map((file) => extractPdfText(file))
  );
  const ok = [];
  const failed = [];
  results.forEach((r, i) => {
    const file = files[i];
    if (r.status === "fulfilled") {
      ok.push({ file, text: r.value.text, pageCount: r.value.pageCount });
    } else {
      failed.push({ file, error: r.reason?.message || "Failed to extract text." });
    }
  });
  return { ok, failed };
}
