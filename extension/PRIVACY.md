# Privacy policy (stub — review before store submission)

BIOS Career Check helps you find and fix gender-coded language in job
descriptions. Because it can read page content on sites like LinkedIn, Gmail,
Google Docs, and Word Online **when you invoke it there**, here is exactly
what happens to that content:

## What is read, and when
- Page text is read **only** when you click "Analyze Current Page",
  "Analyze Selected Text", or the right-click menu item — never automatically,
  never in the background, never on pages where you don't invoke the extension.
- On LinkedIn, only the job-posting container is targeted. On Google Docs, the
  document's plain-text export is fetched from Google using your own session —
  it does not pass through any third party.
- Extracted text is capped at 6,000 characters.

## Where it is sent
- The text is sent to **one place**: the BIOS Career Check analysis server you
  configure in the panel settings (by default your own local/self-hosted
  instance). It is sent over HTTPS when the server address uses https://.
- The server runs the detection pipeline and — if configured with an OpenAI
  API key — sends the text to OpenAI's API to generate explanations and
  rewrites. If no key is configured, no third party ever receives the text.

## What is stored
- On your device: the API server address (`chrome.storage.sync`) and the most
  recent analysis result (`chrome.storage.session`, cleared when the browser
  closes).
- The analysis server does not persist submitted text (no database writes in
  the analyze route).

## What is never collected
- No browsing history, no cookies, no credentials, no analytics, no tracking
  of any kind. The extension makes no network request other than the analysis
  call you trigger.
