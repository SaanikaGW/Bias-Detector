# Permission justifications

manifest.json requests the minimum set below. `host_permissions` is empty —
the extension can never read a page unless you explicitly invoke it there.

| Permission | Why it's needed | What it does NOT allow |
|---|---|---|
| `activeTab` | Grants one-time access to the tab you're looking at, only after you click Analyze (or the context menu). This is how page text is read without any broad host access. | No background reading; no access to other tabs; the grant ends when you navigate away. |
| `scripting` | Injects the small extraction function (`lib/extract.js`) into the active tab to read the job description text. Used only together with the `activeTab` grant. | Cannot run on pages you haven't invoked the extension on. |
| `sidePanel` | Opens the results side panel. | UI only. |
| `contextMenus` | Adds "Analyze selection with BIOS Check" to the right-click menu. The selected text is provided by Chrome's click event itself. | No page access. |
| `storage` | `sync`: remembers the API server address you configured. `session`: passes the pending selection/result between popup, background, and panel (cleared when the browser closes). | Nothing is written to any third-party service. |

Notably absent:
- No `tabs` permission (we don't need URLs/titles of other tabs).
- No `host_permissions` — not even for LinkedIn or Google Docs. Extraction
  there also runs via `activeTab` on your explicit click.
- No `webRequest`, no `cookies`, no `history`.
