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

| `host_permissions` for the BIOS API (`bios-carear-checkers-production.up.railway.app`, `localhost:5001`) | Lets the extension call its own analysis backend. The server's CORS policy is locked to the web app's domain, and extension origins vary per install — a host permission for the API domain is the standard MV3 way to allow this without opening server CORS to everyone. | Applies ONLY to the API server. No permission on LinkedIn, Google, or any other site. Self-hosters on a different domain: add your domain here, or allow your extension origin in the server's `CORS_ORIGIN`. |

Notably absent:
- No `tabs` permission (we don't need URLs/titles of other tabs).
- No host permissions on any site you browse — not even LinkedIn or Google
  Docs. Extraction there runs via `activeTab` on your explicit click.
- No `webRequest`, no `cookies`, no `history`.
