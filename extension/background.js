/**
 * BIOS Career Check — MV3 service worker.
 *
 * Permission usage (full justification in PERMISSIONS.md):
 * - contextMenus: adds "Analyze selection with BIOS Check" to the right-click
 *   menu. The selected text arrives via the click event — no page access.
 * - storage: chrome.storage.session hands the pending selection to the side
 *   panel; chrome.storage.sync remembers the API address the user configured.
 * - sidePanel: opens the results panel (context-menu click is the required
 *   user gesture).
 * - activeTab + scripting are used by the popup/panel (lib/extract.js), never
 *   here — the worker itself reads no page content.
 */

chrome.runtime.onInstalled.addListener(() => {
  chrome.contextMenus.create({
    id: "bios-analyze-selection",
    title: "Analyze selection with BIOS Check",
    contexts: ["selection"],
  });
});

chrome.contextMenus.onClicked.addListener(async (info, tab) => {
  if (info.menuItemId !== "bios-analyze-selection" || !info.selectionText) return;
  // Hand the selection to the side panel, then open it (user gesture).
  await chrome.storage.session.set({
    pendingText: info.selectionText,
    pendingSource: "selection",
  });
  if (tab?.windowId !== undefined) {
    await chrome.sidePanel.open({ windowId: tab.windowId });
  }
});
