chrome.runtime.onInstalled.addListener(() => {
  chrome.contextMenus.removeAll(() => {
    chrome.contextMenus.create({
      id: 'deepfake-analyze-image',
      title: 'Analyze image for deepfake',
      contexts: ['image'],
    });
  });
});

chrome.contextMenus.onClicked.addListener((info) => {
  if (info.menuItemId !== 'deepfake-analyze-image' || !info.srcUrl) {
    return;
  }
  const pageUrl = chrome.runtime.getURL(
    `analyze.html?src=${encodeURIComponent(info.srcUrl)}`,
  );
  chrome.tabs.create({ url: pageUrl });
});
