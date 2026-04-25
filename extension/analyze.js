const API_URL = 'http://127.0.0.1:5002';

function showResult(status, label, detail) {
  const result = document.getElementById('analyze-result');
  result.className = `result show ${status}`;
  result.innerHTML = `
    <div class="result-label">${label}</div>
    <div class="result-detail">${detail}</div>
  `;
}

async function checkServer() {
  const statusDot = document.querySelector('.status-dot');
  const statusText = document.querySelector('.status-text');
  try {
    const response = await fetch(`${API_URL}/`, { method: 'HEAD' });
    if (response.ok) {
      statusDot.className = 'status-dot online';
      statusText.textContent = 'Server connected';
    } else {
      throw new Error('bad');
    }
  } catch {
    statusDot.className = 'status-dot offline';
    statusText.textContent = 'Server offline';
  }
}

async function run() {
  const params = new URLSearchParams(window.location.search);
  const src = params.get('src');
  const hint = document.getElementById('source-hint');
  const loading = document.getElementById('analyze-loading');
  const loadingLabel = document.getElementById('loading-label');
  const preview = document.getElementById('analyze-preview');
  if (!src) {
    loading.classList.remove('show');
    showResult('error', 'Missing image', 'No image URL was provided.');
    return;
  }

  hint.textContent = src.length > 120 ? `${src.slice(0, 120)}…` : src;

  if (src.startsWith('file:')) {
    loading.classList.remove('show');
    showResult(
      'error',
      'Cannot load',
      'Local file URLs cannot be fetched from the extension. Save the image and use the extension popup to upload it.',
    );
    return;
  }

  let blob;
  try {
    loadingLabel.textContent = 'Downloading image…';
    const response = await fetch(src);
    if (!response.ok) {
      throw new Error(`HTTP ${response.status}`);
    }
    blob = await response.blob();
  } catch (e) {
    loading.classList.remove('show');
    showResult(
      'error',
      'Could not load image',
      String(e.message || e) +
        '. The site may block downloads, or the URL may require a login.',
    );
    return;
  }

  if (!blob.type.startsWith('image/') && blob.size > 0) {
    const sniff = await blob.slice(0, 4).arrayBuffer();
    const bytes = new Uint8Array(sniff);
    const isPng = bytes[0] === 0x89 && bytes[1] === 0x50;
    const isJpeg = bytes[0] === 0xff && bytes[1] === 0xd8;
    if (!isPng && !isJpeg && blob.size > 100) {
      loading.classList.remove('show');
      showResult('error', 'Not an image', 'The URL did not return image data.');
      return;
    }
  }

  const pathPart = (() => {
    try {
      return new URL(src).pathname.split('/').pop() || 'image';
    } catch {
      return 'image';
    }
  })();
  const ext =
    blob.type === 'image/png'
      ? 'png'
      : blob.type === 'image/webp'
        ? 'webp'
        : blob.type === 'image/gif'
          ? 'gif'
          : 'jpg';
  const baseName = pathPart.includes('.') ? pathPart : `image.${ext}`;
  const file = new File([blob], baseName, { type: blob.type || 'image/jpeg' });

  const img = document.createElement('img');
  img.src = URL.createObjectURL(blob);
  img.alt = 'Preview';
  preview.appendChild(img);

  loadingLabel.textContent = 'Analyzing…';

  const formData = new FormData();
  formData.append('file', file);
  formData.append('type', 'image');

  try {
    const response = await fetch(`${API_URL}/analyze`, {
      method: 'POST',
      body: formData,
    });
    const data = await response.json();
    loading.classList.remove('show');

    if (data.error) {
      showResult('error', 'Error', data.error);
    } else {
      const isFake = data.is_fake;
      const confidence = Number(data.confidence ?? 0).toFixed(1);
      const label = isFake ? 'Likely Manipulated' : 'Appears Authentic';
      const message = data.message || `Confidence score: ${confidence}%`;
      showResult(isFake ? 'fake' : 'real', label, message);
    }
  } catch {
    loading.classList.remove('show');
    showResult(
      'error',
      'Connection failed',
      'Server offline — run: python3 app.py (listens on http://127.0.0.1:5002)',
    );
  }
}

checkServer();
run();
