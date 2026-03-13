const API_URL = 'http://localhost:5000';
const files = { image: null, video: null, audio: null };

// Tab switching
document.querySelectorAll('.tab').forEach(tab => {
  tab.addEventListener('click', () => {
    document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
    document.querySelectorAll('.tab-content').forEach(c => c.classList.remove('active'));
    
    tab.classList.add('active');
    document.getElementById(`${tab.dataset.tab}-tab`).classList.add('active');
  });
});

// Upload area click handlers
document.querySelectorAll('.upload-area').forEach(area => {
  const type = area.dataset.type;
  const input = document.getElementById(`${type}-input`);
  
  area.addEventListener('click', () => input.click());
  
  area.addEventListener('dragover', (e) => {
    e.preventDefault();
    area.classList.add('dragover');
  });
  
  area.addEventListener('dragleave', () => {
    area.classList.remove('dragover');
  });
  
  area.addEventListener('drop', (e) => {
    e.preventDefault();
    area.classList.remove('dragover');
    handleFile(e.dataTransfer.files[0], type);
  });
});

// File input handlers
document.querySelectorAll('input[type="file"]').forEach(input => {
  const type = input.id.replace('-input', '');
  input.addEventListener('change', (e) => {
    handleFile(e.target.files[0], type);
  });
});

function handleFile(file, type) {
  if (!file) return;
  
  files[type] = file;
  document.getElementById(`${type}-filename`).textContent = file.name;
  document.getElementById(`${type}-btn`).disabled = false;
  document.getElementById(`${type}-result`).classList.remove('show');
  
  const preview = document.getElementById(`${type}-preview`);
  preview.innerHTML = '';
  
  const url = URL.createObjectURL(file);
  
  if (type === 'image') {
    const img = document.createElement('img');
    img.src = url;
    preview.appendChild(img);
  } else if (type === 'video') {
    const video = document.createElement('video');
    video.src = url;
    video.controls = true;
    preview.appendChild(video);
  } else if (type === 'audio') {
    const audio = document.createElement('audio');
    audio.src = url;
    audio.controls = true;
    preview.appendChild(audio);
  }
}

// Analyze button handlers
document.querySelectorAll('.analyze-btn').forEach(btn => {
  const type = btn.id.replace('-btn', '');
  btn.addEventListener('click', () => analyze(type));
});

async function analyze(type) {
  const file = files[type];
  if (!file) return;
  
  const btn = document.getElementById(`${type}-btn`);
  const loading = document.getElementById(`${type}-loading`);
  const result = document.getElementById(`${type}-result`);
  
  btn.style.display = 'none';
  loading.classList.add('show');
  result.classList.remove('show', 'real', 'fake', 'error');
  
  const formData = new FormData();
  formData.append('file', file);
  formData.append('type', type);
  
  try {
    const response = await fetch(`${API_URL}/analyze`, {
      method: 'POST',
      body: formData
    });
    
    const data = await response.json();
    
    loading.classList.remove('show');
    btn.style.display = 'block';
    
    if (data.error) {
      showResult(type, 'error', 'Error', data.error);
    } else {
      const isFake = data.is_fake;
      const confidence = (data.confidence * 100).toFixed(1);
      const label = isFake ? 'Likely Manipulated' : 'Appears Authentic';
      const message = data.message || `Confidence score: ${confidence}%`;
      showResult(type, isFake ? 'fake' : 'real', label, message);
    }
  } catch (error) {
    loading.classList.remove('show');
    btn.style.display = 'block';
    showResult(type, 'error', 'Connection Failed', 'Server offline — run python3 app.py');
  }
}

function showResult(type, status, label, detail) {
  const result = document.getElementById(`${type}-result`);
  result.className = `result show ${status}`;
  result.innerHTML = `
    <div class="result-label">${label}</div>
    <div class="result-detail">${detail}</div>
  `;
}

// Check server status
async function checkServer() {
  const statusDot = document.querySelector('.status-dot');
  const statusText = document.querySelector('.status-text');
  
  try {
    const response = await fetch(`${API_URL}/`, { method: 'HEAD' });
    if (response.ok) {
      statusDot.className = 'status-dot online';
      statusText.textContent = 'Server connected';
    } else {
      throw new Error('Server error');
    }
  } catch {
    statusDot.className = 'status-dot offline';
    statusText.textContent = 'Server offline';
  }
}

checkServer();
