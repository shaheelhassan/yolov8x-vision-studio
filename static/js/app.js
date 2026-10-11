/**
 * YOLOv8x AI Vision Studio - Frontend Application
 */

const STATE = {
  task: 'detect',
  conf: 0.25,
  iou: 0.45,
  device: 'auto',
  currentFile: null,
  currentSampleName: null,
  activeView: 'annotated', // 'annotated' or 'original'
  detectionData: null,
  allCategories: {},
  selectedCategories: new Set(),
  systemInfo: null,
  webcamStream: null
};

// DOM Elements
const elements = {
  deviceBadge: document.getElementById('deviceBadge'),
  deviceText: document.getElementById('deviceText'),
  specDeviceVal: document.getElementById('specDeviceVal'),
  specCudaVal: document.getElementById('specCudaVal'),
  confSlider: document.getElementById('confSlider'),
  confValueLabel: document.getElementById('confValueLabel'),
  iouSlider: document.getElementById('iouSlider'),
  iouValueLabel: document.getElementById('iouValueLabel'),
  taskCards: document.querySelectorAll('.task-card'),
  taskCurrentBadge: document.getElementById('taskCurrentBadge'),
  deviceBtns: document.querySelectorAll('.device-btn'),
  categoryChipsContainer: document.getElementById('categoryChipsContainer'),
  selectedClassesCount: document.getElementById('selectedClassesCount'),
  selectedClassesHint: document.getElementById('selectedClassesHint'),
  btnSelectAllCats: document.getElementById('btnSelectAllCats'),
  btnClearAllCats: document.getElementById('btnClearAllCats'),
  fileInput: document.getElementById('fileInput'),
  dropzoneInput: document.getElementById('dropzoneInput'),
  emptyDropzone: document.getElementById('emptyDropzone'),
  visualizerStage: document.getElementById('visualizerStage'),
  mainDisplayImage: document.getElementById('mainDisplayImage'),
  bboxOverlayLayer: document.getElementById('bboxOverlayLayer'),
  loadingOverlay: document.getElementById('loadingOverlay'),
  samplePillsList: document.getElementById('samplePillsList'),
  sampleGalleryEmpty: document.getElementById('sampleGalleryEmpty'),
  viewToggleAnnotated: document.getElementById('viewToggleAnnotated'),
  viewToggleOriginal: document.getElementById('viewToggleOriginal'),
  btnDownloadImage: document.getElementById('btnDownloadImage'),
  btnExportJson: document.getElementById('btnExportJson'),
  btnReanalyze: document.getElementById('btnReanalyze'),
  btnReset: document.getElementById('btnReset'),
  headerDetCountBadge: document.getElementById('headerDetCountBadge'),
  statTotalObjects: document.getElementById('statTotalObjects'),
  statLatency: document.getElementById('statLatency'),
  statFps: document.getElementById('statFps'),
  categorySummaryStrip: document.getElementById('categorySummaryStrip'),
  detectionsList: document.getElementById('detectionsList'),
  classificationResultsBox: document.getElementById('classificationResultsBox'),
  classificationList: document.getElementById('classificationList'),
  btnOpenCapabilities: document.getElementById('btnOpenCapabilities'),
  capabilitiesModal: document.getElementById('capabilitiesModal'),
  btnCloseModal: document.getElementById('btnCloseModal'),
  modalCocoCategories: document.getElementById('modalCocoCategories'),
  btnWebcam: document.getElementById('btnWebcam'),
  webcamModal: document.getElementById('webcamModal'),
  btnCloseWebcamModal: document.getElementById('btnCloseWebcamModal'),
  webcamVideo: document.getElementById('webcamVideo'),
  webcamCanvas: document.getElementById('webcamCanvas'),
  btnCaptureWebcam: document.getElementById('btnCaptureWebcam'),
  sidebarTabBtns: document.querySelectorAll('.sidebar-tab-btn'),
  tabContents: {
    config: document.getElementById('tabContentConfig'),
    filters: document.getElementById('tabContentFilters'),
    specs: document.getElementById('tabContentSpecs')
  }
};

// Initialize App
async function initApp() {
  setupTabs();
  setupEventListeners();
  await loadSystemInfo();
  await loadSamples();
}

// Tab Switching
function setupTabs() {
  elements.sidebarTabBtns.forEach(btn => {
    btn.addEventListener('click', () => {
      elements.sidebarTabBtns.forEach(b => b.classList.remove('active'));
      btn.classList.add('active');
      const tabName = btn.dataset.tab;
      Object.keys(elements.tabContents).forEach(key => {
        elements.tabContents[key].style.display = (key === tabName) ? 'flex' : 'none';
      });
    });
  });
}

// Fetch System Info
async function loadSystemInfo() {
  try {
    const res = await fetch('/api/info');
    const data = await res.json();
    STATE.systemInfo = data;

    // Hardware status
    const hw = data.hardware;
    if (elements.deviceText) {
      if (hw.cuda_available) {
        elements.deviceText.textContent = `GPU: ${hw.device_name} (${hw.vram_mb} MB)`;
      } else {
        elements.deviceText.textContent = 'CPU Mode Active';
      }
    }
    if (elements.specDeviceVal) elements.specDeviceVal.textContent = hw.cuda_available ? hw.device_name : 'Standard CPU';
    if (elements.specCudaVal) elements.specCudaVal.textContent = hw.cuda_available ? `Available (CUDA ${hw.torch_version})` : 'CUDA Not Detected';

    // Categories
    STATE.allCategories = data.model.capabilities.categories;
    populateCategoryFilters();
    populateModalCategories();
  } catch (err) {
    console.error('Failed to load system info:', err);
    if (elements.deviceText) elements.deviceText.textContent = 'Backend Offline';
  }
}

// Populate Category Filter Chips
function populateCategoryFilters() {
  elements.categoryChipsContainer.innerHTML = '';
  Object.keys(STATE.allCategories).forEach(catName => {
    STATE.selectedCategories.add(catName);
    const cat = STATE.allCategories[catName];
    const chip = document.createElement('button');
    chip.className = 'cat-chip active';
    chip.dataset.category = catName;
    chip.innerHTML = `
      <span class="cat-dot" style="background-color: ${cat.color};"></span>
      <span>${catName}</span>
      <span style="opacity: 0.6; font-size: 0.65rem;">(${cat.classes.length})</span>
    `;
    chip.addEventListener('click', () => toggleCategory(catName, chip));
    elements.categoryChipsContainer.appendChild(chip);
  });
  updateCategoryStats();
}

function toggleCategory(catName, chipEl) {
  if (STATE.selectedCategories.has(catName)) {
    if (STATE.selectedCategories.size === 1) return; // keep at least 1
    STATE.selectedCategories.delete(catName);
    chipEl.classList.remove('active');
  } else {
    STATE.selectedCategories.add(catName);
    chipEl.classList.add('active');
  }
  updateCategoryStats();
  if (STATE.detectionData) {
    runInference();
  }
}

function updateCategoryStats() {
  const allCount = Object.keys(STATE.allCategories).length;
  const selCount = STATE.selectedCategories.size;
  if (selCount === allCount) {
    elements.selectedClassesCount.textContent = 'All 80 Classes';
    elements.selectedClassesHint.textContent = 'Detecting across all 12 COCO categories.';
  } else {
    let classesSum = 0;
    STATE.selectedCategories.forEach(cat => {
      classesSum += STATE.allCategories[cat]?.classes.length || 0;
    });
    elements.selectedClassesCount.textContent = `${classesSum} Classes`;
    elements.selectedClassesHint.textContent = `Filtered to ${selCount} of 12 categories (${classesSum} active target classes).`;
  }
}

// Modal Categories Grid
function populateModalCategories() {
  elements.modalCocoCategories.innerHTML = '';
  Object.keys(STATE.allCategories).forEach(catName => {
    const cat = STATE.allCategories[catName];
    const box = document.createElement('div');
    box.className = 'coco-cat-box';
    const tagsHtml = cat.classes.map(c => `<span class="coco-tag">${c}</span>`).join('');
    box.innerHTML = `
      <div class="coco-cat-header" style="color: ${cat.color};">
        <span class="material-symbols-outlined" style="font-size: 18px;">${cat.icon}</span>
        <span>${catName} (${cat.classes.length})</span>
      </div>
      <div class="coco-cat-tags">${tagsHtml}</div>
    `;
    elements.modalCocoCategories.appendChild(box);
  });
}

// Fetch Sample Images
async function loadSamples() {
  try {
    const res = await fetch('/api/samples');
    const samples = await res.json();
    
    elements.samplePillsList.innerHTML = '';
    elements.sampleGalleryEmpty.innerHTML = '';

    samples.forEach(sample => {
      // Toolbar pill
      const pill = document.createElement('button');
      pill.className = 'btn btn-secondary btn-sm';
      pill.style.padding = '3px 8px';
      pill.style.fontSize = '0.74rem';
      pill.textContent = sample.name.split(' ')[0];
      pill.title = sample.name;
      pill.addEventListener('click', () => loadSampleImage(sample.file));
      elements.samplePillsList.appendChild(pill);

      // Empty dropzone thumbnail card
      const thumb = document.createElement('div');
      thumb.className = 'sample-thumb-item';
      thumb.title = sample.name;
      thumb.innerHTML = `<img src="${sample.url}" alt="${sample.name}">`;
      thumb.addEventListener('click', () => loadSampleImage(sample.file));
      elements.sampleGalleryEmpty.appendChild(thumb);
    });
  } catch (err) {
    console.error('Failed to load sample images:', err);
  }
}

// Load Sample Image
function loadSampleImage(fileName) {
  STATE.currentFile = null;
  STATE.currentSampleName = fileName;
  runInference();
}

// Event Listeners
function setupEventListeners() {
  // Sliders
  elements.confSlider.addEventListener('input', (e) => {
    const val = e.target.value;
    STATE.conf = val / 100;
    elements.confValueLabel.textContent = `${val}%`;
  });

  elements.iouSlider.addEventListener('input', (e) => {
    const val = e.target.value;
    STATE.iou = val / 100;
    elements.iouValueLabel.textContent = `${val}%`;
  });

  // Task Cards
  elements.taskCards.forEach(card => {
    card.addEventListener('click', () => {
      elements.taskCards.forEach(c => c.classList.remove('active'));
      card.classList.add('active');
      STATE.task = card.dataset.task;
      elements.taskCurrentBadge.textContent = card.querySelector('.task-title').textContent;
      if (STATE.currentFile || STATE.currentSampleName) {
        runInference();
      }
    });
  });

  // Device Buttons
  elements.deviceBtns.forEach(btn => {
    btn.addEventListener('click', () => {
      elements.deviceBtns.forEach(b => b.classList.remove('active'));
      btn.classList.add('active');
      STATE.device = btn.dataset.device;
    });
  });

  // Category select all / clear
  elements.btnSelectAllCats.addEventListener('click', () => {
    document.querySelectorAll('.cat-chip').forEach(c => {
      c.classList.add('active');
      STATE.selectedCategories.add(c.dataset.category);
    });
    updateCategoryStats();
    if (STATE.currentFile || STATE.currentSampleName) runInference();
  });

  elements.btnClearAllCats.addEventListener('click', () => {
    STATE.selectedCategories.clear();
    const firstCat = Object.keys(STATE.allCategories)[0];
    STATE.selectedCategories.add(firstCat);
    document.querySelectorAll('.cat-chip').forEach(c => {
      if (c.dataset.category === firstCat) c.classList.add('active');
      else c.classList.remove('active');
    });
    updateCategoryStats();
    if (STATE.currentFile || STATE.currentSampleName) runInference();
  });

  // File Upload Handlers
  elements.fileInput.addEventListener('change', handleFileSelect);
  elements.dropzoneInput.addEventListener('change', handleFileSelect);

  // Drag & Drop
  const dropzone = elements.emptyDropzone;
  ['dragenter', 'dragover'].forEach(name => {
    dropzone.addEventListener(name, (e) => {
      e.preventDefault();
      e.stopPropagation();
      dropzone.classList.add('dragover');
    });
  });
  ['dragleave', 'drop'].forEach(name => {
    dropzone.addEventListener(name, (e) => {
      e.preventDefault();
      e.stopPropagation();
      dropzone.classList.remove('dragover');
    });
  });
  dropzone.addEventListener('drop', (e) => {
    const files = e.dataTransfer.files;
    if (files.length > 0 && files[0].type.startsWith('image/')) {
      STATE.currentFile = files[0];
      STATE.currentSampleName = null;
      runInference();
    }
  });

  // Re-run
  elements.btnReanalyze.addEventListener('click', () => {
    if (STATE.currentFile || STATE.currentSampleName) {
      runInference();
    } else {
      elements.fileInput.click();
    }
  });

  // Reset
  elements.btnReset.addEventListener('click', () => {
    STATE.currentFile = null;
    STATE.currentSampleName = null;
    STATE.detectionData = null;
    elements.visualizerStage.style.display = 'none';
    elements.emptyDropzone.style.display = 'flex';
    elements.mainDisplayImage.src = '';
    elements.bboxOverlayLayer.innerHTML = '';
    elements.statTotalObjects.textContent = '0';
    elements.statLatency.innerHTML = '0<span style="font-size:0.75rem;">ms</span>';
    elements.statFps.textContent = '0';
    elements.headerDetCountBadge.textContent = '0 Found';
    elements.categorySummaryStrip.innerHTML = '';
    elements.detectionsList.innerHTML = `
      <div style="text-align: center; padding: 40px 10px; color: var(--text-muted); font-size: 0.82rem;">
        <span class="material-symbols-outlined" style="font-size: 36px; color: #cbd5e1; margin-bottom: 8px;">image_search</span>
        <div>No detections to display yet.</div>
        <div style="font-size: 0.75rem; color: var(--text-subtle); margin-top: 4px;">Upload an image or pick a sample above to begin.</div>
      </div>
    `;
    elements.classificationResultsBox.style.display = 'none';
  });

  // View toggles (Annotated vs Original)
  elements.viewToggleAnnotated.addEventListener('click', () => {
    STATE.activeView = 'annotated';
    elements.viewToggleAnnotated.classList.add('active');
    elements.viewToggleOriginal.classList.remove('active');
    updateVisualizerView();
  });

  elements.viewToggleOriginal.addEventListener('click', () => {
    STATE.activeView = 'original';
    elements.viewToggleOriginal.classList.add('active');
    elements.viewToggleAnnotated.classList.remove('active');
    updateVisualizerView();
  });

  // Download Annotated Image
  elements.btnDownloadImage.addEventListener('click', () => {
    if (!STATE.detectionData) return;
    const a = document.createElement('a');
    a.href = STATE.detectionData.annotated_image;
    a.download = `yolov8x_detection_${Date.now()}.jpg`;
    a.click();
  });

  // Export JSON
  elements.btnExportJson.addEventListener('click', () => {
    if (!STATE.detectionData) return;
    const blob = new Blob([JSON.stringify(STATE.detectionData, null, 2)], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `yolov8x_analysis_${Date.now()}.json`;
    a.click();
    URL.revokeObjectURL(url);
  });

  // Modal Handlers
  elements.btnOpenCapabilities.addEventListener('click', () => {
    elements.capabilitiesModal.classList.add('open');
  });
  elements.btnCloseModal.addEventListener('click', () => {
    elements.capabilitiesModal.classList.remove('open');
  });
  elements.capabilitiesModal.addEventListener('click', (e) => {
    if (e.target === elements.capabilitiesModal) {
      elements.capabilitiesModal.classList.remove('open');
    }
  });

  // Webcam Handlers
  elements.btnWebcam.addEventListener('click', openWebcamModal);
  elements.btnCloseWebcamModal.addEventListener('click', closeWebcamModal);
  elements.btnCaptureWebcam.addEventListener('click', captureWebcamSnapshot);
}

function handleFileSelect(e) {
  const file = e.target.files[0];
  if (file && file.type.startsWith('image/')) {
    STATE.currentFile = file;
    STATE.currentSampleName = null;
    runInference();
  }
}

// Inference Call
async function runInference() {
  if (!STATE.currentFile && !STATE.currentSampleName) return;

  elements.loadingOverlay.classList.add('visible');

  try {
    const formData = new FormData();
    if (STATE.currentFile) {
      formData.append('file', STATE.currentFile);
    } else if (STATE.currentSampleName) {
      formData.append('sample_name', STATE.currentSampleName);
    }

    formData.append('task', STATE.task);
    formData.append('conf_threshold', STATE.conf.toString());
    formData.append('iou_threshold', STATE.iou.toString());
    formData.append('device', STATE.device);

    // Compute active classes filter if not all 12 are selected
    if (STATE.selectedCategories.size < Object.keys(STATE.allCategories).length) {
      const allowedClasses = [];
      STATE.selectedCategories.forEach(cat => {
        const catObj = STATE.allCategories[cat];
        if (catObj && catObj.classes) {
          allowedClasses.push(...catObj.classes);
        }
      });
      formData.append('classes_filter', allowedClasses.join(','));
    }

    const res = await fetch('/api/detect', {
      method: 'POST',
      body: formData
    });

    if (!res.ok) {
      const err = await res.json();
      throw new Error(err.detail || 'Inference failed');
    }

    const data = await res.json();
    STATE.detectionData = data;
    renderResults(data);
  } catch (err) {
    console.error('Detection Error:', err);
    alert('Detection error: ' + err.message);
  } finally {
    elements.loadingOverlay.classList.remove('visible');
  }
}

// Render Results
function renderResults(data) {
  // Hide empty dropzone, show visualizer
  elements.emptyDropzone.style.display = 'none';
  elements.visualizerStage.style.display = 'flex';

  // Update view
  updateVisualizerView();

  // Metrics
  const m = data.metrics;
  elements.statTotalObjects.textContent = m.total_detections;
  elements.statLatency.innerHTML = `${m.inference_time_ms}<span style="font-size:0.75rem;">ms</span>`;
  elements.statFps.textContent = m.fps;
  elements.headerDetCountBadge.textContent = `${m.total_detections} Found`;

  // Category summary chips
  elements.categorySummaryStrip.innerHTML = '';
  Object.keys(data.category_counts).forEach(catName => {
    const count = data.category_counts[catName];
    const catInfo = STATE.allCategories[catName] || { color: '#64748b', icon: 'label' };
    const chip = document.createElement('div');
    chip.className = 'det-cat-badge';
    chip.style.backgroundColor = `${catInfo.color}15`;
    chip.style.color = catInfo.color;
    chip.innerHTML = `
      <span class="material-symbols-outlined" style="font-size: 14px;">${catInfo.icon}</span>
      <span>${catName}: <strong>${count}</strong></span>
    `;
    elements.categorySummaryStrip.appendChild(chip);
  });

  // Classification view if classification
  if (data.classification_top && data.classification_top.length > 0) {
    elements.classificationResultsBox.style.display = 'flex';
    elements.classificationList.innerHTML = '';
    data.classification_top.forEach(item => {
      const row = document.createElement('div');
      row.style.cssText = 'background: var(--bg-subtle); padding: 6px 10px; border-radius: var(--radius-sm); border: 1px solid var(--border); display: flex; justify-content: space-between; font-size: 0.78rem;';
      row.innerHTML = `
        <span style="font-weight: 600;">${item.rank}. ${item.class_name}</span>
        <span style="color: var(--primary); font-weight: 700;">${item.confidence}%</span>
      `;
      elements.classificationList.appendChild(row);
    });
  } else {
    elements.classificationResultsBox.style.display = 'none';
  }

  // Detections List in right sidebar
  renderDetectionsList(data.detections);
  renderInteractiveOverlay(data.detections);
}

// Update Image View (Annotated vs Original)
function updateVisualizerView() {
  if (!STATE.detectionData) return;
  if (STATE.activeView === 'annotated') {
    elements.mainDisplayImage.src = STATE.detectionData.annotated_image;
  } else {
    elements.mainDisplayImage.src = STATE.detectionData.original_image;
  }
}

// Render Detections List Cards
function renderDetectionsList(detections) {
  elements.detectionsList.innerHTML = '';

  if (detections.length === 0) {
    elements.detectionsList.innerHTML = `
      <div style="text-align: center; padding: 40px 10px; color: var(--text-muted); font-size: 0.82rem;">
        <span class="material-symbols-outlined" style="font-size: 32px; color: #cbd5e1; margin-bottom: 6px;">search_off</span>
        <div>No objects met confidence threshold (${Math.round(STATE.conf * 100)}%).</div>
        <div style="font-size: 0.75rem; color: var(--text-subtle); margin-top: 4px;">Try lowering the confidence slider on the left.</div>
      </div>
    `;
    return;
  }

  detections.forEach(det => {
    const card = document.createElement('div');
    card.className = 'detection-item-card';
    card.id = `detCard_${det.id}`;
    card.dataset.id = det.id;

    const thumbHtml = det.crop_base64
      ? `<img src="${det.crop_base64}" alt="${det.class_name}">`
      : `<span class="material-symbols-outlined" style="color: ${det.color};">${det.icon}</span>`;

    card.innerHTML = `
      <div class="det-crop-thumb">
        ${thumbHtml}
      </div>
      <div class="det-details">
        <div class="det-title-row">
          <span class="det-name">${det.class_name}</span>
          <span class="det-conf">${det.confidence}%</span>
        </div>
        <div class="det-meta-row">
          <span class="det-cat-badge" style="color: ${det.color}; background-color: ${det.color}15;">
            <span class="material-symbols-outlined" style="font-size: 12px;">${det.icon}</span>
            ${det.category}
          </span>
          <span>${det.width}&times;${det.height}px</span>
        </div>
        <div class="conf-bar-bg">
          <div class="conf-bar-fill" style="width: ${det.confidence}%; background-color: ${det.color};"></div>
        </div>
      </div>
    `;

    // Interactive Hover & Focus
    card.addEventListener('mouseenter', () => highlightObject(det.id));
    card.addEventListener('mouseleave', () => unhighlightObject(det.id));

    elements.detectionsList.appendChild(card);
  });
}

// Render Interactive Bounding Box Overlay on Canvas
function renderInteractiveOverlay(detections) {
  elements.bboxOverlayLayer.innerHTML = '';

  detections.forEach(det => {
    const [nx, ny, nw, nh] = det.bbox_normalized;
    const box = document.createElement('div');
    box.className = 'interactive-box';
    box.id = `bboxItem_${det.id}`;
    box.style.left = `${nx * 100}%`;
    box.style.top = `${ny * 100}%`;
    box.style.width = `${nw * 100}%`;
    box.style.height = `${nh * 100}%`;
    box.style.border = `2px dashed ${det.color}`;
    box.title = `${det.class_name} (${det.confidence}%)`;

    // Hover interactions
    box.addEventListener('mouseenter', () => highlightObject(det.id, true));
    box.addEventListener('mouseleave', () => unhighlightObject(det.id));

    elements.bboxOverlayLayer.appendChild(box);
  });
}

function highlightObject(id, scrollSidebar = false) {
  const box = document.getElementById(`bboxItem_${id}`);
  const card = document.getElementById(`detCard_${id}`);

  if (box) {
    box.classList.add('highlighted');
    box.style.border = '2px solid #2563eb';
    box.style.backgroundColor = 'rgba(37, 99, 235, 0.15)';
  }
  if (card) {
    card.classList.add('active');
    if (scrollSidebar) {
      card.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
    }
  }
}

function unhighlightObject(id) {
  const box = document.getElementById(`bboxItem_${id}`);
  const card = document.getElementById(`detCard_${id}`);

  if (box) {
    box.classList.remove('highlighted');
    const det = STATE.detectionData?.detections.find(d => d.id === id);
    box.style.border = `2px dashed ${det?.color || '#3b82f6'}`;
    box.style.backgroundColor = 'transparent';
  }
  if (card) {
    card.classList.remove('active');
  }
}

// Webcam Modal Controls
async function openWebcamModal() {
  elements.webcamModal.classList.add('open');
  try {
    const stream = await navigator.mediaDevices.getUserMedia({
      video: { width: { ideal: 1280 }, height: { ideal: 720 } },
      audio: false
    });
    STATE.webcamStream = stream;
    elements.webcamVideo.srcObject = stream;
  } catch (err) {
    alert('Webcam access error: ' + err.message);
    closeWebcamModal();
  }
}

function closeWebcamModal() {
  if (STATE.webcamStream) {
    STATE.webcamStream.getTracks().forEach(track => track.stop());
    STATE.webcamStream = null;
  }
  elements.webcamModal.classList.remove('open');
}

function captureWebcamSnapshot() {
  const video = elements.webcamVideo;
  const canvas = elements.webcamCanvas;
  canvas.width = video.videoWidth || 640;
  canvas.height = video.videoHeight || 480;
  const ctx = canvas.getContext('2d');
  ctx.drawImage(video, 0, 0, canvas.width, canvas.height);

  canvas.toBlob((blob) => {
    STATE.currentFile = new File([blob], `webcam_snap_${Date.now()}.jpg`, { type: 'image/jpeg' });
    STATE.currentSampleName = null;
    closeWebcamModal();
    runInference();
  }, 'image/jpeg', 0.92);
}

// Launch application
document.addEventListener('DOMContentLoaded', initApp);
