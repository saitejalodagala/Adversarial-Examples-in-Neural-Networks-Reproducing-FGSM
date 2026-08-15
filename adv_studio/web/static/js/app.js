document.addEventListener('DOMContentLoaded', () => {
  // State
  const state = {
    currentImageBase64: null,
    currentAdvImageBase64: null,
    currentPrediction: null,
    currentTrueLabel: null,
    selectedSampleIndex: null,
    models: [],
    attacks: [],
    defenses: [],
  };

  // UI Elements
  const tabButtons = document.querySelectorAll('.tab-btn');
  const tabContents = document.querySelectorAll('.tab-content');
  const sampleGrid = document.getElementById('sampleGrid');
  const modelSelect = document.getElementById('modelSelect');
  const attackSelect = document.getElementById('attackSelect');
  const epsilonSlider = document.getElementById('epsilonSlider');
  const epsilonVal = document.getElementById('epsilonVal');
  const stepsSlider = document.getElementById('stepsSlider');
  const stepsVal = document.getElementById('stepsVal');
  const targetedToggle = document.getElementById('targetedToggle');
  const targetClassGroup = document.getElementById('targetClassGroup');
  const targetClassSelect = document.getElementById('targetClassSelect');

  const attackBtn = document.getElementById('attackBtn');
  const clearBtn = document.getElementById('clearBtn');

  const cleanImgEl = document.getElementById('cleanImg');
  const pertImgEl = document.getElementById('pertImg');
  const advImgEl = document.getElementById('advImg');

  const cleanBadge = document.getElementById('cleanBadge');
  const advBadge = document.getElementById('advBadge');

  const metricLinf = document.getElementById('metricLinf');
  const metricL2 = document.getElementById('metricL2');
  const metricL0 = document.getElementById('metricL0');
  const metricPsnr = document.getElementById('metricPsnr');
  const metricSsim = document.getElementById('metricSsim');

  // Defense Lab Elements
  const defenseTypeSelect = document.getElementById('defenseTypeSelect');
  const defenseParamSlider = document.getElementById('defenseParamSlider');
  const defenseParamVal = document.getElementById('defenseParamVal');
  const defenseParamLabel = document.getElementById('defenseParamLabel');
  const applyDefenseBtn = document.getElementById('applyDefenseBtn');
  const defenseAdvImg = document.getElementById('defenseAdvImg');
  const defenseSanitizedImg = document.getElementById('defenseSanitizedImg');
  const defenseBadge = document.getElementById('defenseBadge');

  // Initialize Canvas & Charts
  const charts = new StudioCharts();
  const visualizer = new StudioVisualizer();

  const digitCanvas = new DigitCanvas('digitCanvas', (dataUrl) => {
    state.currentImageBase64 = dataUrl;
    state.selectedSampleIndex = null;
    document.querySelectorAll('.sample-item').forEach(el => el.classList.remove('active'));
    predictClean(dataUrl);
  });

  // Tab Navigation
  tabButtons.forEach(btn => {
    btn.addEventListener('click', () => {
      tabButtons.forEach(b => b.classList.remove('active'));
      tabContents.forEach(c => c.classList.remove('active'));
      btn.classList.add('active');
      const tabId = btn.getAttribute('data-tab');
      document.getElementById(tabId).classList.add('active');
    });
  });

  // Slider Updates
  epsilonSlider.addEventListener('input', (e) => {
    epsilonVal.textContent = parseFloat(e.target.value).toFixed(2);
  });

  stepsSlider.addEventListener('input', (e) => {
    stepsVal.textContent = e.target.value;
  });

  targetedToggle.addEventListener('change', (e) => {
    targetClassGroup.style.display = e.target.checked ? 'block' : 'none';
  });

  clearBtn.addEventListener('click', () => {
    digitCanvas.clear();
  });

  // Fetch Initial Data
  async function init() {
    try {
      const [modelsRes, attacksRes, defensesRes, samplesRes] = await Promise.all([
        fetch('/api/models').then(r => r.json()),
        fetch('/api/attacks').then(r => r.json()),
        fetch('/api/defenses').then(r => r.json()),
        fetch('/api/samples').then(r => r.json()),
      ]);

      state.models = modelsRes.models;
      state.attacks = attacksRes.attacks;
      state.defenses = defensesRes.defenses;

      populateSelects();
      renderSamples(samplesRes.samples);

      // Load first sample as default
      if (samplesRes.samples && samplesRes.samples.length > 0) {
        selectSample(samplesRes.samples[0], 0);
      }
    } catch (err) {
      console.error('Failed to initialize studio data:', err);
    }
  }

  function populateSelects() {
    modelSelect.innerHTML = state.models.map(m => `<option value="${m.id}">${m.name}</option>`).join('');
    attackSelect.innerHTML = state.attacks.map(a => `<option value="${a.id}">${a.name}</option>`).join('');
    defenseTypeSelect.innerHTML = state.defenses.map(d => `<option value="${d.id}">${d.name}</option>`).join('');
  }

  function renderSamples(samples) {
    sampleGrid.innerHTML = samples.map((s, idx) => `
      <div class="sample-item" data-index="${idx}">
        <img src="${s.image_base64}" alt="Digit ${s.label}" />
        <span>#${s.label}</span>
      </div>
    `).join('');

    sampleGrid.querySelectorAll('.sample-item').forEach((el, idx) => {
      el.addEventListener('click', () => {
        selectSample(samples[idx], idx);
      });
    });
  }

  function selectSample(sample, index) {
    state.selectedSampleIndex = index;
    state.currentTrueLabel = sample.label;
    state.currentImageBase64 = sample.image_base64;

    sampleGrid.querySelectorAll('.sample-item').forEach((el, i) => {
      el.classList.toggle('active', i === index);
    });

    digitCanvas.loadDataURL(sample.image_base64);
  }

  async function predictClean(imageBase64) {
    cleanImgEl.src = imageBase64;
    cleanBadge.textContent = 'Predicting...';

    try {
      const res = await fetch('/api/predict', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          image_base64: imageBase64,
          model_name: modelSelect.value,
        })
      }).then(r => r.json());

      state.currentPrediction = res.prediction;
      cleanBadge.textContent = `Predicted: ${res.prediction} (${(res.confidence * 100).toFixed(1)}%)`;
      charts.updateProbabilities(res.probabilities, null);
    } catch (err) {
      console.error(err);
      cleanBadge.textContent = 'Prediction Error';
    }
  }

  // Attack Action
  attackBtn.addEventListener('click', async () => {
    if (!state.currentImageBase64) return;

    attackBtn.disabled = true;
    attackBtn.textContent = 'Crafting Adversary...';

    try {
      const payload = {
        image_base64: state.currentImageBase64,
        true_label: state.currentTrueLabel !== null ? state.currentTrueLabel : state.currentPrediction,
        model_name: modelSelect.value,
        attack_name: attackSelect.value,
        epsilon: parseFloat(epsilonSlider.value),
        steps: parseInt(stepsSlider.value),
        targeted: targetedToggle.checked,
        target_label: targetedToggle.checked ? parseInt(targetClassSelect.value) : null,
      };

      const res = await fetch('/api/attack', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload)
      }).then(r => r.json());

      // Update UI with adversarial result
      state.currentAdvImageBase64 = res.adversarial_image_base64;
      pertImgEl.src = res.perturbation_image_base64;
      advImgEl.src = res.adversarial_image_base64;

      const advConfPct = (res.adversarial_confidence * 100).toFixed(1);
      const isFlipped = res.adversarial_prediction !== res.original_prediction;
      advBadge.textContent = `Adversarial: ${res.adversarial_prediction} (${advConfPct}%) ${isFlipped ? '⚡' : ''}`;
      advBadge.className = `prediction-badge ${isFlipped ? 'badge-adversarial' : 'badge-clean'}`;

      // Update Charts & Metrics
      charts.updateProbabilities(res.original_probabilities, res.adversarial_probabilities);

      metricLinf.textContent = res.metrics.l_inf.toFixed(3);
      metricL2.textContent = res.metrics.l_2.toFixed(3);
      metricL0.textContent = `${(res.metrics.l_0 * 100).toFixed(1)}%`;
      metricPsnr.textContent = `${res.metrics.psnr_db.toFixed(1)} dB`;
      metricSsim.textContent = res.metrics.ssim.toFixed(3);

      // Update Defense Tab input
      if (defenseAdvImg) {
        defenseAdvImg.src = res.adversarial_image_base64;
      }

      // Compute loss landscape in background
      fetchLossLandscape(state.currentImageBase64, res.adversarial_image_base64, payload.true_label);

    } catch (err) {
      console.error('Attack error:', err);
    } finally {
      attackBtn.disabled = false;
      attackBtn.textContent = '🚀 Generate Adversarial Attack';
    }
  });

  async function fetchLossLandscape(cleanB64, advB64, label) {
    try {
      const res = await fetch('/api/loss-landscape', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          image_base64: cleanB64,
          adv_image_base64: advB64,
          true_label: label,
          model_name: modelSelect.value,
          num_points: 40,
        })
      }).then(r => r.json());

      visualizer.updateLandscape(res);
    } catch (err) {
      console.error('Loss landscape error:', err);
    }
  }

  // Defense Lab Action
  defenseTypeSelect.addEventListener('change', () => {
    const dType = defenseTypeSelect.value;
    const def = state.defenses.find(d => d.id === dType);
    if (def) {
      defenseParamLabel.textContent = def.param_name;
      defenseParamSlider.min = def.min_val;
      defenseParamSlider.max = def.max_val;
      defenseParamSlider.value = def.default_val;
      defenseParamSlider.step = dType === 'bit_depth' ? 1 : 0.01;
      defenseParamVal.textContent = def.default_val;
    }
  });

  defenseParamSlider.addEventListener('input', (e) => {
    defenseParamVal.textContent = parseFloat(e.target.value).toFixed(2);
  });

  applyDefenseBtn.addEventListener('click', async () => {
    const advImg = state.currentAdvImageBase64 || state.currentImageBase64;
    if (!advImg) return;

    applyDefenseBtn.disabled = true;
    defenseBadge.textContent = 'Sanitizing...';

    try {
      const res = await fetch('/api/defense', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          image_base64: advImg,
          defense_type: defenseTypeSelect.value,
          model_name: modelSelect.value,
          param_value: parseFloat(defenseParamSlider.value),
        })
      }).then(r => r.json());

      defenseSanitizedImg.src = res.sanitized_image_base64;
      defenseBadge.textContent = `Sanitized Output: Class ${res.prediction} (${(res.confidence * 100).toFixed(1)}%)`;
      defenseBadge.className = 'prediction-badge badge-sanitized';
    } catch (err) {
      console.error(err);
      defenseBadge.textContent = 'Defense Error';
    } finally {
      applyDefenseBtn.disabled = false;
    }
  });

  init();
});
