/* ══════════════════════════════════════════════════════════════════════
   AutismCare AI — Frontend Application Logic
   Connects to FastAPI backend on port 8000
   Redesigned dashboard with agent insights, score cards,
   signal breakdown, gradient progress chart
   ══════════════════════════════════════════════════════════════════════ */

const API = window.location.hostname === "localhost"
  ? "http://localhost:8000"
  : "";  // same origin on Render

// ── State ─────────────────────────────────────────────────────────────
let latestResult = null;
let allSessions = [];

// ══════════════════════════════════════════════════════════════════════
//  NAVIGATION
// ══════════════════════════════════════════════════════════════════════
document.querySelectorAll(".nav-item").forEach(item => {
  item.addEventListener("click", e => {
    e.preventDefault();
    const page = item.dataset.page;
    document.querySelectorAll(".nav-item").forEach(n => n.classList.remove("active"));
    item.classList.add("active");
    document.querySelectorAll(".page").forEach(p => p.classList.remove("active"));
    document.getElementById("page-" + page).classList.add("active");

    if (page === "dashboard") refreshDashboard();
    if (page === "progress") refreshProgress();
    if (page === "reports") refreshReport();
    if (page === "therapy") refreshTherapy();
    if (page === "profile") refreshProfile();
  });
});

// ══════════════════════════════════════════════════════════════════════
//  API STATUS CHECK
// ══════════════════════════════════════════════════════════════════════
async function checkAPI() {
  const el = document.getElementById("apiStatus");
  try {
    const r = await fetch(API + "/api/status");
    const d = await r.json();
    const dot = el.querySelector(".status-dot");
    dot.className = "status-dot online";
    el.querySelector("span:last-child").textContent = "API Online";

    // Update model pills
    const map = {
      "st-face": d.models.face_classifier,
      "st-behavior": d.models.behavior_lstm,
      "st-questionnaire": d.models.questionnaire_xgb,
      "st-eye": d.models.eye_tracking_xgb,
      "st-pose": d.models.pose_skeleton_xgb,
      "st-cars": d.models.cars_severity,
    };
    for (const [id, loaded] of Object.entries(map)) {
      const pill = document.getElementById(id);
      if (pill) {
        pill.className = "model-pill " + (loaded ? "loaded" : "missing");
      }
    }
  } catch {
    const dot = el.querySelector(".status-dot");
    dot.className = "status-dot offline";
    el.querySelector("span:last-child").textContent = "API Offline";
  }
}
checkAPI();
setInterval(checkAPI, 15000);

// ══════════════════════════════════════════════════════════════════════
//  VISUAL SCREENING (Image/Video Upload)
// ══════════════════════════════════════════════════════════════════════
const uploadZone = document.getElementById("uploadZone");
const fileInput = document.getElementById("fileInput");
const analyzeBtn = document.getElementById("analyzeBtn");
const previewEl = document.getElementById("uploadPreview");
let selectedFile = null;

uploadZone.addEventListener("click", () => fileInput.click());
uploadZone.addEventListener("dragover", e => { e.preventDefault(); uploadZone.classList.add("dragover"); });
uploadZone.addEventListener("dragleave", () => uploadZone.classList.remove("dragover"));
uploadZone.addEventListener("drop", e => {
  e.preventDefault();
  uploadZone.classList.remove("dragover");
  if (e.dataTransfer.files.length) handleFile(e.dataTransfer.files[0]);
});
fileInput.addEventListener("change", () => { if (fileInput.files.length) handleFile(fileInput.files[0]); });

function handleFile(file) {
  selectedFile = file;
  previewEl.classList.remove("hidden");
  previewEl.innerHTML = "";
  if (file.type.startsWith("image/")) {
    const img = document.createElement("img");
    img.src = URL.createObjectURL(file);
    previewEl.appendChild(img);
  } else if (file.type.startsWith("video/")) {
    const vid = document.createElement("video");
    vid.src = URL.createObjectURL(file);
    vid.controls = true;
    vid.muted = true;
    previewEl.appendChild(vid);
  }
  analyzeBtn.disabled = false;
}

analyzeBtn.addEventListener("click", async () => {
  if (!selectedFile) return;
  analyzeBtn.disabled = true;
  document.getElementById("analyzeLoader").classList.remove("hidden");
  document.getElementById("screeningResults").classList.add("hidden");

  const form = new FormData();
  form.append("file", selectedFile);

  try {
    const r = await fetch(API + "/api/analyze", { method: "POST", body: form });
    const d = await r.json();
    latestResult = d;
    allSessions.push(d);
    showScreeningResults(d);
    updateDashboard(d);
  } catch (err) {
    alert("Error: " + err.message);
  } finally {
    analyzeBtn.disabled = false;
    document.getElementById("analyzeLoader").classList.add("hidden");
  }
});

function showScreeningResults(d) {
  if (!d) return;
  const panel = document.getElementById("screeningResults");
  if (!panel) return;
  panel.classList.remove("hidden");

  if (d.original_b64) document.getElementById("resOriginal").src = "data:image/png;base64," + d.original_b64;
  if (d.gradcam_b64) document.getElementById("resGradCAM").src = "data:image/png;base64," + d.gradcam_b64;

  const scoresEl = document.getElementById("resScores");
  scoresEl.innerHTML = "";
  const scores = { "Face": d.face_score, "Behavior": d.behavior_score, "Fused": d.fused_score };
  for (const [label, val] of Object.entries(scores)) {
    if (val == null) continue;
    const cls = val > 0.6 ? "high" : val > 0.3 ? "medium" : "low";
    scoresEl.innerHTML += `<div class="score-chip"><div class="label">${label}</div><div class="value ${cls}">${(val * 100).toFixed(1)}%</div></div>`;
  }

  const state = d.screening.state;
  const riskCls = state === "CLINICAL_REVIEW" ? "risk-high" : state === "MONITOR" ? "risk-monitor" : "risk-low";
  document.getElementById("resRisk").innerHTML = `<span class="risk-badge ${riskCls}">${state.replace("_", " ")}</span>`;

  document.getElementById("resClinical").innerHTML = `<h4>Clinical Assessment</h4><p>${d.clinical.assessment}</p><p><strong>Recommendation:</strong> ${d.clinical.recommendation}</p>`;

  const tEl = document.getElementById("resTherapy");
  tEl.innerHTML = "<h4>Suggested Interventions</h4>";
  if (d.therapy && d.therapy.plan) {
    d.therapy.plan.forEach((item, i) => {
      const priority = d.therapy.priorities[i] || "Low";
      tEl.innerHTML += `<div class="therapy-item"><span class="priority-dot ${priority.toLowerCase()}"></span>${item}</div>`;
    });
  }
}

// ══════════════════════════════════════════════════════════════════════
//  QUESTIONNAIRE
// ══════════════════════════════════════════════════════════════════════
document.querySelectorAll(".toggle-group").forEach(group => {
  group.querySelectorAll(".toggle-btn").forEach(btn => {
    btn.addEventListener("click", () => {
      group.querySelectorAll(".toggle-btn").forEach(b => b.classList.remove("active"));
      btn.classList.add("active");
    });
  });
});

document.getElementById("questForm").addEventListener("submit", async (e) => {
  e.preventDefault();
  document.getElementById("questLoader").classList.remove("hidden");
  document.getElementById("questResults").classList.add("hidden");

  const answers = {};
  document.querySelectorAll(".quest-item").forEach(item => {
    const q = item.dataset.q;
    const active = item.querySelector(".toggle-btn.active");
    answers[q] = parseInt(active.dataset.val);
  });

  const payload = {
    ...answers,
    age: parseFloat(document.getElementById("qAge").value),
    gender: parseInt(document.getElementById("qGender").value),
    jundice: parseInt(document.getElementById("qJaundice").value),
    austim: parseInt(document.getElementById("qFamily").value),
  };

  try {
    const r = await fetch(API + "/api/questionnaire", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    const d = await r.json();
    latestResult = d;
    allSessions.push(d);
    showQuestResults(d);
    updateDashboard(d);
  } catch (err) {
    alert("Error: " + err.message);
  } finally {
    document.getElementById("questLoader").classList.add("hidden");
  }
});

function showQuestResults(d) {
  if (!d) return;
  const panel = document.getElementById("questResults");
  if (!panel) return;
  panel.classList.remove("hidden");

  const state = d.screening.state;
  const riskCls = state === "CLINICAL_REVIEW" ? "risk-high" : state === "MONITOR" ? "risk-monitor" : "risk-low";
  document.getElementById("questRisk").innerHTML = `<span class="risk-badge ${riskCls}">${state.replace("_", " ")}</span>`;

  const prob = d.questionnaire_score;
  const cls = prob > 0.6 ? "high" : prob > 0.3 ? "medium" : "low";
  document.getElementById("questScoreDisplay").innerHTML = `
    <div class="big-score ${cls}">${(prob * 100).toFixed(1)}%</div>
    <div style="color:var(--text-secondary); font-size:13px; margin-top:4px;">ASD Risk Probability</div>
    <div style="font-size:12px; color:var(--text-muted); margin-top:8px;">
      Total Score: ${d.domain_scores.total_score}/10 
      (Social: ${d.domain_scores.social_sum}, 
       Comm: ${d.domain_scores.communication_sum}, 
       Behav: ${d.domain_scores.behavior_sum})
    </div>`;

  const domEl = document.getElementById("questDomainBars");
  domEl.innerHTML = "";
  const domains = d.domain_profile;
  for (const [name, val] of Object.entries(domains)) {
    const pct = (val * 100).toFixed(0);
    domEl.innerHTML += `
      <div class="domain-bar-group">
        <div class="domain-bar-label"><span>${name}</span><span>${pct}%</span></div>
        <div class="domain-bar-track"><div class="domain-bar-fill ${name.toLowerCase()}" style="width:${pct}%"></div></div>
      </div>`;
  }

  document.getElementById("questClinical").innerHTML = `<h4>Clinical Assessment</h4><p>${d.clinical.assessment}</p><p><strong>Recommendation:</strong> ${d.clinical.recommendation}</p>`;

  const tEl = document.getElementById("questTherapy");
  tEl.innerHTML = "<h4>Suggested Interventions</h4>";
  if (d.therapy && d.therapy.plan) {
    d.therapy.plan.forEach((item, i) => {
      const priority = d.therapy.priorities[i] || "Low";
      tEl.innerHTML += `<div class="therapy-item"><span class="priority-dot ${priority.toLowerCase()}"></span>${item}</div>`;
    });
  }
}

// ══════════════════════════════════════════════════════════════════════
//  DASHBOARD UPDATES
// ══════════════════════════════════════════════════════════════════════
function updateDashboard(d) {
  if (!d || !d.screening) return;
  // Update status badge
  const state = d.screening.state;
  const badgeEl = document.getElementById("dashStatusBadge");
  if (!badgeEl) return;
  if (state === "CLINICAL_REVIEW") {
    badgeEl.innerHTML = '<span class="status-pill clinical">⚠ Clinical Review</span>';
  } else if (state === "MONITOR") {
    badgeEl.innerHTML = '<span class="status-pill review">⚡ Under Review</span>';
  } else {
    badgeEl.innerHTML = '<span class="status-pill low-risk">✓ Low Risk</span>';
  }

  // Update last assessment date
  document.getElementById("dashLastDate").textContent = new Date().toLocaleDateString();

  // Update agent insights
  if (d.screening) {
    const scMsg = d.screening.flagged_modalities && d.screening.flagged_modalities.length
      ? `"Flagged: ${d.screening.flagged_modalities.join(', ')}. Agreement: ${((d.screening.cross_modal_agreement || 0) * 100).toFixed(0)}%"`
      : `"Screening complete. Risk state: ${state.replace('_', ' ')}"`;
    document.getElementById("agentScreeningInsight").textContent = scMsg;
  }
  if (d.clinical) {
    const clinMsg = d.clinical.recommendation
      ? `"${d.clinical.recommendation.substring(0, 120)}${d.clinical.recommendation.length > 120 ? '...' : ''}"`
      : '"Clinical assessment complete."';
    document.getElementById("agentClinicalInsight").textContent = clinMsg;
  }
  if (d.therapy) {
    const thMsg = d.therapy.plan && d.therapy.plan.length
      ? `"${d.therapy.plan.length} intervention${d.therapy.plan.length > 1 ? 's' : ''} recommended. Focus: ${d.therapy.plan[0].substring(0, 60)}..."`
      : '"No specific interventions needed at this time."';
    document.getElementById("agentTherapyInsight").textContent = thMsg;
  }
  if (d.monitoring) {
    const monMsg = `"Trajectory: ${d.monitoring.trajectory}. Trend: ${((d.monitoring.trend || 0) * 100).toFixed(1)}%. Alert: ${d.monitoring.alert}"`;
    document.getElementById("agentMonitorInsight").textContent = monMsg;
  }

  // Update score cards
  updateScoreCards(d);

  // Update screening results section
  updateScreeningResultsSection(d);

  // Update signal breakdown
  updateSignalBreakdown(d);

  // Draw progress chart on dashboard
  drawDashTrend();
}

function updateScoreCards(d) {
  // Derive scores from domain_scores or modality_scores
  let social = 0, behavior = 0, comm = 0, index = 0;

  if (d.domain_scores) {
    social = Math.round((d.domain_scores.social_sum / 5) * 100);
    behavior = Math.round((d.domain_scores.behavior_sum / 2) * 100);
    comm = Math.round((d.domain_scores.communication_sum / 3) * 100);
  } else if (d.modality_scores) {
    social = Math.round((d.modality_scores.face || 0) * 100);
    behavior = Math.round((d.modality_scores.behavior || 0) * 100);
    comm = Math.round((d.modality_scores.eye_tracking || 0) * 100);
  }
  index = Math.round((d.fused_score || 0) * 100);

  const setCard = (valId, lvlId, barId, val) => {
    document.getElementById(valId).textContent = val;
    const lvl = val >= 70 ? "High" : val >= 40 ? "Moderate" : "Low";
    document.getElementById(lvlId).textContent = lvl;
    document.getElementById(barId).style.width = val + "%";
  };

  setCard("scSocialVal", "scSocialLvl", "scSocialBar", social);
  setCard("scBehaviorVal", "scBehaviorLvl", "scBehaviorBar", behavior);
  setCard("scCommVal", "scCommLvl", "scCommBar", comm);
  setCard("scIndexVal", "scIndexLvl", "scIndexBar", index);
}

function updateScreeningResultsSection(d) {
  if (!d || !d.screening) return;
  const state = d.screening.state;
  const header = document.getElementById("srHeader");
  if (!header) return;
  const confidence = d.fused_score ? (d.fused_score * 100).toFixed(0) + "%" : "—";

  if (state === "CLINICAL_REVIEW") {
    header.innerHTML = `<span class="sr-title">Recommendation: Clinical Review</span><span class="sr-subtitle">Multi-modal analysis indicates elevated risk patterns</span>`;
  } else if (state === "MONITOR") {
    header.innerHTML = `<span class="sr-title">Recommendation: Monitoring</span><span class="sr-subtitle">Some indicators suggest continued observation</span>`;
  } else {
    header.innerHTML = `<span class="sr-title">Recommendation: Low Risk</span><span class="sr-subtitle">No significant risk indicators detected</span>`;
  }

  document.getElementById("srConfidence").textContent = confidence;
  document.getElementById("srDescription").textContent = d.clinical ? d.clinical.assessment : "Analysis complete.";
}

function updateSignalBreakdown(d) {
  let social = 0, behavior = 0, comm = 0, data = 0;

  if (d.domain_scores) {
    social = Math.round((d.domain_scores.social_sum / 5) * 100);
    behavior = Math.round((d.domain_scores.behavior_sum / 2) * 100);
    comm = Math.round((d.domain_scores.communication_sum / 3) * 100);
  }
  if (d.modality_scores) {
    if (d.modality_scores.face != null) social = Math.round(d.modality_scores.face * 100);
    if (d.modality_scores.behavior != null) behavior = Math.round(d.modality_scores.behavior * 100);
    if (d.modality_scores.eye_tracking != null) comm = Math.round(d.modality_scores.eye_tracking * 100);
  }
  data = Math.round((d.fused_score || 0) * 100);

  const set = (fillId, pctId, val) => {
    document.getElementById(fillId).style.width = val + "%";
    document.getElementById(pctId).textContent = val + "%";
  };
  set("sigSocial", "sigSocialPct", social);
  set("sigBehavior", "sigBehaviorPct", behavior);
  set("sigComm", "sigCommPct", comm);
  set("sigData", "sigDataPct", data);
}

function refreshDashboard() {
  if (latestResult) {
    updateDashboard(latestResult);
  }
}

// ══════════════════════════════════════════════════════════════════════
//  PROGRESS CHART (gradient line from red → green)
// ══════════════════════════════════════════════════════════════════════
function drawGradientChart(canvasId, scores) {
  const canvas = document.getElementById(canvasId);
  const ctx = canvas.getContext("2d");
  const W = canvas.width, H = canvas.height;
  ctx.clearRect(0, 0, W, H);

  if (scores.length === 0) {
    ctx.fillStyle = "#94a3b8";
    ctx.font = "14px Segoe UI";
    ctx.textAlign = "center";
    ctx.fillText("No data yet — complete a screening to see trends", W / 2, H / 2);
    return;
  }

  const pad = { l: 60, r: 30, t: 25, b: 35 };
  const pw = W - pad.l - pad.r;
  const ph = H - pad.t - pad.b;

  // Y-axis labels
  const yLabels = ["High Risk", "Moderate", "Low Risk"];
  ctx.fillStyle = "#94a3b8";
  ctx.font = "11px Segoe UI";
  ctx.textAlign = "right";
  yLabels.forEach((label, i) => {
    const y = pad.t + (i / (yLabels.length - 1)) * ph;
    ctx.fillText(label, pad.l - 12, y + 4);
    ctx.strokeStyle = "#e2e8f0";
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(pad.l, y);
    ctx.lineTo(W - pad.r, y);
    ctx.stroke();
  });

  // Background gradient zones
  const gradient = ctx.createLinearGradient(0, pad.t, 0, pad.t + ph);
  gradient.addColorStop(0, "rgba(220, 38, 38, 0.06)");
  gradient.addColorStop(0.5, "rgba(217, 119, 6, 0.06)");
  gradient.addColorStop(1, "rgba(5, 150, 105, 0.06)");
  ctx.fillStyle = gradient;
  ctx.fillRect(pad.l, pad.t, pw, ph);

  // Line with gradient stroke
  if (scores.length >= 2) {
    const lineGrad = ctx.createLinearGradient(pad.l, 0, pad.l + pw, 0);
    scores.forEach((s, i) => {
      const stop = i / Math.max(scores.length - 1, 1);
      const color = s > 0.6 ? "#dc2626" : s > 0.3 ? "#d97706" : "#059669";
      lineGrad.addColorStop(stop, color);
    });

    ctx.strokeStyle = lineGrad;
    ctx.lineWidth = 3;
    ctx.lineJoin = "round";
    ctx.lineCap = "round";
    ctx.beginPath();
    scores.forEach((s, i) => {
      const x = pad.l + (i / Math.max(scores.length - 1, 1)) * pw;
      const y = pad.t + (1 - s) * ph;
      if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
    });
    ctx.stroke();

    // Fill area under line
    const areaGrad = ctx.createLinearGradient(0, pad.t, 0, pad.t + ph);
    areaGrad.addColorStop(0, "rgba(220, 38, 38, 0.08)");
    areaGrad.addColorStop(0.5, "rgba(217, 119, 6, 0.05)");
    areaGrad.addColorStop(1, "rgba(5, 150, 105, 0.02)");
    ctx.lineTo(pad.l + pw, pad.t + ph);
    ctx.lineTo(pad.l, pad.t + ph);
    ctx.closePath();
    ctx.fillStyle = areaGrad;
    ctx.fill();
  }

  // Points
  scores.forEach((s, i) => {
    const x = pad.l + (i / Math.max(scores.length - 1, 1)) * pw;
    const y = pad.t + (1 - s) * ph;
    const color = s > 0.6 ? "#dc2626" : s > 0.3 ? "#d97706" : "#059669";

    // Glow
    ctx.beginPath();
    ctx.arc(x, y, 6, 0, Math.PI * 2);
    ctx.fillStyle = color + "30";
    ctx.fill();

    // Point
    ctx.beginPath();
    ctx.arc(x, y, 4, 0, Math.PI * 2);
    ctx.fillStyle = "#fff";
    ctx.fill();
    ctx.strokeStyle = color;
    ctx.lineWidth = 2;
    ctx.stroke();
  });

  // X-axis labels
  ctx.fillStyle = "#94a3b8";
  ctx.font = "11px Segoe UI";
  ctx.textAlign = "center";
  scores.forEach((_, i) => {
    const x = pad.l + (i / Math.max(scores.length - 1, 1)) * pw;
    ctx.fillText("Session " + (i + 1), x, H - 8);
  });

  // Annotation if multiple sessions
  if (scores.length >= 2 && scores[scores.length - 1] < scores[0]) {
    const midIdx = Math.floor(scores.length / 2);
    const mx = pad.l + (midIdx / Math.max(scores.length - 1, 1)) * pw;
    const my = pad.t + (1 - scores[midIdx]) * ph - 20;
    ctx.fillStyle = "#2563eb";
    ctx.font = "bold 10px Segoe UI";
    ctx.textAlign = "center";
    ctx.fillText("▼ Therapy Initiated", mx, my);
  }
}

function drawDashTrend() {
  const scores = allSessions.map(s => s.fused_score || 0);
  drawGradientChart("dashTrendCanvas", scores);
}

// ══════════════════════════════════════════════════════════════════════
//  PROFILE PAGE
// ══════════════════════════════════════════════════════════════════════
function refreshProfile() {
  const sessEl = document.getElementById("profileSessions");
  if (sessEl) sessEl.textContent = allSessions.length;

  if (!latestResult) return;
  const d = latestResult;
  const state = d.screening.state;

  // Status badge
  const badgeEl = document.getElementById("profileStatusBadge");
  if (state === "CLINICAL_REVIEW") {
    badgeEl.innerHTML = '<span class="status-pill clinical">⚠ Clinical Review</span>';
  } else if (state === "MONITOR") {
    badgeEl.innerHTML = '<span class="status-pill review">⚡ Under Review</span>';
  } else {
    badgeEl.innerHTML = '<span class="status-pill low-risk">✓ Low Risk</span>';
  }

  // Clinical info
  const clinEl = document.getElementById("profileClinicalInfo");
  let clinHtml = "";
  if (d.clinical) {
    clinHtml += `<p><strong>Assessment:</strong> ${d.clinical.assessment}</p>`;
    if (d.clinical.observations && d.clinical.observations.length) {
      clinHtml += "<p><strong>Observations:</strong></p><ul>";
      d.clinical.observations.forEach(o => { clinHtml += `<li>${o}</li>`; });
      clinHtml += "</ul>";
    }
    clinHtml += `<p style="margin-top:8px"><strong>Recommendation:</strong> ${d.clinical.recommendation}</p>`;
  }
  if (d.screening.flagged_modalities && d.screening.flagged_modalities.length) {
    clinHtml += `<p style="margin-top:8px"><strong>Flagged modalities:</strong> ${d.screening.flagged_modalities.join(", ")}</p>`;
  }
  if (d.screening.cross_modal_agreement != null) {
    clinHtml += `<p style="font-size:12px; color:var(--text-muted); margin-top:4px">Cross-modal agreement: ${(d.screening.cross_modal_agreement * 100).toFixed(1)}%</p>`;
  }
  clinEl.innerHTML = clinHtml || '<p class="empty-state-sm">No clinical data available.</p>';

  // Risk overview cards
  let social = "—", behavior = "—", comm = "—";
  if (d.domain_scores) {
    social = Math.round((d.domain_scores.social_sum / 5) * 100);
    behavior = Math.round((d.domain_scores.behavior_sum / 2) * 100);
    comm = Math.round((d.domain_scores.communication_sum / 3) * 100);
  } else if (d.modality_scores) {
    if (d.modality_scores.face != null) social = Math.round(d.modality_scores.face * 100);
    if (d.modality_scores.behavior != null) behavior = Math.round(d.modality_scores.behavior * 100);
    if (d.modality_scores.eye_tracking != null) comm = Math.round(d.modality_scores.eye_tracking * 100);
  }
  document.getElementById("profSocial").textContent = social;
  document.getElementById("profBehavior").textContent = behavior;
  document.getElementById("profComm").textContent = comm;

  // Assessment history table
  const histEl = document.getElementById("profileHistory");
  if (allSessions.length === 0) {
    histEl.innerHTML = '<div class="empty-state">No assessment history available yet.</div>';
    return;
  }

  let html = `<table class="data-table">
    <thead><tr><th>#</th><th>Date</th><th>Screening Outcome</th><th>Fused Score</th><th>Recommendation</th></tr></thead><tbody>`;
  allSessions.forEach((s, i) => {
    const st = s.screening.state;
    const riskCls = st === "CLINICAL_REVIEW" ? "badge-red" : st === "MONITOR" ? "badge-yellow" : "badge-green";
    html += `<tr>
      <td>S${i + 1}</td>
      <td>${new Date().toLocaleDateString()}</td>
      <td><span class="badge ${riskCls}">${st.replace("_", " ")}</span></td>
      <td>${((s.fused_score || 0) * 100).toFixed(1)}%</td>
      <td style="font-size:12px; max-width:250px; overflow:hidden; text-overflow:ellipsis;">${s.clinical ? s.clinical.recommendation : "—"}</td>
    </tr>`;
  });
  html += "</tbody></table>";
  histEl.innerHTML = html;
}

// ══════════════════════════════════════════════════════════════════════
//  THERAPY PAGE
// ══════════════════════════════════════════════════════════════════════
// Tab switching
document.querySelectorAll(".therapy-tab").forEach(tab => {
  tab.addEventListener("click", () => {
    document.querySelectorAll(".therapy-tab").forEach(t => t.classList.remove("active"));
    tab.classList.add("active");
    refreshTherapy();
  });
});

function refreshTherapy() {
  const el = document.getElementById("therapyItems");
  if (!latestResult || !latestResult.therapy) {
    el.innerHTML = '<div class="empty-state">Run a screening session to generate therapy recommendations.</div>';
    return;
  }

  const t = latestResult.therapy;
  const activeDomain = document.querySelector(".therapy-tab.active").dataset.domain;

  if (!t.plan || t.plan.length === 0) {
    el.innerHTML = '<div class="empty-state">No interventions recommended at this time.</div>';
    return;
  }

  let html = "";
  t.plan.forEach((item, i) => {
    const priority = t.priorities[i] || "Low";
    const itemLower = item.toLowerCase();

    // Filter by domain
    if (activeDomain !== "all") {
      const isSocial = itemLower.includes("social") || itemLower.includes("peer") || itemLower.includes("engagement") || itemLower.includes("play");
      const isComm = itemLower.includes("speech") || itemLower.includes("communicat") || itemLower.includes("language") || itemLower.includes("verbal");
      const isMotor = itemLower.includes("motor") || itemLower.includes("movement") || itemLower.includes("sensory") || itemLower.includes("physical");

      if (activeDomain === "social" && !isSocial) return;
      if (activeDomain === "communication" && !isComm) return;
      if (activeDomain === "motor" && !isMotor) return;
    }

    html += `
      <div class="therapy-checklist-item">
        <div class="therapy-checkbox" onclick="this.classList.toggle('checked')"></div>
        <div class="therapy-check-body">
          <div class="therapy-check-title">
            ${item.split(":")[0] || item}
            <span class="priority-badge ${priority.toLowerCase()}">${priority}</span>
          </div>
          <div class="therapy-check-desc">${item.includes(":") ? item.split(":").slice(1).join(":").trim() : "AI-recommended intervention based on screening indicators."}</div>
        </div>
      </div>`;
  });

  if (!html) {
    html = '<div class="empty-state">No items match this category. Try "All Areas".</div>';
  }

  el.innerHTML = html;
}

// ══════════════════════════════════════════════════════════════════════
//  PROGRESS PAGE
// ══════════════════════════════════════════════════════════════════════
async function refreshProgress() {
  try {
    const r = await fetch(API + "/api/history");
    const d = await r.json();
    drawGradientChart("trendCanvas", d.score_history || []);
    showProgressStats(d);
    showSessionTable(d);
    showParentNotes(d);
  } catch {
    document.getElementById("progressStats").innerHTML = '<div class="empty-state">Cannot reach API.</div>';
  }
}

function showProgressStats(d) {
  const el = document.getElementById("progressStats");
  const scores = d.score_history || [];
  if (scores.length === 0) {
    el.innerHTML = '<div class="empty-state">No sessions recorded yet.</div>';
    return;
  }
  const avg = scores.reduce((a, b) => a + b, 0) / scores.length;
  const last = scores[scores.length - 1];
  const trend = scores.length >= 2 ? scores[scores.length - 1] - scores[scores.length - 2] : 0;
  const trendLabel = trend > 0.05 ? "↑ Increasing" : trend < -0.05 ? "↓ Decreasing" : "→ Stable";

  el.innerHTML = `
    <div class="stat-box"><div class="stat-val">${scores.length}</div><div class="stat-label">Sessions</div></div>
    <div class="stat-box"><div class="stat-val">${(avg * 100).toFixed(1)}%</div><div class="stat-label">Average Risk</div></div>
    <div class="stat-box"><div class="stat-val">${(last * 100).toFixed(1)}%</div><div class="stat-label">Latest Score</div></div>
    <div class="stat-box"><div class="stat-val">${trendLabel}</div><div class="stat-label">Trend</div></div>`;
}

function showSessionTable(d) {
  const el = document.getElementById("sessionTable");
  const scores = d.score_history || [];
  const modHist = d.modality_history || [];

  if (scores.length === 0) {
    el.innerHTML = '<div class="empty-state">No session data.</div>';
    return;
  }

  let html = `<table class="data-table"><thead><tr><th>Session</th><th>Fused Score</th><th>Modalities</th><th>Risk Level</th></tr></thead><tbody>`;
  scores.forEach((s, i) => {
    const state = s >= 0.6 ? "CLINICAL_REVIEW" : s >= 0.3 ? "MONITOR" : "LOW_RISK";
    const riskCls = state === "CLINICAL_REVIEW" ? "badge-red" : state === "MONITOR" ? "badge-yellow" : "badge-green";
    let modStr = "";
    if (modHist[i]) {
      modStr = Object.entries(modHist[i]).map(([k, v]) => `${k}: ${(v * 100).toFixed(0)}%`).join(", ");
    }
    html += `<tr><td>S${i + 1}</td><td>${(s * 100).toFixed(1)}%</td><td style="font-size:12px; color:var(--text-secondary)">${modStr}</td><td><span class="badge ${riskCls}">${state.replace("_", " ")}</span></td></tr>`;
  });
  html += "</tbody></table>";
  el.innerHTML = html;
}

function showParentNotes(d) {
  const el = document.getElementById("parentNotes");
  const scores = d.score_history || [];

  if (scores.length === 0) {
    el.innerHTML = '<p class="empty-state-sm">Notes will appear here after screening sessions are completed.</p>';
    return;
  }

  let html = "";
  if (latestResult && latestResult.clinical) {
    html += `<div class="parent-note-item">${latestResult.clinical.recommendation}</div>`;
  }
  if (latestResult && latestResult.monitoring) {
    html += `<div class="parent-note-item">Trajectory: ${latestResult.monitoring.trajectory}. ${latestResult.monitoring.alert}</div>`;
  }
  if (scores.length >= 2) {
    const trend = scores[scores.length - 1] - scores[scores.length - 2];
    if (trend < -0.05) {
      html += `<div class="parent-note-item">Positive trend observed — risk scores are decreasing across sessions.</div>`;
    } else if (trend > 0.05) {
      html += `<div class="parent-note-item">Risk scores have increased since last session. Consider scheduling follow-up.</div>`;
    } else {
      html += `<div class="parent-note-item">Scores are stable. Continue monitoring at regular intervals.</div>`;
    }
  }
  html += `<div class="parent-note-item">Total screening sessions completed: ${scores.length}</div>`;

  el.innerHTML = html;
}

// ══════════════════════════════════════════════════════════════════════
//  REPORTS PAGE
// ══════════════════════════════════════════════════════════════════════
function refreshReport() {
  const el = document.getElementById("reportContent");
  const exportBtn = document.getElementById("exportBtn");

  if (!latestResult) {
    el.innerHTML = '<div class="empty-state">No report data available yet.</div>';
    exportBtn.disabled = true;
    return;
  }

  const d = latestResult;
  const lines = [];
  lines.push("═════════════════════════════════════════════");
  lines.push("  AUTISM AI SCREENING REPORT");
  lines.push("  AutismCare Multi-Modal Platform v4.0");
  lines.push("═════════════════════════════════════════════");
  lines.push("");
  lines.push(`Date: ${new Date().toLocaleString()}`);
  lines.push(`Sessions completed: ${allSessions.length}`);
  lines.push("");
  lines.push("── RISK CLASSIFICATION ──────────────────────");
  lines.push(`State: ${d.screening.state}`);
  lines.push(`Fused Score: ${(d.fused_score * 100).toFixed(1)}%`);
  if (d.screening.cross_modal_agreement != null) {
    lines.push(`Cross-modal Agreement: ${(d.screening.cross_modal_agreement * 100).toFixed(1)}%`);
  }
  lines.push("");

  lines.push("── MODALITY SCORES ─────────────────────────");
  if (d.modality_scores) {
    for (const [m, v] of Object.entries(d.modality_scores)) {
      if (v != null) lines.push(`  ${m.padEnd(16)}: ${(v * 100).toFixed(1)}%`);
    }
  }
  if (d.domain_scores) {
    lines.push("");
    lines.push("── DOMAIN BREAKDOWN ────────────────────────");
    lines.push(`  Social:        ${d.domain_scores.social_sum}`);
    lines.push(`  Communication: ${d.domain_scores.communication_sum}`);
    lines.push(`  Behavior:      ${d.domain_scores.behavior_sum}`);
    lines.push(`  Total:         ${d.domain_scores.total_score}/10`);
  }
  lines.push("");

  lines.push("── CLINICAL ASSESSMENT ─────────────────────");
  lines.push(d.clinical.assessment);
  if (d.clinical.observations && d.clinical.observations.length) {
    lines.push("");
    lines.push("Observations:");
    d.clinical.observations.forEach(o => lines.push("  • " + o));
  }
  lines.push("");
  lines.push("Recommendation: " + d.clinical.recommendation);
  lines.push("");

  lines.push("── THERAPY PLAN ────────────────────────────");
  if (d.therapy && d.therapy.plan) {
    d.therapy.plan.forEach((item, i) => {
      const p = d.therapy.priorities[i] || "Low";
      lines.push(`  [${p}] ${item}`);
    });
  }
  lines.push("");

  lines.push("── MONITORING ──────────────────────────────");
  if (d.monitoring) {
    lines.push(`Trend: ${(d.monitoring.trend * 100).toFixed(1)}%`);
    lines.push(`Trajectory: ${d.monitoring.trajectory}`);
    lines.push(`Alert: ${d.monitoring.alert}`);
  }
  lines.push("");
  lines.push("═════════════════════════════════════════════");
  lines.push("  Generated by AutismCare AI Platform");
  lines.push("  For research purposes only — not a diagnosis.");
  lines.push("═════════════════════════════════════════════");

  el.textContent = lines.join("\n");
  exportBtn.disabled = false;
}

document.getElementById("exportBtn").addEventListener("click", () => {
  const text = document.getElementById("reportContent").textContent;
  const blob = new Blob([text], { type: "text/plain" });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = "autism_screening_report.txt";
  a.click();
  URL.revokeObjectURL(url);
});

// ══════════════════════════════════════════════════════════════════════
//  INIT
// ══════════════════════════════════════════════════════════════════════
console.log("AutismCare AI Frontend v5.0 loaded");
