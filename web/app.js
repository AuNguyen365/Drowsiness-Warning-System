// DOM Elements
const videoEl = document.getElementById('webcam');
const canvasEl = document.getElementById('output-canvas');
const ctx = canvasEl.getContext('2d');
const spinnerEl = document.getElementById('loading-spinner');
const statusBadge = document.getElementById('system-status');

const metricEarEl = document.getElementById('metric-ear');
const metricFramesEl = document.getElementById('metric-frames');
const earProgressFill = document.getElementById('ear-progress-fill');
const thresholdMarker = document.getElementById('threshold-marker');
const threshLabelEl = document.getElementById('current-threshold-label');

const rangeThresh = document.getElementById('range-threshold');
const valThresh = document.getElementById('val-threshold');
const rangeConsec = document.getElementById('range-consec');
const valConsec = document.getElementById('val-consec');

const btnCalibrate = document.getElementById('btn-calibrate');
const btnResetCalibrate = document.getElementById('btn-reset-calibration');
const calibrationWizard = document.getElementById('calibration-wizard');
const wizardInstruction = document.getElementById('wizard-instruction');
const wizardProgressBar = document.getElementById('wizard-progress-bar');
const btnTestAlarm = document.getElementById('btn-test-alarm');
const btnToggleMonitor = document.getElementById('btn-toggle-monitor');

const chartCanvas = document.getElementById('ear-chart');
const chartCtx = chartCanvas.getContext('2d');

// System Parameters & State
let earThreshold = 0.25;
let consecLimit = 15;
let consecCounter = 0;
let isDrowsy = false;
let earHistory = new Array(100).fill(0.25);
let systemActive = false;

// Monitoring Control State
let cameraInstance = null;
let localStream = null;
let isMonitoringActive = true;

// Audio Alert System State
let alarmActive = false;
let audioCtx = null;
let beepInterval = null;
let speechInterval = null;

// Calibration Wizard State
let calibration = {
    active: false,
    step: 0, // 0 = idle, 1 = open eyes, 2 = closed eyes
    openEarSum: 0,
    openEarCount: 0,
    closedEarSum: 0,
    closedEarCount: 0,
    startTime: 0,
    durationMs: 3000 // 3 seconds per step
};

// MediaPipe Indices
const LEFT_EYE_INDICES = [362, 385, 387, 263, 373, 380];
const RIGHT_EYE_INDICES = [33, 160, 158, 133, 153, 144];

// --- 1. INITIALIZATION & CONFIGURATION ---

// Load saved settings from localStorage
if (localStorage.getItem('earThreshold')) {
    earThreshold = parseFloat(localStorage.getItem('earThreshold'));
    rangeThresh.value = earThreshold;
    valThresh.textContent = earThreshold.toFixed(2);
}
if (localStorage.getItem('consecLimit')) {
    consecLimit = parseInt(localStorage.getItem('consecLimit'));
    rangeConsec.value = consecLimit;
    valConsec.textContent = consecLimit;
}
updateUIThresholdMarker();

// Handle range inputs
rangeThresh.addEventListener('input', (e) => {
    earThreshold = parseFloat(e.target.value);
    valThresh.textContent = earThreshold.toFixed(2);
    localStorage.setItem('earThreshold', earThreshold);
    updateUIThresholdMarker();
});

rangeConsec.addEventListener('input', (e) => {
    consecLimit = parseInt(e.target.value);
    valConsec.textContent = consecLimit;
    localStorage.setItem('consecLimit', consecLimit);
});

// Update threshold position on the progress track
function updateUIThresholdMarker() {
    // Map threshold (0.05 to 0.4) to percentage (0% to 100%)
    const minEar = 0.05;
    const maxEar = 0.4;
    const percentage = ((earThreshold - minEar) / (maxEar - minEar)) * 100;
    thresholdMarker.style.left = `${Math.min(Math.max(percentage, 0), 100)}%`;
    threshLabelEl.textContent = `Ngưỡng: ${earThreshold.toFixed(2)}`;
}

// Ensure audio context is initialized on user interaction
function initAudioContext() {
    if (!audioCtx) {
        audioCtx = new (window.AudioContext || window.webkitAudioContext)();
    }
    if (audioCtx.state === 'suspended') {
        audioCtx.resume();
    }
}

// --- 2. AUDIO ALARM SYSTEM ---

function startAlarm() {
    if (alarmActive) return;
    alarmActive = true;
    document.body.classList.add('alarm-active');
    initAudioContext();
    
    const alarmType = document.querySelector('input[name="sound-type"]:checked').value;
    
    if (alarmType === 'voice') {
        triggerVoiceAlarmLoop();
    } else {
        triggerBeepAlarmLoop();
    }
}

function stopAlarm() {
    if (!alarmActive) return;
    alarmActive = false;
    document.body.classList.remove('alarm-active');
    
    // Stop beep loop
    if (beepInterval) {
        clearInterval(beepInterval);
        beepInterval = null;
    }
    
    // Stop voice loop
    if (speechInterval) {
        clearInterval(speechInterval);
        speechInterval = null;
    }
    if (window.speechSynthesis) {
        window.speechSynthesis.cancel();
    }
}

// Synthesize Beep Sound using Web Audio API
function playSingleBeep(frequency, duration) {
    if (!audioCtx) return;
    try {
        const osc = audioCtx.createOscillator();
        const gainNode = audioCtx.createGain();
        
        osc.type = 'sawtooth';
        osc.frequency.setValueAtTime(frequency, audioCtx.currentTime);
        
        gainNode.gain.setValueAtTime(0.15, audioCtx.currentTime);
        // Exponential decay
        gainNode.gain.exponentialRampToValueAtTime(0.001, audioCtx.currentTime + duration);
        
        osc.connect(gainNode);
        gainNode.connect(audioCtx.destination);
        
        osc.start();
        osc.stop(audioCtx.currentTime + duration);
    } catch (e) {
        console.error("Lỗi phát âm thanh beep:", e);
    }
}

function triggerBeepAlarmLoop() {
    playSingleBeep(2000, 0.4);
    beepInterval = setInterval(() => {
        playSingleBeep(2000, 0.4);
    }, 500);
}

// Synthesize Vietnamese Voice Alarm using SpeechSynthesis API
function speakVietnamese() {
    if (!window.speechSynthesis) return;
    
    // Cancel currently speaking voices to prevent overlapping queues
    window.speechSynthesis.cancel();
    
    const utterance = new SpeechSynthesisUtterance("Cảnh báo! Bạn đang buồn ngủ, hãy tập trung lái xe!");
    utterance.lang = 'vi-VN';
    utterance.volume = 1.0;
    utterance.rate = 1.0;
    
    // Find Vietnamese voice if available
    const voices = window.speechSynthesis.getVoices();
    const viVoice = voices.find(v => v.lang.includes('vi') || v.lang.includes('VI'));
    if (viVoice) {
        utterance.voice = viVoice;
    }
    
    window.speechSynthesis.speak(utterance);
}

function triggerVoiceAlarmLoop() {
    speakVietnamese();
    speechInterval = setInterval(() => {
        if (!window.speechSynthesis.speaking) {
            speakVietnamese();
        }
    }, 2500);
}

// Test Alarm Button toggle
btnTestAlarm.addEventListener('click', () => {
    initAudioContext();
    if (btnTestAlarm.textContent === "Chạy thử còi báo động") {
        btnTestAlarm.textContent = "Dừng chạy thử";
        btnTestAlarm.classList.remove('btn-secondary');
        btnTestAlarm.classList.add('btn-primary');
        startAlarm();
    } else {
        btnTestAlarm.textContent = "Chạy thử còi báo động";
        btnTestAlarm.classList.remove('btn-primary');
        btnTestAlarm.classList.add('btn-secondary');
        stopAlarm();
    }
});

// Load voices once they are populated (needed for Chrome/Edge)
if (window.speechSynthesis) {
    window.speechSynthesis.onvoiceschanged = () => {};
}

// --- 3. CALIBRATION WIZARD STATE MACHINE ---

btnCalibrate.addEventListener('click', () => {
    initAudioContext();
    startCalibrationWizard();
});

btnResetCalibrate.addEventListener('click', () => {
    earThreshold = 0.25;
    rangeThresh.value = earThreshold;
    valThresh.textContent = earThreshold.toFixed(2);
    localStorage.removeItem('earThreshold');
    updateUIThresholdMarker();
    alert("Đã đặt lại ngưỡng EAR mặc định là 0.25!");
});

function startCalibrationWizard() {
    calibration.active = true;
    calibration.step = 1;
    calibration.openEarSum = 0;
    calibration.openEarCount = 0;
    calibration.closedEarSum = 0;
    calibration.closedEarCount = 0;
    calibration.startTime = Date.now();
    
    calibrationWizard.classList.remove('hidden');
    btnCalibrate.disabled = true;
    
    playSingleBeep(1000, 0.2);
}

function processCalibration(currentEar) {
    const elapsed = Date.now() - calibration.startTime;
    
    if (calibration.step === 1) {
        // Step 1: Open Eyes
        const progress = (elapsed / calibration.durationMs) * 100;
        wizardProgressBar.style.width = `${Math.min(progress, 100)}%`;
        wizardInstruction.innerHTML = `Bước 1/2: Hãy <strong>MỞ MẮT</strong> tự nhiên... (${Math.ceil((calibration.durationMs - elapsed)/1000)}s)`;
        
        calibration.openEarSum += currentEar;
        calibration.openEarCount++;
        
        if (elapsed >= calibration.durationMs) {
            calibration.step = 2;
            calibration.startTime = Date.now();
            playSingleBeep(1200, 0.3); // Alert tone transition
        }
    } else if (calibration.step === 2) {
        // Step 2: Closed Eyes
        const progress = (elapsed / calibration.durationMs) * 100;
        wizardProgressBar.style.width = `${Math.min(progress, 100)}%`;
        wizardInstruction.innerHTML = `Bước 2/2: Hãy <strong>NHẮM MẮT</strong> tự nhiên... (${Math.ceil((calibration.durationMs - elapsed)/1000)}s)`;
        
        calibration.closedEarSum += currentEar;
        calibration.closedEarCount++;
        
        if (elapsed >= calibration.durationMs) {
            // End Calibration
            const avgOpen = calibration.openEarSum / calibration.openEarCount;
            const avgClosed = calibration.closedEarSum / calibration.closedEarCount;
            
            // Optimal Threshold is exactly the midpoint between open and closed eye EAR values
            earThreshold = parseFloat(((avgOpen + avgClosed) / 2.0).toFixed(3));
            
            // Validation check
            if (isNaN(earThreshold) || earThreshold < 0.05 || earThreshold > 0.4) {
                alert("Cân chỉnh thất bại. Vui lòng đảm bảo khuôn mặt của bạn đối diện camera và đủ ánh sáng!");
                earThreshold = 0.25;
            } else {
                rangeThresh.value = earThreshold;
                valThresh.textContent = earThreshold.toFixed(2);
                localStorage.setItem('earThreshold', earThreshold);
                updateUIThresholdMarker();
                playSingleBeep(1500, 0.5); // Success tone
            }
            
            // Reset Wizard State
            calibration.active = false;
            calibration.step = 0;
            calibrationWizard.classList.add('hidden');
            btnCalibrate.disabled = false;
        }
    }
}

// --- 4. EAR MATH & CHART RENDERING ---

function getDistance(p1, p2) {
    return Math.sqrt(Math.pow(p1.x - p2.x, 2) + Math.pow(p1.y - p2.y, 2));
}

function calculateEyeEar(landmarks, indices) {
    const p1 = landmarks[indices[0]];
    const p2 = landmarks[indices[1]];
    const p3 = landmarks[indices[2]];
    const p4 = landmarks[indices[3]];
    const p5 = landmarks[indices[4]];
    const p6 = landmarks[indices[5]];
    
    const d2_6 = getDistance(p2, p6);
    const d3_5 = getDistance(p3, p5);
    const d1_4 = getDistance(p1, p4);
    
    if (d1_4 === 0) return 0.0;
    return (d2_6 + d3_5) / (2.0 * d1_4);
}

// Custom High-Performance Canvas Chart
function updateChart(currentEar) {
    // Add value and roll history
    earHistory.push(currentEar);
    if (earHistory.length > 100) {
        earHistory.shift();
    }
    
    const w = chartCanvas.width = chartCanvas.clientWidth;
    const h = chartCanvas.height = chartCanvas.clientHeight;
    
    chartCtx.clearRect(0, 0, w, h);
    
    // Draw background grid lines
    chartCtx.strokeStyle = 'rgba(255, 255, 255, 0.05)';
    chartCtx.lineWidth = 1;
    for (let yVal = 0.1; yVal <= 0.4; yVal += 0.1) {
        const y = h - ((yVal - 0.05) / (0.4 - 0.05)) * h;
        chartCtx.beginPath();
        chartCtx.moveTo(0, y);
        chartCtx.lineTo(w, y);
        chartCtx.stroke();
    }
    
    // Draw threshold boundary line
    const threshY = h - ((earThreshold - 0.05) / (0.4 - 0.05)) * h;
    chartCtx.strokeStyle = 'rgba(255, 184, 0, 0.5)';
    chartCtx.lineWidth = 2;
    chartCtx.setLineDash([6, 4]);
    chartCtx.beginPath();
    chartCtx.moveTo(0, threshY);
    chartCtx.lineTo(w, threshY);
    chartCtx.stroke();
    chartCtx.setLineDash([]);
    
    // Draw EAR history plot path
    chartCtx.beginPath();
    const step = w / 99;
    
    for (let i = 0; i < earHistory.length; i++) {
        const x = i * step;
        // Normalize EAR value (clamped between 0.05 and 0.4) to chart height
        const val = Math.min(Math.max(earHistory[i], 0.05), 0.4);
        const y = h - ((val - 0.05) / (0.4 - 0.05)) * h;
        
        if (i === 0) {
            chartCtx.moveTo(x, y);
        } else {
            chartCtx.lineTo(x, y);
        }
    }
    
    // Glowing gradient effect for plot
    chartCtx.strokeStyle = isDrowsy ? '#ff3838' : '#00d2ff';
    chartCtx.lineWidth = 3;
    chartCtx.stroke();
}

// --- 5. MEDIAPIPE FACE MESH LOOP CALLBACK ---

function onResults(results) {
    if (!systemActive) {
        systemActive = true;
        spinnerEl.classList.add('hidden');
    }
    
    // Reset canvas dimensions to match display container
    canvasEl.width = videoEl.videoWidth || 640;
    canvasEl.height = videoEl.videoHeight || 480;
    
    ctx.clearRect(0, 0, canvasEl.width, canvasEl.height);
    
    // Draw raw video frame mirrored
    ctx.save();
    ctx.translate(canvasEl.width, 0);
    ctx.scale(-1, 1);
    ctx.drawImage(results.image, 0, 0, canvasEl.width, canvasEl.height);
    ctx.restore();
    
    if (results.multiFaceLandmarks && results.multiFaceLandmarks.length > 0) {
        const landmarks = results.multiFaceLandmarks[0];
        
        // Calculate EAR
        const leftEar = calculateEyeEar(landmarks, LEFT_EYE_INDICES);
        const rightEar = calculateEyeEar(landmarks, RIGHT_EYE_INDICES);
        const avgEar = (leftEar + rightEar) / 2.0;
        
        // Update stats UI
        metricEarEl.textContent = avgEar.toFixed(3);
        
        // Map EAR to progress bar width
        const minEar = 0.05;
        const maxEar = 0.4;
        const barPercent = Math.min(Math.max(((avgEar - minEar) / (maxEar - minEar)) * 100, 0), 100);
        earProgressFill.style.width = `${barPercent}%`;
        
        // Change progress bar color based on status
        if (avgEar < earThreshold) {
            earProgressFill.style.background = 'var(--color-danger)';
        } else {
            earProgressFill.style.background = 'linear-gradient(90deg, var(--color-danger) 0%, var(--color-success) 100%)';
        }
        
        // Render Eye Landmarks and Outlines
        drawEyeOverlay(landmarks, LEFT_EYE_INDICES, avgEar < earThreshold);
        drawEyeOverlay(landmarks, RIGHT_EYE_INDICES, avgEar < earThreshold);
        
        // Process calibration if active
        if (calibration.active) {
            processCalibration(avgEar);
        }
        
        // Drowsiness Alert Counter Logic
        if (avgEar < earThreshold) {
            consecCounter++;
            if (consecCounter >= consecLimit) {
                if (!isDrowsy) {
                    isDrowsy = true;
                    statusBadge.textContent = "NGUY HIỂM: BUỒN NGỦ!";
                    statusBadge.className = "status-badge status-drowsy";
                }
                startAlarm();
            } else {
                if (!isDrowsy) {
                    statusBadge.textContent = "CẢNH BÁO";
                    statusBadge.className = "status-badge status-warning";
                }
            }
        } else {
            consecCounter = 0;
            if (isDrowsy) {
                isDrowsy = false;
                stopAlarm();
            }
            statusBadge.textContent = "TỈNH TÁO";
            statusBadge.className = "status-badge status-normal";
        }
        
        metricFramesEl.textContent = `${consecCounter} / ${consecLimit}`;
        updateChart(avgEar);
        
    } else {
        // No face detected
        consecCounter = 0;
        if (isDrowsy) {
            isDrowsy = false;
            stopAlarm();
        }
        statusBadge.textContent = "KHÔNG PHÁT HIỆN KHUÔN MẶT";
        statusBadge.className = "status-badge status-warning";
        metricEarEl.textContent = "0.000";
        metricFramesEl.textContent = "0 / " + consecLimit;
        earProgressFill.style.width = "0%";
        updateChart(0.05);
    }
}

// Draw eye landmark points and connector contour lines
function drawEyeOverlay(landmarks, indices, activeDanger) {
    const color = activeDanger ? '#ff3838' : '#00f076';
    ctx.strokeStyle = color;
    ctx.fillStyle = color;
    ctx.lineWidth = 2;
    
    ctx.beginPath();
    
    for (let i = 0; i < indices.length; i++) {
        // Mirror calculation coordinates for drawing overlay matching the video canvas coordinates
        const lm = landmarks[indices[i]];
        const x = (1.0 - lm.x) * canvasEl.width; // Flip horizontally
        const y = lm.y * canvasEl.height;
        
        if (i === 0) {
            ctx.moveTo(x, y);
        } else {
            ctx.lineTo(x, y);
        }
        
        // Draw keypoint dots
        ctx.fillRect(x - 2, y - 2, 4, 4);
    }
    
    ctx.closePath();
    ctx.stroke();
}

// --- 6. WEBCAM & CAMERA INITIALIZATION ---

async function startSystem() {
    try {
        spinnerEl.classList.remove('hidden');
        spinnerEl.innerHTML = '<div class="spinner"></div><p>Đang khởi động camera và tải mô hình Face Mesh...</p>';
        
        localStream = await navigator.mediaDevices.getUserMedia({
            video: {
                width: { ideal: 640 },
                height: { ideal: 480 },
                facingMode: "user"
            },
            audio: false
        });
        
        videoEl.srcObject = localStream;
        
        // Setup MediaPipe Face Mesh
        const faceMesh = new FaceMesh({
            locateFile: (file) => `https://cdn.jsdelivr.net/npm/@mediapipe/face_mesh/${file}`
        });
        
        faceMesh.setOptions({
            maxNumFaces: 1,
            refineLandmarks: true,
            minDetectionConfidence: 0.5,
            minTrackingConfidence: 0.5
        });
        
        faceMesh.onResults(onResults);
        
        // Set up the camera helper loop
        cameraInstance = new Camera(videoEl, {
            onFrame: async () => {
                if (isMonitoringActive) {
                    await faceMesh.send({ image: videoEl });
                }
            },
            width: 640,
            height: 480
        });
        
        cameraInstance.start();
        
    } catch (err) {
        console.error("Lỗi truy cập Camera hoặc tải mô hình:", err);
        alert("Không thể truy cập camera của bạn. Vui lòng cho phép quyền truy cập camera trong trình duyệt và làm mới trang!");
        statusBadge.textContent = "LỖI CAMERA";
        statusBadge.className = "status-badge status-drowsy";
        spinnerEl.innerHTML = "<p>Lỗi: Không tìm thấy Camera hoặc bị từ chối truy cập.</p>";
    }
}

function stopSystem() {
    stopAlarm();
    
    if (localStream) {
        try {
            localStream.getTracks().forEach(track => track.stop());
        } catch (e) {
            console.error("Lỗi khi giải phóng camera tracks:", e);
        }
        localStream = null;
    }
    
    if (cameraInstance) {
        try {
            cameraInstance.stop();
        } catch (e) {
            console.error("Lỗi khi dừng camera instance:", e);
        }
        cameraInstance = null;
    }
    
    videoEl.srcObject = null;
    ctx.clearRect(0, 0, canvasEl.width, canvasEl.height);
    
    // Reset stats
    consecCounter = 0;
    metricEarEl.textContent = "0.000";
    metricFramesEl.textContent = "0 / " + consecLimit;
    earProgressFill.style.width = "0%";
    updateChart(0.05);
    
    statusBadge.textContent = "ĐÃ TẮT GIÁM SÁT";
    statusBadge.className = "status-badge status-warning";
    
    systemActive = false;
    spinnerEl.classList.add('hidden');
}

// Toggle Monitoring Button
btnToggleMonitor.addEventListener('click', () => {
    if (isMonitoringActive) {
        // Turn off
        isMonitoringActive = false;
        btnToggleMonitor.textContent = "Bật giám sát";
        btnToggleMonitor.classList.remove('btn-primary');
        btnToggleMonitor.style.backgroundColor = 'var(--color-success)';
        btnToggleMonitor.style.color = '#000000';
        stopSystem();
    } else {
        // Turn on
        isMonitoringActive = true;
        btnToggleMonitor.textContent = "Tắt giám sát";
        btnToggleMonitor.style.backgroundColor = '';
        btnToggleMonitor.style.color = '';
        btnToggleMonitor.classList.add('btn-primary');
        startSystem();
    }
});

// Trigger load
window.addEventListener('DOMContentLoaded', () => {
    // Start webcam and load models
    startSystem();
});
