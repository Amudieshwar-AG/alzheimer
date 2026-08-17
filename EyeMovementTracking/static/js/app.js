document.addEventListener('DOMContentLoaded', () => {
    // Elements
    const btnToggleOverlays = document.getElementById('btnToggleOverlays');
    const lblOverlays = document.getElementById('lblOverlays');
    const btnCaptureSnapshot = document.getElementById('btnCaptureSnapshot');
    const blurStatusPill = document.getElementById('blurStatusPill');

    const imgLeftEye = document.getElementById('imgLeftEye');
    const imgLeftSeg = document.getElementById('imgLeftSeg');
    const imgRightEye = document.getElementById('imgRightEye');
    const imgRightSeg = document.getElementById('imgRightSeg');

    const valLeftEar = document.getElementById('valLeftEar');
    const valLeftRadius = document.getElementById('valLeftRadius');
    const valRightEar = document.getElementById('valRightEar');
    const valRightRadius = document.getElementById('valRightRadius');

    const valAvgEar = document.getElementById('valAvgEar');
    const barAvgEar = document.getElementById('barAvgEar');
    const valGazeL = document.getElementById('valGazeL');
    const valGazeR = document.getElementById('valGazeR');
    const valFixationBadge = document.getElementById('valFixationBadge');

    let blurEnabled = true;
    let overlaysEnabled = true;

    // Fetch Telemetry JSON every 150ms
    async function fetchTelemetry() {
        try {
            const res = await fetch('/api/telemetry');
            if (!res.ok) return;

            const data = await res.json();

            // Eye Inspection Images (base64)
            if (data.left_eye_img) imgLeftEye.src = 'data:image/jpeg;base64,' + data.left_eye_img;
            if (data.left_seg_img) imgLeftSeg.src = 'data:image/jpeg;base64,' + data.left_seg_img;
            if (data.right_eye_img) imgRightEye.src = 'data:image/jpeg;base64,' + data.right_eye_img;
            if (data.right_seg_img) imgRightSeg.src = 'data:image/jpeg;base64,' + data.right_seg_img;

            // Metrics
            valLeftEar.textContent = data.left_ear !== undefined ? data.left_ear.toFixed(2) : '0.00';
            valLeftRadius.textContent = (data.left_pupil_radius || 0) + ' px';
            valRightEar.textContent = data.right_ear !== undefined ? data.right_ear.toFixed(2) : '0.00';
            valRightRadius.textContent = (data.right_pupil_radius || 0) + ' px';

            const avgEar = data.avg_ear !== undefined ? data.avg_ear : 0.30;
            valAvgEar.textContent = avgEar.toFixed(2);
            barAvgEar.style.width = Math.min(100, Math.max(5, avgEar * 250)) + '%';

            if (data.left_norm) {
                valGazeL.textContent = `(${data.left_norm[0].toFixed(2)}, ${data.left_norm[1].toFixed(2)})`;
            }
            if (data.right_norm) {
                valGazeR.textContent = `(${data.right_norm[0].toFixed(2)}, ${data.right_norm[1].toFixed(2)})`;
            }

            // Status Badge
            if (avgEar < 0.18) {
                valFixationBadge.textContent = 'BLINK DETECTED';
                valFixationBadge.className = 'badge-status status-saccade';
                valFixationBadge.style.background = 'rgba(244, 63, 94, 0.2)';
                valFixationBadge.style.color = '#f43f5e';
            } else {
                valFixationBadge.textContent = 'TRACKING ACTIVE';
                valFixationBadge.className = 'badge-status status-fixation';
                valFixationBadge.style.background = 'rgba(16, 185, 129, 0.15)';
                valFixationBadge.style.color = '#10b981';
            }

        } catch (err) {
            console.debug('Telemetry poll notice:', err);
        }
    }

    setInterval(fetchTelemetry, 150);



    // Toggle Overlays Handler
    btnToggleOverlays.addEventListener('click', async () => {
        try {
            const res = await fetch('/api/toggle_overlays', { method: 'POST', headers: {'Content-Type': 'application/json'} });
            const data = await res.json();
            overlaysEnabled = data.overlays_enabled;
            lblOverlays.textContent = 'Overlays: ' + (overlaysEnabled ? 'ON' : 'OFF');
        } catch (err) {
            console.error('Error toggling overlays:', err);
        }
    });

    // Capture Snapshot Handler
    btnCaptureSnapshot.addEventListener('click', async () => {
        try {
            btnCaptureSnapshot.disabled = true;
            btnCaptureSnapshot.innerHTML = '<span class="icon">📸</span> Saving...';
            const res = await fetch('/api/capture', { method: 'POST' });
            const data = await res.json();
            alert('🎉 ' + data.message);
        } catch (err) {
            alert('Snapshot request failed.');
        } finally {
            btnCaptureSnapshot.disabled = false;
            btnCaptureSnapshot.innerHTML = '<span class="icon">📸</span> Capture Snapshot';
        }
    });
});
