document.addEventListener('DOMContentLoaded', () => {
    const stats = {
        total: document.getElementById('count-total'),
        ambulance: document.getElementById('count-ambulance'),
        car: document.getElementById('count-car'),
        bike: document.getElementById('count-bike'),
        truck: document.getElementById('count-truck'),
        auto: document.getElementById('count-auto'),
        density: document.getElementById('density-val'),
        speed: document.getElementById('avg-speed'),
        timer: document.getElementById('signal-timer')
    };

    const lights = {
        red: document.getElementById('light-red'),
        yellow: document.getElementById('light-yellow'),
        green: document.getElementById('light-green')
    };

    const alertsContainer = document.getElementById('alerts-container');
    const chartContainer = document.getElementById('chart');

    let currentTimer = 30;
    let isAmbulanceDetected = false;
    let vehicleCounts = { car: 0, bike: 0, truck: 0, auto: 0, ambulance: 0 };

    // Initialize Prediction Chart
    for (let i = 0; i < 12; i++) {
        const bar = document.createElement('div');
        bar.className = 'chart-bar';
        bar.style.height = Math.random() * 80 + 20 + '%';
        chartContainer.appendChild(bar);
    }

    // Update Prediction Chart periodically
    setInterval(() => {
        const bars = document.querySelectorAll('.chart-bar');
        bars.forEach(bar => {
            bar.style.height = Math.random() * 80 + 20 + '%';
        });
    }, 3000);

    // Simulation Logic
    function updateSimulation() {
        if (currentTimer > 0) {
            currentTimer--;
            stats.timer.innerText = currentTimer;
            
            if (currentTimer < 5 && lights.green.classList.contains('active')) {
                setLight('yellow');
            }
        } else {
            if (lights.red.classList.contains('active')) {
                setLight('green');
                currentTimer = isAmbulanceDetected ? 45 : 30;
                isAmbulanceDetected = false; // Reset after priority pass
            } else {
                setLight('red');
                currentTimer = 20;
            }
        }

        // Randomly simulate vehicle detections
        if (Math.random() > 0.7) {
            const types = ['car', 'bike', 'truck', 'auto', 'ambulance'];
            const type = types[Math.floor(Math.random() * types.length)];
            
            vehicleCounts[type]++;
            updateStats();

            if (type === 'ambulance') {
                triggerAmbulancePriority();
            }
        }
    }

    function setLight(color) {
        Object.values(lights).forEach(l => l.classList.remove('active'));
        lights[color].classList.add('active');
        
        const status = document.getElementById('signal-status');
        if (color === 'red') {
            status.innerText = 'WAITING';
            status.style.color = 'var(--error)';
        } else if (color === 'green') {
            status.innerText = 'FLOW OPTIMIZED';
            status.style.color = 'var(--success)';
        } else {
            status.innerText = 'PREPARING';
            status.style.color = 'var(--warning)';
        }
    }

    function updateStats() {
        stats.car.innerText = vehicleCounts.car;
        stats.bike.innerText = vehicleCounts.bike;
        stats.truck.innerText = vehicleCounts.truck;
        stats.auto.innerText = vehicleCounts.auto;
        stats.ambulance.innerText = vehicleCounts.ambulance;
        
        const total = Object.values(vehicleCounts).reduce((a, b) => a + b, 0);
        stats.total.innerText = total;

        // Density logic
        if (total > 50) stats.density.innerText = 'HIGH';
        else if (total > 20) stats.density.innerText = 'MEDIUM';
        else stats.density.innerText = 'LOW';

        stats.speed.innerText = Math.floor(Math.random() * 20 + 30) + ' km/h';
    }

    function triggerAmbulancePriority() {
        if (isAmbulanceDetected) return; // Prevent spam
        isAmbulanceDetected = true;
        addAlert('EMERGENCY: Ambulance Detected! Overriding Signal', 'alert-emergency');
        
        // Immediate green light override
        if (!lights.green.classList.contains('active')) {
            setLight('green');
            currentTimer = 20; // Give time to pass
            stats.timer.innerText = currentTimer;
        }
    }

    function addAlert(msg, className) {
        console.log(`[Alert] ${msg}`);
        const div = document.createElement('div');
        div.className = `alert-item ${className}`;
        div.innerText = msg;
        alertsContainer.prepend(div);
        
        if (alertsContainer.children.length > 8) {
            alertsContainer.lastElementChild.remove();
        }
    }

    function debugLog(msg) {
        const timestamp = new Date().toLocaleTimeString();
        console.log(`[Debug ${timestamp}] ${msg}`);
        // Optionally show in a hidden debug panel if needed
    }

    // AI Detection Integration
    const imageUpload = document.getElementById('image-upload');
    const liveBtn = document.getElementById('live-btn');
    const demoBtn = document.getElementById('demo-btn');
    const videoFeed = document.getElementById('video-feed');
    
    let isLive = false;
    let isDemo = false;
    let activeMode = 'none'; // Track active mode: 'none', 'demo', 'live', 'upload'

    // Helper to update prediction chart with real data
    function updatePredictionsFromData(totalCount) {
        const bars = document.querySelectorAll('.chart-bar');
        bars.forEach((bar, index) => {
            // Simple trend: next intervals fluctuate around current total
            const trend = totalCount * (1 + (Math.random() * 0.4 - 0.2));
            bar.style.height = Math.min(Math.max(trend * 2, 10), 100) + '%';
        });
    }

    imageUpload.addEventListener('change', async (e) => {
        const file = e.target.files[0];
        if (!file) return;

        const isVideo = file.type.startsWith('video/');
        isDemo = false;
        isLive = false;
        activeMode = 'upload';
        stopWebcam(); // Reset any existing stream
        updateModeUI('upload');

        const mediaURL = URL.createObjectURL(file);
        if (isVideo) {
            videoFeed.innerHTML = `
                <video id="uploaded-video" autoplay loop muted style="width: 100%; height: 100%; object-fit: contain; border-radius: 20px;">
                    <source src="${mediaURL}" type="${file.type}">
                </video>
                <div class="overlay-status">AI Vision Initializing...</div>
                <canvas id="live-canvas" style="display: none;"></canvas>
            `;
            // Start real-time processing loop for this video
            setTimeout(() => startLiveProcessing('uploaded-video'), 500);
            addAlert('VIDEO LOADED: Starting AI Vision Monitor', 'alert-info');
        } else {
            videoFeed.innerHTML = `
                <img src="${mediaURL}" style="width: 100%; height: 100%; object-fit: contain; border-radius: 20px;">
                <div class="overlay-status">Static Image Analysis...</div>
            `;
            
            // Single detection for static image
            const formData = new FormData();
            formData.append('image', file);
            try {
                const response = await fetch('/api/detect', { method: 'POST', body: formData });
                const data = await response.json();
                if (data.image) {
                    videoFeed.innerHTML = `<img src="data:image/jpeg;base64,${data.image}" style="width: 100%; height: 100%; object-fit: contain; border-radius: 20px;">`;
                    vehicleCounts = data.counts;
                    updateStats();
                    updatePredictionsFromData(data.counts.total);
                    addAlert(`AI ANALYSIS: Detected ${data.counts.total} vehicles`, 'alert-info');
                }
            } catch (err) { console.error(err); }
        }
    });

    // Unified Live Mode Logic
    let liveInterval = null;

    liveBtn.addEventListener('click', async () => {
        isLive = !isLive;
        isDemo = false;
        activeMode = isLive ? 'live' : 'none';
        stopWebcam(); // Clean shutdown before switching
        updateModeUI(isLive ? 'live' : 'none');

        if (isLive) {
            try {
                const stream = await navigator.mediaDevices.getUserMedia({ video: true });
                videoFeed.innerHTML = `
                    <video id="webcam" autoplay muted playsinline style="width: 100%; height: 100%; object-fit: cover; border-radius: 20px; opacity: 0; position: absolute; top:0; left:0;"></video>
                    <canvas id="live-canvas" style="display: none;"></canvas>
                    <div class="overlay-status">AI Vision Initializing...</div>
                `;
                const videoEl = document.getElementById('webcam');
                videoEl.srcObject = stream;
                
                // Play to ensure we get frames
                videoEl.play();
                
                setTimeout(() => startLiveProcessing('webcam'), 1000);
                addAlert('LIVE MODE: Streaming from Webcam', 'alert-success');
            } catch (err) {
                addAlert('Live Error: Webcam access denied', 'alert-emergency');
                isLive = false;
                updateModeUI('none');
            }
        }
    });

    async function startLiveProcessing(videoElementId) {
        if (liveInterval) clearInterval(liveInterval);
        console.log(`Starting real-time monitoring on: ${videoElementId}`);
        
        liveInterval = setInterval(async () => {
            // Stop logic
            if (activeMode === 'live' && videoElementId !== 'webcam') return;
            if (activeMode === 'upload' && videoElementId !== 'uploaded-video') return;
            if (activeMode === 'none' || activeMode === 'demo') {
                clearInterval(liveInterval);
                return;
            }

            const video = document.getElementById(videoElementId);
            const canvas = document.getElementById('live-canvas');
            if (!video || !canvas) {
                console.warn('Monitoring elements missing, retrying...');
                return;
            }

            // Ensure video is playing and has dimensions
            if (video.paused || video.ended || video.videoWidth === 0) return;

            const context = canvas.getContext('2d');
            canvas.width = video.videoWidth;
            canvas.height = video.videoHeight;
            context.drawImage(video, 0, 0, canvas.width, canvas.height);

            canvas.toBlob(async (blob) => {
                if (!blob) return;
                const formData = new FormData();
                formData.append('image', blob, 'frame.jpg');

                try {
                    const response = await fetch('/api/detect', { method: 'POST', body: formData });
                    if (!response.ok) throw new Error(`HTTP ${response.status}`);
                    
                    const data = await response.json();
                    
                    if (data.counts && data.image) {
                        vehicleCounts = data.counts;
                        updateStats();
                        updatePredictionsFromData(data.counts.total);
                        
                        // Persistent AI Frame
                        let aiImg = document.getElementById('ai-vision-frame');
                        if (!aiImg) {
                            debugLog('Creating AI overlay image');
                            videoFeed.insertAdjacentHTML('beforeend', `<img id="ai-vision-frame" style="position: absolute; top:0; left:0; width: 100%; height: 100%; object-fit: contain; border-radius: 20px; z-index: 5; pointer-events: none;">`);
                            aiImg = document.getElementById('ai-vision-frame');
                        }
                        aiImg.src = "data:image/jpeg;base64," + data.image;
                        aiImg.style.display = 'block';
                        debugLog(`AI Frame updated: ${data.counts.total} vehicles`);

                        if (data.counts.ambulance > 0) triggerAmbulancePriority();
                    } else if (data.error) {
                        console.error('AI Error:', data.error);
                    }
                } catch (err) { 
                    console.error('Detection Loop Error:', err); 
                }
            }, 'image/jpeg', 0.5);
        }, 800); // 800ms for stability
    }

    function stopWebcam() {
        const video = document.getElementById('webcam');
        if (video && video.srcObject) {
            video.srcObject.getTracks().forEach(track => track.stop());
            videoFeed.innerHTML = simulationPlaceholder;
        }
        if (liveInterval) {
            clearInterval(liveInterval);
            liveInterval = null;
        }
    }

    function updateModeUI(mode) {
        [liveBtn, demoBtn].forEach(b => b.classList.remove('active'));
        if (mode === 'live') liveBtn.classList.add('active');
        if (mode === 'demo') demoBtn.classList.add('active');
    }

    const simulationPlaceholder = `
        <svg width="60" height="60" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1" stroke-linecap="round" stroke-linejoin="round" style="opacity: 0.3"><rect x="2" y="2" width="20" height="20" rx="2.18" ry="2.18"></rect><line x1="7" y1="2" x2="7" y2="22"></line><line x1="17" y1="2" x2="17" y2="22"></line><line x1="2" y1="12" x2="22" y2="12"></line><line x1="2" y1="7" x2="7" y2="7"></line><line x1="2" y1="17" x2="7" y2="17"></line><line x1="17" y1="17" x2="22" y2="17"></line><line x1="17" y1="7" x2="22" y2="7"></line></svg>
        <p style="color: var(--text-dim); margin-top: 1rem;">Waiting for Video Input...</p>
    `;

    demoBtn.addEventListener('click', () => {
        isDemo = !isDemo;
        isLive = false;
        activeMode = isDemo ? 'demo' : 'none';
        stopWebcam();
        updateModeUI(isDemo ? 'demo' : 'none');
        
        if (isDemo) {
            addAlert('SIMULATION: Running Demo Logic...', 'alert-info');
        } else {
            addAlert('SIMULATION: Stopped', 'alert-info');
        }
    });

    let simInterval = setInterval(() => {
        // ALWAYS update the signal, regardless of mode
        if (currentTimer > 0) {
            currentTimer--;
            stats.timer.innerText = currentTimer;
            
            // Auto-yellow transition
            if (currentTimer < 5 && lights.green.classList.contains('active')) {
                setLight('yellow');
            }
        } else {
            // Switch lights
            if (lights.red.classList.contains('active')) {
                setLight('green');
                // DYNAMIC DURATION based on AI detection
                const total = parseInt(stats.total.innerText) || 0;
                if (isAmbulanceDetected) {
                    currentTimer = 45;
                    isAmbulanceDetected = false;
                } else if (total > 30) {
                    currentTimer = 60; // Heavy traffic gets more time
                    addAlert('AI LOGIC: Heavy traffic detected, extending green phase', 'alert-info');
                } else if (total < 5 && total > 0) {
                    currentTimer = 15; // Light traffic changes faster
                    addAlert('AI LOGIC: Low traffic detected, shortening green phase', 'alert-info');
                } else {
                    currentTimer = 30;
                }
            } else {
                setLight('red');
                currentTimer = 20;
            }
        }

        // Only do random simulation if in pure Demo mode
        if (isDemo) {
            if (Math.random() > 0.7) {
                const types = ['car', 'bike', 'truck', 'auto'];
                const type = types[Math.floor(Math.random() * types.length)];
                vehicleCounts[type]++;
                updateStats();
            }
        }
    }, 1000);
});
