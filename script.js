document.addEventListener('DOMContentLoaded', () => {
    const directions = ['N', 'S', 'E', 'W'];
    const state = {
        counts: { N: 0, S: 0, E: 0, W: 0 },
        phase: 0, // 0: NS Straight, 1: NS Left, 2: EW Straight, 3: EW Left
        timer: 30,
        isDemo: false,
        activeMode: 'none',
        intervals: { N: null, S: null, E: null, W: null }
    };

    const elements = {
        timer: document.getElementById('signal-timer'),
        status: document.getElementById('signal-status'),
        total: document.getElementById('count-total'),
        density: document.getElementById('density-val'),
        forecast: document.getElementById('forecast-val'),
        chart: document.getElementById('chart'),
        alerts: document.getElementById('alerts-container'),
        lights: {
            ns: { red: document.getElementById('light-ns-red'), green: document.getElementById('light-ns-green') },
            ew: { red: document.getElementById('light-ew-red'), green: document.getElementById('light-ew-green') }
        }
    };

    // Initialize Directional Elements
    directions.forEach(d => {
        elements[`count-${d}-L`] = document.getElementById(`count-${d}-L`);
        elements[`count-${d}-S`] = document.getElementById(`count-${d}-S`);
        elements[`feed-${d}`] = document.getElementById(`video-feed-${d}`);
        elements[`upload-${d}`] = document.getElementById(`upload-${d}`);
        elements[`container-${d}`] = document.getElementById(`container-${d}`);
        elements[`flow-${d}`] = document.getElementById(`flow-${d}`);
        // Arrow Elements
        ['RED', 'L', 'S', 'R'].forEach(type => {
            elements[`signal-${d}-${type}`] = document.getElementById(`signal-${d}-${type}`);
        });
    });

    const phaseConfigs = [
        { name: 'NORTH Clearing (↑ ← → ↩)', arrows: { N: ['S', 'L', 'R'], S: [], E: [], W: [] }, pair: 'n' },
        { name: 'SOUTH Clearing (↑ ← → ↩)', arrows: { N: [], S: ['S', 'L', 'R'], E: [], W: [] }, pair: 's' },
        { name: 'EAST Clearing (↑ ← → ↩)', arrows: { N: [], S: [], E: ['S', 'L', 'R'], W: [] }, pair: 'e' },
        { name: 'WEST Clearing (↑ ← → ↩)', arrows: { N: [], S: [], E: [], W: ['S', 'L', 'R'] }, pair: 'w' }
    ];

    // --- LSTM Prediction Logic ---
    async function updateForecast() {
        try {
            let totalForecast = 0;
            const chartData = [0, 0, 0, 0, 0];

            for (const d of directions) {
                const res = await fetch(`/api/predict?direction=${d}`);
                const data = await res.json();
                if (data.forecast) {
                    totalForecast += data.forecast[0];
                    data.forecast.forEach((v, i) => chartData[i] += v);
                }
            }

            elements.forecast.innerText = Math.round(totalForecast / 4);
            elements.chart.innerHTML = '';
            chartData.forEach((val, i) => {
                const bar = document.createElement('div');
                bar.className = 'chart-bar';
                const height = Math.min(Math.max((val / 4 / 50) * 100, 5), 100);
                bar.style.height = height + '%';
                bar.title = `T+${i+1}h: ${Math.round(val/4)} vehicles`;
                elements.chart.appendChild(bar);
            });
        } catch (err) {
            console.error('Forecast Error:', err);
        }
    }

    // --- Signal Management Logic ---
    function updateSignals() {
        if (state.timer > 0) {
            state.timer--;
            elements.timer.innerText = state.timer;
        } else {
            state.phase = (state.phase + 1) % 4;
            const currentPhase = phaseConfigs[state.phase];
            
            const load = state.counts[currentPhase.pair.toUpperCase()] || 10;
            state.timer = Math.min(60, Math.max(15, Math.ceil(load * 2)));
            addAlert(`PHASE: ${currentPhase.name}`, 'alert-info');
        }

        const config = phaseConfigs[state.phase];
        
        // Update Side Indicators
        elements.lights.ns.green.classList.toggle('active', config.pair === 'n' || config.pair === 's');
        elements.lights.ns.red.classList.toggle('active', config.pair !== 'n' && config.pair !== 's');
        elements.lights.ew.green.classList.toggle('active', config.pair === 'e' || config.pair === 'w');
        elements.lights.ew.red.classList.toggle('active', config.pair !== 'e' && config.pair !== 'w');

        directions.forEach(d => {
            const activeArrows = config.arrows[d];
            const hasGreen = activeArrows.length > 0;

            // Update Physical Lenses
            elements[`signal-${d}-RED`].classList.toggle('active', !hasGreen);
            ['L', 'S', 'R'].forEach(type => {
                const isActive = activeArrows.includes(type);
                elements[`signal-${d}-${type}`].classList.toggle('active', isActive);
            });
            
            // Toggle Specific Flow Icons on Feed
            const flowContainer = elements[`flow-${d}`];
            const straightArrow = flowContainer.querySelector('.arrow-straight');
            const leftArrow = flowContainer.querySelector('.arrow-left');
            
            const hasStraight = activeArrows.includes('S');
            const hasLeft = activeArrows.includes('L');
            
            straightArrow.style.display = hasStraight ? 'block' : 'none';
            leftArrow.style.display = hasLeft ? 'block' : 'none';
            
            flowContainer.classList.toggle('active', hasGreen);
        });
    }

    // --- Video Processing Logic ---
    async function startProcessing(direction, file) {
        if (state.intervals[direction]) clearInterval(state.intervals[direction]);
        
        const videoURL = URL.createObjectURL(file);
        elements[`feed-${direction}`].innerHTML = `
            <video id="vid-${direction}" autoplay loop muted style="width: 100%; height: 100%; object-fit: cover;">
                <source src="${videoURL}" type="${file.type}">
            </video>
            <canvas id="canvas-${direction}" style="display: none;"></canvas>
            <img id="ai-${direction}" style="position: absolute; top:0; left:0; width: 100%; height: 100%; object-fit: cover; z-index: 2; pointer-events: none;">
        `;
        
        const video = document.getElementById(`vid-${direction}`);
        startProcessingLoop(direction, video);
    }

    // --- Data Handling ---
    function updateUI() {
        let total = 0;
        directions.forEach(d => {
            const laneTotal = state.counts[d];
            total += laneTotal;
        });
        elements.total.innerText = total;
        
        const density = total > 40 ? 'CRITICAL' : (total > 20 ? 'BUSY' : 'NORMAL');
        elements.density.innerText = density;
        elements.density.style.color = total > 40 ? 'var(--error)' : (total > 20 ? 'var(--warning)' : 'var(--success)');
    }

    function addAlert(msg, className) {
        const div = document.createElement('div');
        div.className = `alert-item ${className}`;
        div.innerText = msg;
        elements.alerts.prepend(div);
        if (elements.alerts.children.length > 5) elements.alerts.lastElementChild.remove();
    }

    // --- Event Listeners ---
    directions.forEach(d => {
        elements[`upload-${d}`].addEventListener('change', (e) => {
            const file = e.target.files[0];
            if (file) {
                state.isDemo = false;
                addAlert(`UPLOAD: Direction ${d} video active`, 'alert-info');
                startProcessing(d, file);
            }
        });
        
        // Prevent click bubble to trigger file input multiple times
        elements[`container-${d}`].addEventListener('click', (e) => {
            if (e.target.tagName !== 'INPUT') elements[`upload-${d}`].click();
        });
    });

    document.getElementById('bulk-upload').addEventListener('change', (e) => {
        const files = Array.from(e.target.files);
        files.forEach((file, i) => {
            if (i < 4) startProcessing(directions[i], file);
        });
    });

    document.getElementById('demo-btn').addEventListener('click', () => {
        state.isDemo = !state.isDemo;
        if (state.isDemo) {
            directions.forEach(d => {
                if (state.intervals[d]) clearInterval(state.intervals[d]);
                state.counts[d] = 10;
                elements[`feed-${d}`].innerHTML = `<p style="color: var(--primary);">SIMULATED FEED ACTIVE</p>`;
            });
            addAlert('SIMULATION: 4-Way Traffic Patterns Active', 'alert-success');
        }
    });

    document.getElementById('live-btn').addEventListener('click', () => {
        state.isDemo = false;
        addAlert('LIVE: Initializing system camera...', 'alert-info');
        startLiveStream('N'); // Default live to North feed
    });

    async function startLiveStream(direction) {
        try {
            const stream = await navigator.mediaDevices.getUserMedia({ video: true });
            if (state.intervals[direction]) clearInterval(state.intervals[direction]);
            
            elements[`feed-${direction}`].innerHTML = `
                <video id="vid-${direction}" autoplay playsinline style="width: 100%; height: 100%; object-fit: cover;"></video>
                <canvas id="canvas-${direction}" style="display: none;"></canvas>
                <img id="ai-${direction}" style="position: absolute; top:0; left:0; width: 100%; height: 100%; object-fit: cover; z-index: 2; pointer-events: none;">
            `;
            
            const video = document.getElementById(`vid-${direction}`);
            video.srcObject = stream;
            
            // Start detection loop for live stream
            startProcessingLoop(direction, video);
            addAlert('LIVE: North camera stream active', 'alert-success');
        } catch (err) {
            console.error('Webcam Error:', err);
            addAlert('LIVE ERROR: Camera access denied', 'alert-error');
        }
    }

    state.pending = { N: false, S: false, E: false, W: false };

    function startProcessingLoop(direction, video) {
        const canvas = document.getElementById(`canvas-${direction}`);
        const aiImg = document.getElementById(`ai-${direction}`);
        
        state.intervals[direction] = setInterval(async () => {
            if (video.paused || video.ended || video.videoWidth === 0 || state.pending[direction]) return;
            
            // Downscale for performance (640px width is sweet spot for speed vs clarity)
            const scale = 640 / video.videoWidth;
            canvas.width = 640;
            canvas.height = video.videoHeight * scale;
            
            const ctx = canvas.getContext('2d');
            ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
            
            state.pending[direction] = true;
            canvas.toBlob(async (blob) => {
                const formData = new FormData();
                formData.append('image', blob);
                formData.append('direction', direction);
                
                try {
                    const res = await fetch('/api/detect', { method: 'POST', body: formData });
                    const data = await res.json();
                    
                    const total = data.counts.total;
                    const left = Math.floor(total * 0.4);
                    const straight = total - left;
                    
                    state.counts[direction] = total;
                    elements[`count-${direction}-L`].innerText = left;
                    elements[`count-${direction}-S`].innerText = straight;
                    
                    if (data.image) aiImg.src = 'data:image/png;base64,' + data.image;
                    updateUI();
                } catch (err) { console.error(`AI Error (${direction}):`, err); }
                finally { state.pending[direction] = false; }
            }, 'image/png', 0.5); // Lower quality PNG for faster wire transfer
        }, 150); // ~7 FPS for smooth tracking
    }

    // --- Loops ---
    setInterval(updateSignals, 1000);
    state.intervals.demo = setInterval(() => {
        if (state.isDemo) {
            directions.forEach(d => {
                const change = Math.floor(Math.random() * 5) - 2;
                state.counts[d] = Math.max(0, state.counts[d] + change);
                
                // Lane Distribution Simulation
                const total = state.counts[d];
                const left = Math.floor(total * 0.4);
                const straight = total - left;
                
                elements[`count-${d}-L`].innerText = left;
                elements[`count-${d}-S`].innerText = straight;
            });
            updateUI();
        }
    }, 2000);

    setInterval(updateForecast, 15000);
    updateForecast();
});
