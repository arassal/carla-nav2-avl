// AVROS Console - Joystick WebSocket Client

const WS_URL = `wss://${location.host}/ws`;

// Cache DOM elements
const statusEl = document.getElementById('status');
const valuesEl = document.getElementById('values');
const estopBtn = document.getElementById('estop');
const autoBtn = document.getElementById('autobtn');
const joystickZone = document.getElementById('joystick-zone');
const modeButtons = document.querySelectorAll('.modes button');

let ws = null;
let estop = false;
let mode = 'D';
let autonomous = false;
let joystickX = 0;
let joystickY = 0;
let sendInterval = null;

function clamp(value, min, max) {
    return Math.max(min, Math.min(max, value));
}

// ===== WEBSOCKET =====
function connect() {
    ws = new WebSocket(WS_URL);

    ws.onopen = () => {
        statusEl.textContent = 'Connected';
        statusEl.className = '';
        send({ type: 'mode', value: mode });
        send({ type: 'estop', value: estop });
        send({ type: 'autonomous', value: autonomous });
        startSending();
    };

    ws.onclose = () => {
        statusEl.textContent = 'Disconnected';
        statusEl.className = 'disconnected';
        stopSending();
        setTimeout(connect, 1000);
    };

    ws.onmessage = (e) => {
        const data = JSON.parse(e.data);
        if (data.error) {
            statusEl.textContent = data.error;
            statusEl.className = 'disconnected';
            stopSending();
            return;
        }
        valuesEl.textContent = `T: ${data.t.toFixed(2)} | S: ${data.s.toFixed(2)} | B: ${data.b.toFixed(2)}`;
    };
}

function send(data) {
    if (ws && ws.readyState === WebSocket.OPEN) {
        ws.send(JSON.stringify(data));
    }
}

function startSending() {
    if (sendInterval) return;
    sendInterval = setInterval(() => {
        if (autonomous) {
            // Hand control to Nav2: stop asserting actuator_command so it goes
            // stale and /cmd_vel takes over. Keepalive keeps telemetry flowing.
            send({ type: 'keepalive' });
        } else {
            send({ type: 'control', x: joystickX, y: joystickY });
        }
    }, 50);
}

function stopSending() {
    if (sendInterval) {
        clearInterval(sendInterval);
        sendInterval = null;
    }
}

// ===== JOYSTICK =====
function resetJoystick() {
    joystickX = 0;
    joystickY = 0;
}

// Pointer-events based joystick: no external library, no fixed anchor point.
// A drag starting anywhere in the zone tracks relative to where it started,
// so there's no dependency on network reachability (previously loaded from
// an external CDN, which silently failed with no visible error whenever the
// device had no general internet access) and no requirement to touch one
// exact pixel (previously nipplejs 'static' mode only responded to drags
// starting on its fixed anchor point).
function createJoystick() {
    const maxDist = 75;
    const handle = document.createElement('div');
    handle.style.cssText = [
        'position:absolute',
        'left:50%',
        'top:50%',
        'width:90px',
        'height:90px',
        'margin-left:-45px',
        'margin-top:-45px',
        'border-radius:50%',
        'background:#1976d2',
        'box-shadow:0 0 24px rgba(25,118,210,0.6)',
        'pointer-events:none',
        'touch-action:none'
    ].join(';');
    joystickZone.appendChild(handle);

    // 2026-08-19: touch-action is NOT an inherited CSS property. index.html
    // sets it on <body>, which does NOT cover this element -- so mobile
    // browsers were free to claim a drag here as a pan/scroll gesture. That
    // gesture claim is what spontaneously revoked pointer capture mid-drag
    // (see the lostpointercapture note below). Set it explicitly on the zone.
    joystickZone.style.touchAction = 'none';
    joystickZone.style.overscrollBehavior = 'none';

    let activePointerId = null;
    let originX = 0; // px, zone-relative — set to wherever the drag started
    let originY = 0;

    function updateHandle() {
        handle.style.left = `${originX}px`;
        handle.style.top = `${originY}px`;
        handle.style.transform = `translate(${joystickX * maxDist}px, ${-joystickY * maxDist}px)`;
    }

    function recenterHandle() {
        const rect = joystickZone.getBoundingClientRect();
        originX = rect.width / 2;
        originY = rect.height / 2;
        updateHandle();
    }

    joystickZone.addEventListener('pointerdown', (event) => {
        activePointerId = event.pointerId;
        joystickZone.setPointerCapture(event.pointerId);
        const rect = joystickZone.getBoundingClientRect();
        originX = event.clientX - rect.left;
        originY = event.clientY - rect.top;
        joystickX = 0;
        joystickY = 0;
        updateHandle();
    });

    // 2026-08-19: listeners moved from joystickZone to window.
    //
    // Bug this fixes (measured: stick registered ~150 ms of a 30 s hold, then
    // read 0.0 for the rest while the finger was still down):
    //   - 'lostpointercapture' was wired to release(). Mobile browsers revoke
    //     pointer capture spontaneously mid-drag; that is NOT the finger
    //     lifting, but release() zeroed the stick anyway and never recovered,
    //     because subsequent pointermove events were filtered out by the
    //     activePointerId check that release() had just nulled. Handler
    //     deleted outright -- losing capture is not an end-of-touch signal.
    //   - Tracking on the zone alone also dropped input if the finger left the
    //     zone's bounds once capture was gone. window-level listeners keep
    //     tracking regardless of capture state or finger position.
    //
    // SAFETY: this makes stop-on-release MORE reliable, not less. pointerup
    // and pointercancel anywhere on the page now zero the stick, including
    // releases outside the zone that the old zone-scoped listener could miss.
    // Real end-of-touch is still exactly pointerup/pointercancel, and the
    // actuator's own command-freshness watchdog remains the backstop.
    window.addEventListener('pointermove', (event) => {
        if (event.pointerId !== activePointerId) return;
        const rect = joystickZone.getBoundingClientRect();
        const dx = (event.clientX - rect.left) - originX;
        const dy = (event.clientY - rect.top) - originY;
        joystickX = clamp(dx / maxDist, -1, 1);
        joystickY = clamp(-dy / maxDist, -1, 1);
        updateHandle();
    });

    function release(event) {
        // Ignore stray releases from a different finger than the active drag.
        if (event && activePointerId !== null && event.pointerId !== activePointerId) return;
        activePointerId = null;
        resetJoystick();
        recenterHandle();
    }

    window.addEventListener('pointerup', release);
    window.addEventListener('pointercancel', release);

    recenterHandle();
}

createJoystick();

// ===== E-STOP =====
estopBtn.addEventListener('click', () => {
    estop = !estop;
    estopBtn.classList.toggle('active', estop);
    estopBtn.textContent = estop ? 'E-STOP ACTIVE' : 'E-STOP';
    send({ type: 'estop', value: estop });
    // E-stop is not autonomous — drop the AUTO toggle (server forces it too).
    if (estop && autonomous) {
        autonomous = false;
        updateAutoBtn();
    }
});

// ===== AUTONOMOUS TOGGLE (IGVC §I.2 safety light) =====
function updateAutoBtn() {
    autoBtn.classList.toggle('active', autonomous);
    autoBtn.textContent = autonomous ? 'AUTO ON' : 'AUTO';
}

autoBtn.addEventListener('click', () => {
    autonomous = !autonomous;
    updateAutoBtn();
    send({ type: 'autonomous', value: autonomous });
});

// ===== MODE BUTTONS =====
modeButtons.forEach(btn => {
    btn.addEventListener('click', () => {
        modeButtons.forEach(b => b.classList.remove('active'));
        btn.classList.add('active');
        mode = btn.dataset.mode;
        send({ type: 'mode', value: mode });
    });
});

// ===== START =====
connect();
