import eventlet
eventlet.monkey_patch()

from flask import Flask, render_template_string, request, jsonify
from flask_socketio import SocketIO
import threading
import time

class CameraMonitor:
    def __init__(self, host='0.0.0.0', port=5000, api_port=5152, timeout=60):
        self.host = host
        self.port = port
        self.api_port = api_port
        self.timeout = timeout  # seconds
        self.camera_states = {}
        self.last_update = {}  # Track last update time per camera
        self.app = Flask(__name__)
        self.socketio = SocketIO(self.app, async_mode='eventlet')
        self.server_thread = None
        self.api_thread = None
        self.timeout_thread = None
        self.running = False
        self._server_started = threading.Event()
        self._api_started = threading.Event()

        self._setup_routes()
        self._setup_api()

    def _setup_routes(self):
        @self.app.route('/')
        def index():
            return render_template_string(self._html_template(), camera_states=self.camera_states)

    def _setup_api(self):
        api_app = Flask("CameraMonitorAPI")

        @api_app.route('/update', methods=['POST'])
        def update_camera_states():
            data = request.get_json()
            if not isinstance(data, dict):
                return jsonify({'error': 'Invalid payload, must be a JSON object'}), 400
            self.update_camera_states(data)
            return jsonify({'status': 'updated', 'states': self.camera_states})

        self.api_app = api_app

    def _html_template(self):
        return """
        <!DOCTYPE html>
        <html>
        <head>
            <title>Camera Monitor</title>
            <script src="https://cdn.socket.io/4.3.2/socket.io.min.js"></script>
            <style>
                body { font-family: 'Segoe UI', sans-serif; background: #f4f4f4; margin: 0; padding: 0; }
                h1 { text-align: center; padding: 1rem; margin: 0; background: #333; color: white; }
                .container { display: flex; flex-wrap: wrap; justify-content: center; padding: 2rem; gap: 1rem; }
                .camera-card { background: white; border-radius: 12px; padding: 1rem 1.5rem; box-shadow: 0 4px 8px rgba(0,0,0,0.1); width: 200px; text-align: center; }
                .camera-name { font-size: 1.2rem; margin-bottom: 0.5rem; }
                .status { padding: 0.5rem 1rem; border-radius: 20px; font-weight: bold; display: inline-block; text-transform: capitalize; }
                .active { background-color: #d4edda; color: #155724; }
                .inactive { background-color: #e2e3e5; color: #6c757d; }
                .recording { background-color: #f8d7da; color: #721c24; }
                .dead { background-color: #343a40; color: #ffc107; }
            </style>
        </head>
        <body>
            <h1>Camera Monitor</h1>
            <div class="container" id="camera-list">
                {% for cam, state in camera_states.items() %}
                <div class="camera-card" id="{{ cam }}">
                    <div class="camera-name">{{ cam }}</div>
                    <div class="status {{ state }}">{{ state }}</div>
                </div>
                {% endfor %}
            </div>

            <script>
                const socket = io();
                socket.on('update', function (data) {
                    for (const [cam, state] of Object.entries(data)) {
                        const el = document.getElementById(cam);
                        if (el) {
                            el.querySelector('.status').textContent = state;
                            el.querySelector('.status').className = 'status ' + state;
                        }
                    }
                });
            </script>
        </body>
        </html>
        """

    def update_camera_states(self, new_states: dict):
        now = time.time()
        for cam, state in new_states.items():
            self.camera_states[cam] = state
            self.last_update[cam] = now
        self.socketio.emit('update', self.camera_states)

    def _timeout_watcher(self):
        while self.running:
            now = time.time()
            changed = False
            for cam in list(self.camera_states.keys()):
                last = self.last_update.get(cam)
                if last is not None and self.camera_states[cam] != "dead":
                    if now - last > self.timeout:
                        self.camera_states[cam] = "dead"
                        changed = True
            if changed:
                self.socketio.emit('update', self.camera_states)
            time.sleep(1)

    def get_url(self):
        return f"http://{self.host}:{self.port}"

    def start(self):
        """Start the camera monitor server and API server asynchronously"""
        if self.running:
            return

        def run_server():
            self._server_started.set()
            self.socketio.run(self.app, host=self.host, port=self.port, debug=False)

        def run_api():
            self._api_started.set()
            self.api_app.run(host=self.host, port=self.api_port, debug=False, use_reloader=False)

        self.running = True
        self.server_thread = threading.Thread(target=run_server, daemon=True)
        self.api_thread = threading.Thread(target=run_api, daemon=True)
        self.timeout_thread = threading.Thread(target=self._timeout_watcher, daemon=True)
        self.server_thread.start()
        self.api_thread.start()
        self.timeout_thread.start()

        self._server_started.wait(timeout=2.0)
        self._api_started.wait(timeout=2.0)
        time.sleep(0.1)

    def stop(self):
        """Stop the camera monitor server"""
        self.running = False

    def is_running(self):
        return self.running and self.server_thread and self.server_thread.is_alive()

