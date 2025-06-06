import eventlet
eventlet.monkey_patch()

from flask import Flask, render_template_string
from flask_socketio import SocketIO
import threading
import time

class CameraMonitor:
    def __init__(self, host='0.0.0.0', port=5000):
        self.host = host
        self.port = port
        self.camera_states = {}
        self.app = Flask(__name__)
        self.socketio = SocketIO(self.app, async_mode='eventlet')
        self.server_thread = None
        self.running = False
        self._server_started = threading.Event()  # Add this event

        self._setup_routes()

    def _setup_routes(self):
        @self.app.route('/')
        def index():
            return render_template_string(self._html_template(), camera_states=self.camera_states)

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
        self.camera_states.update(new_states)
        self.socketio.emit('update', self.camera_states)

    def get_url(self):
        return f"http://{self.host}:{self.port}"
    
    def start(self):
        """Start the camera monitor server asynchronously"""
        if self.running:
            return
        
        def run_server():
            self._server_started.set()  # Signal that server is starting
            self.socketio.run(self.app, host=self.host, port=self.port, debug=False)

        self.server_thread = threading.Thread(target=run_server, daemon=False)
        self.server_thread.start()
        self.running = True
        
        # Wait for server to actually start instead of just sleeping
        self._server_started.wait(timeout=2.0)
        time.sleep(0.1)  # Small additional delay for socket binding}

    def stop(self):
        """Stop the camera monitor server"""
        if self.running:
            self.running = False

    def is_running(self):
        """Check if the server is running"""
        return self.running and self.server_thread and self.server_thread.is_alive()