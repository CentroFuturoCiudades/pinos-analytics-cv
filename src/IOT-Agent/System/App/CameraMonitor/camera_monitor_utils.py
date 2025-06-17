import requests

def update_camera_states(states: dict, url="http://localhost:5152/update"):
    """
    Update the camera states in the monitor.
    
    Args:
        states (dict): Dictionary with camera names as keys and their states as values.
    """
    response = requests.post(url, json=states)