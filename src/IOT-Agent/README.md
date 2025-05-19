# IOT-Agent

## Requirements
To download the Python requirements, run the following command:

```bash
  pip install -r requirements.txt
```

To use the IOT-Agent with GStreamer for video capture, it is necessary to install the following packages:

```bash
  apt-get install libgstreamer1.0-dev libgstreamer-plugins-base1.0-dev libgstreamer-plugins-bad1.0-dev gstreamer1.0-plugins-base gstreamer1.0-plugins-good gstreamer1.0-plugins-bad gstreamer1.0-plugins-ugly gstreamer1.0-libav gstreamer1.0-tools gstreamer1.0-x gstreamer1.0-alsa gstreamer1.0-gl gstreamer1.0-gtk3 gstreamer1.0-qt5 gstreamer1.0-pulseaudio libcairo2-dev libxt-dev libgirepository1.0-dev

```

The OpenCV option can be activated in the [RTSP-Recorder](System/App/RTSPRecorder/RTSPRecorder.py) file.

## Configuration
The credential and stream configurations are located in the [configuration](Public/config/system.release.standard.config.ini) file.

## Execution
To run the IOT-Agent, run the following command:

```bash
  python3 main.py
```