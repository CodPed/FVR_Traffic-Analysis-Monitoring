#     FVR_Traffic Analysis Monitoring with NVIDIA DeepStream

Real-time traffic analytics system powered by NVIDIA DeepStream and Python.  
This project performs vehicle detection and classification, estimates instantaneous speed, and identifies wrong-way driving violations. It is designed for high-performance NVIDIA GPUs and can be used as a scalable base for Smart Cities and road safety monitoring applications.

## Features

- Vehicle detection using NVIDIA DeepStream primary inference
- Object tracking with the DeepStream multi-object tracker
- Vehicle type classification with an optional Secondary GIE (SGIE)
- Instantaneous speed estimation using two virtual ROI lines
- Wrong-way driving detection based on vehicle movement direction
- On-screen display with object ID, vehicle type, speed, and alert state
- Console alerts for normal and wrong-way traffic events

## Repository structure

```text
.
├── README.md
├── traffic_analytics_deepstream.py
├── config_analytics.txt
├── scripts/
│   ├── run_container.sh
│   └── install_dependencies.sh
└── .gitignore
```

## Requirements

- Ubuntu/Linux host with X11 display support
- NVIDIA GPU
- NVIDIA driver compatible with Docker GPU passthrough
- Docker
- NVIDIA Container Toolkit
- Webcam available as `/dev/video0` or a local video file named `traffic.mp4`

This project was prepared to run inside:

```bash
nvcr.io/nvidia/deepstream:6.4-triton-multiarch
```

## Expected files inside the container

The application uses the following paths inside the DeepStream container:

```text
/data/traffic_analytics_deepstream.py
/data/config_analytics.txt
/data/traffic.mp4
/data/config_infer_secondary_vehicletypes_runtime.txt   # optional
```

The SGIE configuration file is optional. If it is not available, the application still runs, but vehicle type labels may appear as `unknown type`.

## Before running DeepStream

On the host machine, allow the Docker container to access the display:

```bash
export DISPLAY=:0
xhost +
```

## Start the DeepStream Docker container

From the repository folder:

```bash
docker run -it --rm \
  --gpus all \
  --device /dev/video0:/dev/video0 \
  --network host \
  -e DISPLAY=$DISPLAY \
  -v /tmp/.X11-unix:/tmp/.X11-unix \
  -v "$PWD":/data \
  nvcr.io/nvidia/deepstream:6.4-triton-multiarch
```

Alternative, using your current Downloads folder:

```bash
docker run -it --rm \
  --gpus all \
  --device /dev/video0:/dev/video0 \
  --network host \
  -e DISPLAY=$DISPLAY \
  -v /tmp/.X11-unix:/tmp/.X11-unix \
  -v "/home/pedro-sousa/Transferências":/data \
  nvcr.io/nvidia/deepstream:6.4-triton-multiarch
```

## Install dependencies inside the container

After entering the container, install the required Python/GStreamer dependencies:

```bash
apt update && apt install -y python3-gi python3-dev python3-gst-1.0 python-gi-dev \
    python3-pip libglib2.0-dev libgstrtspserver-1.0-0 libgstreamer-plugins-base1.0-dev wget
```

Download and install the DeepStream Python bindings:

```bash
wget https://github.com/NVIDIA-AI-IOT/deepstream_python_apps/releases/download/v1.1.10/pyds-1.1.10-py3-none-linux_x86_64.whl
pip3 install pyds-1.1.10-py3-none-linux_x86_64.whl
```

You can also run:

```bash
bash /data/scripts/install_dependencies.sh
```

## Run the application

Make sure `traffic.mp4` is available in the mounted folder as:

```text
/data/traffic.mp4
```

Then run:

```bash
python3 /data/traffic_analytics_deepstream.py
```

## Analytics configuration

The `config_analytics.txt` file defines the virtual ROI lines used to estimate speed and detect movement direction.

Current ROI groups:

- `L_Topo` and `L_Base`
- `R_Topo` and `R_Base`

The script assumes a reference distance of `16.0` meters between the top and base ROI lines. If the camera position or road geometry changes, update this value in the Python file:

```python
DISTANCE_METERS = 16.0
```

## Notes

- The video input path is currently hardcoded as `/data/traffic.mp4`.
- The analytics configuration path is currently hardcoded as `/data/config_analytics.txt`.
- The optional SGIE vehicle type configuration path is `/data/config_infer_secondary_vehicletypes_runtime.txt`.
- If the SGIE file is missing, the application will still run without vehicle type classification.
- The primary inference configuration uses the default DeepStream sample path:
  `/opt/nvidia/deepstream/deepstream/samples/configs/deepstream-app/config_infer_primary.txt`

This project is intended for academic and research purposes. Add a license file before making the repository public if needed.
