# AI Fatigue Driver Detection

Real-time driver fatigue detection using computer vision and TensorFlow Lite.

<img src="assets/driver.gif" width="300" />

*Simulation of the application running in real time on a Raspberry Pi 5, fully offline. When a yawn is detected, the yellow LED starts blinking. If the eyes close, the speed decreases and the red LED turns on (simulating an emergency stop).*

## Project Structure

```
.
├── main.py                     # Unified entry point (Windows / Raspberry Pi)
├── main-pi.py                  # Compatibility alias (imports main.py)
├── src/                        # Clean architecture source code
│   ├── config.py              # Global settings
│   ├── domain/                # Domain layer (entities & business rules)
│   │   ├── eye.py            # Eye blink & gaze analysis
│   │   ├── iris.py           # Iris position tracking
│   │   ├── lips.py           # Yawn detection
│   │   └── monitor.py        # Speed & alert logic
│   ├── application/           # Use cases (orchestration)
│   │   ├── facial_tracker.py # Facial tracking pipeline
│   │   └── tflite_tracker.py # TFLite classification
│   ├── infrastructure/        # External adapters (MediaPipe, GPIO)
│   │   ├── face_mesh.py      # MediaPipe FaceLandmarker wrapper
│   │   └── gpio_controller.py# GPIO LED control (Raspberry Pi)
│   └── presentation/          # User interface
│       └── speedometer.py    # OpenCV speedometer gauge
├── models/
│   └── model_random.tflite   # Compiled model (Alertness/Fatigue)
├── assets/
│   └── driver.mp4            # Sample test video
├── facial_tracking/          # Legacy module (kept for compatibility)
├── requirements.txt          # Common dependencies
├── requirements-pi.txt       # Raspberry Pi (+ tflite-runtime, RPi.GPIO, gpiozero)
├── requirements-win.txt      # Windows (+ tensorflow)
└── README.md
```

## Requirements

```
opencv-python
mediapipe
numpy
flatbuffers
```

## Installation

**Windows / Desktop (development):**
```bash
pip install -r requirements-win.txt
```

**Raspberry Pi (production):**
```bash
pip install -r requirements-pi.txt
```

`requirements-pi.txt` adds `tflite-runtime`, `RPi.GPIO`, `gpiozero` (Linux/ARM wheels).
`requirements-win.txt` adds `tensorflow` (required on Windows since `tflite-runtime` has no official wheels for this platform).

## Usage

```bash
python main.py
```

The webcam will activate and display:
- **FPS**: Frames per second
- **Eye Status**: Open, closed, or gazing direction
- **Yawn Status**: Yawn detection
- **Prediction**: Alertness/Fatigue classification with confidence %

### Controls

- **q**: Quit the application

## Model

- **File**: `models/model_random.tflite`
- **Size**: 8.46 MB
- **Input**: 224x224x3 (RGB normalized -1 to 1)
- **Output**: 2 classes (Alertness, Fatigue)
- **Framework**: TensorFlow Lite (CPU-optimized)

## Notes

- Clean architecture with layered separation: domain, application, infrastructure, and presentation
- The model was converted from Keras H5 to TFLite with maximum compatibility
- Uses XNNPACK delegate for CPU acceleration
- Compatible with TensorFlow 2.19.0+
