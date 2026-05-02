# XIAO ESP32-S3 — On-Device TinyML Vibration Classifier

A PlatformIO/Arduino project that performs **on-device training** of a
feedforward neural network directly on the Seeed XIAO ESP32-S3, inspired by
the methodology presented in:

> Llisterri Giménez et al., *"On-Device Training of Machine Learning Models on
> Microcontrollers with Federated Learning"*, Electronics 2022, 11, 573.

The MFCC audio pipeline from the paper is replaced with a vibration-angle
pipeline: the MPU-6050 IMU is sampled at 100 Hz, a complementary filter
produces pitch/roll angles, and every 1-second window (500 features) feeds
an online-trained neural network to classify three vibration states.

---

## Hardware

| Component | Details |
|-----------|---------|
| MCU board | Seeed XIAO ESP32-S3 |
| IMU | MPU-6050 (I2C) |
| I2C SDA | D4 → GPIO 5 |
| I2C SCL | D5 → GPIO 6 |
| Supply | 3.3 V from XIAO 3V3 pin |

**Wiring**

```
MPU-6050   XIAO ESP32-S3
---------  ---------------
VCC     →  3V3
GND     →  GND
SDA     →  D4  (GPIO 5)
SCL     →  D5  (GPIO 6)
AD0     →  GND  (I2C addr 0x68)
INT        not connected
```

---

## Project Structure

```
xiao_vibration_tinyml/
├── platformio.ini
├── README.md
├── src/
│   └── main.cpp                 — Application entry point
└── lib/
    ├── MPU6050_driver/
    │   ├── MPU6050_driver.h     — I2C driver + complementary filter
    │   └── MPU6050_driver.cpp
    ├── FeatureExtractor/
    │   ├── FeatureExtractor.h   — 1-second window → 500-element vector
    │   └── FeatureExtractor.cpp
    └── NeuralNetwork/
        ├── NeuralNetwork.h      — 500→16→3 feedforward NN
        └── NeuralNetwork.cpp    — Forward + backprop (manual)
```

---

## Pipeline

```
MPU-6050 (100 Hz)
    │
    ▼
CalibratedIMU  ──►  Complementary Filter  ──►  Angles (pitch, roll)
    │                                               │
    └───────────────────────────────────────────────┘
                          │
                          ▼
               FeatureExtractor (100-sample window)
               [ pitch×100 | roll×100 | gx×100 | gy×100 | gz×100 ]
                          │  500 floats
                          ▼
               NeuralNetwork  500 → 16 → 3
                 forward()  →  output [p0, p1, p2]
                 backward() →  MSE gradient descent
                 updateW1() →  update input→hidden weights
                          │
                          ▼
               Serial: loss, angles, predictions
```

---

## Neural Network

| Property | Value |
|----------|-------|
| Input size | 500 |
| Hidden neurons | 16 (sigmoid) |
| Output neurons | 3 (sigmoid) |
| Activation | Sigmoid |
| Loss | Mean Squared Error |
| Optimiser | Gradient descent (online, 1 sample/step) |
| Learning rate | 0.01 |
| Weight init | Xavier uniform |
| Weights storage | Static RAM (no heap) |
| NN RAM footprint | ≈ 32 KB |

**Output classes**

| Index | Vibration State |
|-------|----------------|
| 0 | Idle / stationary |
| 1 | Low-frequency vibration |
| 2 | High-frequency vibration |

---

## RAM Budget

| Segment | Size |
|---------|------|
| W1 (500×16 floats) | 32 000 B |
| W2 (16×3 floats) | 192 B |
| Biases + activations | ~280 B |
| Feature vector (stack) | 2 000 B |
| IMU buffers + misc | ~200 B |
| **Total static/stack** | **≈ 34.7 KB** |

The XIAO ESP32-S3 has 512 KB SRAM; this project uses well under 200 KB.

---

## Build & Flash

**Prerequisites**

- [PlatformIO Core](https://docs.platformio.org/en/latest/core/installation/index.html) ≥ 6.x  
  or PlatformIO IDE extension in VS Code.

**Build**

```bash
cd xiao_vibration_tinyml
pio run
```

**Flash**

```bash
pio run --target upload
```

**Monitor**

```bash
pio device monitor --baud 115200
```

---

## Runtime Label Switching

During operation, send a single character over the Serial monitor to switch
the active training label without reflashing:

| Key | Label |
|-----|-------|
| `0` | Idle |
| `1` | Low-frequency vibration |
| `2` | High-frequency vibration |

Example session:

```
[MPU6050] Initialised OK — 100 Hz, ±2g, ±250 dps
[NN] Weights initialised — W1:8000 W2:48 params
[MAIN] Training label: 0  LR: 0.0100
[FEX] Window full: 100 samples captured
[TRAIN] step=   1  label=0  out=[0.4921 0.5103 0.4879]  loss=0.082341
...
[CMD] Label set to 2        ← user sent '2'
[TRAIN] step=  23  label=2  out=[0.1820 0.2011 0.7641]  loss=0.011023
```

---

## Configuration Reference

Key compile-time constants:

| Constant | File | Default | Purpose |
|----------|------|---------|---------|
| `I2C_SDA_PIN` | main.cpp | 5 | SDA GPIO |
| `I2C_SCL_PIN` | main.cpp | 6 | SCL GPIO |
| `SAMPLE_INTERVAL_US` | main.cpp | 10 000 | 100 Hz interval |
| `LEARNING_RATE` | main.cpp | 0.01 | Gradient descent LR |
| `CURRENT_LABEL` | main.cpp | 0 | Initial training class |
| `DEBUG_ANGLES` | main.cpp | 1 | Log angles per sample |
| `DEBUG_LOSS` | main.cpp | 1 | Log loss per window |
| `MPU6050_ADDR` | MPU6050_driver.h | 0x68 | I2C address |
| `COMP_FILTER_ALPHA` | MPU6050_driver.h | 0.98 | Complementary filter α |
| `WINDOW_SAMPLES` | FeatureExtractor.h | 100 | Samples per window |
| `NN_HIDDEN_SIZE` | NeuralNetwork.h | 16 | Hidden neurons |

---

## Extending to Federated Learning

Following the paper's federated learning architecture, this project can be
extended as follows:

1. **Server** — Run a Python script (PC or SBC) that:
   - Listens for serialised weight arrays over USB Serial / Wi-Fi.
   - Averages received models (`FedAvg`).
   - Sends the global model back to each client.

2. **Client (this firmware)** — Add:
   - `nn.serializeWeights(buf)` — flatten weights to a byte buffer.
   - `nn.loadWeights(buf)` — restore a received global model.
   - Trigger serialise/load at configurable round intervals.

The XIAO ESP32-S3's built-in Wi-Fi eliminates the bandwidth bottleneck
noted in the paper (64 B USB serial buffer), enabling much higher throughput
for model exchange.

---

## License

MIT — see individual source files for author attribution.
