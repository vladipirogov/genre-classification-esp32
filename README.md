# Genre Classification on ESP32

This project implements a **real-time audio genre classification system** on an ESP32 microcontroller. It uses a microphone (e.g., MAX9814) to capture audio, computes MFCCs (Mel-Frequency Cepstral Coefficients), and performs inference using a TensorFlow Lite model. The predicted genre is displayed on an QT1306P82 OLED screen.

---

## Features
- **Audio Sampling**: Captures audio data from an ADC pin at a sampling rate of 22,050 Hz.
- **MFCC Computation**: Extracts MFCC features from the audio signal.
- **TensorFlow Lite Inference**: Classifies audio into one of 10 genres using a pre-trained model.
- **OLED Display**: Displays the predicted genre on an QT1306P82 OLED screen.

---

## Hardware Requirements
1. **ESP32-C3 Development Board**
2. **MAX9814 Microphone Module**
3. **QT1306P82 OLED Display**
4. **Wires and Breadboard**

---

## Software Requirements
- **ESP-IDF** (v4.4)
- **TensorFlow Lite Micro**
- **U8G2 Library** for OLED display
- **MFCC Library** for feature extraction

---

## Wiring Diagram
| Component       | ESP32-C3 Pin       |
|------------------|-----------------|
| MAX9814 OUT      | GPIO0 (ADC1_CHANNEL_0) |
| MAX9814 VCC      | 3.3V      |
| MAX9814 GND      | GND             |
| QT1306P82 SDA      | GPIO5           |
| QT1306P82 SCL      | GPIO6           |
| QT1306P82 VCC      | 3.3V            |
| QT1306P82 GND      | GND             |

---

## Installation and Setup

### 1. Clone the Repository
```bash
git clone https://github.com/vladipirogov/genre-classification-esp32
cd genre-classification-esp32
```

### 2. Configure the ESP-IDF Environment
Follow the [ESP-IDF setup guide](https://docs.espressif.com/projects/esp-idf/en/latest/esp32/get-started/index.html) to configure your development environment.

### 3. Build and Flash the Project
```bash
idf.py set-target esp32c3
idf.py build
idf.py flash monitor
```

---

## Project Structure
```
.
├── main
│   ├── main.cc                # Main application logic
│   ├── u8g2_esp32_hal.h       # HAL for U8G2 library
│   ├── u8g2_esp32_hal.c       # Implementation of U8G2 HAL
│   ├── MFCC_Q15.cc            # MFCC computation logic
│   ├── model.h                # TensorFlow Lite model header
│   ├── dct_wei_mtx_q15_T.h    # DCT weight matrix for MFCC computation
│   ├── hann_lut_q15.h         # Hanning window lookup table
│   ├── log_lut_q13_3.h        # Logarithm lookup table
│   ├── mel_wei_mtx_q15_T.h    # Mel filter bank weights
│   ├── mfccs_consts.h         # Constants for MFCC computation
├── components
│   ├── esp-tflite-micro       # TensorFlow Lite Micro component
│   ├── esp-mfcc               # MFCC computation library
│   ├── u8g2                   # U8G2 library for OLED display
├── CMakeLists.txt             # Build configuration
├── README.md                  # Project documentation
```

---

## How It Works
1. **Audio Sampling**:
   - The ADC samples audio data from the MAX9814 microphone at 22,050 Hz.
   - A bias offset is subtracted to center the audio signal around zero.

2. **MFCC Computation**:
   - MFCC features are extracted from the audio buffer using the `MFCC_Q15` library.

3. **TensorFlow Lite Inference**:
   - The MFCC features are fed into a TensorFlow Lite model to classify the audio into one of 10 genres:
     - `blues`, `classical`, `country`, `disco`, `hiphop`, `jazz`, `metal`, `pop`, `reggae`, `rock`.

4. **Display Output**:
   - The predicted genre is displayed on the QT1306P82 OLED screen.

---

## Configuration

### ADC Calibration
The ADC is calibrated using the ESP32's eFuse data. If calibration data is unavailable, the default Vref (1100 mV) is used. You can adjust the default Vref in `main.cc`:
```cpp
#define DEFAULT_VREF 1100
```

### Sampling Rate
The audio sampling rate is set to 22,050 Hz. You can modify it in `main.cc`:
```cpp
#define SAMPLE_RATE 22050
```

---

## Troubleshooting

### Common Issues
1. **Bias Offset Measured as 4095**:
   - Ensure the ADC pin is connected to the microphone output.
   - Verify the microphone output voltage is within the ADC's input range (0–3.6V for `ADC_ATTEN_DB_11`).

2. **Display Not Working**:
   - Check the I2C connections (SDA and SCL).
   - Ensure the display address is set correctly:
     ```cpp
     u8x8_SetI2CAddress(&u8g2.u8x8, 0x78);
     ```

3. **TensorFlow Lite Initialization Fails**:
   - Ensure the TensorFlow Lite model is correctly included in `model.h`.

---

## License
This project is licensed under the MIT License. See the `LICENSE` file for details.

---

## Acknowledgments
- [ESP-IDF](https://github.com/espressif/esp-idf)
- [TensorFlow Lite Micro](https://www.tensorflow.org/lite/microcontrollers)
- [U8G2 Library](https://github.com/olikraus/u8g2)