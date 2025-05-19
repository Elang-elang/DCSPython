# Document Classification with OCR & QR Detection

A Flask-based web application that classifies documents (ID cards, letters, receipts, etc.) using OCR and detects QR codes in real-time via camera or uploaded images.

## ✨ Features

- 📄 **Document Classification**: Categorizes documents into KTP, Kartu Keluarga, Surat, Nota, or Lainnya
- 🔍 **QR Code Detection**: Real-time QR code scanning with visual overlay
- 📷 **Camera Integration**: Live OCR processing with stability detection
- 🖼️ **Image Upload**: Supports JPG/PNG uploads up to 16MB
- 🚀 **Lightweight**: Uses TensorFlow Lite for efficient inference

## 📂 Directory Structure

```
.
├── app.py                 # Main Flask application
├── calibrate_stability.py # Camera stability calibration tool
├── generate_qr.py         # QR code generator for testing
├── test_qr.py             # QR detection tester
├── test_camera.py         # Camera tester
├── templates/             # HTML templates
│   ├── index.html         # Homepage
│   ├── camera.html        # Camera interface
│   └── result.html        # Results page
├── static/                # Static files
│   └── style.css          # CSS styles
└── uploads/               # Temporary upload storage
```

## 🛠️ Installation

1. Clone the repository:
   ```bash
   git clone [your-repository-url-here]
   cd [repository-name]
   ```

2. Install dependencies:
   ```bash
   pip install flask numpy opencv-python pytesseract pyzbar pillow qrcode
   ```

3. Install Tesseract OCR (required for text extraction):
   - **Windows**: Download from [Tesseract installer](https://github.com/UB-Mannheim/tesseract/wiki)
   - **Mac**: `brew install tesseract`
   - **Linux**: `sudo apt install tesseract-ocr`

## 🚀 Running the Application

1. Start the Flask server:
   ```bash
   python app.py
   ```

2. Open your browser to:
   ```
   http://localhost:5000
   ```

## 🔧 Testing Tools

- Generate test QR codes:
  ```bash
  python generate_qr.py
  ```

- Test camera stability:
  ```bash
  python calibrate_stability.py
  ```

- Test QR detection:
  ```bash
  python test_qr.py
  ```

## 📸 Usage Tips

1. For best OCR results:
   - Ensure good lighting
   - Keep the camera steady
   - Position documents flat

2. Camera mode automatically processes every 5 seconds when stable

3. Switch between normal OCR mode and dedicated QR stream mode
