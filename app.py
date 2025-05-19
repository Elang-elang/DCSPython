import os
import re
import string
import numpy as np
import cv2
import base64
import threading
import time
from flask import Flask, request, render_template, flash, redirect, url_for, jsonify, Response
from PIL import Image
import pytesseract
import tflite_runtime.interpreter as tflite
from werkzeug.utils import secure_filename
from io import BytesIO
from pyzbar import pyzbar

app = Flask(__name__)
app.secret_key = 'your-secret-key-change-this'

# Konfigurasi
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16MB max file size

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = MAX_CONTENT_LENGTH

# Pastikan folder upload ada
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Kategori dokumen
DOCUMENT_CATEGORIES = [
    'Surat',
    'Nota', 
    'KTP',
    'Kartu Keluarga',
    'QR Code',
    'Lainnya'
]

# Variabel global untuk model dan kamera
interpreter = None
camera_active = False
last_detection_time = 0
detection_interval = 5  # deteksi setiap 5 detik
stability_threshold = 30  # threshold untuk deteksi guncangan
last_frame_gray = None

def load_model():
    """Load TensorFlow Lite model"""
    global interpreter
    model_path = 'document_classifier.tflite'
    
    try:
        if os.path.exists(model_path):
            interpreter = tflite.Interpreter(model_path=model_path)
            interpreter.allocate_tensors()
            print(f"Model berhasil dimuat dari {model_path}")
        else:
            print(f"Model tidak ditemukan di {model_path}. Menggunakan dummy classifier.")
            interpreter = None
    except Exception as e:
        print(f"Error loading model: {e}")
        interpreter = None

def allowed_file(filename):
    """Cek apakah file yang diupload diizinkan"""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def preprocess_text(text):
    """Preprocessing teks hasil OCR"""
    if not text:
        return ""
    
    # Konversi ke lowercase
    text = text.lower()
    
    # Hapus karakter non-alfanumerik kecuali spasi
    text = re.sub(r'[^a-zA-Z0-9\s]', ' ', text)
    
    # Hapus multiple spaces
    text = re.sub(r'\s+', ' ', text)
    
    # Trim whitespace
    text = text.strip()
    
    return text

def detect_qr_code(image):
    """Deteksi QR code dari gambar OpenCV"""
    try:
        # Deteksi QR codes
        qr_codes = pyzbar.decode(image)
        
        qr_data = []
        for qr in qr_codes:
            # Decode data QR code
            data = qr.data.decode('utf-8')
            qr_type = qr.type
            
            # Ambil koordinat QR code untuk overlay
            points = qr.polygon
            if len(points) > 4:
                # Jika polygon kompleks, buat rectangle
                rect = qr.rect
                points = [
                    (rect.left, rect.top),
                    (rect.left + rect.width, rect.top),
                    (rect.left + rect.width, rect.top + rect.height),
                    (rect.left, rect.top + rect.height)
                ]
            
            qr_data.append({
                'data': data,
                'type': qr_type,
                'points': [(int(point.x), int(point.y)) for point in points] if hasattr(points[0], 'x') else points
            })
        
        return qr_data
    except Exception as e:
        print(f"Error detecting QR code: {e}")
        return []

def draw_qr_overlay(image, qr_data):
    """Gambar overlay pada QR code yang terdeteksi"""
    overlay_image = image.copy()
    
    for qr_info in qr_data:
        points = qr_info['points']
        
        # Gambar polygon di sekitar QR code
        points_array = np.array(points, dtype=np.int32)
        cv2.polylines(overlay_image, [points_array], True, (0, 255, 0), 3)
        
        # Tambahkan text label
        x = points[0][0]
        y = points[0][1] - 10
        cv2.putText(overlay_image, f"QR: {qr_info['type']}", 
                   (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    
    return overlay_image

def extract_text_from_image(image_path):
    """Ekstrak teks dari gambar menggunakan pytesseract (update untuk QR code)"""
    try:
        # Buka gambar dengan PIL
        pil_image = Image.open(image_path)
        
        # Konversi ke OpenCV format untuk deteksi QR
        cv_image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
        
        # Deteksi QR code terlebih dahulu
        qr_data = detect_qr_code(cv_image)
        
        # Jika ada QR code, return data QR
        if qr_data:
            qr_texts = []
            for qr_info in qr_data:
                qr_texts.append(f"QR Code Data: {qr_info['data']}")
            return "\n".join(qr_texts)
        
        # Jika tidak ada QR code, lakukan OCR biasa
        text = pytesseract.image_to_string(pil_image, lang='ind+eng')
        return text
        
    except Exception as e:
        print(f"Error extracting text: {e}")
        return None

def extract_text_from_cv_image(cv_image):
    """Ekstrak teks dari OpenCV image (update untuk QR code)"""
    try:
        # Deteksi QR code terlebih dahulu
        qr_data = detect_qr_code(cv_image)
        
        # Jika ada QR code, return data QR
        if qr_data:
            qr_texts = []
            for qr_info in qr_data:
                qr_texts.append(f"QR Code Data: {qr_info['data']}")
            return "\n".join(qr_texts)
        
        # Jika tidak ada QR code, lakukan OCR biasa
        # Konversi BGR ke RGB
        rgb_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
        
        # Konversi ke PIL Image
        pil_image = Image.fromarray(rgb_image)
        
        # Ekstrak teks menggunakan OCR
        text = pytesseract.image_to_string(pil_image, lang='ind+eng')
        
        return text
    except Exception as e:
        print(f"Error extracting text from camera: {e}")
        return None

def text_to_features(text, max_length=100):
    """Konversi teks ke features untuk model (dummy implementation)"""
    words = text.split()
    
    # Buat feature vector sederhana (bag of words)
    feature_vector = np.zeros(max_length, dtype=np.float32)
    
    for i, word in enumerate(words[:max_length]):
        # Konversi kata ke angka (hash sederhana)
        feature_vector[i] = hash(word) % 1000 / 1000.0
    
    return feature_vector

def predict_document_category(text):
    """Prediksi kategori dokumen (update untuk QR code)"""
    global interpreter
    
    # Cek apakah ini QR code
    if text and text.startswith("QR Code Data:"):
        return 'QR Code', 0.95
    
    if interpreter is None:
        # Dummy prediction jika model tidak tersedia
        return dummy_predict(text)
    
    try:
        # Preprocessing teks menjadi features
        features = text_to_features(text)
        features = np.expand_dims(features, axis=0)  # Add batch dimension
        
        # Get input and output details
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        
        # Set input tensor
        interpreter.set_tensor(input_details[0]['index'], features)
        
        # Jalankan inferensi
        interpreter.invoke()
        
        # Get output
        output_data = interpreter.get_tensor(output_details[0]['index'])
        
        # Get predicted class
        predicted_class = np.argmax(output_data[0])
        confidence = float(output_data[0][predicted_class])
        
        return DOCUMENT_CATEGORIES[predicted_class], confidence
        
    except Exception as e:
        print(f"Error during prediction: {e}")
        return dummy_predict(text)

def dummy_predict(text):
    """Dummy prediction untuk testing"""
    # Implementasi dummy berdasarkan kata kunci
    text_lower = text.lower()
    
    if any(word in text_lower for word in ['nik', 'tempat lahir', 'golongan darah']):
        return 'KTP', 0.85
    elif any(word in text_lower for word in ['kepala keluarga', 'alamat', 'rt/rw']):
        return 'Kartu Keluarga', 0.80
    elif any(word in text_lower for word in ['nota', 'receipt', 'total', 'bayar']):
        return 'Nota', 0.75
    elif any(word in text_lower for word in ['surat', 'kepada', 'hormat', 'tanda tangan']):
        return 'Surat', 0.70
    else:
        return 'Lainnya', 0.60

def check_camera_stability(current_frame):
    """Cek apakah kamera stabil (tidak guncang)"""
    global last_frame_gray, stability_threshold
    
    if current_frame is None:
        return False
    
    # Konversi ke grayscale
    gray = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY)
    
    if last_frame_gray is None:
        last_frame_gray = gray
        return False
    
    # Hitung perbedaan frame
    diff = cv2.absdiff(last_frame_gray, gray)
    mean_diff = np.mean(diff)
    
    # Update last frame
    last_frame_gray = gray
    
    # Return True jika kamera stabil (perbedaan kecil)
    return mean_diff < stability_threshold

def enhance_image_for_ocr(image):
    """Enhance image untuk meningkatkan akurasi OCR"""
    # Konversi ke grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Terapkan Gaussian blur untuk mengurangi noise
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Terapkan adaptive threshold
    enhanced = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                   cv2.THRESH_BINARY, 11, 2)
    
    # Konversi kembali ke BGR untuk kompatibilitas
    enhanced_bgr = cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR)
    
    return enhanced_bgr

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    # Cek apakah file ada dalam request
    if 'file' not in request.files:
        flash('Tidak ada file yang dipilih')
        return redirect(url_for('index'))
    
    file = request.files['file']
    
    # Cek apakah ada file yang dipilih
    if file.filename == '':
        flash('Tidak ada file yang dipilih')
        return redirect(url_for('index'))
    
    if file and allowed_file(file.filename):
        try:
            # Simpan file
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            
            # Ekstrak teks dari gambar
            extracted_text = extract_text_from_image(filepath)
            
            if extracted_text is None:
                flash('Gagal mengekstrak teks dari gambar')
                os.remove(filepath)  # Hapus file yang gagal diproses
                return redirect(url_for('index'))
            
            # Preprocessing teks
            processed_text = preprocess_text(extracted_text)
            
            # Prediksi kategori dokumen
            category, confidence = predict_document_category(processed_text)
            
            # Hapus file setelah diproses
            os.remove(filepath)
            
            return render_template('result.html', 
                                 original_text=extracted_text,
                                 processed_text=processed_text,
                                 category=category,
                                 confidence=confidence,
                                 source='upload')
            
        except Exception as e:
            flash(f'Error memproses file: {str(e)}')
            return redirect(url_for('index'))
    else:
        flash('Format file tidak diizinkan. Gunakan .jpg, .png, atau .jpeg')
        return redirect(url_for('index'))

@app.route('/camera')
def camera():
    return render_template('camera.html')

@app.route('/process_camera_frame', methods=['POST'])
def process_camera_frame():
    global last_detection_time, detection_interval
    
    try:
        # Cek apakah sudah waktunya untuk deteksi
        current_time = time.time()
        if current_time - last_detection_time < detection_interval:
            return jsonify({
                'success': False,
                'message': f'Tunggu {detection_interval - (current_time - last_detection_time):.1f} detik lagi'
            })
        
        # Ambil data image dari request
        data = request.get_json()
        image_data = data['image'].split(',')[1]  # Remove data:image/jpeg;base64,
        
        # Decode base64 image
        image_bytes = base64.b64decode(image_data)
        nparr = np.frombuffer(image_bytes, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if frame is None:
            return jsonify({'success': False, 'message': 'Gagal memproses gambar'})
        
        # Deteksi QR code terlebih dahulu
        qr_data = detect_qr_code(frame)
        
        if qr_data:
            # Jika ada QR code, proses langsung
            qr_texts = []
            for qr_info in qr_data:
                qr_texts.append(f"QR Code Data: {qr_info['data']}")
            
            extracted_text = "\n".join(qr_texts)
            processed_text = preprocess_text(extracted_text)
            category, confidence = predict_document_category(extracted_text)
            
            # Update last detection time
            last_detection_time = current_time
            
            return jsonify({
                'success': True,
                'original_text': extracted_text,
                'processed_text': processed_text,
                'category': category,
                'confidence': round(confidence * 100, 2),
                'qr_detected': True,
                'qr_count': len(qr_data)
            })
        
        # Jika tidak ada QR code, lanjutkan dengan OCR biasa
        # Cek stabilitas kamera
        if not check_camera_stability(frame):
            return jsonify({
                'success': False,
                'message': 'Kamera tidak stabil. Harap keep kamera steady.'
            })
        
        # Enhance image untuk OCR
        enhanced_frame = enhance_image_for_ocr(frame)
        
        # Ekstrak teks
        extracted_text = extract_text_from_cv_image(enhanced_frame)
        
        if not extracted_text or len(extracted_text.strip()) < 5:
            return jsonify({
                'success': False,
                'message': 'Tidak ada teks atau QR code yang terdeteksi. Coba arahkan kamera ke dokumen atau QR code.'
            })
        
        # Preprocessing teks
        processed_text = preprocess_text(extracted_text)
        
        if len(processed_text.strip()) < 3:
            return jsonify({
                'success': False,
                'message': 'Teks terlalu pendek untuk diklasifikasi.'
            })
        
        # Prediksi kategori dokumen
        category, confidence = predict_document_category(processed_text)
        
        # Update last detection time
        last_detection_time = current_time
        
        return jsonify({
            'success': True,
            'original_text': extracted_text,
            'processed_text': processed_text,
            'category': category,
            'confidence': round(confidence * 100, 2),
            'qr_detected': False
        })
        
    except Exception as e:
        print(f"Error processing camera frame: {e}")
        return jsonify({'success': False, 'message': f'Error: {str(e)}'})


# Route baru untuk real-time QR detection feed
@app.route('/qr_feed')
def qr_feed():
    """Stream real-time QR detection"""
    def generate_frames():
        cap = cv2.VideoCapture(0)
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Deteksi QR codes
            qr_data = detect_qr_code(frame)
            
            # Gambar overlay jika ada QR code
            if qr_data:
                frame = draw_qr_overlay(frame, qr_data)
                
                # Tambahkan text informasi QR
                for i, qr_info in enumerate(qr_data):
                    y_pos = 30 + (i * 25)
                    text = f"QR {i+1}: {qr_info['data'][:30]}..."
                    cv2.putText(frame, text, (10, y_pos), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            
            # Encode frame
            ret, buffer = cv2.imencode('.jpg', frame)
            frame_bytes = buffer.tobytes()
            
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
        
        cap.release()
    
    return Response(generate_frames(),
                   mimetype='multipart/x-mixed-replace; boundary=frame')

@app.errorhandler(413)
def too_large(e):
    flash('File terlalu besar. Maksimal 16MB.')
    return redirect(url_for('index'))

if __name__ == '__main__':
    # Load model saat aplikasi dimulai
    load_model()
    
    # Jalankan aplikasi
    app.run(host='127.0.0.1', port=5000, debug=True)
