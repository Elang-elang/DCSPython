import cv2
import numpy as np
from pyzbar import pyzbar

def test_qr_detection():
    """Test script untuk QR code detection"""
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Error: Tidak dapat mengakses kamera")
        return
    
    print("QR Code Detection Test")
    print("Arahkan kamera ke QR code. Tekan 'q' untuk keluar")
    
    while True:
        ret, frame = cap.read()
        
        if not ret:
            print("Error: Tidak dapat membaca frame")
            break
        
        # Deteksi QR codes
        qr_codes = pyzbar.decode(frame)
        
        # Gambar overlay untuk setiap QR code
        for qr in qr_codes:
            # Decode data
            data = qr.data.decode('utf-8')
            qr_type = qr.type
            
            # Gambar rectangle di sekitar QR code
            points = qr.polygon
            if len(points) > 4:
                # Kompleks polygon, gunakan bounding rect
                hull = cv2.convexHull(np.array([point for point in points], dtype=np.float32))
                points = hull
            
            # Convert points ke integer
            points = np.array(points, dtype=np.int32)
            
            # Gambar polygon
            cv2.polylines(frame, [points], True, (0, 255, 0), 3)
            
            # Tambahkan text
            x = qr.rect.left
            y = qr.rect.top - 10
            cv2.putText(frame, f"{qr_type}: {data}", (x, y), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            
            print(f"QR Detected - Type: {qr_type}, Data: {data}")
        
        # Tampilkan frame
        cv2.imshow('QR Code Detection Test', frame)
        
        # Tekan 'q' untuk keluar
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    test_qr_detection()