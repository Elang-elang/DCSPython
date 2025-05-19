import cv2
import time

def test_camera():
    """Test script untuk memeriksa apakah kamera dapat diakses"""
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Error: Tidak dapat mengakses kamera")
        return False
    
    print("Kamera berhasil diakses!")
    print("Tekan 'q' untuk keluar")
    
    while True:
        ret, frame = cap.read()
        
        if not ret:
            print("Error: Tidak dapat membaca frame")
            break
        
        # Tampilkan frame
        cv2.imshow('Test Kamera', frame)
        
        # Tekan 'q' untuk keluar
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()
    print("Test kamera selesai")
    return True

if __name__ == "__main__":
    test_camera()
