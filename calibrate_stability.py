import cv2
import numpy as np
import time

def calibrate_stability():
    """Script untuk mengkalibrasi threshold stabilitas kamera"""
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Error: Tidak dapat mengakses kamera")
        return
    
    print("Kalibrasi stabilitas kamera")
    print("1. Pertama, letakkan kamera di posisi stabil selama 10 detik")
    print("2. Kemudian, goyangkan kamera selama 10 detik")
    print("Tekan SPACE untuk mulai kalibrasi, ESC untuk keluar")
    
    last_frame = None
    stable_diffs = []
    unstable_diffs = []
    mode = "waiting"  # waiting, stable, unstable, done
    start_time = 0
    
    while True:
        ret, frame = cap.read()
        
        if not ret:
            print("Error: Tidak dapat membaca frame")
            break
        
        # Konversi ke grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Hitung perbedaan frame jika ada frame sebelumnya
        if last_frame is not None:
            diff = cv2.absdiff(last_frame, gray)
            mean_diff = np.mean(diff)
            
            # Simpan data berdasarkan mode
            if mode == "stable":
                stable_diffs.append(mean_diff)
                elapsed = time.time() - start_time
                cv2.putText(frame, f"STABLE MODE: {10-elapsed:.1f}s", 
                           (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                
                if elapsed >= 10:
                    mode = "unstable"
                    start_time = time.time()
                    print("Sekarang goyangkan kamera!")
            
            elif mode == "unstable":
                unstable_diffs.append(mean_diff)
                elapsed = time.time() - start_time
                cv2.putText(frame, f"UNSTABLE MODE: {10-elapsed:.1f}s", 
                           (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                
                if elapsed >= 10:
                    mode = "done"
                    break
            
            # Tampilkan current diff
            cv2.putText(frame, f"Current diff: {mean_diff:.2f}", 
                       (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Tampilkan instruksi berdasarkan mode
        if mode == "waiting":
            cv2.putText(frame, "Press SPACE to start calibration", 
                       (10, frame.shape[0] - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        
        last_frame = gray
        cv2.imshow('Kalibrasi Stabilitas', frame)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord(' ') and mode == "waiting":
            mode = "stable"
            start_time = time.time()
            print("Mode stabil dimulai - jangan gerakkan kamera!")
        elif key == 27:  # ESC
            break
    
    cap.release()
    cv2.destroyAllWindows()
    
    if mode == "done" and stable_diffs and unstable_diffs:
        # Analisis hasil
        stable_mean = np.mean(stable_diffs)
        stable_max = np.max(stable_diffs)
        unstable_mean = np.mean(unstable_diffs)
        unstable_min = np.min(unstable_diffs)
        
        print("\n=== HASIL KALIBRASI ===")
        print(f"Stable - Mean: {stable_mean:.2f}, Max: {stable_max:.2f}")
        print(f"Unstable - Mean: {unstable_mean:.2f}, Min: {unstable_min:.2f}")
        
        # Rekomendasikan threshold
        recommended_threshold = stable_max + (unstable_min - stable_max) * 0.3
        print(f"\nRekomendasi threshold: {recommended_threshold:.2f}")
        print(f"Update variabel 'stability_threshold' di app.py dengan nilai ini")
    else:
        print("Kalibrasi tidak lengkap")

if __name__ == "__main__":
    calibrate_stability()
