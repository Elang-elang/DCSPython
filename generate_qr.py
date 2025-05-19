import qrcode
from PIL import Image

def generate_test_qr():
    """Generate test QR codes"""
    # Test URLs
    test_data = [
        "https://www.google.com",
        "https://www.github.com",
        "Hello World! This is a test QR code.",
        "https://docs.python.org/3/",
        "Contact: phone=+6281234567890, email=test@example.com"
    ]
    
    for i, data in enumerate(test_data):
        # Buat QR code
        qr = qrcode.QRCode(
            version=1,
            error_correction=qrcode.constants.ERROR_CORRECT_L,
            box_size=10,
            border=4,
        )
        qr.add_data(data)
        qr.make(fit=True)
        
        # Buat image
        img = qr.make_image(fill_color="black", back_color="white")
        
        # Simpan
        filename = f"test_qr_{i+1}.png"
        img.save(filename)
        print(f"QR code saved: {filename} - Data: {data}")

if __name__ == "__main__":
    generate_test_qr()