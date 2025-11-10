import cv2
import numpy as np
import matplotlib.pyplot as plt
import pywt
import urllib.request

def load_image_from_url(url):
    """
    Mengunduh gambar dari URL dan memuatnya sebagai gambar grayscale.
    """
    try:
        with urllib.request.urlopen(url) as resp:
            image_data = np.asarray(bytearray(resp.read()), dtype="uint8")
            # Muat sebagai gambar BGR terlebih dahulu
            img_bgr = cv2.imdecode(image_data, cv2.IMREAD_COLOR)
            if img_bgr is None:
                print("Gagal men-decode gambar dari URL.")
                return None
            # Konversi ke grayscale
            img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
            return img_gray
    except Exception as e:
        print(f"Error saat mengunduh gambar: {e}")
        return None

def get_fourier_spectrum(image_gray):
    """
    Menghitung spektrum Fourier dari sebuah gambar grayscale.
    Input: gambar (numpy array) grayscale.
    Output: Spektrum Fourier (untuk ditampilkan).
    """
    f = np.fft.fft2(image_gray)
    
    fshift = np.fft.fftshift(f)
    
    spectrum = 20 * np.log(np.abs(fshift) + 1)
    
    return spectrum

def get_dct_spectrum(image_gray):
    """
    Menghitung spektrum Discrete Cosine Transform (DCT) dari sebuah gambar.
    Input: gambar (numpy array) grayscale.
    Output: Spektrum DCT (untuk ditampilkan).
    """
    image_float = image_gray.astype('float32')
    
    dct = cv2.dct(image_float)
    
    spectrum = np.log(np.abs(dct) + 1)
    
    return spectrum

def get_wavelet_coeffs(image_gray, wavelet_type='haar'):
    """
    Menghitung koefisien Wavelet (DWT) dari sebuah gambar.
    Input: 
        - image_gray (numpy array) grayscale.
        - wavelet_type (string): Tipe wavelet, cth: 'haar', 'db1', 'sym4'
    Output: 
        - LL: Koefisien Aproksimasi
        - (LH, HL, HH): Koefisien Detail (Horizontal, Vertikal, Diagonal)
    """
    LL, (LH, HL, HH) = pywt.dwt2(image_gray, wavelet_type)
    
    return LL, LH, HL, HH

def main():
    """
    Fungsi utama untuk menjalankan demo.
    """
    # URL gambar placeholder (pengganti 'lena.jpg')
    IMAGE_URL = "https://placehold.co/512x512/999999/FFFFFF?text=Contoh+Gambar"
    
    print(f"Mengunduh gambar contoh dari: {IMAGE_URL}")
    img_gray = load_image_from_url(IMAGE_URL)
    
    if img_gray is None:
        print("Gagal memuat gambar. Membuat gambar noise acak sebagai fallback.")
        # Buat gambar noise acak jika unduhan gagal
        img_gray = np.random.randint(0, 256, (256, 256), dtype=np.uint8)

    print("Menghitung transformasi...")
    
    fourier_spec = get_fourier_spectrum(img_gray)
    print("Menampilkan plot Spektrum Fourier...")
    plt.figure(1, figsize=(7, 7)) # Buat figur baru
    plt.imshow(fourier_spec, cmap='gray')
    plt.title('Spektrum Fourier (Skala Log)')
    plt.axis('off')
    plt.show() # Tampilkan plot ini
    
    dct_spec = get_dct_spectrum(img_gray)
    print("Menampilkan plot Spektrum DCT...")
    plt.figure(2, figsize=(7, 7)) # Buat figur baru
    plt.imshow(dct_spec, cmap='gray')
    plt.title('Spektrum DCT (Skala Log)')
    plt.axis('off')
    plt.show() # Tampilkan plot ini
    
    LL, LH, HL, HH = get_wavelet_coeffs(img_gray, 'haar')
    print("Menampilkan plot Wavelet (LL)...")
    plt.figure(3, figsize=(7, 7))
    plt.imshow(LL, cmap='gray')
    plt.title('Wavelet Aproksimasi (LL - Haar)')
    plt.axis('off')
    plt.show()
    
    print("Demo selesai.")

if __name__ == "__main__":
    main()