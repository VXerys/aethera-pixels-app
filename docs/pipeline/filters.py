"""
Implementasi Filter Edge-Preserving dan Sharpening untuk Aethera Pixelis
- Bilateral Filter: Edge-Preserving Smoothing
- Unsharp Mask: Sharpening Klasik
- Guided Filter: Detail Extraction
"""

import cv2
import numpy as np
from scipy.ndimage import gaussian_filter


class BilateralFilter:
    """
    Bilateral Filter untuk edge-preserving smoothing.
    
    Formula:
    I_p = (1/W_p) * Σ I_q * G_σs(|p-q|) * G_σr(|I_p - I_q|)
    
    Dimana:
    - G_σs adalah Gaussian kernel spatial (jarak euclidean)
    - G_σr adalah Gaussian kernel range (perbedaan intensitas)
    """
    
    @staticmethod
    def apply(image, d=9, sigma_color=75, sigma_space=75):
        """
        Aplikasi Bilateral Filter
        
        Args:
            image: Input image (BGR format)
            d: Diameter pixel neighborhood
            sigma_color: Filter sigma untuk range/warna
            sigma_space: Filter sigma untuk spatial/ruang
            
        Returns:
            Filtered image
        """
        if len(image.shape) == 3:
            # Jika warna, apply ke setiap channel
            return cv2.bilateralFilter(image, d, sigma_color, sigma_space)
        else:
            # Jika grayscale
            return cv2.bilateralFilter(image, d, sigma_color, sigma_space)


class UnsharpMask:
    """
    Unsharp Mask untuk sharpening klasik.
    
    Formula:
    I_sharpened = I_original + α * (I_original - I_blurred)
    
    Dimana:
    - α adalah sharpening amount (kekuatan)
    - I_blurred adalah versi blur dari image asli
    """
    
    @staticmethod
    def apply(image, amount=1.2, radius=2.0):
        """
        Aplikasi Unsharp Mask
        
        Args:
            image: Input image (BGR atau grayscale)
            amount: Sharpening strength (typically 0.5 - 2.0)
            radius: Radius untuk Gaussian blur
            
        Returns:
            Sharpened image
        """
        # Convert ke float untuk operasi
        img_float = image.astype(np.float32) / 255.0
        
        # Create blurred version
        gaussian_blur = cv2.GaussianBlur(img_float, (0, 0), radius)
        
        # Calculate unsharp mask
        sharpened = img_float + amount * (img_float - gaussian_blur)
        
        # Clip values dan convert kembali ke uint8
        sharpened = np.clip(sharpened * 255, 0, 255).astype(np.uint8)
        
        return sharpened


class GuidedFilter:
    """
    Guided Filter untuk edge-aware smoothing dan detail extraction.
    
    Memisahkan citra menjadi:
    - Base layer (low frequency)
    - Detail layer (high frequency)
    
    Fungsi energi yang diminimalkan:
    E = Σ(G_i - a_i * I_i - b_i)^2 + ε * (a_i^2 + b_i^2)
    """
    
    @staticmethod
    def apply(image, guidance_image=None, radius=30, epsilon=0.01):
        """
        Aplikasi Guided Filter
        
        Args:
            image: Input image yang akan di-filter
            guidance_image: Citra panduan (jika None, gunakan image itu sendiri)
            radius: Kernel radius
            epsilon: Regularization parameter
            
        Returns:
            Filtered image
        """
        if guidance_image is None:
            guidance_image = image
        
        # Convert ke float
        if image.dtype != np.float32:
            image = image.astype(np.float32) / 255.0
        if guidance_image.dtype != np.float32:
            guidance_image = guidance_image.astype(np.float32) / 255.0
        
        # Implementasi Guided Filter menggunakan scipy
        # Ini adalah implementasi sederhana, untuk production bisa gunakan:
        # https://github.com/wuhuikx/Guided-Filter-PyTorch
        
        # Method: Menggunakan box filter untuk rapid approximation
        mean_I = cv2.boxFilter(guidance_image, -1, (radius, radius))
        mean_p = cv2.boxFilter(image, -1, (radius, radius))
        mean_Ip = cv2.boxFilter(guidance_image * image, -1, (radius, radius))
        mean_II = cv2.boxFilter(guidance_image * guidance_image, -1, (radius, radius))
        
        var_I = mean_II - mean_I * mean_I
        cov_Ip = mean_Ip - mean_I * mean_p
        
        a = cov_Ip / (var_I + epsilon)
        b = mean_p - a * mean_I
        
        mean_a = cv2.boxFilter(a, -1, (radius, radius))
        mean_b = cv2.boxFilter(b, -1, (radius, radius))
        
        q = mean_a * guidance_image + mean_b
        
        # Konversi kembali ke uint8 jika perlu
        if q.dtype != np.uint8:
            q = np.clip(q * 255, 0, 255).astype(np.uint8)
        
        return q
    
    @staticmethod
    def extract_detail_layer(image, guidance_image=None, radius=30, epsilon=0.01):
        """
        Extract detail layer dari image menggunakan Guided Filter
        
        Detail layer = Original - Base layer (filtered)
        
        Args:
            image: Input image
            guidance_image: Citra panduan
            radius: Kernel radius
            epsilon: Regularization parameter
            
        Returns:
            Detail layer
        """
        if image.dtype != np.uint8:
            image = np.clip(image * 255, 0, 255).astype(np.uint8)
        
        base_layer = GuidedFilter.apply(image, guidance_image, radius, epsilon)
        
        # Konversi ke float untuk operasi
        img_float = image.astype(np.float32)
        base_float = base_layer.astype(np.float32)
        
        detail_layer = img_float - base_float
        
        return detail_layer, base_layer


# Convenience functions
def bilateral_denoise(image, d=9, sigma_color=75, sigma_space=75):
    """Quick bilateral filter application"""
    return BilateralFilter.apply(image, d, sigma_color, sigma_space)

def unsharp_sharpen(image, amount=1.2, radius=2.0):
    """Quick unsharp mask application"""
    return UnsharpMask.apply(image, amount, radius)

def guided_filter(image, guidance_image=None, radius=30, epsilon=0.01):
    """Quick guided filter application"""
    return GuidedFilter.apply(image, guidance_image, radius, epsilon)
