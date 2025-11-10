"""
Implementasi Color Grading dan Texture Enhancement untuk Aethera Pixelis
- 3D LUT: Color Grading non-linear
- Histogram Matching: Penyesuaian distribusi warna
- Film Grain Simulation: Tekstur simulasi butiran film
"""

import cv2
import numpy as np
from scipy import ndimage


class LUT3D:
    """
    3D Look-Up Table untuk Color Grading
    
    Mengubah skema warna citra secara cepat dengan memetakan
    triplet (R, G, B) input ke (R', G', B') output melalui kubus 3D.
    
    Kompleksitas: O(1) lookup time (sangat cepat)
    """
    
    @staticmethod
    def create_cinematic_lut(size=16):
        """
        Buat LUT preset untuk efek sinematik
        
        Efek: Shadow menjadi lebih blue/teal, highlight menjadi warm/orange
        """
        lut = np.zeros((size, size, size, 3), dtype=np.uint8)
        
        for r in range(size):
            for g in range(size):
                for b in range(size):
                    # Normalize values 0-1
                    r_norm = r / (size - 1)
                    g_norm = g / (size - 1)
                    b_norm = b / (size - 1)
                    
                    # Calculate luma (brightness)
                    luma = 0.299 * r_norm + 0.587 * g_norm + 0.114 * b_norm
                    
                    # Shadow enhancement (Teal): increase blue, reduce red
                    shadow_boost = 1 - luma  # Shadow mask
                    r_out = r_norm * (1 - 0.2 * shadow_boost)
                    g_out = g_norm * (1 + 0.1 * shadow_boost)
                    b_out = b_norm * (1 + 0.3 * shadow_boost)
                    
                    # Highlight enhancement (Orange): increase red, reduce blue
                    highlight_boost = luma  # Highlight mask
                    r_out = r_out * (1 + 0.2 * highlight_boost)
                    g_out = g_out * (1 + 0.1 * highlight_boost)
                    b_out = b_out * (1 - 0.2 * highlight_boost)
                    
                    # Convert to 0-255 range
                    lut[r, g, b, 0] = np.clip(r_out * 255, 0, 255)
                    lut[r, g, b, 1] = np.clip(g_out * 255, 0, 255)
                    lut[r, g, b, 2] = np.clip(b_out * 255, 0, 255)
        
        return lut
    
    @staticmethod
    def apply_lut(image, lut, lut_size=16):
        """
        Aplikasi 3D LUT ke image
        
        Args:
            image: Input image (BGR format, uint8)
            lut: 3D LUT array (shape: (size, size, size, 3))
            lut_size: Ukuran LUT (default 16)
            
        Returns:
            LUT-applied image
        """
        # Quantize image values ke LUT grid
        image_quantized = (image.astype(np.float32) * (lut_size - 1) / 255.0).astype(np.int32)
        image_quantized = np.clip(image_quantized, 0, lut_size - 1)
        
        # Apply LUT
        result = np.zeros_like(image, dtype=np.uint8)
        for c in range(3):  # RGB channels
            result[:, :, c] = lut[image_quantized[:, :, 0], 
                                   image_quantized[:, :, 1], 
                                   image_quantized[:, :, 2], c]
        
        return result


class HistogramMatching:
    """
    Histogram Matching untuk Color Correction
    
    Menyesuaikan histogram input agar sesuai dengan histogram
    referensi target film.
    """
    
    @staticmethod
    def calculate_histogram(image, bins=256):
        """Calculate histogram dari image"""
        hist = cv2.calcHist([image], [0], None, [bins], [0, 256])
        hist = cv2.normalize(hist, hist).flatten()
        return hist
    
    @staticmethod
    def calculate_cdf(hist):
        """Calculate cumulative distribution function dari histogram"""
        cdf = hist.cumsum()
        cdf = (cdf - cdf.min()) * 255 / (cdf.max() - cdf.min())
        return cdf.astype(np.uint8)
    
    @staticmethod
    def match_histogram(image, reference_image):
        """
        Match histogram dari image ke reference_image
        
        Args:
            image: Source image yang akan diubah
            reference_image: Reference image untuk target histogram
            
        Returns:
            Image dengan histogram yang match reference
        """
        if len(image.shape) == 3:
            # Multi-channel image
            result = np.zeros_like(image)
            for i in range(image.shape[2]):
                src_hist = HistogramMatching.calculate_histogram(image[:, :, i])
                ref_hist = HistogramMatching.calculate_histogram(reference_image[:, :, i])
                
                src_cdf = HistogramMatching.calculate_cdf(src_hist)
                ref_cdf = HistogramMatching.calculate_cdf(ref_hist)
                
                # Build mapping
                mapping = np.zeros(256, dtype=np.uint8)
                for j in range(256):
                    mapping[j] = np.argmin(np.abs(ref_cdf - src_cdf[j]))
                
                # Apply mapping
                result[:, :, i] = cv2.LUT(image[:, :, i], mapping)
            
            return result
        else:
            # Grayscale image
            src_hist = HistogramMatching.calculate_histogram(image)
            ref_hist = HistogramMatching.calculate_histogram(reference_image)
            
            src_cdf = HistogramMatching.calculate_cdf(src_hist)
            ref_cdf = HistogramMatching.calculate_cdf(ref_hist)
            
            mapping = np.zeros(256, dtype=np.uint8)
            for j in range(256):
                mapping[j] = np.argmin(np.abs(ref_cdf - src_cdf[j]))
            
            return cv2.LUT(image, mapping)
    
    @staticmethod
    def teal_orange_preset(image):
        """
        Apply Teal-Orange color grading preset
        
        Shadow: Teal (cyan/blue)
        Midtone: Neutral
        Highlight: Orange/warm
        """
        # Convert to LAB color space untuk manipulasi lebih baik
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        
        # Adjust a channel (green-red axis)
        # Shadow: negative (green/cyan), Highlight: positive (red/orange)
        a_adjusted = a.astype(np.float32)
        
        # Create mask berdasarkan luminance
        l_norm = l.astype(np.float32) / 255.0
        shadow_mask = (1 - l_norm)
        highlight_mask = l_norm
        
        # Apply adjustment
        a_adjusted -= 30 * shadow_mask  # Shadow: shift to green
        a_adjusted += 20 * highlight_mask  # Highlight: shift to red
        
        # Adjust b channel (yellow-blue axis)
        # Shadow: positive (blue), Highlight: neutral
        b_adjusted = b.astype(np.float32)
        b_adjusted += 20 * shadow_mask  # Shadow: shift to blue
        
        # Combine dan convert kembali
        a_adjusted = np.clip(a_adjusted, 0, 255).astype(np.uint8)
        b_adjusted = np.clip(b_adjusted, 0, 255).astype(np.uint8)
        
        lab_result = cv2.merge([l, a_adjusted, b_adjusted])
        result = cv2.cvtColor(lab_result, cv2.COLOR_LAB2BGR)
        
        return result


class FilmGrainSimulation:
    """
    Film Grain Simulation untuk menambahkan tekstur butiran film
    
    Menambahkan lapisan noise frekuensi tinggi Gaussian atau Perlin noise
    untuk memberikan kesan otentik dan tekstural pada hasil akhir.
    """
    
    @staticmethod
    def add_gaussian_grain(image, intensity=0.15, size=2.0):
        """
        Tambah Gaussian grain ke image
        
        Args:
            image: Input image
            intensity: Grain intensity (0.0 - 1.0)
            size: Grain size (kernel size)
            
        Returns:
            Image dengan grain
        """
        # Generate Gaussian noise
        noise = np.random.normal(0, intensity * 255, image.shape)
        
        # Smooth noise slightly untuk lebih natural
        if size > 1:
            noise = cv2.GaussianBlur(noise, (int(size * 2) | 1, int(size * 2) | 1), 1)
        
        # Combine with original image
        result = image.astype(np.float32) + noise
        result = np.clip(result, 0, 255).astype(np.uint8)
        
        return result
    
    @staticmethod
    def add_perlin_grain(image, intensity=0.15, scale=8):
        """
        Tambah Perlin-like noise grain ke image
        
        Menggunakan sine wave untuk simulasi Perlin noise yang lebih smooth
        
        Args:
            image: Input image
            intensity: Grain intensity (0.0 - 1.0)
            scale: Noise scale / frequency
            
        Returns:
            Image dengan grain
        """
        h, w = image.shape[:2]
        
        # Create 2D Perlin-like noise menggunakan sine waves
        x = np.linspace(0, 4 * np.pi, w)
        y = np.linspace(0, 4 * np.pi, h)
        X, Y = np.meshgrid(x, y)
        
        noise1 = np.sin(X / scale) * np.cos(Y / scale)
        noise2 = np.sin(X / (scale * 2)) * np.cos(Y / (scale * 2))
        noise = (noise1 + noise2) / 2  # Blend multiple scales
        
        # Normalize noise ke range [-intensity*255, intensity*255]
        noise = (noise / noise.max()) * intensity * 255
        
        # Add to image
        if len(image.shape) == 3:
            noise = np.stack([noise] * image.shape[2], axis=2)
        
        result = image.astype(np.float32) + noise
        result = np.clip(result, 0, 255).astype(np.uint8)
        
        return result
    
    @staticmethod
    def add_chromatic_grain(image, intensity=0.15, color_shift=0.5):
        """
        Tambah chromatic aberration grain untuk efek film vintage
        
        Args:
            image: Input image (BGR)
            intensity: Grain intensity
            color_shift: Color shift amount
            
        Returns:
            Image dengan chromatic grain
        """
        if len(image.shape) != 3 or image.shape[2] != 3:
            # If not color image, use regular grain
            return FilmGrainSimulation.add_gaussian_grain(image, intensity)
        
        # Split channels
        b, g, r = cv2.split(image)
        
        # Add different grain ke setiap channel
        b_grain = FilmGrainSimulation.add_gaussian_grain(b, intensity * 1.2)
        g_grain = FilmGrainSimulation.add_gaussian_grain(g, intensity * 1.0)
        r_grain = FilmGrainSimulation.add_gaussian_grain(r, intensity * 0.8)
        
        # Merge channels
        result = cv2.merge([b_grain, g_grain, r_grain])
        
        return result


# Convenience functions
def apply_lut_cinematic(image):
    """Quick apply cinematic LUT"""
    lut = LUT3D.create_cinematic_lut()
    return LUT3D.apply_lut(image, lut)

def apply_teal_orange(image):
    """Quick apply teal-orange color grading"""
    return HistogramMatching.teal_orange_preset(image)

def add_film_grain(image, intensity=0.15, grain_type='gaussian'):
    """Quick add film grain"""
    if grain_type == 'gaussian':
        return FilmGrainSimulation.add_gaussian_grain(image, intensity)
    elif grain_type == 'perlin':
        return FilmGrainSimulation.add_perlin_grain(image, intensity)
    elif grain_type == 'chromatic':
        return FilmGrainSimulation.add_chromatic_grain(image, intensity)
    else:
        return FilmGrainSimulation.add_gaussian_grain(image, intensity)
