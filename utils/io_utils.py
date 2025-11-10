"""
Utility functions untuk I/O dan Image Processing
"""

import cv2
import numpy as np
from PIL import Image
import io
from pathlib import Path


class ImageIO:
    """
    Image Input/Output utilities
    """
    
    SUPPORTED_FORMATS = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}
    
    @staticmethod
    def load_image(image_path):
        """
        Load image dari file path
        
        Args:
            image_path: Path ke image file
            
        Returns:
            Image dalam format BGR (OpenCV)
        """
        image = cv2.imread(str(image_path))
        if image is None:
            raise ValueError(f"Tidak bisa load image dari {image_path}")
        return image
    
    @staticmethod
    def load_image_from_pil(pil_image):
        """
        Convert PIL Image ke OpenCV format (BGR)
        
        Args:
            pil_image: PIL Image object
            
        Returns:
            Image dalam format BGR
        """
        # Convert PIL (RGB) ke OpenCV (BGR)
        numpy_array = np.array(pil_image)
        if len(numpy_array.shape) == 3 and numpy_array.shape[2] == 3:
            return cv2.cvtColor(numpy_array, cv2.COLOR_RGB2BGR)
        elif len(numpy_array.shape) == 3 and numpy_array.shape[2] == 4:
            # RGBA to BGR
            rgba = cv2.cvtColor(numpy_array, cv2.COLOR_RGBA2BGR)
            return rgba
        else:
            # Grayscale
            return numpy_array
    
    @staticmethod
    def load_image_from_bytes(image_bytes):
        """
        Load image dari bytes
        
        Args:
            image_bytes: Image data dalam bytes
            
        Returns:
            Image dalam format BGR
        """
        nparr = np.frombuffer(image_bytes, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        return image
    
    @staticmethod
    def save_image(image, output_path, quality=95):
        """
        Save image ke file
        
        Args:
            image: Image dalam format BGR
            output_path: Path untuk menyimpan
            quality: JPEG quality (1-100)
        """
        if output_path.lower().endswith(('.jpg', '.jpeg')):
            cv2.imwrite(str(output_path), image, [cv2.IMWRITE_JPEG_QUALITY, quality])
        elif output_path.lower().endswith('.png'):
            cv2.imwrite(str(output_path), image, [cv2.IMWRITE_PNG_COMPRESSION, 9])
        else:
            cv2.imwrite(str(output_path), image)
    
    @staticmethod
    def save_image_to_bytes(image, format='PNG'):
        """
        Save image ke bytes buffer
        
        Args:
            image: Image dalam format BGR
            format: Format output ('PNG', 'JPEG', etc.)
            
        Returns:
            BytesIO buffer
        """
        # Convert BGR ke RGB untuk PIL
        if len(image.shape) == 3 and image.shape[2] == 3:
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        else:
            image_rgb = image
        
        # Convert ke PIL
        pil_image = Image.fromarray(image_rgb)
        
        # Save ke buffer
        buf = io.BytesIO()
        pil_image.save(buf, format=format)
        buf.seek(0)
        
        return buf
    
    @staticmethod
    def convert_bgr_to_rgb(image_bgr):
        """Convert BGR (OpenCV) ke RGB"""
        if len(image_bgr.shape) == 3 and image_bgr.shape[2] == 3:
            return cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        return image_bgr
    
    @staticmethod
    def convert_rgb_to_bgr(image_rgb):
        """Convert RGB ke BGR (OpenCV)"""
        if len(image_rgb.shape) == 3 and image_rgb.shape[2] == 3:
            return cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
        return image_rgb
    
    @staticmethod
    def get_image_info(image):
        """
        Get informasi tentang image
        
        Returns:
            Dictionary dengan info
        """
        h, w = image.shape[:2]
        channels = 1 if len(image.shape) == 2 else image.shape[2]
        dtype = str(image.dtype)
        size_mb = (image.nbytes / 1024 / 1024)
        
        return {
            'width': w,
            'height': h,
            'channels': channels,
            'dtype': dtype,
            'size_mb': size_mb,
            'resolution': f"{w}x{h}"
        }
    
    @staticmethod
    def resize_image(image, width=None, height=None, scale=None):
        """
        Resize image dengan berbagai opsi
        
        Args:
            image: Input image
            width: Target width (jika None, maintain aspect ratio)
            height: Target height (jika None, maintain aspect ratio)
            scale: Scaling factor (jika ini diberikan, ignore width/height)
            
        Returns:
            Resized image
        """
        h, w = image.shape[:2]
        
        if scale:
            new_w, new_h = int(w * scale), int(h * scale)
        elif width and height:
            new_w, new_h = width, height
        elif width:
            new_w = width
            new_h = int(h * (width / w))
        elif height:
            new_h = height
            new_w = int(w * (height / h))
        else:
            return image
        
        return cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)


class ImageMetrics:
    """
    Image quality metrics untuk evaluasi
    """
    
    @staticmethod
    def psnr(image_true, image_pred):
        """
        Peak Signal-to-Noise Ratio (PSNR)
        
        Formula:
        PSNR = 20 * log10(MAX_PIXEL / sqrt(MSE))
        
        Lebih tinggi lebih baik. Typical range: 20-40 dB
        
        Args:
            image_true: Reference image (ground truth)
            image_pred: Predicted/processed image
            
        Returns:
            PSNR value in dB
        """
        # Ensure same shape
        if image_true.shape != image_pred.shape:
            return None
        
        # Convert to float
        true_float = image_true.astype(np.float32)
        pred_float = image_pred.astype(np.float32)
        
        # Calculate MSE
        mse = np.mean((true_float - pred_float) ** 2)
        
        if mse == 0:
            return float('inf')
        
        # Calculate PSNR
        max_pixel = 255.0
        psnr_value = 20 * np.log10(max_pixel / np.sqrt(mse))
        
        return psnr_value
    
    @staticmethod
    def ssim(image_true, image_pred, window_size=11, sigma=1.5):
        """
        Structural Similarity Index (SSIM)
        
        Formula:
        SSIM = (2*μx*μy + C1) * (2*σxy + C2) / ((μx^2 + μy^2 + C1) * (σx^2 + σy^2 + C2))
        
        Range: -1 to 1. Nilai 1 = identik. Typical range: 0.5-1.0
        
        Args:
            image_true: Reference image
            image_pred: Predicted/processed image
            window_size: Gaussian window size
            sigma: Gaussian standard deviation
            
        Returns:
            SSIM value (mean across channels)
        """
        # Ensure same shape
        if image_true.shape != image_pred.shape:
            return None
        
        # Convert to float
        true_float = image_true.astype(np.float32) / 255.0
        pred_float = image_pred.astype(np.float32) / 255.0
        
        if len(image_true.shape) == 3:
            # Multi-channel: calculate SSIM untuk setiap channel, lalu average
            ssim_values = []
            for c in range(image_true.shape[2]):
                ssim_val = ImageMetrics._ssim_single_channel(
                    true_float[:, :, c],
                    pred_float[:, :, c],
                    window_size,
                    sigma
                )
                ssim_values.append(ssim_val)
            return np.mean(ssim_values)
        else:
            # Grayscale
            return ImageMetrics._ssim_single_channel(true_float, pred_float, window_size, sigma)
    
    @staticmethod
    def _ssim_single_channel(img1, img2, window_size=11, sigma=1.5):
        """SSIM untuk single channel"""
        from scipy import signal
        
        # Constants
        C1 = 0.01 ** 2
        C2 = 0.03 ** 2
        
        # Create Gaussian window
        kernel = cv2.getGaussianKernel(window_size, sigma)
        window = kernel @ kernel.T
        window = window.astype(np.float32) / window.sum()
        
        # Calculate local means
        mu1 = signal.correlate2d(img1, window, mode='valid')
        mu2 = signal.correlate2d(img2, window, mode='valid')
        
        # Calculate local variances and covariance
        mu1_sq = mu1 ** 2
        mu2_sq = mu2 ** 2
        mu1_mu2 = mu1 * mu2
        
        img1_sq = signal.correlate2d(img1 ** 2, window, mode='valid')
        img2_sq = signal.correlate2d(img2 ** 2, window, mode='valid')
        img1_img2 = signal.correlate2d(img1 * img2, window, mode='valid')
        
        sigma1_sq = img1_sq - mu1_sq
        sigma2_sq = img2_sq - mu2_sq
        sigma12 = img1_img2 - mu1_mu2
        
        # Calculate SSIM
        numerator = (2 * mu1_mu2 + C1) * (2 * sigma12 + C2)
        denominator = (mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2)
        
        ssim_map = numerator / denominator
        
        return np.mean(ssim_map)
    
    @staticmethod
    def mse(image_true, image_pred):
        """
        Mean Squared Error
        
        Args:
            image_true: Reference image
            image_pred: Predicted/processed image
            
        Returns:
            MSE value
        """
        if image_true.shape != image_pred.shape:
            return None
        
        true_float = image_true.astype(np.float32)
        pred_float = image_pred.astype(np.float32)
        
        mse_value = np.mean((true_float - pred_float) ** 2)
        return mse_value
    
    @staticmethod
    def calculate_all_metrics(image_true, image_pred):
        """
        Calculate semua metrics sekaligus
        
        Returns:
            Dictionary dengan semua metric values
        """
        return {
            'psnr': ImageMetrics.psnr(image_true, image_pred),
            'ssim': ImageMetrics.ssim(image_true, image_pred),
            'mse': ImageMetrics.mse(image_true, image_pred),
        }


# Convenience functions
def load_img(path):
    """Quick load image"""
    return ImageIO.load_image(path)

def save_img(image, path):
    """Quick save image"""
    ImageIO.save_image(image, path)

def get_psnr(img_true, img_pred):
    """Quick calculate PSNR"""
    return ImageMetrics.psnr(img_true, img_pred)

def get_ssim(img_true, img_pred):
    """Quick calculate SSIM"""
    return ImageMetrics.ssim(img_true, img_pred)
