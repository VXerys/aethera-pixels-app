"""
Advanced metrics untuk image quality assessment
"""

import numpy as np
import cv2


class EdgeMetrics:
    """
    Metrics berdasarkan analisis edge/kontur
    """
    
    @staticmethod
    def edge_response(image):
        """
        Calculate edge response strength
        
        Mengukur seberapa jelas edge/detail dalam image
        
        Args:
            image: Input image
            
        Returns:
            Edge response score (higher = more detailed)
        """
        # Convert ke grayscale jika perlu
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        
        # Calculate gradient magnitude menggunakan Sobel
        sobelx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        sobely = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
        
        gradient_magnitude = np.sqrt(sobelx ** 2 + sobely ** 2)
        
        # Edge response = mean gradient magnitude
        edge_response = np.mean(gradient_magnitude)
        
        return edge_response
    
    @staticmethod
    def laplacian_variance(image):
        """
        Laplacian Variance - indikator sharpness
        
        Nilai tinggi = image lebih sharp. Nilai rendah = image lebih blur
        
        Args:
            image: Input image
            
        Returns:
            Laplacian variance
        """
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        variance = laplacian.var()
        
        return variance
    
    @staticmethod
    def local_contrast(image, region_size=8):
        """
        Calculate local contrast
        
        Args:
            image: Input image
            region_size: Size dari local region
            
        Returns:
            Mean local contrast score
        """
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        
        gray_float = gray.astype(np.float32) / 255.0
        
        # Calculate local min dan max
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (region_size, region_size))
        
        local_min = cv2.erode(gray_float, kernel)
        local_max = cv2.dilate(gray_float, kernel)
        
        local_contrast_map = (local_max - local_min)
        
        return np.mean(local_contrast_map)


class ColorMetrics:
    """
    Metrics untuk color distribution dan saturation
    """
    
    @staticmethod
    def color_saturation(image):
        """
        Calculate average color saturation
        
        Args:
            image: Input image (BGR)
            
        Returns:
            Mean saturation (0-100)
        """
        # Convert BGR ke HSV
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        
        # Extract S channel
        saturation = hsv[:, :, 1]
        
        # Normalize ke 0-100
        mean_saturation = np.mean(saturation) / 255.0 * 100
        
        return mean_saturation
    
    @staticmethod
    def color_entropy(image):
        """
        Calculate color entropy (measure of color diversity)
        
        Tinggi entropy = diverse colors. Rendah entropy = limited color palette
        
        Args:
            image: Input image
            
        Returns:
            Entropy value
        """
        if len(image.shape) == 3:
            # Multi-channel: combine channels
            hist_r = cv2.calcHist([image], [2], None, [256], [0, 256])
            hist_g = cv2.calcHist([image], [1], None, [256], [0, 256])
            hist_b = cv2.calcHist([image], [0], None, [256], [0, 256])
            
            hist = (hist_r + hist_g + hist_b) / 3
        else:
            hist = cv2.calcHist([image], [0], None, [256], [0, 256])
        
        # Normalize histogram
        hist = hist.flatten() / hist.sum()
        
        # Remove zeros
        hist = hist[hist > 0]
        
        # Calculate entropy
        entropy = -np.sum(hist * np.log2(hist))
        
        return entropy
    
    @staticmethod
    def histogram_spread(image):
        """
        Calculate histogram spread (how well distributed is color range)
        
        Args:
            image: Input image
            
        Returns:
            Spread score (0-1, where 1 = full range utilized)
        """
        if len(image.shape) == 3:
            spread_values = []
            for c in range(3):
                hist = cv2.calcHist([image], [c], None, [256], [0, 256])
                non_zero = np.count_nonzero(hist)
                spread = non_zero / 256.0
                spread_values.append(spread)
            return np.mean(spread_values)
        else:
            hist = cv2.calcHist([image], [0], None, [256], [0, 256])
            non_zero = np.count_nonzero(hist)
            return non_zero / 256.0


class NoiseMetrics:
    """
    Metrics untuk mengukur noise level
    """
    
    @staticmethod
    def noise_estimation(image):
        """
        Estimate noise level menggunakan Laplacian
        
        Args:
            image: Input image
            
        Returns:
            Estimated noise standard deviation
        """
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        
        # Apply Laplacian
        laplacian = cv2.Laplacian(gray, cv2.CV_32F)
        
        # Estimate noise std
        noise_sigma = np.std(laplacian)
        
        return noise_sigma
    
    @staticmethod
    def high_frequency_energy(image):
        """
        Calculate high-frequency energy (indicator of detail/noise ratio)
        
        Args:
            image: Input image
            
        Returns:
            High frequency energy
        """
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        
        # Create high-pass filter (Laplacian)
        hf = cv2.Laplacian(gray, cv2.CV_32F)
        
        # Energy = sum of squared high-frequency components
        energy = np.sum(hf ** 2)
        
        # Normalize by image size
        normalized_energy = energy / (gray.shape[0] * gray.shape[1])
        
        return normalized_energy


class QualityAssessment:
    """
    Comprehensive quality assessment
    """
    
    @staticmethod
    def calculate_quality_score(original_image, processed_image):
        """
        Calculate comprehensive quality score
        
        Returns:
            Dictionary dengan berbagai quality metrics
        """
        scores = {
            'edge_response_original': EdgeMetrics.edge_response(original_image),
            'edge_response_processed': EdgeMetrics.edge_response(processed_image),
            
            'laplacian_variance_original': EdgeMetrics.laplacian_variance(original_image),
            'laplacian_variance_processed': EdgeMetrics.laplacian_variance(processed_image),
            
            'local_contrast_original': EdgeMetrics.local_contrast(original_image),
            'local_contrast_processed': EdgeMetrics.local_contrast(processed_image),
            
            'saturation_original': ColorMetrics.color_saturation(original_image),
            'saturation_processed': ColorMetrics.color_saturation(processed_image),
            
            'entropy_original': ColorMetrics.color_entropy(original_image),
            'entropy_processed': ColorMetrics.color_entropy(processed_image),
            
            'histogram_spread_original': ColorMetrics.histogram_spread(original_image),
            'histogram_spread_processed': ColorMetrics.histogram_spread(processed_image),
            
            'noise_estimation_original': NoiseMetrics.noise_estimation(original_image),
            'noise_estimation_processed': NoiseMetrics.noise_estimation(processed_image),
            
            'hf_energy_original': NoiseMetrics.high_frequency_energy(original_image),
            'hf_energy_processed': NoiseMetrics.high_frequency_energy(processed_image),
        }
        
        return scores
    
    @staticmethod
    def format_quality_report(scores):
        """
        Format quality scores menjadi readable report
        """
        report = """
        === IMAGE QUALITY ASSESSMENT REPORT ===
        
        Edge Detection:
        - Original Edge Response: {:.2f}
        - Processed Edge Response: {:.2f}
        - Improvement: {:.2f}%
        
        Sharpness (Laplacian Variance):
        - Original: {:.2f}
        - Processed: {:.2f}
        - Improvement: {:.2f}%
        
        Local Contrast:
        - Original: {:.4f}
        - Processed: {:.4f}
        - Change: {:.2f}%
        
        Color Saturation:
        - Original: {:.2f}%
        - Processed: {:.2f}%
        - Change: {:.2f}%
        
        Color Entropy:
        - Original: {:.2f}
        - Processed: {:.2f}
        - Change: {:.2f}%
        
        Histogram Spread:
        - Original: {:.4f}
        - Processed: {:.4f}
        - Change: {:.2f}%
        
        Noise Level:
        - Original: {:.2f}
        - Processed: {:.2f}
        - Reduction: {:.2f}%
        
        High-Frequency Energy:
        - Original: {:.2f}
        - Processed: {:.2f}
        - Change: {:.2f}%
        """.format(
            scores['edge_response_original'],
            scores['edge_response_processed'],
            (scores['edge_response_processed'] - scores['edge_response_original']) / scores['edge_response_original'] * 100 if scores['edge_response_original'] != 0 else 0,
            
            scores['laplacian_variance_original'],
            scores['laplacian_variance_processed'],
            (scores['laplacian_variance_processed'] - scores['laplacian_variance_original']) / scores['laplacian_variance_original'] * 100 if scores['laplacian_variance_original'] != 0 else 0,
            
            scores['local_contrast_original'],
            scores['local_contrast_processed'],
            (scores['local_contrast_processed'] - scores['local_contrast_original']) / scores['local_contrast_original'] * 100 if scores['local_contrast_original'] != 0 else 0,
            
            scores['saturation_original'],
            scores['saturation_processed'],
            (scores['saturation_processed'] - scores['saturation_original']) / scores['saturation_original'] * 100 if scores['saturation_original'] != 0 else 0,
            
            scores['entropy_original'],
            scores['entropy_processed'],
            (scores['entropy_processed'] - scores['entropy_original']) / scores['entropy_original'] * 100 if scores['entropy_original'] != 0 else 0,
            
            scores['histogram_spread_original'],
            scores['histogram_spread_processed'],
            (scores['histogram_spread_processed'] - scores['histogram_spread_original']) / scores['histogram_spread_original'] * 100 if scores['histogram_spread_original'] != 0 else 0,
            
            scores['noise_estimation_original'],
            scores['noise_estimation_processed'],
            (scores['noise_estimation_original'] - scores['noise_estimation_processed']) / scores['noise_estimation_original'] * 100 if scores['noise_estimation_original'] != 0 else 0,
            
            scores['hf_energy_original'],
            scores['hf_energy_processed'],
            (scores['hf_energy_processed'] - scores['hf_energy_original']) / scores['hf_energy_original'] * 100 if scores['hf_energy_original'] != 0 else 0,
        )
        
        return report
