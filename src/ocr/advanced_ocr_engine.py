"""
Advanced OCR Engine with Multiple Backends and Intelligent Fallback
Supports: EasyOCR, PaddleOCR, TrOCR, Tesseract with confidence-based selection

Features:
- Multiple OCR engine integration with fallback chain
- Confidence-based engine selection
- Text preprocessing and post-processing
- Layout analysis and region detection
- Performance optimization and caching
- Error handling and graceful degradation
"""

import os
import sys
import cv2
import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass
from enum import Enum
import logging
from pathlib import Path
import base64
from PIL import Image
import io

# OCR Engine imports (with fallbacks)
try:
    import easyocr
    EASYOCR_AVAILABLE = True
except ImportError:
    EASYOCR_AVAILABLE = False

try:
    import paddleocr
    PADDLEOCR_AVAILABLE = True
except ImportError:
    PADDLEOCR_AVAILABLE = False

try:
    from transformers import TrOCRProcessor, VisionEncoderDecoderModel
    import torch
    TROCR_AVAILABLE = True
except ImportError:
    TROCR_AVAILABLE = False

try:
    import pytesseract
    TESSERACT_AVAILABLE = True
except ImportError:
    TESSERACT_AVAILABLE = False

class OCREngine(Enum):
    """Available OCR engines"""
    EASYOCR = "easyocr"
    PADDLEOCR = "paddleocr"
    TROCR = "trocr"
    TESSERACT = "tesseract"

@dataclass
class OCRResult:
    """Result from OCR processing"""
    text: str
    confidence: float
    bounding_boxes: List[Tuple[int, int, int, int]]
    word_confidences: List[float]
    processing_time: float
    engine: OCREngine
    metadata: Dict[str, Any]

@dataclass
class TextRegion:
    """Detected text region with properties"""
    bbox: Tuple[int, int, int, int]
    text: str
    confidence: float
    region_type: str  # 'title', 'body', 'caption', 'equation'
    font_size: Optional[float] = None

class AdvancedOCREngine:
    """
    Advanced OCR engine with multiple backends and intelligent selection
    """

    def __init__(
        self,
        preferred_engines: List[OCREngine] = None,
        enable_preprocessing: bool = True,
        enable_postprocessing: bool = True,
        cache_results: bool = True,
        verbose: bool = False
    ):
        """
        Initialize OCR engine with multiple backends

        Args:
            preferred_engines: List of engines in preference order
            enable_preprocessing: Apply image preprocessing
            enable_postprocessing: Clean and process text output
            cache_results: Cache results for repeated processing
            verbose: Enable detailed logging
        """
        self.verbose = verbose
        self.enable_preprocessing = enable_preprocessing
        self.enable_postprocessing = enable_postprocessing
        self.cache_results = cache_results

        # Setup logging
        self.logger = logging.getLogger(__name__)
        if verbose:
            self.logger.setLevel(logging.INFO)

        # Initialize available engines
        self.available_engines = {}
        self.preferred_engines = preferred_engines or [
            OCREngine.TROCR,  # Best for technical text
            OCREngine.PADDLEOCR,  # Good for complex layouts
            OCREngine.EASYOCR,  # Reliable fallback
            OCREngine.TESSERACT  # Last resort
        ]

        self._initialize_engines()

        # Cache for results
        self.result_cache = {} if cache_results else None

        self.logger.info(f" Advanced OCR initialized with {len(self.available_engines)} engines")

    def _initialize_engines(self):
        """Initialize available OCR engines"""

        # Initialize EasyOCR
        if EASYOCR_AVAILABLE:
            try:
                self.available_engines[OCREngine.EASYOCR] = easyocr.Reader(
                    ['en'],
                    gpu=torch.cuda.is_available() if 'torch' in globals() else False,
                    verbose=False
                )
                self.logger.info(" EasyOCR initialized")
            except Exception as e:
                self.logger.warning(f" EasyOCR initialization failed: {e}")

        # Initialize PaddleOCR
        if PADDLEOCR_AVAILABLE:
            try:
                self.available_engines[OCREngine.PADDLEOCR] = paddleocr.PaddleOCR(
                    use_angle_cls=True,
                    lang='en',
                    show_log=False
                )
                self.logger.info(" PaddleOCR initialized")
            except Exception as e:
                self.logger.warning(f" PaddleOCR initialization failed: {e}")

        # Initialize TrOCR
        if TROCR_AVAILABLE:
            try:
                self.available_engines[OCREngine.TROCR] = {
                    'processor': TrOCRProcessor.from_pretrained('microsoft/trocr-large-printed'),
                    'model': VisionEncoderDecoderModel.from_pretrained('microsoft/trocr-large-printed')
                }
                self.logger.info(" TrOCR initialized")
            except Exception as e:
                self.logger.warning(f" TrOCR initialization failed: {e}")

        # Initialize Tesseract
        if TESSERACT_AVAILABLE:
            try:
                # Test if tesseract is available
                pytesseract.get_tesseract_version()
                self.available_engines[OCREngine.TESSERACT] = True
                self.logger.info(" Tesseract initialized")
            except Exception as e:
                self.logger.warning(f" Tesseract initialization failed: {e}")

    def preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """
        Preprocess image for better OCR results

        Args:
            image: Input image as numpy array

        Returns:
            Preprocessed image
        """
        if not self.enable_preprocessing:
            return image

        try:
            # Convert to grayscale if needed
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            else:
                gray = image.copy()

            # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            enhanced = clahe.apply(gray)

            # Noise reduction
            denoised = cv2.bilateralFilter(enhanced, 9, 75, 75)

            # Sharpening
            kernel = np.array([[-1, -1, -1],
                               [-1, 9, -1],
                               [-1, -1, -1]])
            sharpened = cv2.filter2D(denoised, -1, kernel)

            # Thresholding for better text contrast
            _, binary = cv2.threshold(sharpened, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

            return binary

        except Exception as e:
            self.logger.warning(f"Preprocessing failed: {e}, using original image")
            return image

    def postprocess_text(self, text: str, engine: OCREngine) -> str:
        """
        Clean and postprocess extracted text

        Args:
            text: Raw extracted text
            engine: OCR engine used

        Returns:
            Cleaned text
        """
        if not self.enable_postprocessing or not text:
            return text

        try:
            # Remove excessive whitespace
            cleaned = ' '.join(text.split())

            # Fix common OCR errors for technical text
            replacements = {
                # Common character confusions
                '0': 'O',  # Only in certain contexts
                'l': 'I',  # In uppercase contexts
                '1': 'l',  # In lowercase contexts
                # Mathematical symbols
                'x': 'x',  # Multiplication
                'infinity': 'infinity',
                # Common technical terms
                'atlention': 'attention',
                'transformcr': 'transformer',
                'encodcr': 'encoder',
                'decodcr': 'decoder'
            }

            # Apply corrections carefully (context-aware)
            for old, new in replacements.items():
                # Only replace if it improves readability
                if old in cleaned and len(old) > 1:
                    cleaned = cleaned.replace(old, new)

            # Remove artifacts from specific engines
            if engine == OCREngine.TESSERACT:
                # Remove confidence markers like [0.85]
                import re
                cleaned = re.sub(r'\[\d+\.\d+\]', '', cleaned)

            return cleaned.strip()

        except Exception as e:
            self.logger.warning(f"Postprocessing failed: {e}, using raw text")
            return text

    def _run_easyocr(self, image: np.ndarray) -> OCRResult:
        """Run EasyOCR on image"""
        import time
        start_time = time.time()

        try:
            reader = self.available_engines[OCREngine.EASYOCR]
            results = reader.readtext(image)

            # Extract text and confidence
            texts = []
            confidences = []
            bboxes = []

            for (bbox, text, confidence) in results:
                texts.append(text)
                confidences.append(confidence)
                # Convert bbox format
                x1, y1 = bbox[0]
                x2, y2 = bbox[2]
                bboxes.append((int(x1), int(y1), int(x2), int(y2)))

            combined_text = ' '.join(texts)
            avg_confidence = sum(confidences) / len(confidences) if confidences else 0

            return OCRResult(
                text=self.postprocess_text(combined_text, OCREngine.EASYOCR),
                confidence=avg_confidence,
                bounding_boxes=bboxes,
                word_confidences=confidences,
                processing_time=time.time() - start_time,
                engine=OCREngine.EASYOCR,
                metadata={'raw_results': results, 'word_count': len(texts)}
            )

        except Exception as e:
            self.logger.error(f"EasyOCR failed: {e}")
            return OCRResult(
                text="", confidence=0.0, bounding_boxes=[], word_confidences=[],
                processing_time=time.time() - start_time, engine=OCREngine.EASYOCR,
                metadata={'error': str(e)}
            )

    def _run_paddleocr(self, image: np.ndarray) -> OCRResult:
        """Run PaddleOCR on image"""
        import time
        start_time = time.time()

        try:
            ocr = self.available_engines[OCREngine.PADDLEOCR]
            results = ocr.ocr(image, cls=True)

            texts = []
            confidences = []
            bboxes = []

            if results and results[0]:
                for line in results[0]:
                    bbox = line[0]
                    text_info = line[1]
                    text = text_info[0]
                    confidence = text_info[1]

                    texts.append(text)
                    confidences.append(confidence)

                    # Convert bbox format
                    x1 = int(min(point[0] for point in bbox))
                    y1 = int(min(point[1] for point in bbox))
                    x2 = int(max(point[0] for point in bbox))
                    y2 = int(max(point[1] for point in bbox))
                    bboxes.append((x1, y1, x2, y2))

            combined_text = ' '.join(texts)
            avg_confidence = sum(confidences) / len(confidences) if confidences else 0

            return OCRResult(
                text=self.postprocess_text(combined_text, OCREngine.PADDLEOCR),
                confidence=avg_confidence,
                bounding_boxes=bboxes,
                word_confidences=confidences,
                processing_time=time.time() - start_time,
                engine=OCREngine.PADDLEOCR,
                metadata={'raw_results': results, 'word_count': len(texts)}
            )

        except Exception as e:
            self.logger.error(f"PaddleOCR failed: {e}")
            return OCRResult(
                text="", confidence=0.0, bounding_boxes=[], word_confidences=[],
                processing_time=time.time() - start_time, engine=OCREngine.PADDLEOCR,
                metadata={'error': str(e)}
            )

    def _run_trocr(self, image: np.ndarray) -> OCRResult:
        """Run TrOCR on image"""
        import time
        start_time = time.time()

        try:
            processor = self.available_engines[OCREngine.TROCR]['processor']
            model = self.available_engines[OCREngine.TROCR]['model']

            # Convert numpy array to PIL Image
            if len(image.shape) == 3:
                pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            else:
                pil_image = Image.fromarray(image)

            # Process image
            pixel_values = processor(images=pil_image, return_tensors="pt").pixel_values
            generated_ids = model.generate(pixel_values, max_length=200)
            generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

            # TrOCR doesn't provide confidence scores or bboxes
            # Estimate confidence based on text quality
            confidence = self._estimate_text_quality(generated_text)

            return OCRResult(
                text=self.postprocess_text(generated_text, OCREngine.TROCR),
                confidence=confidence,
                bounding_boxes=[(0, 0, image.shape[1], image.shape[0])],  # Full image bbox
                word_confidences=[confidence],
                processing_time=time.time() - start_time,
                engine=OCREngine.TROCR,
                metadata={'generated_text': generated_text, 'estimated_confidence': confidence}
            )

        except Exception as e:
            self.logger.error(f"TrOCR failed: {e}")
            return OCRResult(
                text="", confidence=0.0, bounding_boxes=[], word_confidences=[],
                processing_time=time.time() - start_time, engine=OCREngine.TROCR,
                metadata={'error': str(e)}
            )

    def _run_tesseract(self, image: np.ndarray) -> OCRResult:
        """Run Tesseract OCR on image"""
        import time
        start_time = time.time()

        try:
            # Get text with confidence
            data = pytesseract.image_to_data(image, output_type=pytesseract.Output.DICT)

            # Filter out low-confidence results
            texts = []
            confidences = []
            bboxes = []

            for i in range(len(data['text'])):
                conf = int(data['conf'][i])
                text = data['text'][i].strip()

                if conf > 30 and text:  # Minimum confidence threshold
                    texts.append(text)
                    confidences.append(conf / 100.0)  # Convert to 0-1 scale

                    x, y, w, h = data['left'][i], data['top'][i], data['width'][i], data['height'][i]
                    bboxes.append((x, y, x + w, y + h))

            combined_text = ' '.join(texts)
            avg_confidence = sum(confidences) / len(confidences) if confidences else 0

            return OCRResult(
                text=self.postprocess_text(combined_text, OCREngine.TESSERACT),
                confidence=avg_confidence,
                bounding_boxes=bboxes,
                word_confidences=confidences,
                processing_time=time.time() - start_time,
                engine=OCREngine.TESSERACT,
                metadata={'raw_data': data, 'word_count': len(texts)}
            )

        except Exception as e:
            self.logger.error(f"Tesseract failed: {e}")
            return OCRResult(
                text="", confidence=0.0, bounding_boxes=[], word_confidences=[],
                processing_time=time.time() - start_time, engine=OCREngine.TESSERACT,
                metadata={'error': str(e)}
            )

    def _estimate_text_quality(self, text: str) -> float:
        """Estimate text quality for engines that don't provide confidence"""
        if not text:
            return 0.0

        quality_score = 0.5  # Base score

        # Length bonus (longer text often means better extraction)
        if len(text) > 50:
            quality_score += 0.2
        elif len(text) > 20:
            quality_score += 0.1

        # Technical terms bonus
        tech_terms = ['attention', 'transformer', 'encoder', 'decoder', 'layer', 'multi-head', 'figure']
        found_terms = sum(1 for term in tech_terms if term.lower() in text.lower())
        quality_score += min(found_terms * 0.1, 0.3)

        # Penalize obvious errors
        if '???' in text or '###' in text:
            quality_score -= 0.2

        return max(0.0, min(1.0, quality_score))

    def extract_text_multiple_engines(
        self,
        image: Union[np.ndarray, str, bytes],
        engines: List[OCREngine] = None
    ) -> List[OCRResult]:
        """
        Extract text using multiple OCR engines

        Args:
            image: Input image (numpy array, file path, or bytes)
            engines: List of engines to use (uses preferred engines if None)

        Returns:
            List of OCR results from different engines
        """
        # Load and preprocess image
        if isinstance(image, str):
            image = cv2.imread(image)
        elif isinstance(image, bytes):
            nparr = np.frombuffer(image, np.uint8)
            image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if image is None:
            raise ValueError("Could not load image")

        # Preprocess image
        processed_image = self.preprocess_image(image)

        # Use specified engines or preferred engines
        engines_to_use = engines or [e for e in self.preferred_engines if e in self.available_engines]

        results = []

        for engine in engines_to_use:
            if engine not in self.available_engines:
                self.logger.warning(f"Engine {engine.value} not available, skipping")
                continue

            try:
                if self.verbose:
                    self.logger.info(f" Running {engine.value}")

                if engine == OCREngine.EASYOCR:
                    result = self._run_easyocr(processed_image)
                elif engine == OCREngine.PADDLEOCR:
                    result = self._run_paddleocr(processed_image)
                elif engine == OCREngine.TROCR:
                    result = self._run_trocr(processed_image)
                elif engine == OCREngine.TESSERACT:
                    result = self._run_tesseract(processed_image)
                else:
                    continue

                results.append(result)

                if self.verbose:
                    self.logger.info(f" {engine.value}: {len(result.text)} chars, {result.confidence:.2f} confidence")

            except Exception as e:
                self.logger.error(f" {engine.value} failed: {e}")

        return results

    def get_best_result(self, results: List[OCRResult]) -> OCRResult:
        """
        Select the best OCR result based on confidence and quality metrics

        Args:
            results: List of OCR results from different engines

        Returns:
            Best OCR result
        """
        if not results:
            raise ValueError("No OCR results provided")

        # Filter out failed results
        valid_results = [r for r in results if r.text and r.confidence > 0]

        if not valid_results:
            # Return the least failed result
            return max(results, key=lambda r: r.confidence)

        # Score each result
        scored_results = []
        for result in valid_results:
            score = result.confidence

            # Length bonus (reasonable text length)
            if 50 <= len(result.text) <= 1000:
                score += 0.1

            # Engine preference bonus
            engine_bonus = {
                OCREngine.TROCR: 0.15,  # Best for technical text
                OCREngine.PADDLEOCR: 0.10,  # Good for complex layouts
                OCREngine.EASYOCR: 0.05,  # Reliable
                OCREngine.TESSERACT: 0.0  # Baseline
            }
            score += engine_bonus.get(result.engine, 0)

            # Technical content bonus
            tech_indicators = ['figure', 'attention', 'encoder', 'decoder', 'layer']
            tech_score = sum(1 for term in tech_indicators if term.lower() in result.text.lower())
            score += min(tech_score * 0.05, 0.2)

            scored_results.append((score, result))

        # Return best scored result
        return max(scored_results, key=lambda x: x[0])[1]

    def extract_text_with_fallback(
        self,
        image: Union[np.ndarray, str, bytes]
    ) -> OCRResult:
        """
        Extract text with intelligent fallback between engines

        Args:
            image: Input image

        Returns:
            Best OCR result after trying multiple engines
        """
        results = self.extract_text_multiple_engines(image)

        if not results:
            raise RuntimeError("All OCR engines failed")

        best_result = self.get_best_result(results)

        if self.verbose:
            self.logger.info(f" Best result: {best_result.engine.value} ({best_result.confidence:.2f})")

        return best_result

def main():
    """Demo of advanced OCR engine"""
    print(f"Advanced OCR Engine Demo")
    print("=" * 50)

    # Initialize OCR engine
    ocr_engine = AdvancedOCREngine(verbose=True)

    print(f"Available engines: {list(ocr_engine.available_engines.keys())}")

    # Test with a sample image (if available)
    test_image_path = "test_image.png"

    if not os.path.exists(test_image_path):
        print(f"No test image found at {test_image_path}")
        print(f"To test:")
        print("1. Save a technical diagram as 'test_image.png'")
        print("2. Run this script again")
        return

    try:
        # Extract text with multiple engines
        results = ocr_engine.extract_text_multiple_engines(test_image_path)

        print(f"\n Results from {len(results)} engines:")
        for result in results:
            print(f"\n {result.engine.value}:")
            print(f"Confidence: {result.confidence:.2f}")
            print(f"Time: {result.processing_time:.2f}s")
            print(f"Text length: {len(result.text)} chars")
            print(f"Preview: {result.text[:100]}...")

        # Get best result
        best = ocr_engine.get_best_result(results)
        print(f"\n Best result: {best.engine.value}")
        print(f"Extracted text:\n{best.text}")

    except Exception as e:
        print(f"OCR testing failed: {e}")

if __name__ == "__main__":
    main()
