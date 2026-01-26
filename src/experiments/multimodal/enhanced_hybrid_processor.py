"""
Enhanced Hybrid Multimodal Processor: OCR + Vision LLM Integration
Combines multiple OCR engines with LLaVA for comprehensive technical document understanding

Key Features:
- Multi-engine OCR with intelligent fallback
- Context-aware LLaVA prompting using OCR results
- Confidence-based fusion of OCR and vision understanding
- Comprehensive quality assessment and debugging
- Robust error handling and graceful degradation
- Performance monitoring and optimization
"""

import os
import sys
import time
import json
import base64
from typing import Dict, List, Any, Optional, Union, Tuple
from dataclasses import dataclass, asdict
from pathlib import Path
import numpy as np
import cv2
from PIL import Image

# Add project paths
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

try:
    import fitz # PyMuPDF
    import ollama
    from src.ocr.advanced_ocr_engine import AdvancedOCREngine, OCRResult, OCREngine
    from langchain.schema import Document
except ImportError as e:
    print(f"Missing dependencies: {e}")
    print("Install required packages first")
    sys.exit(1)

# Optional debugging support
try:
    from src.debugging.comprehensive_debugger import ComprehensiveDebugger, PerformanceMetrics
    DEBUGGER_AVAILABLE = True
except ImportError:
    DEBUGGER_AVAILABLE = False
    ComprehensiveDebugger = None
    PerformanceMetrics = None

@dataclass
class HybridAnalysisResult:
    """Result from hybrid OCR + Vision analysis"""
    final_description: str
    ocr_result: OCRResult
    llava_description: str
    confidence_score: float
    fusion_method: str
    quality_metrics: Dict[str, float]
    processing_time: float
    metadata: Dict[str, Any]

@dataclass
class ProcessingConfig:
    """Configuration for hybrid processing"""
    # OCR settings
    preferred_ocr_engines: List[OCREngine] = None
    ocr_confidence_threshold: float = 0.5
    enable_ocr_preprocessing: bool = True

    # LLaVA settings
    llava_model: str = "llava:34b"
    enable_context_prompting: bool = True

    # Fusion settings
    fusion_strategy: str = "confidence_weighted" # "ocr_priority", "llava_priority", "confidence_weighted"
    min_confidence_for_fusion: float = 0.3

    # Quality settings
    enable_quality_assessment: bool = True
    debug_mode: bool = False

class EnhancedHybridProcessor:
    """
    Enhanced hybrid processor combining OCR and Vision LLM capabilities
    """

    def __init__(self, config: ProcessingConfig = None, verbose: bool = True):
        """
        Initialize enhanced hybrid processor

        Args:
            config: Processing configuration
            verbose: Enable detailed logging
        """
        self.config = config or ProcessingConfig()
        self.verbose = verbose

        # Initialize components
        self._initialize_components()

        # Performance tracking
        self.processing_stats = {
            'total_images': 0,
            'successful_extractions': 0,
            'ocr_successes': 0,
            'llava_successes': 0,
            'fusion_successes': 0
        }

        if self.verbose:
            print(f"Enhanced Hybrid Processor Initialized")
            print(f"OCR Engines: {len(self.ocr_engine.available_engines)}")
            print(f"LLaVA Model: {self.config.llava_model}")
            print(f"Fusion Strategy: {self.config.fusion_strategy}")

    def _initialize_components(self):
        """Initialize OCR and debugging components"""
        # Initialize OCR engine
        self.ocr_engine = AdvancedOCREngine(
            preferred_engines=self.config.preferred_ocr_engines,
            enable_preprocessing=self.config.enable_ocr_preprocessing,
            verbose=self.verbose
        )

        # Initialize debugger if in debug mode and available
        if self.config.debug_mode and DEBUGGER_AVAILABLE and ComprehensiveDebugger:
            self.debugger = ComprehensiveDebugger(verbose=self.verbose)
        else:
            self.debugger = None

        # Check LLaVA availability
        self._check_llava_availability()

    def _check_llava_availability(self):
        """Check if LLaVA model is available"""
        try:
            models = ollama.list()
            available_models = [m.model for m in models.models]

            if self.config.llava_model not in available_models:
                print(f"LLaVA model {self.config.llava_model} not found")
                print(f"Available models: {available_models}")

                # Try to find alternative
                llava_models = [m for m in available_models if 'llava' in m]
                if llava_models:
                    self.config.llava_model = llava_models[0]
                    print(f"Using alternative: {self.config.llava_model}")
                else:
                    raise ValueError("No LLaVA model available")

            if self.verbose:
                print(f"LLaVA model ready: {self.config.llava_model}")

        except Exception as e:
            raise RuntimeError(f"LLaVA check failed: {e}")

    def extract_text_with_ocr(self, image: np.ndarray) -> OCRResult:
        """
        Extract text using multi-engine OCR with fallback

        Args:
            image: Input image as numpy array

        Returns:
            OCR result with best text extraction
        """
        if self.debugger:
            with self.debugger.monitor_performance("OCR_Extraction"):
                result = self.ocr_engine.extract_text_with_fallback(image)
        else:
            result = self.ocr_engine.extract_text_with_fallback(image)

        # Update stats
        self.processing_stats['total_images'] += 1
        if result.confidence > self.config.ocr_confidence_threshold:
            self.processing_stats['ocr_successes'] += 1

        if self.verbose:
            print(f"OCR Result: {result.engine.value}, {result.confidence:.2f} confidence")
            print(f"Text: {result.text[:100]}..." if len(result.text) > 100 else f" Text: {result.text}")

        return result

    def analyze_with_llava(self, image: np.ndarray, ocr_result: OCRResult = None) -> str:
        """
        Analyze image with LLaVA using context-aware prompting

        Args:
            image: Input image as numpy array
            ocr_result: OCR result to inform prompting

        Returns:
            LLaVA description of the image
        """
        try:
            # Convert image to base64
            if len(image.shape) == 3:
                # Convert BGR to RGB
                rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            else:
                rgb_image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

            pil_image = Image.fromarray(rgb_image)

            # Convert to bytes
            import io
            img_buffer = io.BytesIO()
            pil_image.save(img_buffer, format='PNG')
            img_bytes = img_buffer.getvalue()
            img_b64 = base64.b64encode(img_bytes).decode('utf-8')

            # Create context-aware prompt
            prompt = self._create_context_aware_prompt(ocr_result)

            if self.debugger:
                with self.debugger.monitor_performance("LLaVA_Analysis"):
                    response = ollama.chat(
                        model=self.config.llava_model,
                        messages=[{
                            'role': 'user',
                            'content': prompt,
                            'images': [img_b64]
                        }]
                    )
            else:
                response = ollama.chat(
                    model=self.config.llava_model,
                    messages=[{
                        'role': 'user',
                        'content': prompt,
                        'images': [img_b64]
                    }]
                )

            result = response['message']['content']

            # Update stats
            if len(result) > 50 and 'cannot see' not in result.lower():
                self.processing_stats['llava_successes'] += 1

            if self.verbose:
                print(f"LLaVA Result: {len(result)} chars")
                print(f"Preview: {result[:150]}..." if len(result) > 150 else f" Content: {result}")

            return result

        except Exception as e:
            error_msg = f"LLaVA analysis failed: {e}"
            if self.verbose:
                print(f"{error_msg}")

            if self.debugger:
                self.debugger.record_error("LLaVA_Analysis", e)

            return error_msg

    def _create_context_aware_prompt(self, ocr_result: OCRResult = None) -> str:
        """
        Create intelligent prompt based on OCR results

        Args:
            ocr_result: OCR result to inform prompting

        Returns:
            Context-aware prompt for LLaVA
        """
        if not self.config.enable_context_prompting or not ocr_result or not ocr_result.text:
            # Fallback to generic technical prompt
            return """You are analyzing a technical figure from an academic research paper. Describe this image in detail for a document search system.

CRITICAL: Even if text appears small or unclear, make your best effort to read and transcribe it.

Include ALL of the following:
1. Figure type: (architecture diagram, flowchart, chart, table, etc.)
2. ALL visible text, labels, and numbers - transcribe everything you can see, even if small
3. Structural elements: boxes, arrows, connections, layers
4. Technical components: if this appears to be a neural network or AI architecture, identify specific components
5. Data flow: how information moves through the diagram
6. Key technical terms: any ML/AI terminology visible
7. Mathematical notation: any formulas, equations, or symbols

Be extremely detailed and technical. This description will be used to answer specific questions about neural network architectures."""

        # Create context-informed prompt
        ocr_text = ocr_result.text

        # Analyze OCR content to customize prompt
        has_tech_terms = any(term in ocr_text.lower() for term in ['attention', 'encoder', 'decoder', 'transformer', 'layer'])
        has_figure_ref = any(term in ocr_text.lower() for term in ['figure', 'fig'])
        has_math = any(char in ocr_text for char in ['=', '+', '-', '*', '/', '×', '∑'])

        prompt_parts = [
            "You are analyzing a technical diagram from an academic research paper.",
            "",
            f"OCR has extracted the following text from this image: \"{ocr_text}\"",
            "",
            "Your task is to provide a comprehensive visual analysis that COMPLEMENTS the OCR text.",
            "Focus on the visual structure, layout, and relationships that OCR cannot capture.",
            "",
            "Include:"
        ]

        if has_tech_terms:
            prompt_parts.extend([
                "1. Neural network architecture details (this appears to be an AI/ML diagram)",
                "2. How the components mentioned in the OCR text are visually connected",
                "3. Data flow directions and transformations"
            ])
        else:
            prompt_parts.extend([
                "1. Type of diagram or visualization",
                "2. Visual structure and layout",
                "3. Relationships between elements"
            ])

        if has_figure_ref:
            prompt_parts.append("4. Specific details about this figure that would help answer questions about it")

        if has_math:
            prompt_parts.append("5. Mathematical notation and equations visible in the diagram")

        prompt_parts.extend([
            "6. Colors, shapes, and visual indicators used",
            "7. Any text or labels that OCR might have missed",
            "",
            "Be precise and technical. Your description combined with the OCR text should provide complete understanding of this technical figure."
        ])

        return '\n'.join(prompt_parts)

    def fuse_ocr_and_llava_results(
        self,
        ocr_result: OCRResult,
        llava_description: str,
        image: np.ndarray = None
    ) -> HybridAnalysisResult:
        """
        Intelligently fuse OCR and LLaVA results

        Args:
            ocr_result: Result from OCR processing
            llava_description: Description from LLaVA
            image: Original image (for additional analysis if needed)

        Returns:
            Fused analysis result
        """
        start_time = time.time()

        try:
            # Assess quality of both inputs
            ocr_quality = self._assess_ocr_quality(ocr_result)
            llava_quality = self._assess_llava_quality(llava_description)

            # Choose fusion strategy
            if self.config.fusion_strategy == "ocr_priority":
                final_description, method = self._fuse_ocr_priority(ocr_result, llava_description, ocr_quality, llava_quality)
            elif self.config.fusion_strategy == "llava_priority":
                final_description, method = self._fuse_llava_priority(ocr_result, llava_description, ocr_quality, llava_quality)
            else: # confidence_weighted
                final_description, method = self._fuse_confidence_weighted(ocr_result, llava_description, ocr_quality, llava_quality)

            # Calculate overall confidence
            confidence = self._calculate_fusion_confidence(ocr_quality, llava_quality, method)

            # Quality metrics
            quality_metrics = {
                'ocr_quality': ocr_quality,
                'llava_quality': llava_quality,
                'text_length': len(final_description),
                'technical_content_score': self._score_technical_content(final_description),
                'coherence_score': self._score_coherence(final_description)
            }

            # Update stats
            if confidence > self.config.min_confidence_for_fusion:
                self.processing_stats['fusion_successes'] += 1
                self.processing_stats['successful_extractions'] += 1

            result = HybridAnalysisResult(
                final_description=final_description,
                ocr_result=ocr_result,
                llava_description=llava_description,
                confidence_score=confidence,
                fusion_method=method,
                quality_metrics=quality_metrics,
                processing_time=time.time() - start_time,
                metadata={
                    'ocr_engine': ocr_result.engine.value,
                    'llava_model': self.config.llava_model,
                    'fusion_strategy': self.config.fusion_strategy
                }
            )

            if self.verbose:
                print(f"Fusion Result: {method}, {confidence:.2f} confidence")
                print(f"Final length: {len(final_description)} chars")

            return result

        except Exception as e:
            if self.debugger:
                self.debugger.record_error("Fusion", e)

            # Fallback result
            return HybridAnalysisResult(
                final_description=llava_description if llava_description else ocr_result.text,
                ocr_result=ocr_result,
                llava_description=llava_description,
                confidence_score=0.1,
                fusion_method="fallback",
                quality_metrics={'error': str(e)},
                processing_time=time.time() - start_time,
                metadata={'error': str(e)}
            )

    def _assess_ocr_quality(self, ocr_result: OCRResult) -> float:
        """Assess quality of OCR result"""
        if not ocr_result or not ocr_result.text:
            return 0.0

        quality = ocr_result.confidence

        # Length assessment
        text_length = len(ocr_result.text)
        if 20 <= text_length <= 500:
            quality += 0.1
        elif text_length > 500:
            quality += 0.05

        # Technical content assessment
        tech_terms = ['attention', 'transformer', 'encoder', 'decoder', 'layer', 'multi-head', 'figure']
        tech_score = sum(1 for term in tech_terms if term.lower() in ocr_result.text.lower())
        quality += min(tech_score * 0.05, 0.2)

        return min(1.0, quality)

    def _assess_llava_quality(self, description: str) -> float:
        """Assess quality of LLaVA description"""
        if not description:
            return 0.0

        quality = 0.5 # Base score

        # Length assessment
        if len(description) > 200:
            quality += 0.2
        elif len(description) > 100:
            quality += 0.1

        # Content quality indicators
        positive_indicators = [
            'diagram', 'architecture', 'component', 'flow', 'structure',
            'attention', 'encoder', 'decoder', 'transformer', 'layer'
        ]

        negative_indicators = [
            'cannot see clearly', 'resolution', 'image quality', 'unclear',
            'difficult to read', 'blurry'
        ]

        positive_score = sum(1 for term in positive_indicators if term in description.lower())
        negative_score = sum(1 for term in negative_indicators if term in description.lower())

        quality += min(positive_score * 0.05, 0.3)
        quality -= min(negative_score * 0.1, 0.3)

        return max(0.0, min(1.0, quality))

    def _fuse_confidence_weighted(
        self,
        ocr_result: OCRResult,
        llava_description: str,
        ocr_quality: float,
        llava_quality: float
    ) -> Tuple[str, str]:
        """Fuse results using confidence weighting"""

        if ocr_quality > 0.7 and llava_quality > 0.7:
            # Both high quality - comprehensive fusion
            method = "comprehensive_fusion"
            return self._create_comprehensive_description(ocr_result.text, llava_description), method

        elif ocr_quality > llava_quality + 0.2:
            # OCR significantly better
            method = "ocr_enhanced"
            return self._enhance_with_llava(ocr_result.text, llava_description), method

        elif llava_quality > ocr_quality + 0.2:
            # LLaVA significantly better
            method = "llava_enhanced"
            return self._enhance_with_ocr(llava_description, ocr_result.text), method

        else:
            # Similar quality - balanced fusion
            method = "balanced_fusion"
            return self._create_balanced_description(ocr_result.text, llava_description), method

    def _fuse_ocr_priority(
        self,
        ocr_result: OCRResult,
        llava_description: str,
        ocr_quality: float,
        llava_quality: float
    ) -> Tuple[str, str]:
        """OCR-priority fusion"""
        if ocr_quality > 0.5:
            return self._enhance_with_llava(ocr_result.text, llava_description), "ocr_primary"
        else:
            return llava_description, "llava_fallback"

    def _fuse_llava_priority(
        self,
        ocr_result: OCRResult,
        llava_description: str,
        ocr_quality: float,
        llava_quality: float
    ) -> Tuple[str, str]:
        """LLaVA-priority fusion"""
        if llava_quality > 0.5:
            return self._enhance_with_ocr(llava_description, ocr_result.text), "llava_primary"
        else:
            return ocr_result.text, "ocr_fallback"

    def _create_comprehensive_description(self, ocr_text: str, llava_desc: str) -> str:
        """Create comprehensive description from both high-quality sources"""
        return f"""**Technical Figure Analysis**

**Visual Structure and Layout:**
{llava_desc}

**Extracted Text and Labels:**
{ocr_text}

**Combined Analysis:**
This technical diagram combines the visual structure described above with the specific text elements extracted through OCR. The visual components and textual labels work together to represent a complete technical system or architecture."""

    def _enhance_with_llava(self, ocr_text: str, llava_desc: str) -> str:
        """Enhance OCR text with LLaVA visual understanding"""
        if not llava_desc or 'cannot see' in llava_desc.lower():
            return ocr_text

        # Extract visual insights from LLaVA
        visual_info = self._extract_visual_insights(llava_desc)

        if visual_info:
            return f"{ocr_text}\n\n**Visual Context:** {visual_info}"
        else:
            return ocr_text

    def _enhance_with_ocr(self, llava_desc: str, ocr_text: str) -> str:
        """Enhance LLaVA description with OCR text details"""
        if not ocr_text or len(ocr_text.strip()) < 10:
            return llava_desc

        return f"{llava_desc}\n\n**Extracted Text Labels:** {ocr_text}"

    def _create_balanced_description(self, ocr_text: str, llava_desc: str) -> str:
        """Create balanced description from both sources"""
        return f"**Visual Analysis:** {llava_desc}\n\n**Text Content:** {ocr_text}"

    def _extract_visual_insights(self, llava_desc: str) -> str:
        """Extract relevant visual insights from LLaVA description"""
        # Look for structural information
        insights = []

        for line in llava_desc.split('.'):
            line = line.strip()
            if any(keyword in line.lower() for keyword in ['box', 'arrow', 'connect', 'flow', 'structure', 'layout']):
                insights.append(line)

        return '. '.join(insights[:3]) if insights else "" # Limit to top 3 insights

    def _calculate_fusion_confidence(self, ocr_quality: float, llava_quality: float, method: str) -> float:
        """Calculate overall confidence for fused result"""
        if method == "comprehensive_fusion":
            return (ocr_quality + llava_quality) / 2
        elif method in ["ocr_enhanced", "ocr_primary"]:
            return ocr_quality * 0.8 + llava_quality * 0.2
        elif method in ["llava_enhanced", "llava_primary"]:
            return llava_quality * 0.8 + ocr_quality * 0.2
        elif method == "balanced_fusion":
            return (ocr_quality + llava_quality) / 2
        else: # fallback methods
            return max(ocr_quality, llava_quality) * 0.7

    def _score_technical_content(self, text: str) -> float:
        """Score technical content relevance"""
        tech_terms = [
            'attention', 'transformer', 'encoder', 'decoder', 'layer',
            'neural', 'network', 'architecture', 'model', 'figure',
            'diagram', 'flow', 'component', 'multi-head', 'self-attention'
        ]

        found_terms = sum(1 for term in tech_terms if term.lower() in text.lower())
        return min(found_terms / 5, 1.0) # Normalize to 0-1

    def _score_coherence(self, text: str) -> float:
        """Score text coherence and readability"""
        if not text or len(text) < 20:
            return 0.0

        # Simple coherence metrics
        sentences = text.split('.')
        avg_sentence_length = sum(len(s.split()) for s in sentences) / len(sentences)

        # Ideal sentence length for technical content
        length_score = 1.0 - abs(avg_sentence_length - 15) / 15 if avg_sentence_length > 0 else 0

        # Check for repeated words (sign of poor quality)
        words = text.lower().split()
        unique_ratio = len(set(words)) / len(words) if words else 0

        return (length_score + unique_ratio) / 2

    def process_image(
        self,
        image: Union[np.ndarray, str, bytes],
        image_metadata: Dict[str, Any] = None
    ) -> HybridAnalysisResult:
        """
        Process single image with hybrid OCR + LLaVA approach

        Args:
            image: Input image (numpy array, file path, or bytes)
            image_metadata: Additional metadata about the image

        Returns:
            Comprehensive hybrid analysis result
        """
        if self.debugger:
            with self.debugger.monitor_performance("Hybrid_Image_Processing"):
                return self._process_image_internal(image, image_metadata)
        else:
            return self._process_image_internal(image, image_metadata)

    def _process_image_internal(
        self,
        image: Union[np.ndarray, str, bytes],
        image_metadata: Dict[str, Any] = None
    ) -> HybridAnalysisResult:
        """Internal image processing method"""
        # Load image
        if isinstance(image, str):
            img_array = cv2.imread(image)
            if img_array is None:
                raise ValueError(f"Could not load image from {image}")
        elif isinstance(image, bytes):
            nparr = np.frombuffer(image, np.uint8)
            img_array = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        else:
            img_array = image.copy()

        if img_array is None:
            raise ValueError("Invalid image input")

        # Step 1: OCR extraction
        ocr_result = self.extract_text_with_ocr(img_array)

        # Step 2: LLaVA analysis
        llava_description = self.analyze_with_llava(img_array, ocr_result)

        # Step 3: Fusion
        hybrid_result = self.fuse_ocr_and_llava_results(ocr_result, llava_description, img_array)

        # Add metadata
        if image_metadata:
            hybrid_result.metadata.update(image_metadata)

        return hybrid_result

    def get_processing_summary(self) -> Dict[str, Any]:
        """Get processing statistics and summary"""
        total = max(self.processing_stats['total_images'], 1)

        return {
            'total_images_processed': self.processing_stats['total_images'],
            'success_rates': {
                'ocr_success_rate': self.processing_stats['ocr_successes'] / total,
                'llava_success_rate': self.processing_stats['llava_successes'] / total,
                'fusion_success_rate': self.processing_stats['fusion_successes'] / total,
                'overall_success_rate': self.processing_stats['successful_extractions'] / total
            },
            'configuration': asdict(self.config),
            'available_ocr_engines': list(self.ocr_engine.available_engines.keys()),
            'performance_stats': self.processing_stats
        }

def main():
    """Demo of enhanced hybrid processor"""
    print(f"Enhanced Hybrid Processor Demo")
    print("=" * 50)

    # Configuration
    config = ProcessingConfig(
        debug_mode=True,
        fusion_strategy="confidence_weighted",
        enable_context_prompting=True
    )

    # Initialize processor
    processor = EnhancedHybridProcessor(config=config, verbose=True)

    # Test with sample image
    test_image = "test_technical_diagram.png"

    if not os.path.exists(test_image):
        print(f"No test image found at {test_image}")
        print(f"To test the hybrid processor:")
        print("1. Save a technical diagram as 'test_technical_diagram.png'")
        print("2. Run this script again")
        return

    try:
        # Process image
        result = processor.process_image(test_image)

        print("\nProcessing complete")
        print(f"Confidence: {result.confidence_score:.2f}")
        print(f"Fusion Method: {result.fusion_method}")
        print(f"Processing Time: {result.processing_time:.2f}s")

        print(f"\n Final Description:")
        print(result.final_description)

        print(f"\n Quality Metrics:")
        for metric, value in result.quality_metrics.items():
            print(f"{metric}: {value}")

        # Processing summary
        summary = processor.get_processing_summary()
        print(f"\n Processing Summary:")
        for rate_name, rate_value in summary['success_rates'].items():
            print(f"{rate_name}: {rate_value:.1%}")

    except Exception as e:
        print(f"Processing failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
