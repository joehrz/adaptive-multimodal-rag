"""
LLaVA Multimodal RAG Implementation

Gives RAG "eyes" to understand images, figures, tables, charts, and diagrams.

Features:
- Extract images from PDFs
- Use LLaVA to describe images
- Extract tables and convert to text
- Unified search across text and visual content

Requirements:
    pip install pymupdf pillow camelot-py[cv] ollama
    ollama pull llava:13b
"""

import sys
from pathlib import Path
from typing import List, Dict

# Third-party imports
try:
    from langchain.schema import Document
    import ollama
except ImportError as e:
    print(f"Missing dependency: {e}")
    print("Install with: pip install pymupdf pillow camelot-py[cv] ollama langchain")
    sys.exit(1)

# Local imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src.core.ollama_rag import OllamaRAG

# Re-export data models so existing code that imports them from here still works
from src.experiments.multimodal.models import ContentType, ExtractedContent  # noqa: F401
from src.experiments.multimodal.pdf_processor import PDFProcessor

# Import enhanced hybrid processor
try:
    from src.experiments.multimodal.enhanced_hybrid_processor import (
        EnhancedHybridProcessor,
        ProcessingConfig,
        HybridAnalysisResult  # noqa: F401
    )
    HYBRID_AVAILABLE = True
except ImportError:
    HYBRID_AVAILABLE = False

# Optional debugging support
try:
    from src.debugging.comprehensive_debugger import ComprehensiveDebugger
    DEBUGGER_AVAILABLE = True
except ImportError:
    DEBUGGER_AVAILABLE = False
    ComprehensiveDebugger = None


class LLaVAMultimodalRAG:
    """
    Multimodal RAG using LLaVA for vision understanding of:
    - Figures and charts
    - Tables
    - Diagrams
    - Screenshots
    - Any visual content in PDFs
    """

    def __init__(
        self,
        llava_model: str = "llava:34b",
        verbose: bool = True,
        use_hybrid: bool = True,
        debug_mode: bool = False
    ):
        """
        Initialize multimodal RAG.

        Args:
            llava_model: LLaVA model to use (llava:7b, llava:13b, llava:34b)
            verbose: Print progress messages
            use_hybrid: Use enhanced hybrid OCR + LLaVA processing
            debug_mode: Enable comprehensive debugging
        """
        self.llava_model = llava_model
        self.verbose = verbose
        self.use_hybrid = use_hybrid and HYBRID_AVAILABLE
        self.debug_mode = debug_mode

        # Initialize RAG system
        self.rag = OllamaRAG(verbose=verbose)

        # Initialize hybrid processor if available
        if self.use_hybrid:
            config = ProcessingConfig(
                llava_model=llava_model,
                enable_context_prompting=True,
                fusion_strategy="confidence_weighted",
                debug_mode=debug_mode
            )
            self.hybrid_processor = EnhancedHybridProcessor(config=config, verbose=verbose)
        else:
            self.hybrid_processor = None

        # Initialize debugger if in debug mode and available
        if debug_mode and DEBUGGER_AVAILABLE and ComprehensiveDebugger:
            self.debugger = ComprehensiveDebugger(verbose=verbose)
        else:
            self.debugger = None

        # Initialize PDF processor (delegates text/image/table extraction)
        self._pdf_processor = PDFProcessor(
            llava_model=self.llava_model,
            verbose=self.verbose,
            hybrid_processor=self.hybrid_processor,
            use_hybrid=self.use_hybrid,
            debugger=self.debugger,
        )

        # Check if LLaVA is available
        self._check_llava_available()

        if self.verbose:
            print(f"\n{'='*60}")
            print("Enhanced Multimodal RAG Initialized")
            print(f"{'='*60}")
            print(f"Vision Model: {llava_model}")
            print(f"Hybrid OCR+LLaVA: {'Enabled' if self.use_hybrid else 'Disabled'}")
            print(f"Debug Mode: {'Enabled' if debug_mode else 'Disabled'}")
            print(f"{'='*60}\n")

    def _check_llava_available(self):
        """Check if LLaVA model is available in Ollama."""
        try:
            models = ollama.list()
            model_list = models.get('models', []) if isinstance(models, dict) else getattr(models, 'models', [])
            model_names = [m.get('name', '') if isinstance(m, dict) else getattr(m, 'model', '') for m in model_list]

            if not any(self.llava_model in name for name in model_names):
                print(f"\nWARNING: {self.llava_model} not found in Ollama")
                print(f"Available models: {model_names}")
                print(f"\n Install with: ollama pull {self.llava_model}")
                print(f"Continuing anyway, but vision features will fail.\n")
        except Exception as e:
            print(f"Warning: Could not check Ollama models: {e}")

    def process_pdf(self, pdf_path: str) -> List[Document]:
        """
        Process entire PDF including text, images, and tables.

        Args:
            pdf_path: Path to PDF file

        Returns:
            List of Document objects with all content types
        """
        if self.verbose:
            print(f"\n{'='*60}")
            print(f"Processing: {Path(pdf_path).name}")
            print(f"{'='*60}\n")

        # Extract all content types via the PDF processor
        text_content = self._pdf_processor.extract_text(pdf_path)
        image_content = self._pdf_processor.extract_and_describe_images(pdf_path)
        table_content = self._pdf_processor.extract_tables(pdf_path)

        # Convert to Document objects
        documents = []
        for content in text_content + image_content + table_content:
            doc = Document(
                page_content=content.content,
                metadata={
                    'type': content.content_type.value,
                    'page': content.page_number,
                    'source': pdf_path,
                    **content.metadata
                }
            )
            documents.append(doc)

        if self.verbose:
            print(f"\nSUCCESS: Extracted {len(documents)} content pieces:")
            print(f"- Text sections: {len(text_content)}")
            print(f"- Images/Figures: {len(image_content)}")
            print(f"- Tables: {len(table_content)}")

        return documents

    def add_documents(self, pdf_paths: List[str]):
        """
        Add multiple PDFs to the multimodal knowledge base.

        Args:
            pdf_paths: List of paths to PDF files
        """
        all_documents = []

        for pdf_path in pdf_paths:
            docs = self.process_pdf(pdf_path)
            all_documents.extend(docs)

        # Add to RAG system
        self.rag.add_documents(all_documents)

        if self.verbose:
            print(f"\n{'='*60}")
            print(f"SUCCESS: Added {len(all_documents)} content pieces from {len(pdf_paths)} PDF(s)")
            print(f"{'='*60}\n")

    def query(self, question: str) -> Dict:
        """
        Query the multimodal knowledge base.

        This searches across ALL content types: text, images, and tables.

        Args:
            question: Natural language question

        Returns:
            Dict with answer, metadata, and timing
        """
        return self.rag.query(question)

    def print_statistics(self):
        """Print cache and usage statistics."""
        self.rag.print_cache_stats()

        if self.use_hybrid and self.hybrid_processor:
            print("\n" + "="*60)
            print("HYBRID PROCESSOR STATISTICS")
            print("="*60)
            stats = self.hybrid_processor.get_processing_summary()
            print(f"Images Processed: {stats['total_images_processed']}")
            print(f"OCR Success Rate: {stats['success_rates']['ocr_success_rate']:.1%}")
            print(f"LLaVA Success Rate: {stats['success_rates']['llava_success_rate']:.1%}")
            print(f"Fusion Success Rate: {stats['success_rates']['fusion_success_rate']:.1%}")
            print(f"Overall Success Rate: {stats['success_rates']['overall_success_rate']:.1%}")
            print("="*60)

    def generate_debug_report(self, pdf_path: str = None) -> Dict:
        """Generate debugging report."""
        if not self.debugger:
            return {"error": "Debug mode not enabled"}

        if pdf_path:
            return self.debugger.generate_comprehensive_report(pdf_path)
        else:
            return {
                'session_id': self.debugger.session_id,
                'performance_metrics': [vars(m) for m in self.debugger.performance_metrics],
                'quality_metrics': [vars(m) for m in self.debugger.quality_metrics],
                'errors': self.debugger.errors,
                'recommendations': self.debugger._generate_recommendations()
            }


# ============================================================================
# Demo and Testing Functions
# ============================================================================

def demo_with_sample_pdf():
    """
    Demo showing how to use multimodal RAG with a sample PDF

    """
    print("\n" + "="*60)
    print("MULTIMODAL RAG DEMO")
    print("="*60)

    # Initialize
    rag = LLaVAMultimodalRAG(verbose=True)

    # For demo, we'll show what WOULD happen with a real PDF
    print("\n Example: Processing research paper with figures and tables")
    print("\nWhat the system does:")
    print("1. Extracts text: 'Introduction: Deep learning has...'")
    print("2. Finds Figure 1: [neural network diagram]")
    print(f"-> Asks LLaVA: 'What's in this image?'")
    print(f"-> LLaVA describes: 'Neural network architecture with...'")
    print("3. Finds Table 1: [results table]")
    print(f"-> Converts to text: 'Table showing Method, Accuracy...'")
    print("4. Stores ALL content as searchable text")
    print("\nNow queries like 'What was the architecture in Figure 1?' work! SUCCESS")


def compare_standard_vs_multimodal():
    """
    Compare what standard RAG sees vs multimodal RAG
    """
    print("\n" + "="*70)
    print("COMPARISON: Standard RAG vs Multimodal RAG")
    print("="*70)

    print("\n Sample Document: Research Paper")
    print("-" * 70)

    print("\nTEXT (both systems see this):")
    print(f"'Figure 1 shows our proposed architecture...'")
    print(f"'Table 2 presents the experimental results...'")

    print("\n FIGURE 1 (only multimodal sees this):")
    print(f"Standard RAG: FAILS [Ignores image completely]")
    print(f"Multimodal RAG: SUCCESS 'Architecture diagram showing input layer")
    print(f"with 512 neurons, hidden layers with...")

    print("\nTABLE 2 (only multimodal sees this):")
    print(f"Standard RAG: FAILS [Ignores table completely]")
    print(f"Multimodal RAG: SUCCESS 'Table with columns: Method, Accuracy,")
    print(f"Time. Row 1: CNN, 92.3%, 45s...'")

    print("\n" + "-" * 70)
    print("USER QUESTION: 'What was the accuracy of the CNN method?'")
    print("-" * 70)
    print("\n Standard RAG: 'The paper mentions experimental results...")
    print(f"but I cannot provide specific numbers.'")
    print(f"FAILED to answer\n")

    print(f"Multimodal RAG: 'According to Table 2, the CNN method")
    print(f"achieved 92.3% accuracy with 45s runtime.'")
    print("Answer verified")


if __name__ == "__main__":
    print("\n" + "="*70)
    print("LLAVA MULTIMODAL RAG - Vision-Enabled Document Understanding")
    print("="*70)

    print("\nWhat This Does:")
    print(f"Gives your RAG system 'eyes' to see and understand:")
    print(f"- Figures and charts")
    print(f"- Tables and data")
    print(f"- Diagrams and flowcharts")
    print(f"- Screenshots and photos")
    print(f"- ANY visual content in PDFs")

    print("\n Requirements:")
    print(f"1. Install: ollama pull llava:13b")
    print(f"2. Install: pip install pymupdf camelot-py[cv] pillow")
    print(f"3. Have PDFs with images/tables to process")

    print("\n" + "="*70 + "\n")

    # Run demos
    demo_with_sample_pdf()
    print("\n")
    compare_standard_vs_multimodal()

    print("\n" + "="*70)
    print("Ready to process your PDFs with vision understanding!")
    print("="*70 + "\n")
