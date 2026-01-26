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

import os
import sys
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import base64
import io
import json
import numpy as np
import cv2
from dataclasses import dataclass
from enum import Enum
from contextlib import contextmanager

# Third-party imports
try:
    import fitz # PyMuPDF
    import ollama
    from PIL import Image
    import camelot
    from langchain.schema import Document
except ImportError as e:
    print(f"Missing dependency: {e}")
    print("Install with: pip install pymupdf pillow camelot-py[cv] ollama langchain")
    sys.exit(1)

# Local imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src.core.ollama_rag import OllamaRAG

# Import enhanced hybrid processor
try:
    from src.experiments.multimodal.enhanced_hybrid_processor import (
        EnhancedHybridProcessor,
        ProcessingConfig,
        HybridAnalysisResult
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


class ContentType(Enum):
    """Types of content in multimodal documents"""
    TEXT = "text"
    IMAGE = "image"
    TABLE = "table"
    CHART = "chart"
    DIAGRAM = "diagram"


@dataclass
class ExtractedContent:
    """Represents extracted content from a document"""
    content_type: ContentType
    content: str # Text description or actual text
    metadata: Dict
    page_number: int
    source_path: Optional[str] = None


class LLaVAMultimodalRAG:
    """
    Multimodal RAG using LLaVA for vision understanding

    This class gives your RAG system "eyes" to see and understand:
    - Figures and charts
    - Tables
    - Diagrams
    - Screenshots
    - Any visual content in PDFs
    """

    def __init__(
        self,
        llava_model: str = "llava:34b",
        use_caching: bool = True,
        verbose: bool = True,
        use_hybrid: bool = True,
        debug_mode: bool = False
    ):
        """
        Initialize multimodal RAG

        Args:
            llava_model: LLaVA model to use (llava:7b, llava:13b, llava:34b)
            use_caching: Enable caching for faster repeated queries
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

        # Check if LLaVA is available
        self._check_llava_available()

        if self.verbose:
            print(f"\n{'='*60}")
            print("Enhanced Multimodal RAG Initialized")
            print(f"{'='*60}")
            print(f"Vision Model: {llava_model}")
            print(f"Caching: {'Enabled' if use_caching else 'Disabled'}")
            print(f"Hybrid OCR+LLaVA: {'Enabled' if self.use_hybrid else 'Disabled'}")
            print(f"Debug Mode: {'Enabled' if debug_mode else 'Disabled'}")
            print(f"{'='*60}\n")

    @contextmanager
    def _null_context(self):
        """Null context manager for when debugger is not available"""
        yield

    def _check_llava_available(self):
        """Check if LLaVA model is available in Ollama"""
        try:
            models = ollama.list()
            model_names = [m['name'] for m in models.get('models', [])]

            if not any(self.llava_model in name for name in model_names):
                print(f"\nWARNING: {self.llava_model} not found in Ollama")
                print(f"Available models: {model_names}")
                print(f"\n Install with: ollama pull {self.llava_model}")
                print(f"Continuing anyway, but vision features will fail.\n")
        except Exception as e:
            print(f"Warning: Could not check Ollama models: {e}")

    def process_pdf(self, pdf_path: str) -> List[Document]:
        """
        Process entire PDF including text, images, and tables

        Args:
            pdf_path: Path to PDF file

        Returns:
            List of Document objects with all content types
        """
        if self.verbose:
            print(f"\n{'='*60}")
            print(f"Processing: {Path(pdf_path).name}")
            print(f"{'='*60}\n")

        documents = []

        # Extract all content types
        text_content = self._extract_text(pdf_path)
        image_content = self._extract_and_describe_images(pdf_path)
        table_content = self._extract_tables(pdf_path)

        # Convert to Document objects
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

    def _extract_text(self, pdf_path: str) -> List[ExtractedContent]:
        """Extract text content from PDF with filtering and chunking"""
        contents = []
        doc = fitz.open(pdf_path)

        for page_num, page in enumerate(doc):
            text = page.get_text()
            if text.strip():
                # Clean and filter the text
                cleaned_text = self._clean_text(text)

                if cleaned_text and len(cleaned_text.strip()) > 50: # Skip very short content
                    # Split long text into chunks to improve retrieval
                    chunks = self._split_text_into_chunks(cleaned_text, max_chunk_size=1000)

                    for chunk_idx, chunk in enumerate(chunks):
                        contents.append(ExtractedContent(
                            content_type=ContentType.TEXT,
                            content=chunk,
                            metadata={
                                'extraction_method': 'pymupdf',
                                'chunk_index': chunk_idx,
                                'total_chunks': len(chunks)
                            },
                            page_number=page_num + 1
                        ))

        return contents

    def _clean_text(self, text: str) -> str:
        """Clean text by removing copyright notices and repetitive content"""
        lines = text.split('\n')
        cleaned_lines = []

        for line in lines:
            line = line.strip()

            # Skip copyright notices and common headers/footers
            skip_patterns = [
                'provided proper attribution is provided',
                'google hereby grants permission',
                'reproduce the tables and figures',
                'solely for use in journalistic',
                'arxiv:',
                'preprint',
                'submitted to',
                'under review'
            ]

            should_skip = any(pattern in line.lower() for pattern in skip_patterns)

            # Skip very short lines and whitespace
            if not should_skip and len(line) > 5:
                cleaned_lines.append(line)

        return '\n'.join(cleaned_lines)

    def _split_text_into_chunks(self, text: str, max_chunk_size: int = 1000) -> List[str]:
        """Split text into smaller chunks for better retrieval"""
        if len(text) <= max_chunk_size:
            return [text]

        chunks = []
        sentences = text.split('. ')
        current_chunk = ""

        for sentence in sentences:
            if len(current_chunk) + len(sentence) + 2 <= max_chunk_size:
                current_chunk += sentence + ". "
            else:
                if current_chunk.strip():
                    chunks.append(current_chunk.strip())
                current_chunk = sentence + ". "

        if current_chunk.strip():
            chunks.append(current_chunk.strip())

        return chunks

    def _extract_and_describe_images(self, pdf_path: str) -> List[ExtractedContent]:
        """
        Extract images and use enhanced hybrid OCR + LLaVA processing
        """
        contents = []
        doc = fitz.open(pdf_path)

        for page_num, page in enumerate(doc):
            image_list = page.get_images()

            if self.verbose and image_list:
                print(f"\nPage {page_num + 1}: Found {len(image_list)} image(s)")

            for img_index, img in enumerate(image_list):
                try:
                    # Extract image bytes
                    xref = img[0]
                    base_image = doc.extract_image(xref)
                    image_bytes = base_image["image"]
                    image_ext = base_image["ext"]

                    # Convert bytes to numpy array for processing
                    nparr = np.frombuffer(image_bytes, np.uint8)
                    image_cv = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

                    if image_cv is None:
                        if self.verbose:
                            print(f"Failed to decode image {img_index + 1}")
                        continue

                    # Use enhanced hybrid processing if available
                    if self.use_hybrid and self.hybrid_processor:
                        if self.verbose:
                            print(f"-> Image {img_index + 1}: Using hybrid OCR + LLaVA processing...")

                        # Process with hybrid approach
                        with (self.debugger.monitor_performance(f"Hybrid_Image_{page_num + 1}_{img_index + 1}")
                              if self.debugger else self._null_context()):
                            result = self.hybrid_processor.process_image(
                                image_cv,
                                image_metadata={
                                    'page': page_num + 1,
                                    'image_index': img_index,
                                    'image_format': image_ext,
                                    'pdf_path': pdf_path
                                }
                            )

                        description = result.final_description
                        extraction_method = f"hybrid_{result.fusion_method}"

                        # Record quality metrics if debugging
                        if self.debugger:
                            self.debugger.record_quality_metrics(
                                component=f"Hybrid_Processing_Page_{page_num + 1}",
                                input_size=len(image_bytes),
                                output_size=len(description),
                                confidence_score=result.confidence_score,
                                quality_indicators={
                                    'fusion_successful': result.confidence_score > 0.5,
                                    'adequate_length': len(description) > 200,
                                    'ocr_contributed': bool(result.ocr_result.text),
                                    'llava_contributed': len(result.llava_description) > 100
                                },
                                extracted_entities={
                                    'ocr_words': len(result.ocr_result.text.split()) if result.ocr_result.text else 0,
                                    'llava_length': len(result.llava_description),
                                    'processing_time': int(result.processing_time * 1000) # ms
                                },
                                metadata=result.metadata
                            )

                        if self.verbose:
                            print(f"Hybrid Result: {result.fusion_method}, confidence: {result.confidence_score:.2f}")
                            print(f"Description: {description[:100]}...")

                    else:
                        # Fallback to LLaVA-only processing
                        if self.verbose:
                            print(f"-> Image {img_index + 1}: Using LLaVA-only processing...")

                        image_b64 = base64.b64encode(image_bytes).decode('utf-8')
                        description = self._describe_image_with_llava(image_b64)
                        extraction_method = 'llava_only'

                        if self.verbose:
                            print(f"Description: {description[:100]}...")

                    # Create content with description
                    # Get confidence score only if hybrid processing was used and result exists
                    confidence_score = None
                    if self.use_hybrid and self.hybrid_processor:
                        try:
                            confidence_score = result.confidence_score
                        except NameError:
                            confidence_score = None

                    contents.append(ExtractedContent(
                        content_type=ContentType.IMAGE,
                        content=f"Figure {img_index + 1} on page {page_num + 1}: {description}",
                        metadata={
                            'extraction_method': extraction_method,
                            'image_format': image_ext,
                            'image_index': img_index,
                            'confidence_score': confidence_score
                        },
                        page_number=page_num + 1
                    ))

                except Exception as e:
                    if self.verbose:
                        print(f"Failed to process image {img_index + 1}: {e}")

                    if self.debugger:
                        self.debugger.record_error(f"Image_Processing_Page_{page_num + 1}", e, {
                            'image_index': img_index,
                            'image_format': image_ext if 'image_ext' in locals() else 'unknown'
                        })
                    continue

        return contents

    def _describe_image_with_llava(self, image_b64: str) -> str:
        """
        Use LLaVA to generate a detailed description of an image.

        Args:
            image_b64: Base64 encoded image data

        Returns:
            Text description of the image content
        """
        try:
            response = ollama.chat(
                model=self.llava_model,
                messages=[{
                    'role': 'user',
                    'content': '''You are analyzing a technical figure from an academic research paper. Describe this image in detail for a document search system.

CRITICAL: Even if text appears small or unclear, make your best effort to read and transcribe it.

Include ALL of the following:
1. Figure type: (architecture diagram, flowchart, chart, table, etc.)
2. ALL visible text, labels, and numbers - transcribe everything you can see, even if small
3. Structural elements: boxes, arrows, connections, layers
4. Technical components: if this appears to be a neural network or AI architecture, identify specific components
5. Data flow: how information moves through the diagram
6. Key technical terms: any ML/AI terminology visible
7. Mathematical notation: any formulas, equations, or symbols

If this appears to be Figure 1 or Figure 2 from a Transformer/attention paper, pay special attention to:
- Encoder/decoder components
- Attention mechanisms
- Multi-head attention structures
- Input/output flows
- Layer normalization
- Feed-forward networks

Be extremely detailed and technical. This description will be used to answer specific questions about neural network architectures.''',
                    'images': [image_b64]
                }]
            )

            return response['message']['content']

        except Exception as e:
            return f"[Could not describe image: {e}]"

    def _extract_tables(self, pdf_path: str) -> List[ExtractedContent]:
        """Extract and describe tables from PDF"""
        contents = []

        try:
            # Use Camelot to extract tables
            tables = camelot.read_pdf(pdf_path, pages='all', flavor='lattice')

            if self.verbose and len(tables) > 0:
                print(f"\nExtracted {len(tables)} table(s)")

            for i, table in enumerate(tables):
                # Convert table to natural language description
                description = self._table_to_text(table.df, i + 1)

                if self.verbose:
                    print(f"-> Table {i + 1} on page {table.page}")

                contents.append(ExtractedContent(
                    content_type=ContentType.TABLE,
                    content=description,
                    metadata={
                        'extraction_method': 'camelot',
                        'table_index': i,
                        'accuracy': table.accuracy
                    },
                    page_number=table.page
                ))

        except Exception as e:
            if self.verbose:
                print(f"\nWARNING: Table extraction failed: {e}")
                print(f"(This is often due to PDF format - not a critical error)")

        return contents

    def _table_to_text(self, df, table_num: int) -> str:
        """
        Convert table DataFrame to searchable natural language

        Example:
        Table with headers [Name, Score, Grade]
        Row 1: Alice, 95, A
        Row 2: Bob, 87, B

        Becomes:
        "Table 1 contains 3 columns: Name, Score, Grade. It has 2 data rows.
        Row 1: Name=Alice, Score=95, Grade=A.
        Row 2: Name=Bob, Score=87, Grade=B."
        """
        # Build description
        description = f"Table {table_num} contains {len(df.columns)} columns: "
        description += ", ".join([str(col) for col in df.columns])
        description += f". It has {len(df)} data rows.\n\n"

        # Add row-by-row data (limit to first 20 rows to avoid huge text)
        for idx, row in df.head(20).iterrows():
            row_text = ", ".join([f"{col}={val}" for col, val in row.items()])
            description += f"Row {idx + 1}: {row_text}.\n"

        if len(df) > 20:
            description += f"\n[Table continues for {len(df) - 20} more rows...]"

        return description

    def add_documents(self, pdf_paths: List[str]):
        """
        Add multiple PDFs to the multimodal knowledge base

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
        Query the multimodal knowledge base

        This searches across ALL content types: text, images, and tables!

        Args:
            question: Natural language question

        Returns:
            Dict with answer, metadata, and timing
        """
        return self.rag.query(question)

    def print_statistics(self):
        """Print cache and usage statistics"""
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
        """Generate comprehensive debugging report"""
        if not self.debugger:
            return {"error": "Debug mode not enabled"}

        if pdf_path:
            return self.debugger.generate_comprehensive_report(pdf_path)
        else:
            # Generate partial report on current session
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

    This function demonstrates the complete workflow
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

    # If you have actual PDFs, uncomment:
    # rag.add_documents(['path/to/paper1.pdf', 'path/to/paper2.pdf'])
    #
    # Then query:
    # result = rag.query("What were the accuracy results in the comparison table?")
    # print(result['answer'])


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
