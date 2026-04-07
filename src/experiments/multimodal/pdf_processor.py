"""
PDF processing: text extraction, image extraction, and table extraction.
"""

import base64
from contextlib import contextmanager
from typing import List

import cv2
import numpy as np

try:
    import fitz  # PyMuPDF
    import ollama
    import camelot
except ImportError:
    pass

from src.experiments.multimodal.models import ContentType, ExtractedContent


class PDFProcessor:
    """Handles extraction of text, images, and tables from PDF documents."""

    def __init__(
        self,
        llava_model: str,
        verbose: bool,
        hybrid_processor=None,
        use_hybrid: bool = False,
        debugger=None,
    ):
        self.llava_model = llava_model
        self.verbose = verbose
        self.hybrid_processor = hybrid_processor
        self.use_hybrid = use_hybrid
        self.debugger = debugger

    @contextmanager
    def _null_context(self):
        """Null context manager for when debugger is not available."""
        yield

    # ------------------------------------------------------------------
    # Text extraction
    # ------------------------------------------------------------------

    def extract_text(self, pdf_path: str) -> List[ExtractedContent]:
        """Extract text content from PDF with filtering and chunking."""
        contents = []
        doc = fitz.open(pdf_path)

        for page_num, page in enumerate(doc):
            text = page.get_text()
            if text.strip():
                cleaned_text = self._clean_text(text)

                if cleaned_text and len(cleaned_text.strip()) > 50:
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
        """Clean text by removing copyright notices and repetitive content."""
        lines = text.split('\n')
        cleaned_lines = []

        for line in lines:
            line = line.strip()

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

            if not should_skip and len(line) > 5:
                cleaned_lines.append(line)

        return '\n'.join(cleaned_lines)

    def _split_text_into_chunks(self, text: str, max_chunk_size: int = 1000) -> List[str]:
        """Split text into smaller chunks for better retrieval."""
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

    # ------------------------------------------------------------------
    # Image extraction
    # ------------------------------------------------------------------

    def extract_and_describe_images(self, pdf_path: str) -> List[ExtractedContent]:
        """Extract images and use enhanced hybrid OCR + LLaVA processing."""
        contents = []
        doc = fitz.open(pdf_path)

        for page_num, page in enumerate(doc):
            image_list = page.get_images()

            if self.verbose and image_list:
                print(f"\nPage {page_num + 1}: Found {len(image_list)} image(s)")

            for img_index, img in enumerate(image_list):
                try:
                    xref = img[0]
                    base_image = doc.extract_image(xref)
                    image_bytes = base_image["image"]
                    image_ext = base_image["ext"]

                    nparr = np.frombuffer(image_bytes, np.uint8)
                    image_cv = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

                    if image_cv is None:
                        if self.verbose:
                            print(f"Failed to decode image {img_index + 1}")
                        continue

                    if self.use_hybrid and self.hybrid_processor:
                        if self.verbose:
                            print(f"-> Image {img_index + 1}: Using hybrid OCR + LLaVA processing...")

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
                                    'processing_time': int(result.processing_time * 1000)
                                },
                                metadata=result.metadata
                            )

                        if self.verbose:
                            print(f"Hybrid Result: {result.fusion_method}, confidence: {result.confidence_score:.2f}")
                            print(f"Description: {description[:100]}...")

                    else:
                        if self.verbose:
                            print(f"-> Image {img_index + 1}: Using LLaVA-only processing...")

                        image_b64 = base64.b64encode(image_bytes).decode('utf-8')
                        description = self._describe_image_with_llava(image_b64)
                        extraction_method = 'llava_only'

                        if self.verbose:
                            print(f"Description: {description[:100]}...")

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

    # ------------------------------------------------------------------
    # Table extraction
    # ------------------------------------------------------------------

    def extract_tables(self, pdf_path: str) -> List[ExtractedContent]:
        """Extract and describe tables from PDF."""
        contents = []

        try:
            tables = camelot.read_pdf(pdf_path, pages='all', flavor='lattice')

            if self.verbose and len(tables) > 0:
                print(f"\nExtracted {len(tables)} table(s)")

            for i, table in enumerate(tables):
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
        Convert table DataFrame to searchable natural language.

        Example:
        Table with headers [Name, Score, Grade]
        Row 1: Alice, 95, A
        Row 2: Bob, 87, B

        Becomes:
        "Table 1 contains 3 columns: Name, Score, Grade. It has 2 data rows.
        Row 1: Name=Alice, Score=95, Grade=A.
        Row 2: Name=Bob, Score=87, Grade=B."
        """
        description = f"Table {table_num} contains {len(df.columns)} columns: "
        description += ", ".join([str(col) for col in df.columns])
        description += f". It has {len(df)} data rows.\n\n"

        for idx, row in df.head(20).iterrows():
            row_text = ", ".join([f"{col}={val}" for col, val in row.items()])
            description += f"Row {idx + 1}: {row_text}.\n"

        if len(df) > 20:
            description += f"\n[Table continues for {len(df) - 20} more rows...]"

        return description
