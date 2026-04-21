"""Document processing pipeline for ATLAS.

Supports single-file and folder-wide processing:
1. Extraction (PDF/DOCX/PPTX/image)
2. OCR fallback for scanned content
3. Optional page enrichment
4. Chunking
5. Optional result persistence
6. Optional DB ingestion (disabled by default for batch/Slurm usage)
"""

import argparse
import json
import re
import shutil
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

from document_ingestion.chunker import DocumentChunker
from document_ingestion.enricher import DocumentEnricher
from document_ingestion.extractors import PDFExtractor, get_extractor
from document_ingestion.ocr_engine import DocumentOCREngine

try:
    from document_ingestion.ingest_documents import DocumentIngester

    HAS_DB = True
except ImportError:
    HAS_DB = False

PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_DOCUMENTS_DIR = PROJECT_ROOT / "documents"
DEFAULT_OUTPUT_BASE = PROJECT_ROOT / "processed" / "documents"


SUPPORTED_EXTENSIONS = {
    ".pdf",
    ".docx",
    ".doc",
    ".pptx",
    ".ppt",
    ".png",
    ".jpg",
    ".jpeg",
    ".tiff",
}


def _get_pymupdf():
    try:
        import fitz as pymupdf
        return pymupdf
    except ImportError:
        try:
            import pymupdf
            return pymupdf
        except ImportError as exc:
            raise ImportError(
                "PyMuPDF is not installed. Run: pip install pymupdf"
            ) from exc


class DocumentPipeline:
    def __init__(
        self,
        force_ocr: bool = False,
        skip_ingest: bool = True,
        save_results: bool = True,
        enrichment_model: Optional[str] = None,
        enrichment_load_in_4bit: Optional[bool] = None,
        enrichment_device: str = "auto",
        enrichment_dtype: str = "auto",
        enrichment_trust_remote_code: Optional[bool] = None,
    ):
        self.force_ocr = force_ocr
        self.skip_ingest = skip_ingest
        self.save_results = save_results
        self.enrichment_model = enrichment_model
        self.enrichment_load_in_4bit = enrichment_load_in_4bit
        self.enrichment_device = enrichment_device
        self.enrichment_dtype = enrichment_dtype
        self.enrichment_trust_remote_code = enrichment_trust_remote_code
        self.chunker = DocumentChunker(max_tokens=512, overlap=64)
        self.ocr_engine = None
        self.enricher = None

    def _ensure_ocr(self):
        if self.ocr_engine is None:
            self.ocr_engine = DocumentOCREngine()
        return self.ocr_engine

    def _ensure_enricher(self):
        if self.enricher is None:
            self.enricher = DocumentEnricher(
                model_name=self.enrichment_model,
                load_in_4bit=self.enrichment_load_in_4bit,
                device=self.enrichment_device,
                torch_dtype=self.enrichment_dtype,
                trust_remote_code=self.enrichment_trust_remote_code,
            )
        return self.enricher

    @staticmethod
    def _is_supported_file(path: Path) -> bool:
        return path.suffix.lower() in SUPPORTED_EXTENSIONS

    @staticmethod
    def _build_output_paths(path: Path, output_base: Path):
        # Windows paths cannot end with space/dot; normalize for output folders.
        safe_stem = path.stem.strip(" .")
        if not safe_stem:
            safe_stem = "document"
        doc_root = output_base / safe_stem
        results_dir = doc_root / "results"
        temp_dir = doc_root / "temp"
        return doc_root, results_dir, temp_dir

    @staticmethod
    def _normalize_for_dedupe(text: str) -> str:
        if not text:
            return ""
        lowered = text.lower()
        lowered = re.sub(r"\s+", " ", lowered)
        # Keep unicode word characters and spaces for robust approximate checks.
        return re.sub(r"[^\w\s]", "", lowered, flags=re.UNICODE).strip()

    def _merge_page_text_with_ocr(self, extracted_text: str, ocr_text: str) -> str:
        """Append only OCR lines that are not already present in extracted text."""
        extracted_text = (extracted_text or "").strip()
        ocr_text = (ocr_text or "").strip()
        if not extracted_text:
            return ocr_text
        if not ocr_text:
            return extracted_text

        extracted_norm = self._normalize_for_dedupe(extracted_text)
        novel_lines: List[str] = []
        seen_norm = set()

        for raw_line in ocr_text.splitlines():
            line = raw_line.strip()
            if len(line) < 3:
                continue
            nline = self._normalize_for_dedupe(line)
            if not nline or nline in seen_norm:
                continue
            seen_norm.add(nline)
            if nline in extracted_norm:
                continue
            novel_lines.append(line)

        if not novel_lines:
            return extracted_text

        merged_ocr = "\n".join(novel_lines)
        return (
            f"{extracted_text}\n\n"
            f"[OCR Image/Text Layer]\n"
            f"{merged_ocr}"
        )

    def process_file(self, file_path: str, output_base: str = str(DEFAULT_OUTPUT_BASE)):
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {path}")
        if not self._is_supported_file(path):
            raise ValueError(f"Unsupported file type: {path.suffix.lower()}")

        output_base_path = Path(output_base)
        doc_root, results_dir, temp_dir = self._build_output_paths(path, output_base_path)
        temp_dir.mkdir(parents=True, exist_ok=True)

        print(f"\n{'=' * 50}")
        print(f"Processing Document: {path.name}")
        print(f"{'=' * 50}")

        start_time = time.time()

        # 1. Extraction
        ext = path.suffix.lower()
        extractor = get_extractor(str(path))

        pages: List[Dict] = []
        needs_ocr = False
        extraction_method = "text"

        if extractor is not None:
            print("1. Extracting text directly...")
            if isinstance(extractor, PDFExtractor):
                pages, needs_ocr = extractor.extract(str(path))
            else:
                pages = extractor.extract(str(path))
        else:
            # Image-only flow
            needs_ocr = True

        if self.force_ocr or needs_ocr:
            print("1b. Performing OCR on scanned/image content...")
            extraction_method = "mixed" if pages else "ocr"
            ocr = self._ensure_ocr()

            if ext == ".pdf":
                # OCR only pages that need it unless force_ocr is enabled.
                pymupdf = _get_pymupdf()

                doc = pymupdf.open(str(path))
                for i in range(len(doc)):
                    if (
                        i < len(pages)
                        and not self.force_ocr
                        and not pages[i].get("needs_ocr")
                    ):
                        continue

                    print(f"   OCR on page {i + 1}...")
                    text, conf = ocr.extract_from_pdf_page(str(path), i)

                    if i < len(pages):
                        if self.force_ocr:
                            pages[i]["text"] = text
                        else:
                            pages[i]["text"] = self._merge_page_text_with_ocr(
                                pages[i].get("text", ""),
                                text,
                            )
                        pages[i]["ocr_confidence"] = conf
                    else:
                        pages.append(
                            {
                                "page_num": i + 1,
                                "text": text,
                                "ocr_confidence": conf,
                            }
                        )
                doc.close()
            elif ext in {".png", ".jpg", ".jpeg", ".tiff"}:
                from PIL import Image

                img = Image.open(str(path))
                text, conf = ocr.extract_text_from_image(img)
                pages = [{"page_num": 1, "text": text, "ocr_confidence": conf}]

        # 2. Enrichment
        enricher = self._ensure_enricher()
        if enricher.enabled and ext == ".pdf":
            print("2. Enriching pages with Qwen2-VL...")
            pymupdf = _get_pymupdf()
            from PIL import Image

            doc = pymupdf.open(str(path))
            for i, page_data in enumerate(pages):
                try:
                    img_path = temp_dir / f"page_{i + 1}.png"
                    zoom = 2
                    mat = pymupdf.Matrix(zoom, zoom)
                    pix = doc[i].get_pixmap(matrix=mat)
                    img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
                    img.save(img_path)

                    summary = enricher.enrich_page(str(img_path))
                    if summary:
                        page_data["summary"] = summary
                except Exception as e:
                    print(f"Skipping enrichment for page {i + 1}: {e}")
                finally:
                    if img_path.exists():
                        img_path.unlink()
            doc.close()

        # 3. Chunking
        print("3. Chunking text...")
        chunks = self.chunker.chunk_document(pages)

        stat = path.stat()
        results = {
            "metadata": {
                "filename": path.name,
                "file_path": str(path.absolute()),
                "file_type": ext.lstrip("."),
                "file_size_mb": round(stat.st_size / (1024 * 1024), 2),
                "total_pages": len(pages),
                "extraction_method": extraction_method,
                "ocr_model": "surya/easyocr" if (self.force_ocr or needs_ocr) else None,
                "visual_enrichment_model": (
                    enricher.model_name if (enricher.enabled and ext == ".pdf") else None
                ),
                "language": "en",
            },
            "chunks": chunks,
        }

        # 4. Save results (optional)
        out_file = None
        if self.save_results:
            results_dir.mkdir(parents=True, exist_ok=True)
            out_file = results_dir / "results.json"
            with open(out_file, "w", encoding="utf-8") as f:
                json.dump(results, f, indent=2, ensure_ascii=False)

        processing_time = time.time() - start_time
        if out_file is not None:
            print(f"[OK] Results saved to {out_file} in {processing_time:.2f}s")
        else:
            print(f"[OK] Document processed in {processing_time:.2f}s")

        # 5. Ingestion (optional)
        if self.skip_ingest:
            print("4. Skipping database ingestion (extraction-only mode).")
        elif not HAS_DB:
            print("4. Skipping database ingestion (ingester module unavailable).")
        else:
            print("4. Ingesting to database...")
            try:
                with DocumentIngester() as ingester:
                    ingester.ingest_document(results)
            except Exception as e:
                # Keep extraction pipeline successful even when DB is unreachable.
                print(f"[WARN] Database ingestion failed, continuing: {e}")

        # Cleanup document temp directory if empty.
        if temp_dir.exists():
            try:
                shutil.rmtree(temp_dir)
            except Exception:
                pass

        return results

    def process_folder(
        self,
        documents_dir: str = str(DEFAULT_DOCUMENTS_DIR),
        output_base: str = str(DEFAULT_OUTPUT_BASE),
        recursive: bool = False,
    ):
        docs_path = Path(documents_dir)
        if not docs_path.exists():
            raise FileNotFoundError(f"Documents folder not found: {docs_path}")

        glob_pattern = "**/*" if recursive else "*"
        files = sorted(
            p for p in docs_path.glob(glob_pattern) if p.is_file() and self._is_supported_file(p)
        )

        if not files:
            print(f"No supported document files found in: {docs_path}")
            return []

        print(f"\n{'=' * 60}")
        print(f"Starting document batch processing: {len(files)} files")
        print(f"Source folder: {docs_path}")
        print(f"{'=' * 60}")

        summary = []
        for idx, file_path in enumerate(files, 1):
            print(f"\nDocument {idx}/{len(files)}: {file_path.name}")
            try:
                result = self.process_file(str(file_path), output_base=output_base)
                summary.append(
                    {
                        "file": str(file_path),
                        "success": True,
                        "chunks": len(result.get("chunks", [])),
                        "pages": result.get("metadata", {}).get("total_pages", 0),
                    }
                )
            except Exception as e:
                print(f"[ERROR] Failed to process {file_path.name}: {e}")
                summary.append({"file": str(file_path), "success": False, "error": str(e)})

        ok_count = sum(1 for item in summary if item["success"])
        fail_count = len(summary) - ok_count
        print(f"\n{'=' * 60}")
        print("DOCUMENT BATCH SUMMARY")
        print(f"{'=' * 60}")
        print(f"Success: {ok_count}/{len(summary)}")
        print(f"Failed:  {fail_count}/{len(summary)}")
        return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Document processing pipeline")
    parser.add_argument("--file", type=str, help="Process a single document file")
    parser.add_argument(
        "--folder",
        type=str,
        default=str(DEFAULT_DOCUMENTS_DIR),
        help="Process all supported files in this folder",
    )
    parser.add_argument(
        "--output-base",
        type=str,
        default=str(DEFAULT_OUTPUT_BASE),
        help="Base output folder for saved results",
    )
    parser.add_argument("--force-ocr", action="store_true", help="Force OCR on all pages")
    parser.add_argument(
        "--skip-db",
        action="store_true",
        help="Skip database ingestion (default behavior; kept for compatibility)",
    )
    parser.add_argument(
        "--ingest-db",
        action="store_true",
        help="Enable database ingestion after extraction",
    )
    parser.add_argument("--no-save", action="store_true", help="Do not save results JSON")
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Recursively process documents in nested folders",
    )
    parser.add_argument(
        "--enrichment-model",
        type=str,
        default=None,
        help=(
            "Vision-language model for page enrichment "
            "(e.g., Qwen/Qwen2-VL-7B-Instruct, Qwen/Qwen2.5-VL-72B-Instruct, "
            "or any compatible HF VLM)"
        ),
    )
    parser.add_argument(
        "--enrichment-device",
        type=str,
        default="auto",
        choices=["auto", "cuda", "cpu"],
        help="Device for enrichment model",
    )
    parser.add_argument(
        "--enrichment-dtype",
        type=str,
        default="auto",
        choices=["auto", "bf16", "fp16", "fp32", "bfloat16", "float16", "float32"],
        help="Torch dtype for enrichment model",
    )
    parser.add_argument(
        "--enrichment-load-in-4bit",
        dest="enrichment_load_in_4bit",
        action="store_true",
        help="Enable 4-bit quantization for enrichment model",
    )
    parser.add_argument(
        "--enrichment-no-4bit",
        dest="enrichment_load_in_4bit",
        action="store_false",
        help="Disable 4-bit quantization for enrichment model",
    )
    parser.add_argument(
        "--enrichment-trust-remote-code",
        dest="enrichment_trust_remote_code",
        action="store_true",
        help="Allow trust_remote_code when loading enrichment model",
    )
    parser.add_argument(
        "--enrichment-no-trust-remote-code",
        dest="enrichment_trust_remote_code",
        action="store_false",
        help="Disable trust_remote_code when loading enrichment model",
    )
    parser.set_defaults(
        enrichment_load_in_4bit=None,
        enrichment_trust_remote_code=None,
    )

    args = parser.parse_args()

    # Default mode is extraction + save only (no DB).
    # --ingest-db can enable ingestion; --skip-db keeps it disabled.
    skip_ingest = True
    if args.ingest_db:
        skip_ingest = False
    if args.skip_db:
        skip_ingest = True

    pipeline = DocumentPipeline(
        force_ocr=args.force_ocr,
        skip_ingest=skip_ingest,
        save_results=not args.no_save,
        enrichment_model=args.enrichment_model,
        enrichment_load_in_4bit=args.enrichment_load_in_4bit,
        enrichment_device=args.enrichment_device,
        enrichment_dtype=args.enrichment_dtype,
        enrichment_trust_remote_code=args.enrichment_trust_remote_code,
    )

    if args.file:
        pipeline.process_file(args.file, output_base=args.output_base)
    else:
        pipeline.process_folder(
            documents_dir=args.folder,
            output_base=args.output_base,
            recursive=args.recursive,
        )
