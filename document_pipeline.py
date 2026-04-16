"""Document processing pipeline for ATLAS.

Supports single-file and folder-wide processing:
1. Extraction (PDF/DOCX/PPTX/image)
2. OCR fallback for scanned content
3. Optional page enrichment
4. Chunking
5. Optional result persistence
6. Optional DB ingestion
"""

import argparse
import json
import shutil
import sys
import time
from pathlib import Path
from typing import Dict, List

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
        skip_ingest: bool = False,
        save_results: bool = True,
    ):
        self.force_ocr = force_ocr
        self.skip_ingest = skip_ingest
        self.save_results = save_results
        self.chunker = DocumentChunker(max_tokens=512, overlap=64)
        self.ocr_engine = None
        self.enricher = None

    def _ensure_ocr(self):
        if self.ocr_engine is None:
            self.ocr_engine = DocumentOCREngine()
        return self.ocr_engine

    def _ensure_enricher(self):
        if self.enricher is None:
            self.enricher = DocumentEnricher()
        return self.enricher

    @staticmethod
    def _is_supported_file(path: Path) -> bool:
        return path.suffix.lower() in SUPPORTED_EXTENSIONS

    @staticmethod
    def _build_output_paths(path: Path, output_base: Path):
        doc_root = output_base / path.stem
        results_dir = doc_root / "results"
        temp_dir = doc_root / "temp"
        return doc_root, results_dir, temp_dir

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
                        pages[i]["text"] = text
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

        # 5. Ingestion
        if HAS_DB and not self.skip_ingest:
            print("4. Ingesting to database...")
            with DocumentIngester() as ingester:
                ingester.ingest_document(results)

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
    parser.add_argument("--skip-db", action="store_true", help="Skip database ingestion")
    parser.add_argument("--no-save", action="store_true", help="Do not save results JSON")
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Recursively process documents in nested folders",
    )

    args = parser.parse_args()

    pipeline = DocumentPipeline(
        force_ocr=args.force_ocr,
        skip_ingest=args.skip_db,
        save_results=not args.no_save,
    )

    if args.file:
        pipeline.process_file(args.file, output_base=args.output_base)
    else:
        pipeline.process_folder(
            documents_dir=args.folder,
            output_base=args.output_base,
            recursive=args.recursive,
        )
