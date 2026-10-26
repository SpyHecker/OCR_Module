"""Command-Line Interface for OCR Module."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from ocr_module.config import PipelineConfig, EngineConfig, PreprocessingConfig
from ocr_module.pipeline import OCRPipeline


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="ocr-module",
        description="OCR Module: Extract text and structured metadata from images."
    )
    parser.add_argument(
        "-i", "--input",
        required=True,
        type=str,
        help="Path to input image file (PNG, JPG, TIFF, etc.)."
    )
    parser.add_argument(
        "-e", "--engine",
        default="auto",
        choices=["auto", "baseline", "tesseract", "transformers", "mock"],
        help="OCR engine backend to utilize (default: 'auto')."
    )
    parser.add_argument(
        "-l", "--languages",
        default="eng",
        help="Comma-separated language codes, e.g. 'eng,fra' (default: 'eng')."
    )
    parser.add_argument(
        "--no-preprocess",
        action="store_true",
        help="Disable automatic computer vision preprocessing (deskew, binarize, etc.)."
    )
    parser.add_argument(
        "-f", "--output-format",
        default="text",
        choices=["text", "json"],
        help="Output format: 'text' (default) or 'json' with bounding boxes and confidence."
    )
    parser.add_argument(
        "-o", "--out-file",
        type=str,
        default=None,
        help="Optional path to write output file instead of stdout."
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    input_path = Path(args.input)
    if not input_path.exists():
        sys.stderr.write(f"Error: File not found: '{args.input}'\n")
        return 1

    lang_list = [lang.strip() for lang in args.languages.split(",") if lang.strip()]

    config = PipelineConfig(
        preprocessing=PreprocessingConfig(enabled=not args.no_preprocess),
        engine=EngineConfig(
            engine_type=args.engine,
            languages=lang_list
        )
    )

    try:
        pipeline = OCRPipeline(config=config)
        result = pipeline.process(input_path)
    except Exception as exc:
        sys.stderr.write(f"OCR Processing Error: {exc}\n")
        return 2

    if args.output_format == "json":
        output_str = json.dumps(result.to_dict(), indent=2)
    else:
        output_str = result.text

    if args.out_file:
        out_path = Path(args.out_file)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(output_str, encoding="utf-8")
        print(f"Output saved to {out_path}")
    else:
        print(output_str)

    return 0


if __name__ == "__main__":
    sys.exit(main())
