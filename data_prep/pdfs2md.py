"""Script for converting PDFs to markdown using Docling (https://docling-project.github.io/docling/)."""

import os
from docling.document_converter import DocumentConverter
from docling.datamodel.accelerator_options import AcceleratorDevice, AcceleratorOptions
import argparse
import csv
from tqdm import tqdm

accelerator_options = AcceleratorOptions(device=AcceleratorDevice.CUDA)

def main(args):

    input_path = args.input_path
    output_path = args.output_path
    log_path = args.log_path
    os.makedirs(output_path, exist_ok=True)
    os.makedirs(os.path.dirname(log_path), exist_ok=True)

    if not os.path.exists(log_path):
        with open(log_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["pdf_path", "error"])

    converter = DocumentConverter()

    pdf_files = [f for f in os.listdir(input_path) if f.lower().endswith(".pdf")]
    for filename in tqdm(pdf_files, desc="Converting PDFs", unit="file"):
        pdf_path = os.path.join(input_path, filename)
        md_filename = os.path.splitext(filename)[0] + ".md"
        md_path = os.path.join(output_path, md_filename)
        try:
            result = converter.convert(pdf_path)
            markdown_text = result.document.export_to_markdown()

            with open(md_path, "w", encoding="utf-8") as f:
                f.write(markdown_text)
            print(f"Converted {filename} to {md_filename}")
                
        except Exception as e:
            error = str(e)
            print(f"Failed to convert {filename}: {error}")
            writer.writerow([pdf_path, error])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path", required=True)
    parser.add_argument(
        "--log_path", required=True, help="CSV file to log failed conversions"
    )
    parser.add_argument("--output_path", required=True)
    args = parser.parse_args()
    main(args)
