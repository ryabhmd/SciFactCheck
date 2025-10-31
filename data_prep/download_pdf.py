"""Script for downloading paper PDFs from URL."""

import pandas as pd
import requests
import os
import csv
from time import sleep
from urllib.parse import urlparse
import re
import argparse


def get_filename(response, fallback_url):
    """
    Extracts filename from the Content-Disposition header if present.
    Otherwise, falls back to the URL basename or a default name.
    """
    cd = response.headers.get("Content-Disposition")
    if cd:
        match = re.findall('filename="?([^"]+)"?', cd)
        if match:
            return match[0]
    name = os.path.basename(urlparse(fallback_url).path)
    return name


def download_file(url: str, output_dir: str) -> tuple:
    try:
        with requests.get(url, timeout=30) as response:
            if response.status_code == 200:
                filename = get_filename(response, url)
                filepath = os.path.join(output_dir, filename)
                os.makedirs(output_dir, exist_ok=True)
                with open(filepath, "wb") as f:
                    f.write(response.content)
                return filepath, f"{response.status_code}, ok", ""
            elif response.status_code == 404:
                return "", f"{response.status_code}, missing", "404 not found"
            else:
                return "", "error", f"HTTP {response.status_code}"
    except Exception as e:
        return "", "error", str(e)


def main(args):

    df = pd.read_csv(args.csv_path)
    df = df[df["field"] != "Arts and Humanities"]

    log_file = args.log_path
    output_dir = args.output_path
    os.makedirs(os.path.dirname(log_file), exist_ok=True)

    if not os.path.exists(log_file):
        with open(log_file, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(
                ["pdf_url", "paper_doi", "s2orc_url", "pdf_path", "pdf_status", "note"]
            )

    os.makedirs(output_dir, exist_ok=True)

    for i, (_, row) in enumerate(df.iterrows(), start=1):
        pdf_status, pdf_note = "skipped", ""
        pdf_url = row["pdf_url"]

        pdf_path, pdf_status, pdf_note = download_file(pdf_url, output_dir)
        print(f"Downloading PDF from: {pdf_url}")

        with open(log_file, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    pdf_url,
                    row["paper_doi"],
                    row["s2orc_url"],
                    pdf_path,
                    pdf_status,
                    pdf_note or "",
                ]
            )

            sleep(5)
            print("Taking a 5-seconds break...")
            if i % 300 == 0:
                print("Taking a 2-minute break...")
                sleep(120)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv_path", required=True)
    parser.add_argument("--log_path", required=True)
    parser.add_argument("--output_path", required=True)
    args = parser.parse_args()
    main(args)
