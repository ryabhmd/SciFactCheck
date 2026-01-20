"""Script to create JSONL files from IEP articles saved as HTML."""

import os
from bs4 import BeautifulSoup
import json
import pandas as pd
import argparse


def main(args):
    output_dir = args.output_path
    html_dir = args.html_path
    data_path = args.data_path

    os.makedirs(output_dir, exist_ok=True)
    df = pd.read_pickle(data_path)

    for filename in os.listdir(html_dir):
        if not filename.endswith(".html"):
            continue

        idx = int(filename.split("_", 1)[0])
        title = df.loc[idx, "title"]

        file_path = os.path.join(html_dir, filename)
        with open(file_path, "r", encoding="utf-8") as f:
            soup = BeautifulSoup(f, "lxml")

        paragraphs = [
            p.get_text(strip=True) for p in soup.find_all("p") if p.get_text(strip=True)
        ]

        data_jsonl = {"title": "", "text": []}
        data_jsonl["title"] = title
        data_jsonl["text"] = paragraphs
        output_file_path = os.path.join(output_dir, filename.replace(".html", ".jsonl"))
        print(f"Saving {output_file_path}")
        with open(output_file_path, "w", encoding="utf-8") as out_f:
            json.dump(data_jsonl, out_f, ensure_ascii=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", required=True)
    parser.add_argument("--html_path", required=True)
    parser.add_argument("--output_path", required=True)
    args = parser.parse_args()
    main(args)
