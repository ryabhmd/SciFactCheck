"""Script to create JSONL files from IEP articles saved as HTML."""

import os
from bs4 import BeautifulSoup
import json
import pandas as pd
import argparse
from tabulate import tabulate


def extract_text_and_tables(element: BeautifulSoup) -> list:
    content = []
    for child in element.children:
        if getattr(child, "name", None) == "p":
            text = child.get_text(strip=True)
            if text:
                content.append(text)
        elif getattr(child, "name", None) == "table":
            dfs = pd.read_html(str(child))
            if dfs:
                table = dfs[0]
                table = table.fillna("")
                if all(str(h).isdigit() for h in table.columns):
                    table.columns = [""] * len(table.columns)
                table_md = table.to_markdown(index=False, tablefmt="github")
                content.append(table_md)
        elif getattr(child, "children", None):
            content.extend(extract_text_and_tables(child))
    return content


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

        content_list = extract_text_and_tables(soup.body)

        if content_list and "Email:" in content_list[-1]:
            content_list.pop()

        data_jsonl = {"title": "", "text": []}
        data_jsonl["title"] = title
        data_jsonl["text"] = content_list
        output_file_path = os.path.join(
            output_dir, filename.replace(".html", ".jsonl").lower()
        )
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
