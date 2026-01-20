"""Script for scraping IEP articles based on URLs and saving them as HTML files."""

import requests
from bs4 import BeautifulSoup
import time
import os
import argparse
import pandas as pd
import re


def scrape_iep_article(url: str) -> tuple:
    headers = {"User-Agent": "Mozilla/5.0 (compatible; IEP-Scraper/1.0)"}
    try:
        response = requests.get(url, headers=headers, timeout=30)
        response.raise_for_status()
        soup = BeautifulSoup(response.content, "lxml")
        entry_content = soup.find("div", class_="entry-content")
        if entry_content is None:
            return None, "entry-content not found"
        return entry_content.prettify(), None
    except Exception as e:
        return None, f"parse_error: {e}"


def clean_title(name: str) -> str:
    return re.sub(r'[<>:"/\\|?*,.()]', "", name)


def main(args):
    output_dir = args.output_path
    log_path = args.log_path
    data_path = args.data_path

    error_log = []
    os.makedirs(output_dir, exist_ok=True)

    df = pd.read_pickle(data_path)
    iep = df[df["field"] == "Arts and Humanities"]

    for _, row in iep.iterrows():
        url = row["text_url"]
        title = clean_title(row["title"].replace(" ", "_"))
        idx = row.name
        filename = os.path.join(output_dir, f"{idx}_{title}.html")

        if os.path.exists(filename):
            print(f"Skipping {filename}, already exists)")
            continue

        result, error = scrape_iep_article(url)
        if result:
            with open(filename, "w", encoding="utf-8") as f:
                f.write(result)
            print(f"Saved: {filename}")
        else:
            error_log.append({"idx": idx, "url": url, "error": error})
            print(f"Error at {idx}: {error}")
        time.sleep(30)

    if error_log:
        log_df = pd.DataFrame(error_log)
        log_df.to_csv(log_path, index=False)
        print(f"Error log saved to {log_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", required=True)
    parser.add_argument("--log_path", required=True)
    parser.add_argument("--output_path", required=True)
    args = parser.parse_args()
    main(args)
