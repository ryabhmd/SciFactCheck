"""Create jsonl files from converted pdf json files for KB."""

import pandas as pd
import os
import json


def main():
    scifactcheck_data = pd.read_csv("scifactcheck_20251218.csv")
    log_data = pd.read_csv("log_pdf_download_with_organized_paths_20251218.csv")

    merged_data = pd.merge(
        scifactcheck_data,
        log_data[["paper_doi", "s2orc_url", "organized_pdf_path"]],
        on=["paper_doi", "s2orc_url"],
        how="left",
    )
    merged_data["json_file"] = merged_data["organized_pdf_path"].apply(
        lambda x: (
            os.path.splitext(os.path.basename(x))[0] + ".json" if pd.notna(x) else None
        )
    )

    title_lookup = dict(zip(merged_data["json_file"], merged_data["title"]))
    save_dir = "./jsonl_files"
    folder_path = "converted_pdfs_scifact_md"

    for file_name in os.listdir(folder_path):
        print(f"Processing file: {file_name}")
        data_jsonl = {"title": "", "text": []}

        if not file_name.endswith(".json"):
            continue

        file_path = os.path.join(folder_path, file_name)
        data_jsonl["title"] = title_lookup.get(file_name, "")

        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        for item in data:
            text = item.get("paragraphs")
            data_jsonl["text"].extend(text)
        output_file_path = os.path.join(save_dir, file_name.replace(".json", ".jsonl"))
        with open(output_file_path, "w", encoding="utf-8") as out_f:
            json.dump(data_jsonl, out_f, ensure_ascii=False)


if __name__ == "__main__":
    main()
