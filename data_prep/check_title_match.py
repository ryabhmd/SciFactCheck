"""Check that correct papers were downloaded and parsed based on title match."""

import os
import json
import pandas as pd
from rapidfuzz import process, fuzz
import os
import datetime
import argparse

def fuzzy_match(row, threshold, df1, df2_grouped):
    json_file = row.get("json_file")
    title = row.get("title")
    if pd.isna(json_file):
        return False, None, None
    if json_file not in df2_grouped.groups:
        return False, None, None
    candidates = df1.loc[df2_grouped.groups[json_file], "title_lc"].tolist()
    match = process.extractOne(
        title.lower(),
        candidates,
        scorer=fuzz.token_sort_ratio
    )
    if match and match[1] >= threshold:
        score = match[1]
        return True, "token_sort_ratio", score
    else:
        score = match[1] if match else None
        return False, None, score

def main(args):
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    threshold = float(args.fuzzy_threshold)

    json_folder = args.json_folder
    save_path = args.save_path
    scifactcheck_data = pd.read_csv(args.data_path)
    log_data = pd.read_csv(args.log_path)

    # Create a df with all titles from JSON files
    rows = []
    for file_name in os.listdir(json_folder):
        if not file_name.endswith(".json"):
            continue
        file_path = os.path.join(json_folder, file_name)
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, list):
            continue
        for item in data:
            title = item.get("title")
            if isinstance(title, str) and title.strip() != "":
                rows.append({
                    "json_file": file_name,
                    "title": title
                })

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(save_path, "titles_extracted.csv"), index=False)
    df = pd.read_csv('titles_extracted.csv')

    # Merge dataframes to get json file names
    merged_data = pd.merge(scifactcheck_data, log_data[['paper_doi', 's2orc_url', 'organized_pdf_path']], on=['paper_doi', 's2orc_url'], how='left')
    merged_data["json_file"] = merged_data["organized_pdf_path"].apply(
        lambda x: os.path.splitext(os.path.basename(x))[0] + ".json" if pd.notna(x) else None
    )

    df = df.copy()
    df["title_lc"] = df["title"].str.lower()
    json_grouped = df.groupby("json_file")

    # Match titles using fuzzy matching
    merged_data[["matched", "match_method", "fuzzy_score"]] = merged_data.apply(
        lambda row: pd.Series(fuzzy_match(row, threshold, df, json_grouped)),
        axis=1
    )
    merged_data.to_csv(os.path.join(save_path, f"scifactcheck_data_{timestamp}.csv"), index=False)
    
    # Save subset of not matched excluding Arts and Humanities
    subset = merged_data[merged_data["field"] != "Arts and Humanities"]
    subset_missing = subset[subset["matched"] == False]
    subset_missing.to_csv(os.path.join(save_path, f"not_matched_{timestamp}.csv"), index=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--json_folder", required=True)
    parser.add_argument("--data_path", required=True)
    parser.add_argument("--log_path", required=True)
    parser.add_argument("--fuzzy_threshold", required=True)
    parser.add_argument("--save_path", required=True)
    args = parser.parse_args()
    main(args)
