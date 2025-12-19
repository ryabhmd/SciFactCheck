"""Remove empty strings and very short paragraphs from jsonl files."""

import json
import re
import os


def is_valid_paragraph(s: str, min_len: int) -> bool:
    s = s.strip()
    if not s:
        return False
    if len(s) < min_len:
        return False
    if re.fullmatch(r"[\W\d_]+", s):
        return False
    return True

def main():
    min_len = 5
    input_path = "./jsonl_files"
    output_path = "./jsonl_clean_files"

    for file in os.listdir(input_path):
        file_path = os.path.join(input_path, file)
        output_file = os.path.join(output_path, file)

        with open(file_path, "r", encoding="utf-8") as fin, open(
            output_file, "w", encoding="utf-8"
        ) as fout:
            for line in fin:
                record = json.loads(line)

                record["text"] = [
                    p for p in record["text"] if is_valid_paragraph(p, min_len)
                ]

            fout.write(json.dumps(record, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
