import os
from factscore.factscorer import FactScorer


def main():
    fs = FactScorer()
    input_path = "./jsonl_clean_files"
    for file in os.listdir(input_path):
        file_path = os.path.join(input_path, file)

        db_name = file.replace(".jsonl", "")
        db_path = os.path.join("./kb_files", db_name + ".db")
        fs.register_knowledge_source(db_name, data_path=file_path, db_path=db_path)


if __name__ == "__main__":
    main()
