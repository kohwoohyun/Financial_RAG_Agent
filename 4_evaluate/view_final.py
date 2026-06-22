import csv
import glob
import os

files = glob.glob("results/final_comparison_*.csv")
latest_file = max(files, key=os.path.getctime)
print(f"파일: {latest_file}\n")
print("=" * 80)

with open(latest_file, "r", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    for i, row in enumerate(reader, 1):
        print(f"\n[{i}] ({row['type']}) {row['question']}")
        print(f"  LLM 단독: {row['llm_only'][:150]}")
        print(f"  RAG: {row['rag'][:150]}")
        print(f"  Agent: {row['agent'][:150]}")
        print("-" * 80)
