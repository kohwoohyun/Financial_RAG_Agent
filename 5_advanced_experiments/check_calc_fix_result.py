import csv
import glob
import os
from collections import defaultdict

files = glob.glob("results/calc_tool_fix_*.csv")
latest_file = max(files, key=os.path.getctime)
print(f"파일: {latest_file}\n")

by_question = defaultdict(list)
with open(latest_file, "r", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    for row in reader:
        by_question[row["question"]].append(row)

total_success, total = 0, 0
for q, rows in by_question.items():
    success = sum(1 for r in rows if r["diagnosis"] == "성공")
    total_success += success
    total += len(rows)
    print(f"[{q}]")
    for r in rows:
        print(f"  시도{r['trial']}: {r['diagnosis']} (답변: {r['final_answer'][:60]})")
    print(f"  → {success}/{len(rows)} 성공\n")

print("=" * 50)
print(f"전체: {total_success}/{total} ({total_success/total*100:.0f}%)")
print(f"수정 전 (기존 실험): 2/15 (13%)")
