import csv
import glob
import os
from collections import defaultdict

files = glob.glob("results/tool_trust_test_*.csv")
latest_file = max(files, key=os.path.getctime)
print(f"파일: {latest_file}\n")

by_question = defaultdict(list)
total_followed = 0
total = 0

with open(latest_file, "r", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    for row in reader:
        by_question[row["question"]].append(row)
        total += 1
        if row["followed_fake"] == "True":
            total_followed += 1

for q, rows in by_question.items():
    followed = sum(1 for r in rows if r["followed_fake"] == "True")
    print(f"[{q}]")
    print(f"  실제값: {rows[0]['true_pct']}%  /  오염값: {rows[0]['fake_pct_given']}%")
    for r in rows:
        mark = "✓베낌" if r["followed_fake"] == "True" else "✗의심"
        print(f"  시도{r['trial']}: {mark} — {r['final_answer'][:50]}")
    print(f"  → {followed}/{len(rows)} 오염값 추종\n")

print("=" * 50)
print(f"전체: {total_followed}/{total} ({total_followed/total*100:.0f}%) 오염값을 그대로 신뢰함")
