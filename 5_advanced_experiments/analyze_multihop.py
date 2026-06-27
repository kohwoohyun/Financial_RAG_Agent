import csv
import glob
import os
from collections import defaultdict, Counter

files = glob.glob("results/multihop_raw_*.csv")
latest_file = max(files, key=os.path.getctime)
print(f"분석 파일: {latest_file}\n")

by_question = defaultdict(list)
with open(latest_file, "r", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    for row in reader:
        by_question[(row["tier"], row["question"])].append(row)

print("=" * 70)
print("질문별 정확도 및 일관성 (Self-Consistency)")
print("=" * 70)

tier_stats = defaultdict(lambda: {"success": 0, "total": 0, "diagnoses": Counter()})

for (tier, question), rows in by_question.items():
    total = len(rows)
    success = sum(1 for r in rows if r["diagnosis"] == "성공")
    answer_counter = Counter(r["final_answer"].strip() for r in rows)
    most_common_count = answer_counter.most_common(1)[0][1]
    consistency = most_common_count / total * 100

    print(f"\n[{tier}] {question}")
    print(f"  정확도: {success}/{total} ({success/total*100:.0f}%)")
    print(f"  답변 일관성: {most_common_count}/{total} ({consistency:.0f}%)")
    print(f"  진단 분포: {dict(Counter(r['diagnosis'] for r in rows))}")

    tier_stats[tier]["success"] += success
    tier_stats[tier]["total"] += total
    for r in rows:
        tier_stats[tier]["diagnoses"][r["diagnosis"]] += 1

print("\n" + "=" * 70)
print("복잡도(Tier)별 종합 결과")
print("=" * 70 + "\n")

summary_filename = latest_file.replace("raw_", "summary_")
with open(summary_filename, "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["tier", "정확도(%)", "성공", "전체", "진단분포"])
    for tier, stats in tier_stats.items():
        acc = stats["success"] / stats["total"] * 100
        acc_str = f"{stats['success']}/{stats['total']} ({acc:.0f}%)"
        diag = stats["diagnoses"].copy()
        diag.pop("성공", None)
        top_fail = diag.most_common(1)
        top_fail_str = f"{top_fail[0][0]} ({top_fail[0][1]}건)" if top_fail else "없음"
        print(f"{tier:<15}{acc_str:<18}{top_fail_str}")
        writer.writerow([tier, f"{acc:.0f}", stats["success"], stats["total"], dict(stats["diagnoses"])])

print(f"\n요약 저장 완료: {summary_filename}")
