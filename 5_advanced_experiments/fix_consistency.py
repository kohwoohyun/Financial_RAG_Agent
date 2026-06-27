import csv
import glob
import os
from collections import Counter, defaultdict

files = glob.glob("results/multihop_raw_*.csv")
latest_file = max(files, key=os.path.getctime)

companies = ["삼성전자", "SK하이닉스", "현대자동차", "LG에너지솔루션", "NAVER"]

def extract_decision(answer, candidates):
    mentioned = [c for c in candidates if c in answer]
    # 가장 먼저 결론으로 언급된 기업 추정: 문장 앞부분에 등장한 기업
    if not mentioned:
        return None
    return min(mentioned, key=lambda c: answer.find(c))

by_question = defaultdict(list)
with open(latest_file, "r", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    for row in reader:
        if row["tier"] == "2hop_비교":
            by_question[row["question"]].append(row["final_answer"])

print("=== 결정 수준 일관성 (2hop_비교) ===\n")
for q, answers in by_question.items():
    candidates = [c for c in companies if c in q]
    decisions = [extract_decision(a, candidates) for a in answers]
    counter = Counter(decisions)
    top = counter.most_common(1)[0][1]
    print(f"[{q}]")
    print(f"  결정들: {decisions}")
    print(f"  결정 일관성: {top}/{len(answers)} ({top/len(answers)*100:.0f}%)\n")
