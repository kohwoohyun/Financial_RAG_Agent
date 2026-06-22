import csv
import json
import glob
import os
import re

def format_amount(amount_str):
    try:
        amount = int(amount_str.replace(",", ""))
        jo = amount // 1_000_000_000_000
        eok = (amount % 1_000_000_000_000) // 100_000_000
        return f"{jo}조 {eok}억원"
    except (ValueError, TypeError):
        return amount_str

def extract_year(question):
    m = re.search(r"(\d{4})년", question)
    return m.group(1) if m else None

companies = ["삼성전자", "SK하이닉스", "현대자동차", "LG에너지솔루션", "NAVER"]
accounts = ["유동자산", "비유동자산", "자산총계", "유동부채", "비유동부채",
            "부채총계", "자본금", "이익잉여금", "자본총계", "매출액",
            "영업이익", "법인세차감전 순이익", "당기순이익(손실)", "총포괄손익"]
account_synonyms = {"순이익": "당기순이익(손실)", "총자산": "자산총계", "총부채": "부채총계"}

def extract_company(q):
    for c in companies:
        if c in q: return c
    return None

def extract_account(q):
    for a in accounts:
        if a in q: return a
    for k, v in account_synonyms.items():
        if k in q: return v
    return None

def get_ground_truth(company, year, account):
    filename = f"data/{company}_{year}.json"
    if not os.path.exists(filename): return None
    with open(filename, "r", encoding="utf-8") as f:
        data = json.load(f)
    if data.get("status") != "000": return None
    for item in data.get("list", []):
        if item.get("fs_div") == "CFS" and item.get("account_nm") == account:
            return format_amount(item.get("thstrm_amount", ""))
    return None

files = glob.glob("results/final_comparison_*.csv")
latest_file = max(files, key=os.path.getctime)

rows = []
llm_correct, rag_correct, agent_correct, total = 0, 0, 0, 0

with open(latest_file, "r", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    for row in reader:
        question, qtype = row["question"], row["type"]

        if qtype == "재무수치":
            company, year, account = extract_company(question), extract_year(question), extract_account(question)
            ground_truth = get_ground_truth(company, year, account)
            llm_ok = (ground_truth in row["llm_only"]) if ground_truth else None
            rag_ok = (ground_truth in row["rag"]) if ground_truth else None
            agent_ok = (ground_truth in row["agent"]) if ground_truth else None
            if ground_truth:
                total += 1
                llm_correct += int(llm_ok)
                rag_correct += int(rag_ok)
                agent_correct += int(agent_ok)
        else:
            ground_truth, llm_ok, rag_ok, agent_ok = None, None, None, None  # 뉴스/동향은 정성 평가 대상

        rows.append({"question": question, "type": qtype, "ground_truth": ground_truth,
                      "llm_correct": llm_ok, "rag_correct": rag_ok, "agent_correct": agent_ok})

        print(f"[{qtype}] {question}\n  정답: {ground_truth}\n  LLM: {llm_ok} | RAG: {rag_ok} | Agent: {agent_ok}\n")

print("=" * 50)
print(f"재무수치 질문 {total}개 중 정확도")
print(f"LLM 단독: {llm_correct}/{total} ({llm_correct/total*100:.0f}%)")
print(f"RAG:      {rag_correct}/{total} ({rag_correct/total*100:.0f}%)")
print(f"Agent:    {agent_correct}/{total} ({agent_correct/total*100:.0f}%)")

out_file = latest_file.replace(".csv", "_scored.csv")
with open(out_file, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=["question", "type", "ground_truth", "llm_correct", "rag_correct", "agent_correct"])
    writer.writeheader()
    writer.writerows(rows)
print(f"\n채점 결과 저장: {out_file}")
