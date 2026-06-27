import csv
import glob
import os
import json

files = glob.glob("results/multihop_raw_*.csv")
latest_file = max(files, key=os.path.getctime)

def get_raw(company, year, account, field):
    with open(f"data/{company}_{year}.json", "r", encoding="utf-8") as f:
        data = json.load(f)
    for item in data.get("list", []):
        if item.get("fs_div") == "CFS" and item.get("account_nm") == account:
            return int(item.get(field, "0").replace(",", ""))
    return None

meta_map = {
    "현대자동차의 2024년 영업이익은 전년 대비 몇 퍼센트 증가했어?": ("현대자동차", "2024", "영업이익"),
    "삼성전자의 2024년 매출액은 전년 대비 몇 퍼센트 증가했어?": ("삼성전자", "2024", "매출액"),
    "NAVER의 2023년 영업이익은 전년 대비 몇 퍼센트 증가했어?": ("NAVER", "2023", "영업이익"),
}

with open(latest_file, "r", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    for row in reader:
        if row["tier"] == "2hop_계산" and row["diagnosis"] == "추론_실패":
            company, year, account = meta_map[row["question"]]
            curr = get_raw(company, year, account, "thstrm_amount")
            prev = get_raw(company, year, account, "frmtrm_amount")
            correct = (curr - prev) / abs(prev) * 100
            print(f"[시도 {row['trial']}] {row['question']}")
            print(f"  정답 증감률: {correct:.1f}% (당기 {curr:,} / 전기 {prev:,})")
            print(f"  모델 답변: {row['final_answer']}\n")
