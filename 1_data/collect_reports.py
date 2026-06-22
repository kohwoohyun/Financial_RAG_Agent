import requests
import os
import json
import time

api_key = os.getenv("DART_API_KEY")

companies = {
    "삼성전자": "00126380",
    "SK하이닉스": "00164779",
    "현대자동차": "00164742",
    "LG에너지솔루션": "01515323",
    "NAVER": "00266961",
}

def get_business_report(corp_code, year):
    url = "https://opendart.fss.or.kr/api/fnlttSinglAcnt.json"
    params = {
        "crtfc_key": api_key,
        "corp_code": corp_code,
        "bsns_year": year,
        "reprt_code": "11011"
    }
    response = requests.get(url, params=params)
    return response.json()

for name, code in companies.items():
    for year in ["2022", "2023", "2024"]:
        data = get_business_report(code, year)
        filename = f"data/{name}_{year}.json"
        with open(filename, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        print(f"저장 완료: {filename}")
        time.sleep(0.5)
