import json
import os
from langchain_core.documents import Document
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

def format_amount(amount_str):
    try:
        amount = int(amount_str.replace(",", ""))
        jo = amount // 1_000_000_000_000
        eok = (amount % 1_000_000_000_000) // 100_000_000
        return f"{jo}조 {eok}억원"
    except (ValueError, TypeError):
        return amount_str

documents = []

for filename in sorted(os.listdir("data")):
    if not filename.endswith(".json"):
        continue

    company_year = filename.replace(".json", "")
    company = company_year.split("_")[0]

    with open(f"data/{filename}", "r", encoding="utf-8") as f:
        data = json.load(f)

    if data.get("status") != "000":
        continue

    for item in data.get("list", []):
        if item.get("fs_div") != "CFS":
            continue

        account = item.get("account_nm")
        year = item.get("bsns_year")
        thstrm = format_amount(item.get("thstrm_amount", ""))
        frmtrm = format_amount(item.get("frmtrm_amount", ""))

        sentence = f"{company}의 {year}년 {account}은(는) {thstrm}이며, 전년도는 {frmtrm}입니다."

        documents.append(Document(
            page_content=sentence,
            metadata={"company": company, "year": year, "account": account}
        ))

print(f"문서(문장) 수: {len(documents)}")

print("임베딩 모델 로딩 중...")
embeddings = HuggingFaceEmbeddings(
    model_name="snunlp/KR-SBERT-V40K-klueNLI-augSTS"
)

print("벡터 DB 생성 중...")
vectorstore = FAISS.from_documents(documents, embeddings)
vectorstore.save_local("data/faiss_index")

print("완료: data/faiss_index 에 저장됨")
