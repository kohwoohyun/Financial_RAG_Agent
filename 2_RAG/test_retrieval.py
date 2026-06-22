from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

embeddings = HuggingFaceEmbeddings(
    model_name="snunlp/KR-SBERT-V40K-klueNLI-augSTS"
)
vectorstore = FAISS.load_local(
    "data/faiss_index", embeddings, allow_dangerous_deserialization=True
)

companies = ["삼성전자", "SK하이닉스", "현대자동차", "LG에너지솔루션", "NAVER"]

def extract_company(question):
    for c in companies:
        if c in question:
            return c
    return None

question = "삼성전자 매출액이 얼마야?"
target_company = extract_company(question)

if target_company:
    results = vectorstore.similarity_search(
        question, k=3, filter={"company": target_company}
    )
else:
    results = vectorstore.similarity_search(question, k=3)

print(f"질문: {question}")
print(f"감지된 기업: {target_company}\n")
print("검색된 관련 문서:")
for i, doc in enumerate(results, 1):
    print(f"{i}. {doc.page_content}")
