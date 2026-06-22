import csv
import os
import re
from datetime import datetime
from collections import namedtuple
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_ollama import OllamaLLM

embeddings = HuggingFaceEmbeddings(model_name="snunlp/KR-SBERT-V40K-klueNLI-augSTS")
vectorstore = FAISS.load_local("data/faiss_index", embeddings, allow_dangerous_deserialization=True)
llm = OllamaLLM(model="gemma3:4b")

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

def ask_llm_only(question):
    return llm.invoke(question)

def ask_rag(question, k=5):
    target_company = extract_company(question)
    target_account = extract_account(question)
    filter_dict = {}
    if target_company: filter_dict["company"] = target_company
    if target_account: filter_dict["account"] = target_account
    docs = vectorstore.similarity_search(question, k=k, filter=filter_dict, fetch_k=200) if filter_dict else vectorstore.similarity_search(question, k=k, fetch_k=200)
    docs = sorted(docs, key=lambda d: d.metadata.get("year", ""), reverse=True)
    context = "\n".join([d.page_content for d in docs]) if docs else "관련 데이터 없음"
    prompt = f"""다음은 참고할 재무 정보입니다:
{context}

위 정보를 바탕으로 질문에 답변하세요. 정보가 없으면 모른다고 답하세요.
질문에 특정 연도가 명시되지 않았다면 최신 연도 기준으로 답변하세요.
질문: {question}"""
    return llm.invoke(prompt)

web_search = DuckDuckGoSearchRun()

def rag_search_tool(query):
    target_company = extract_company(query)
    target_account = extract_account(query)
    filter_dict = {}
    if target_company: filter_dict["company"] = target_company
    if target_account: filter_dict["account"] = target_account
    docs = vectorstore.similarity_search(query, k=5, filter=filter_dict, fetch_k=200) if filter_dict else vectorstore.similarity_search(query, k=5, fetch_k=200)
    if not docs: return "관련된 재무 데이터를 찾을 수 없습니다."
    docs = sorted(docs, key=lambda d: d.metadata.get("year", ""), reverse=True)
    return "\n".join([d.page_content for d in docs])

ToolDef = namedtuple("ToolDef", ["name", "description", "func"])
tools = [
    ToolDef("재무데이터검색", "5개 기업의 2022~2024년 매출액, 영업이익, 자산총계 등 재무 수치 조회", rag_search_tool),
    ToolDef("웹검색", "최신 뉴스, 시장 동향 등 재무데이터검색에 없는 정보 검색", web_search.run),
]
tool_map = {t.name: t.func for t in tools}

def run_agent(question, max_iterations=5, verbose=False):
    tools_desc = "\n".join([f"{t.name}: {t.description}" for t in tools])
    tool_names = ", ".join([t.name for t in tools])
    prompt = f"""다음 질문에 최선을 다해 답변하세요. 아래 도구들을 사용할 수 있습니다:

{tools_desc}

다음 형식을 정확히 따르세요. 절대 Action 없이 바로 Final Answer를 쓰지 마세요.

예시:
Question: 2023년 SK하이닉스 매출액이 얼마야?
Thought: SK하이닉스 매출액 정보가 필요하므로 재무데이터검색을 사용해야 한다.
Action: 재무데이터검색
Action Input: 2023년 SK하이닉스 매출액
Observation: SK하이닉스의 2023년 매출액은(는) 32조 7657억원이며, 전년도는 44조 6215억원입니다.
Thought: 이제 최종 답변을 알겠다.
Final Answer: 2023년 SK하이닉스 매출액은 32조 7657억원입니다.

Question: {question}
Thought:"""
    for i in range(max_iterations):
        response = llm.invoke(prompt, stop=["\nObservation:"])
        if verbose: print(f"--- 반복 {i+1} ---\n{response}")
        if "Final Answer:" in response:
            return response.split("Final Answer:")[-1].strip()
        action_match = re.search(r"Action:\s*(.+)", response)
        input_match = re.search(r"Action Input:\s*(.+)", response)
        if action_match and input_match:
            action = action_match.group(1).strip()
            action_input = input_match.group(1).strip()
            observation = tool_map[action](action_input) if action in tool_map else f"'{action}'은 존재하지 않는 도구입니다."
            prompt += response + f"\nObservation: {observation}\nThought:"
        else:
            return response.strip()
    return "최대 반복 횟수 도달"

questions = [
    ("2024년 삼성전자 매출액이 얼마야?", "재무수치"),
    ("2023년 SK하이닉스 영업이익은 얼마야?", "재무수치"),
    ("2024년 현대자동차 자산총계는?", "재무수치"),
    ("2024년 LG에너지솔루션 영업이익은?", "재무수치"),
    ("2022년 NAVER 매출액이 얼마야?", "재무수치"),
    ("삼성전자 최근 반도체 업황 관련 뉴스 알려줘", "뉴스/동향"),
    ("SK하이닉스 최근 실적 전망은 어때?", "뉴스/동향"),
    ("현대자동차 전기차 시장 관련 최근 동향은?", "뉴스/동향"),
    ("LG에너지솔루션 최근 배터리 시장 이슈는?", "뉴스/동향"),
    ("NAVER 최근 AI 사업 관련 소식은?", "뉴스/동향"),
]

os.makedirs("results", exist_ok=True)
filename = f"results/final_comparison_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"

with open(filename, "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["question", "type", "llm_only", "rag", "agent"])
    for q, qtype in questions:
        print(f"처리 중 [{qtype}]: {q}")
        llm_ans = ask_llm_only(q)
        rag_ans = ask_rag(q)
        agent_ans = run_agent(q)
        writer.writerow([q, qtype, llm_ans, rag_ans, agent_ans])
        print("완료\n")

print(f"저장 완료: {filename}")
