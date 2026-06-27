import re
import csv
import json
import os
import time
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

def rag_search_tool(query):
    target_company = extract_company(query)
    target_account = extract_account(query)
    filter_dict = {}
    if target_company: filter_dict["company"] = target_company
    if target_account: filter_dict["account"] = target_account
    docs = vectorstore.similarity_search(query, k=5, filter=filter_dict, fetch_k=200) if filter_dict else vectorstore.similarity_search(query, k=5, fetch_k=200)
    if not docs:
        return "관련된 재무 데이터를 찾을 수 없습니다."
    docs = sorted(docs, key=lambda d: d.metadata.get("year", ""), reverse=True)
    return "\n".join([d.page_content for d in docs])

web_search = DuckDuckGoSearchRun()

ToolDef = namedtuple("ToolDef", ["name", "description", "func"])
tools = [
    ToolDef("재무데이터검색", "5개 기업의 2022~2024년 매출액, 영업이익, 자산총계 등 재무 수치 조회 (한 번에 한 기업만 조회 가능, 두 기업을 비교하려면 두 번 따로 검색)", rag_search_tool),
    ToolDef("웹검색", "최신 뉴스, 시장 동향 등 재무데이터검색에 없는 정보 검색", web_search.run),
]
tool_map = {t.name: t.func for t in tools}

def format_amount(amount_str):
    try:
        amount = int(str(amount_str).replace(",", ""))
        jo = amount // 1_000_000_000_000
        eok = (amount % 1_000_000_000_000) // 100_000_000
        return f"{jo}조 {eok}억원"
    except (ValueError, TypeError):
        return str(amount_str)

def get_raw_amount(company, year, account, field="thstrm_amount"):
    filename = f"data/{company}_{year}.json"
    if not os.path.exists(filename):
        return None
    with open(filename, "r", encoding="utf-8") as f:
        data = json.load(f)
    if data.get("status") != "000":
        return None
    for item in data.get("list", []):
        if item.get("fs_div") == "CFS" and item.get("account_nm") == account:
            try:
                return int(item.get(field, "").replace(",", ""))
            except ValueError:
                return None
    return None

def run_agent_traced(question, max_iterations=6):
    tools_desc = "\n".join([f"{t.name}: {t.description}" for t in tools])
    tool_names = ", ".join([t.name for t in tools])
    prompt = f"""다음 질문에 최선을 다해 답변하세요. 아래 도구들을 사용할 수 있습니다:

{tools_desc}

다음 형식을 정확히 따르세요. 절대 Action 없이 바로 Final Answer를 쓰지 마세요.
비교나 계산이 필요한 질문은 필요한 모든 데이터를 각각 검색한 뒤에 답변하세요.

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

    action_log = []
    for i in range(max_iterations):
        response = llm.invoke(prompt, stop=["\nObservation:"])
        if "Final Answer:" in response:
            return response.split("Final Answer:")[-1].strip(), action_log

        action_match = re.search(r"Action:\s*(.+)", response)
        input_match = re.search(r"Action Input:\s*(.+)", response)
        if action_match and input_match:
            action = action_match.group(1).strip()
            action_input = input_match.group(1).strip()
            action_log.append((action, action_input))
            observation = tool_map[action](action_input) if action in tool_map else f"'{action}'은 존재하지 않는 도구입니다."
            prompt += response + f"\nObservation: {observation}\nThought:"
        else:
            return response.strip(), action_log

    return "최대 반복 횟수 도달", action_log

def diagnose(meta, final_answer, action_log):
    rag_calls = [a for a in action_log if a[0] == "재무데이터검색"]
    if len(rag_calls) == 0:
        return "tool_선택_실패"

    qtype = meta["type"]

    if qtype == "lookup":
        raw = get_raw_amount(meta["companies"][0], meta["year"], meta["account"])
        if raw is None:
            return "데이터_없음"
        return "성공" if format_amount(raw) in final_answer else "추론_실패"

    elif qtype == "compare":
        searched = set(c for _, inp in rag_calls for c in meta["companies"] if c in inp)
        if len(searched) < 2:
            return "검색_불완전"
        v1 = get_raw_amount(meta["companies"][0], meta["year"], meta["account"])
        v2 = get_raw_amount(meta["companies"][1], meta["year"], meta["account"])
        if v1 is None or v2 is None:
            return "데이터_없음"
        winner = meta["companies"][0] if v1 > v2 else meta["companies"][1]
        return "성공" if winner in final_answer else "추론_실패"

    elif qtype == "calc":
        company = meta["companies"][0]
        curr = get_raw_amount(company, meta["year"], meta["account"], "thstrm_amount")
        prev = get_raw_amount(company, meta["year"], meta["account"], "frmtrm_amount")
        if curr is None or prev is None or prev == 0:
            return "데이터_없음"
        correct_pct = (curr - prev) / abs(prev) * 100
        numbers = re.findall(r"-?\d+\.?\d*", final_answer.replace(",", ""))
        for n in numbers:
            try:
                if abs(float(n) - correct_pct) <= 1.0:
                    return "성공"
            except ValueError:
                continue
        return "추론_실패"

    return "알수없음"

questions = [
    {"q": "2024년 삼성전자 매출액이 얼마야?", "tier": "1-hop", "type": "lookup",
     "companies": ["삼성전자"], "year": "2024", "account": "매출액"},
    {"q": "2023년 SK하이닉스 영업이익은 얼마야?", "tier": "1-hop", "type": "lookup",
     "companies": ["SK하이닉스"], "year": "2023", "account": "영업이익"},
    {"q": "2024년 현대자동차 자산총계는?", "tier": "1-hop", "type": "lookup",
     "companies": ["현대자동차"], "year": "2024", "account": "자산총계"},

    {"q": "삼성전자와 SK하이닉스 중 2024년 매출액이 더 큰 기업은?", "tier": "2hop_비교", "type": "compare",
     "companies": ["삼성전자", "SK하이닉스"], "year": "2024", "account": "매출액"},
    {"q": "현대자동차와 NAVER 중 2023년 영업이익이 더 큰 기업은?", "tier": "2hop_비교", "type": "compare",
     "companies": ["현대자동차", "NAVER"], "year": "2023", "account": "영업이익"},
    {"q": "LG에너지솔루션과 SK하이닉스 중 2024년 자산총계가 더 큰 기업은?", "tier": "2hop_비교", "type": "compare",
     "companies": ["LG에너지솔루션", "SK하이닉스"], "year": "2024", "account": "자산총계"},

    {"q": "현대자동차의 2024년 영업이익은 전년 대비 몇 퍼센트 증가했어?", "tier": "2hop_계산", "type": "calc",
     "companies": ["현대자동차"], "year": "2024", "account": "영업이익"},
    {"q": "삼성전자의 2024년 매출액은 전년 대비 몇 퍼센트 증가했어?", "tier": "2hop_계산", "type": "calc",
     "companies": ["삼성전자"], "year": "2024", "account": "매출액"},
    {"q": "NAVER의 2023년 영업이익은 전년 대비 몇 퍼센트 증가했어?", "tier": "2hop_계산", "type": "calc",
     "companies": ["NAVER"], "year": "2023", "account": "영업이익"},
]

REPEATS = 5

os.makedirs("results", exist_ok=True)
raw_filename = f"results/multihop_raw_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"

with open(raw_filename, "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["tier", "question", "trial", "final_answer", "diagnosis", "num_rag_calls", "elapsed_sec"])

    for meta in questions:
        for trial in range(1, REPEATS + 1):
            print(f"[{meta['tier']}] (시도 {trial}/{REPEATS}) {meta['q']}")
            start = time.time()
            final_answer, action_log = run_agent_traced(meta["q"])
            elapsed = round(time.time() - start, 1)
            diagnosis = diagnose(meta, final_answer, action_log)
            rag_count = len([a for a in action_log if a[0] == "재무데이터검색"])
            writer.writerow([meta["tier"], meta["q"], trial, final_answer, diagnosis, rag_count, elapsed])
            f.flush()
            print(f"  → {diagnosis} ({elapsed}초)\n")

print(f"\n저장 완료: {raw_filename}")
