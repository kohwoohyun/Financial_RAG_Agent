import re
import csv
import json
import os
import time
from datetime import datetime
from collections import namedtuple
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
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

def extract_year(q):
    m = re.search(r"(\d{4})년", q)
    return m.group(1) if m else None

def format_amount(amount):
    try:
        amount = int(amount)
        jo = amount // 1_000_000_000_000
        eok = (amount % 1_000_000_000_000) // 100_000_000
        return f"{jo}조 {eok}억원"
    except (ValueError, TypeError):
        return str(amount)

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

def growth_calc_tool(query):
    """핵심 수정: 증감률을 LLM이 아니라 코드가 직접 계산해서 돌려줌"""
    company = extract_company(query)
    account = extract_account(query)
    year = extract_year(query)
    if not (company and account and year):
        return "기업명, 항목명, 연도를 정확히 포함해서 다시 질문해주세요."

    curr = get_raw_amount(company, year, account, "thstrm_amount")
    prev = get_raw_amount(company, year, account, "frmtrm_amount")
    if curr is None or prev is None or prev == 0:
        return "해당 데이터를 찾을 수 없습니다."

    pct = (curr - prev) / abs(prev) * 100
    direction = "증가" if pct > 0 else "감소"
    return (f"{company}의 {year}년 {account}은 {format_amount(curr)}이며, "
            f"전년도는 {format_amount(prev)}입니다. "
            f"전년 대비 정확히 {abs(pct):.1f}% {direction}했습니다 (이 숫자를 그대로 답변에 사용하세요).")

ToolDef = namedtuple("ToolDef", ["name", "description", "func"])
tools = [
    ToolDef("재무데이터검색", "5개 기업의 2022~2024년 매출액, 영업이익, 자산총계 등 재무 수치를 단순 조회할 때 사용 (증감률 계산은 절대 직접 하지 말고 증감률계산 도구를 사용)", rag_search_tool),
    ToolDef("증감률계산", "특정 기업의 특정 항목이 전년 대비 몇 퍼센트 증가/감소했는지 물을 때 반드시 사용. 직접 계산하지 말고 이 도구가 계산한 값을 그대로 사용", growth_calc_tool),
]
tool_map = {t.name: t.func for t in tools}

def run_agent_traced(question, max_iterations=6):
    tools_desc = "\n".join([f"{t.name}: {t.description}" for t in tools])
    tool_names = ", ".join([t.name for t in tools])
    prompt = f"""다음 질문에 최선을 다해 답변하세요. 아래 도구들을 사용할 수 있습니다:

{tools_desc}

다음 형식을 정확히 따르세요. 절대 Action 없이 바로 Final Answer를 쓰지 마세요.
증감률, 퍼센트 변화를 묻는 질문은 절대 스스로 계산하지 말고 반드시 증감률계산 도구를 사용하세요.

예시:
Question: NAVER의 2023년 영업이익은 전년 대비 몇 퍼센트 증가했어?
Thought: 증감률을 묻는 질문이므로 증감률계산 도구를 사용해야 한다.
Action: 증감률계산
Action Input: NAVER 2023년 영업이익
Observation: NAVER의 2023년 영업이익은 1조 4888억원이며, 전년도는 1조 3046억원입니다. 전년 대비 정확히 14.1% 증가했습니다 (이 숫자를 그대로 답변에 사용하세요).
Thought: 이제 최종 답변을 알겠다.
Final Answer: NAVER의 2023년 영업이익은 전년 대비 14.1% 증가했습니다.

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

def extract_pct_with_sign(text):
    results = []
    for m in re.finditer(r"(-?\d+\.?\d*)\s*%\s*(증가|상승|감소|하락)?", text):
        num = float(m.group(1))
        if m.group(2) in ("감소", "하락") and num > 0:
            num = -num
        results.append(num)
    return results

def diagnose_calc(company, year, account, final_answer):
    curr = get_raw_amount(company, year, account, "thstrm_amount")
    prev = get_raw_amount(company, year, account, "frmtrm_amount")
    if curr is None or prev is None:
        return "데이터_없음", None
    correct_pct = (curr - prev) / abs(prev) * 100
    candidates = extract_pct_with_sign(final_answer)
    for c in candidates:
        if abs(c - correct_pct) <= 1.0:
            return "성공", correct_pct
    return "추론_실패", correct_pct

questions = [
    {"q": "현대자동차의 2024년 영업이익은 전년 대비 몇 퍼센트 증가했어?", "company": "현대자동차", "year": "2024", "account": "영업이익"},
    {"q": "삼성전자의 2024년 매출액은 전년 대비 몇 퍼센트 증가했어?", "company": "삼성전자", "year": "2024", "account": "매출액"},
    {"q": "NAVER의 2023년 영업이익은 전년 대비 몇 퍼센트 증가했어?", "company": "NAVER", "year": "2023", "account": "영업이익"},
]

REPEATS = 5
os.makedirs("results", exist_ok=True)
filename = f"results/calc_tool_fix_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"

with open(filename, "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["question", "trial", "final_answer", "correct_pct", "diagnosis", "elapsed_sec"])
    for meta in questions:
        for trial in range(1, REPEATS + 1):
            print(f"(시도 {trial}/{REPEATS}) {meta['q']}")
            start = time.time()
            final_answer, action_log = run_agent_traced(meta["q"])
            elapsed = round(time.time() - start, 1)
            diagnosis, correct_pct = diagnose_calc(meta["company"], meta["year"], meta["account"], final_answer)
            writer.writerow([meta["q"], trial, final_answer, correct_pct, diagnosis, elapsed])
            f.flush()
            print(f"  → {diagnosis} (정답:{correct_pct:.1f}%, {elapsed}초)\n")

print(f"\n저장 완료: {filename}")
