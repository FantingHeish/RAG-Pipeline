# graders.py
# - DocScoreItem / DocumentScoreBatch： LLM-as-Judge 多維度評分（batch）
#   → 現在只有 offline_evaluate.py 會用到，線上問答不再每題呼叫這個
# - RAG Responder：有文件時用文件內容生成答案
# - Plain LLM：直接用 LLM 自身知識回答
# - Answer Quality Grader：生成後的唯一一次線上品質檢查
#   （原本是 hallucination_grader + answer_grader 兩次呼叫，合併成一次，降低線上延遲/成本）

from typing import List

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai.chat_models import ChatOpenAI
from pydantic import BaseModel, Field

# OPTIMIZE/ 目前所有 grader/generation chain 都寫死 gpt-3.5-turbo，
#          之後可以拉出一個 config.py 的 GRADER_MODEL / GENERATION_MODEL 變數，
#          方便換模型或針對不同任務用不同等級的模型（例如評分用便宜模型，生成用較強模型）

# ============================================================
# Retrieval Grader 評分標準（batch 版本，現在只有 offline_evaluate.py 會呼叫）
# ============================================================
# factual_relevance x 0.5 (事實相關性): 文件是否包含與問題直接相關的具體事實？
# information_sufficiency x 0.3 (資訊充分度): 文件的資訊量是否足以回答問題？
# specificity x 0.2 (具體程度): 文件的內容是否具體針對問題情境？
# 得分越高越好，加權分數低於 config.OFFLINE_RELEVANCE_THRESHOLD -> 標記為不相關（訓練資料的負例標籤）

GRADER_CRITERIA_TEXT = """評分標準（每項 1-5 分）：
- factual_relevance（事實相關性，權重 0.5）：文件是否包含與問題直接相關的具體事實？1=完全無關，5=高度相關
- information_sufficiency（資訊充分度，權重 0.3）：文件的資訊量是否足以回答問題？1=嚴重不足，5=完全充分
- specificity（具體程度，權重 0.2）：文件的內容是否具體針對問題情境，而非過於泛泛？1=非常籠統，5=非常具體"""

WEIGHTS = {
    "factual_relevance": 0.5,
    "information_sufficiency": 0.3,
    "specificity": 0.2,
}


def compute_weighted_score(score) -> float:
    """計算加權分數（DocScoreItem 或其他有相同欄位的物件都可以用）"""
    return (
        score.factual_relevance * WEIGHTS["factual_relevance"] +
        score.information_sufficiency * WEIGHTS["information_sufficiency"] +
        score.specificity * WEIGHTS["specificity"]
    )


# ============================================================
# Batch Retrieval Grader（LLM as Judge，離線用）
# 一次 LLM call 評分多份文件；現在只有 offline_evaluate.py 拿它來幫 reranker 訓練資料打標籤，
# 線上問答的檢索過濾已經改用 Layer 2 reranker 自己的 relevance_score（見 pipeline.py）。
# ============================================================

class DocScoreItem(BaseModel):
    doc_id: int = Field(description="對應輸入時的文件編號（從 0 開始）")
    factual_relevance: int = Field(ge=1, le=5, description="事實相關性 1-5")
    information_sufficiency: int = Field(ge=1, le=5, description="資訊充分度 1-5")
    specificity: int = Field(ge=1, le=5, description="具體程度 1-5")
    reasoning: str = Field(description="簡短說明評分理由")


class DocumentScoreBatch(BaseModel):
    scores: List[DocScoreItem] = Field(
        description="每一份輸入文件對應一筆評分，數量必須等於輸入的文件數"
    )


def build_batch_retrieval_grader():
    grader_prompt = ChatPromptTemplate.from_messages([
        ("system", f"""你是一個文件品質評審，負責評估多份檢索到的文件對使用者問題的相關程度。

{GRADER_CRITERIA_TEXT}

你會收到多份帶編號的文件（格式：[文件 N] 內容...）。
請對「每一份」文件分別評分，doc_id 必須對應輸入時的編號，
輸出的 scores 陣列長度必須等於輸入的文件數量，不可以省略任何一份。"""),
        ("human", "使用者問題：{question}\n\n文件列表：\n{documents_block}")
    ])
    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
    return grader_prompt | llm.with_structured_output(DocumentScoreBatch)


# ============================================================
# 其他 LLM Chains（RAG Responder / Plain LLM / 線上答案品質檢查）
# ============================================================

def build_rag_chain():
    rag_prompt = ChatPromptTemplate.from_messages([
        ("system", (
            "你是一位負責處理使用者問題的助手，請利用提取出來的文件內容來回應問題。\n"
            "若問題的答案無法從文件內取得，請直接回覆你不知道，禁止虛構答案。"
        )),
        ("system", "文件: \n\n {documents}"),
        ("human", "問題: {question}"),
    ])
    return rag_prompt | ChatOpenAI(model="gpt-3.5-turbo", temperature=0) | StrOutputParser()


def build_llm_chain():
    """Plain LLM：(1) 備用方案 (2) MAX_RETRIES 用盡後的 give_up 路徑"""
    plain_prompt = ChatPromptTemplate.from_messages([
        ("system", (
            "你是一位負責處理使用者問題的助手，請利用你的知識來回應問題，勿虛構答案。\n"
            "如果你不確定答案，請誠實說明，不要編造。"
        )),
        ("human", "問題: {question}"),
    ])
    return plain_prompt | ChatOpenAI(model="gpt-3.5-turbo", temperature=0) | StrOutputParser()


class AnswerQuality(BaseModel):
    is_grounded: bool = Field(description="答案內容是否有依據提供的文件，沒有虛構額外事實")
    addresses_question: bool = Field(description="答案是否有實際回應使用者的問題")
    reasoning: str = Field(description="簡短說明理由")


def build_answer_quality_grader():
    """
    生成後的唯一一次線上品質檢查：合併「有沒有幻覺」跟「有沒有回答到問題」成一次 LLM 呼叫，
    取代原本 hallucination_grader + answer_grader 兩次呼叫，降低線上延遲與 API 成本。
    """
    prompt = ChatPromptTemplate.from_messages([
        ("system", (
            "你是一個回答品質評審。請根據提供的文件內容，檢查 LLM 的回答：\n"
            "1) is_grounded：回答的內容是否都能在文件裡找到依據，沒有虛構文件沒提到的事實\n"
            "2) addresses_question：回答是否有實際回應使用者的問題（不是答非所問或迴避）\n"
            "兩者都要基於文件內容跟問題本身客觀判斷。"
        )),
        ("human", "文件: \n\n{documents}\n\n使用者問題: {question}\n\nLLM 回答: {generation}"),
    ])
    return prompt | ChatOpenAI(model="gpt-3.5-turbo", temperature=0).with_structured_output(AnswerQuality)
