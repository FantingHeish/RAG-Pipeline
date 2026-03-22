# graders.py
# 負責：所有 grader 定義
#   - DocumentScore：Layer 3 LLM-as-Judge 多維度評分
#   - RAG Responder：有文件時用文件內容生成答案
#   - Plain LLM：直接用 LLM 自身知識回答
#   - Hallucination Grader：檢查生成答案是否有幻覺
#   - Answer Grader：檢查生成答案是否回應了問題

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai.chat_models import ChatOpenAI
from pydantic import BaseModel, Field

from config import RELEVANCE_THRESHOLD

# ============================================================
# Retrieval Grader — 多維度加權評分 (1-5)
# LLM-as-Judge → Retrieval 的 Layer 3（針對 doc）
# ============================================================
'''
# 從以下 3 個維度各給 1-5 分，加權成 relevance score：
#   factual_relevance       × 0.5（事實直接相關性）
#   information_sufficiency × 0.3（資訊量是否夠）
#   specificity             × 0.2（是否具體針對問題）
# RELEVANCE_THRESHOLD：加權分數低於此值 -> 過濾
# 進到Layer3(Retrieval Grader)的文件已經經過：
#   Layer 1：向量相似度初篩（k=LAYER1_K）
#   Layer 2：Cross-Encoder reranker 重排序（top_n=LAYER2_TOP_N）
# 這裡做最終品質把關：三維度加權評分，低於 RELEVANCE_THRESHOLD -> 過濾掉
'''

# --- Retrieval Grader ---
class DocumentScore(BaseModel):
    """Layer 3：LLM-as-Judge 多維度評分"""
    factual_relevance: int = Field(
        description="文件是否包含與問題直接相關的事實？1=完全無關, 5=高度相關",
        ge=1, le=5
    )
    information_sufficiency: int = Field(
        description="文件提供的資訊是否足以回答問題？1=嚴重不足, 5=完全充分",
        ge=1, le=5
    )
    specificity: int = Field(
        description="文件的內容是否具體針對問題，而非過於泛泛？1=非常籠統, 5=非常具體",
        ge=1, le=5
    )
    reasoning: str = Field(description="簡短說明評分理由")

# 計算加權
WEIGHTS = {
    "factual_relevance":       0.5,
    "information_sufficiency": 0.3,
    "specificity":             0.2,
}

def compute_weighted_score(score: DocumentScore) -> float:
    return (
        score.factual_relevance       * WEIGHTS["factual_relevance"] +
        score.information_sufficiency * WEIGHTS["information_sufficiency"] +
        score.specificity             * WEIGHTS["specificity"]
    )

def build_retrieval_grader():
    grader_prompt = ChatPromptTemplate.from_messages([
        ("system", """你是一個文件品質評審，負責評估檢索到的文件對使用者問題的相關程度。

請從三個維度各給 1-5 分：

1. factual_relevance（事實相關性）：文件是否包含與問題直接相關的具體事實？
   - 1 = 文件完全沒提到問題相關的內容
   - 3 = 文件有提到相關主題，但不是直接答案
   - 5 = 文件直接包含回答問題所需的事實

2. information_sufficiency（資訊充分度）：文件的資訊量是否足以回答問題？
   - 1 = 資訊嚴重不足，幾乎無法作答
   - 3 = 有部分資訊，但需要補充
   - 5 = 資訊完整，單靠這份文件就能回答

3. specificity（具體程度）：文件的內容是否具體針對問題情境？
   - 1 = 內容非常籠統，適用於任何情況
   - 3 = 有一定具體性
   - 5 = 內容非常具體，針對性強

請輸出 JSON 格式的評分結果，包含 factual_relevance、information_sufficiency、specificity、reasoning。"""),
        ("human", "文件內容：\n\n{document}\n\n使用者問題：{question}")
    ])
    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
    return grader_prompt | llm.with_structured_output(DocumentScore)


# ============================================================
# 其他 LLMs Graders（RAG Responder / Plain LLM / Hallucination / Answer）
# ============================================================

# --- RAG Responder ---
def build_rag_chain():
    rag_prompt = ChatPromptTemplate.from_messages([
        ("system", (
            "你是一位負責處理使用者問題的助手，請利用提取出來的文件內容來回應問題。\n"
            "若問題的答案無法從文件內取得，請直接回覆你不知道，禁止虛構答案。"
        )),
        ("system", "文件: \n\n {documents}"),
        ("human",  "問題: {question}"),
    ])
    return rag_prompt | ChatOpenAI(model="gpt-3.5-turbo", temperature=0) | StrOutputParser()


# --- Plain LLM（直接用 LLM 自身知識回答）---
def build_llm_chain():
    plain_prompt = ChatPromptTemplate.from_messages([
        ("system", "你是一位負責處理使用者問題的助手，請利用你的知識來回應問題，勿虛構答案。"),
        ("human",  "問題: {question}"),
    ])
    return plain_prompt | ChatOpenAI(model="gpt-3.5-turbo", temperature=0) | StrOutputParser()


# --- Hallucination Grader ---
class GradeHallucinations(BaseModel):
    binary_score: str = Field(description="答案是否為虛構。('yes' or 'no')")


def build_hallucination_grader():
    hallucination_prompt = ChatPromptTemplate.from_messages([
        ("system", (
            "你是一個評分的人員，負責確認LLM的回應是否為虛構的。\n"
            "'Yes' 代表LLM的回答是虛構的；'No' 代表回答基於文件內容。"
        )),
        ("human", "文件: \n\n {documents} \n\n LLM 回應: {generation}"),
    ])
    return (
        hallucination_prompt
        | ChatOpenAI(model="gpt-3.5-turbo", temperature=0).with_structured_output(GradeHallucinations)
    )


# --- Answer Grader ---
class GradeAnswer(BaseModel):
    binary_score: str = Field(description="答案是否回應問題。('yes' or 'no')")


def build_answer_grader():
    answer_prompt = ChatPromptTemplate.from_messages([
        ("system", "你是一個評分的人員，負責確認答案是否回應了問題。輸出 'yes' or 'no'。"),
        ("human", "使用者問題: \n\n {question} \n\n 答案: {generation}"),
    ])
    return (
        answer_prompt
        | ChatOpenAI(model="gpt-3.5-turbo", temperature=0).with_structured_output(GradeAnswer)
    )
