# router.py
# 負責：Router structured JSON output + confidence score + heuristic fallback
# 這支檔案邏輯沒有變動（原本設計沒問題），只搬過來保持專案完整可跑。
#
# 目前沒有任何地方 import 這支檔案（見 README「拿掉 Router」）。
# 注意：如果之後要拿回來用，CONFIDENCE_THRESHOLD 和 INDEX_DESCRIPTIONS 這兩個
# 設定值在目前的 config.py 裡已經不存在了，需要先補回去，否則 import 這支檔案會直接 NameError/AttributeError。
# OPTIMIZE/ 拿回來用之前，順便把 model 從 gpt-3.5-turbo 評估要不要換成更新的模型

from typing import List

from langchain_core.prompts import ChatPromptTemplate
from langchain_openai.chat_models import ChatOpenAI
from pydantic import BaseModel, Field

from config import CONFIDENCE_THRESHOLD, INDEX_DESCRIPTIONS

# Router 輸出
class RouteDecision(BaseModel):
    sources: List[str] = Field(
        description=(
            "選擇的資料來源，可以是多個。"
            "可選值：'technical'、'business'、'legal'、'healthcare'、'web_search'。"
            "例如 ['technical', 'business']"
        )
    )
    confidence: float = Field(
        description="對這個路由決策的信心分數，0.0 到 1.0 之間"
    )
    reasoning: str = Field(
        description="為什麼選這些 source，一句話說明"
    )


def build_router_prompt() -> ChatPromptTemplate:
    """動態注入 INDEX_DESCRIPTIONS，新增 source 只需改 config.py"""
    source_list = "\n".join(f'- "{k}": {v}' for k, v in INDEX_DESCRIPTIONS.items())
    return ChatPromptTemplate.from_messages([
        ("system", f"""你是一個資料來源路由專家，負責判斷每個問題應該去哪裡找答案。

你有以下幾個資料來源：
{source_list}
- "web_search": 即時網路搜尋。用於上述來源都找不到答案、需要最新資訊、或問題是一般知識的情況。

根據使用者的問題，選擇最合適的一個或多個資料來源：
- 如果問題橫跨多個領域，可以選多個 source。
- 如果不確定，優先選 web_search。
- 如果是閒聊或非常簡單的一般知識，選 web_search。

confidence 反映你對這個分類的確定程度：
- 0.9+ = 非常確定
- 0.7-0.9 = 有把握
- 0.5-0.7 = 不確定
- <0.5 = 很不確定（系統會自動 fallback 到 web_search）

請輸出 JSON，包含 sources（list）、confidence（0.0-1.0）、reasoning（一句話）。
"""),
        ("human", "{question}")
    ])


def build_question_router():
    """建立並回傳 question_router chain"""
    llm_router = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
    return build_router_prompt() | llm_router.with_structured_output(RouteDecision)
