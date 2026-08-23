# gold_standard.py
#
# 對應 smart_healthcare_docs/ 資料夾內容的測試集
# ground_truth 用於 RAGAS 的 context_recall 計算
# 如果你之後加入自己找的 PDF 文獻，記得也對應補幾題進來，這樣評估才涵蓋你實際的資料。#TODO
#
# 這 19 題的 ground_truth 都已經對照過 project 裡實際的文件內容（用 project_knowledge_search
# 逐篇確認過），不是憑標題推測的，可以放心用在 RAGAS 的 context_recall 評估上。

GOLD_STANDARD = [
    {
        "question":        "台灣健康台灣深耕計畫的預算跟期程是多少？",
        "answer_keywords": ["489", "2025", "2029", "健康台灣"],
        "ground_truth":     "健康台灣深耕計畫期程為114-118年（2025-2029），總經費約新台幣489億元，"
                             "分成優化醫療工作條件、規劃多元人才培訓、導入智慧科技醫療、"
                             "社會責任醫療永續四大範疇，共18項具體目標。",
    },
    {
        "question":        "AI在放射科的臨床採用率跟效率提升的具體數據有哪些？",
        "answer_keywords": ["54%", "放射", "週轉", "53%"],
        "ground_truth":     "2025年美國百床以上醫院約54%已在放射科使用AI，主要用於影像判讀（82%）"
                             "與工作清單排序（48%）。AI輔助分流已將平均報告週轉時間從11.2天縮短到2.7天，"
                             "並可將放射科醫師工作量減輕最多53%。",
    },
    {
        "question":        "FDA對AI醫材的核准數量趨勢跟PCCP機制是什麼？",
        "answer_keywords": ["PCCP", "FDA", "331", "變更控制"],
        "ground_truth":     "FDA在2025年單年核准331款AI醫材為歷史新高，累計核准超過1250款，76%集中在放射科。"
                             "PCCP（Predetermined Change Control Plan，預先設定變更控制計畫）允許廠商"
                             "在上市申請中說明模型未來可能的更新範圍與驗證方式，經核准後可在一定範圍內"
                             "迭代模型而不需每次重新申請，2025年約10%的AI醫材核准案採用此機制。",
    },
    {
        "question":        "台灣的通訊診療辦法適用於哪些病人？除了偏鄉地區之外？",
        "answer_keywords": ["通訊診療", "出院", "長照", "急性住院"],
        "ground_truth":     "除了山地離島偏僻地區，通訊診察治療辦法也適用於：急性住院病人於出院後三個月內"
                             "依出院準備服務計畫的追蹤治療；以及長照機構住民持有效期內慢性病連續處方箋、"
                             "因病情需要原醫師診療的情形。執行時需事先擬具實施計畫並經主管機關核准。",
    },
    {
        "question":        "WHO對健康人工智慧的倫理治理指引主要在講什麼？",
        "answer_keywords": ["WHO", "倫理", "2021", "2024"],
        "ground_truth":     "WHO於2021年發布首份國際健康AI倫理共識文件，2024年針對生成式AI與大型多模態模型"
                             "發布更新指引，提出超過40項建議，核心原則包括人權保護、透明可解釋性、"
                             "全生命週期治理與公平可及性，2025年進一步成立GI-AI4H推動全球治理協調。",
    },
    {
        "question":        "環境式AI病歷紀錄（Ambient AI Scribe）目前有哪些爭議？",
        "answer_keywords": ["遺漏率", "生產力悖論", "過勞", "準確"],
        "ground_truth":     "Ambient AI Scribe目前仍有偏高的遺漏率與間歇性事實錯誤問題；"
                             "部分醫師擔心節省的時間會被用來增加看診量而非減少負擔（生產力悖論）；"
                             "也有醫師擔心AI生成的逐字病歷若被脫離脈絡引用，可能在訴訟中造成不利影響。",
    },
    {
        # 對應 A Drug-Centric Dual-Layer Knowledge Graph Framework for Drug Combination Prediction.pdf
        "question":        "藥物組合預測的雙層知識圖譜框架（DualKG-DC）是怎麼設計的？",
        "answer_keywords": ["雙層", "知識圖譜", "BiologicalKG", "適應症", "藥物組合"],
        "ground_truth":     "DualKG-DC 使用雙層知識圖譜結構：第一層 BiologicalKG 是基礎生物知識圖譜，"
                             "涵蓋基因-疾病關聯、藥物-標靶交互作用、蛋白質-蛋白質網絡等，共 45,709 個實體、"
                             "935,694 個跨 8 種關係類型的連結；第二層是針對特定藥物組合建立的任務專屬子圖，"
                             "以代表藥物配對的虛擬組合節點為中心，向外擴展兩跳（two-hop）取得相關生物脈絡。"
                             "模型用訊息傳遞（message passing）與注意力機制學習如何聚合生物證據，"
                             "目的是預測藥物組合可能適用的新疾病適應症，並能對未曾出現過的藥物配對做推論。",
    },
    {
        # 對應 Calibration of Self-Reported Confidence and Accuracy of Large Language Models in Medical Question Answering.pdf
        "question":        "LLM在醫療問答中的自陳信心程度跟實際準確率之間的校準情況如何？",
        "answer_keywords": ["MedMCQA", "校準", "ECE", "過度自信", "專科"],
        "ground_truth":     "這篇研究用 MedMCQA 醫學選擇題基準（20 個專科，每科抽樣 100 題，"
                             "6 個 LLM 共 12,000 筆回應），比較 GPT-5、GPT-5-mini、GPT-5-nano、GPT-4o、"
                             "Gemini 2.5、Claude Sonnet 4.5 的自陳信心與實際答對率之間的校準誤差（ECE）。"
                             "結果顯示不同模型間的 ECE 差異最高達 0.064，不同專科間差異最高達 0.10"
                             "（代表信心與實際正確率有 10% 的系統性落差，換算下來至少每 10 個回答就有"
                             "1 個是「自信但答錯」）。整體而言較大的模型「自我認知」較好，"
                             "但高準確率不代表沒有過度自信的問題，且校準品質因專科而異，"
                             "只看整體平均分數可能掩蓋特定專科的安全風險。",
    },
    {
        # 對應 A Deep Learning Approach to Estimate Corrected QT Intervals from Multi-Lead Conventional ECG Waveforms in Pediatric Patients.pdf
        "question":        "深度學習如何從兒童心電圖預測校正後的QT間期（pedQTNet）？",
        "answer_keywords": ["pedQTNet", "QTc", "LQTS", "敏感度", "兒童"],
        "ground_truth":     "研究團隊開發了 pedQTNet 深度神經網路模型，用 37,992 名 0-18 歲病患、"
                             "65,370 筆由兒科心臟電生理專科醫師標註的心電圖（2010-2020年，"
                             "費城兒童醫院）訓練，從標準心電圖波形估計校正QT間期（QTc）並偵測"
                             "長QT症候群（LQTS）。交叉驗證結果：平均絕對誤差18.8毫秒，"
                             "在470毫秒閾值下偵測LQTS的敏感度85%、特異度87%，效能優於商用的"
                             "GE Marquette 12SL演算法；在前瞻性測試集（200筆心電圖）中，"
                             "pedQTNet的敏感度（100%）甚至高於人類專科醫師（71%）。",
    },
    {
        # 對應 A Python Framework for Visualizing and Labelling High-Resolution Physiological Data for Critical Care Machine Learning.pdf
        "question":        "Vitabel這個 Python 框架如何協助標註重症照護的高解析度生理數據？",
        "answer_keywords": ["Vitabel", "Python", "標註", "時間序列", "重症照護"],
        "ground_truth":     "Vitabel 是一個開源 Python 框架，用來對重症照護產生的醫療時間序列資料"
                             "進行事後（post hoc）載入、視覺化、對齊與標註。設計目標是處理重症照護"
                             "常見的資料雜訊多、高解析度資料量龐大的問題，提供合理的預設值與"
                             "互動元件方便標準化工作流程使用，同時保留彈性讓使用者能客製化分析與"
                             "標註流程。可以無縫整合進 Jupyter Notebook，以 MIT 授權開源釋出，"
                             "論文中展示了三個實際應用案例。",
    },
    {
        # 對應 AI-enabled clinical decision support in breast cancer care.pdf
        "question":        "AI臨床決策支援系統在乳癌照護上的表現如何？跟通用型系統比較結果是什麼？",
        "answer_keywords": ["乳癌", "臨床決策支援", "Prof. Valmed", "篩檢", "17.6%"],
        "ground_truth":     "這是一篇盲性多中心對照研究，比較醫療專用的 RAG 臨床決策支援系統"
                             "（Prof. Valmed，歐盟第一個取得 Class IIb 醫療器材認證的 LLM/RAG系統）"
                             "跟通用型 LLM 系統在乳癌照護上的表現。背景數據：FDA核准的AI醫材數過去"
                             "五年成長約四倍，2025年底累計達1,357款，約四分之三集中在放射科；"
                             "德國一項涵蓋46萬筆乳房攝影的真實世界研究顯示，AI輔助篩檢的癌症偵測率"
                             "比標準專家篩檢高17.6%。但這篇研究的結論是：法規認證與領域專用化"
                             "本身並不保證表現更好，醫療專用系統目前較適合當作專家監督下的輔助工具，"
                             "仍需要更多真實世界驗證。",
    },
    {
        # 對應 Development and Validation of an Automated Acute Kidney Injury E-Alert System Integrated with Clinical Decision Support for Hospitalized Patients.pdf
        "question":        "急性腎損傷（AKI）自動電子預警系統的效能跟驗證結果如何？",
        "answer_keywords": ["AKI", "KDIGO", "敏感度", "長庚", "預警"],
        "ground_truth":     "這是台灣高雄長庚紀念醫院（2,768床）開發的AKI電子預警系統，鎖定"
                             "KDIGO第2-3期（中重度）急性腎損傷，整合用藥導向的臨床決策支援，"
                             "以近即時方式每天處理4次，並設有透析中病人、即將轉出/出院病人的"
                             "抑制規則以減少警示疲勞。用2018年3月到2023年5月的病歷資料驗證，"
                             "系統共產生3,946筆第2-3期AKI警示，對照回溯性參考演算法達到90.94%"
                             "敏感度與99.65%準確率；警示率有季節性（冬季4.48%最高、夏季3.17%最低）。"
                             "醫師問卷調查（n=78）顯示63.0%認為附帶的用藥建議在臨床上有幫助。",
    },
    {
        # 對應 02_ai_healthcare_market_growth.md
        "question":        "全球AI醫療市場的成長趨勢跟規模預測是什麼？",
        "answer_keywords": ["市場規模", "16.7億", "262.3億", "35%"],
        "ground_truth":     "全球醫學影像AI市場規模預估將從2025年的16.7億美元成長到2034年的262.3億美元，"
                             "年複合成長率超過35%，成長動能主要來自FDA核准數量快速增加（2025年單年"
                             "核准331款AI醫材）與臨床採用率提升（美國百床以上醫院54%已在放射科使用AI）。"
                             "2025年也出現多起大型醫療體系的指標性AI投資案例（如Kaiser Permanente、"
                             "Advocate Health、Mayo Clinic），顯示AI投資已從單點試驗轉向機構層級的"
                             "策略性投資；但環境式AI病歷紀錄這類生成式AI應用的採用率成長曲線可能開始趨緩。",
    },
    {
        # 對應 03_remote_patient_monitoring_wearables.md
        "question":        "遠距病患監測與穿戴式裝置在智慧醫療中的應用現況如何？",
        "answer_keywords": ["遠距病患監測", "5G", "穿戴式裝置", "心血管", "COVID-19"],
        "ground_truth":     "COVID-19疫情是遠距病患監測（RPM）與穿戴式裝置快速普及的關鍵轉折點，"
                             "疫情後這兩項技術持續發展成醫療體系的常態化基礎建設。目前的瓶頸是網路延遲，"
                             "業界預期未來5G搭配邊緣運算能把感測器到電子病歷系統的傳輸時間壓縮到"
                             "100毫秒以下。心血管疾病是AI穿戴式裝置監測研究最密集的領域，一項心臟"
                             "復健相關的臨床決策支援系統研究顯示，搭配App與穿戴裝置的介入措施"
                             "潛在效益包括降低死亡率20-47%、降低再住院率18%。產業觀察也指出，"
                             "RPM資料要直接嵌入既有EHR工作流程才能提高臨床採用率，單純做成獨立"
                             "監控儀表板通常成效不佳。",
    },
    {
        "question":        "目前 FDA 核准的醫療 AI 產品中，影像類佔了幾成？",
        "answer_keywords": ["76%", "75%", "影像", "放射", "FDA"],
        "ground_truth":     "根據 2025 至 2026 年初的最新數據，在 FDA 核准的所有醫療 AI 產品中，"
                             "醫學影像（Radiology/Imaging）類別依然佔據最大宗，比例高達約 76%。",
    },
    {
        "question":        "台灣健康台灣深耕計畫的預算跟期程是多少？",
        "answer_keywords": ["489億", "2025", "2029", "114", "118"],
        "ground_truth":     "健康台灣深耕計畫期程為 114 至 118 年（2025-2029 年），總經費約新台幣 489 億元，"
                             "旨在導入智慧科技醫療與優化醫療條件。",
    },
    {
        "question":        "Neuralink 的 N1 晶片含有多少個微型電極？",
        "answer_keywords": ["1024", "64", "電極", "細絲", "Threads"],
        "ground_truth":     "Neuralink 的 N1 晶片（Telemetry BCI）共含有 1024 個微型電極，"
                             "分佈在 64 條比頭髮還細的彈性細絲（Threads）上，用以捕捉神經元訊號。",
    },
    {
        "question":        "環境臨床語音整合（Ambient AI Scribe）是什麼技術？",
        "answer_keywords": ["語音識別", "自然語言處理", "生成式", "自動", "病歷"],
        "ground_truth":     "環境臨床語音整合技術結合了語音識別、自然語言處理（NLP）與生成式 AI，"
                             "能在診間自動監聽醫病對話，並將其轉化為結構化的醫療病歷紀錄（如 SOAP Notes）。",
    },
    {
        "question":        "什麼是 Neuralink 的 Blindsight（盲視）專案？",
        "answer_keywords": ["視覺皮質", "微電極陣列", "失明", "視覺訊號"],
        "ground_truth":     "Blindsight（盲視）是 Neuralink 開發的神經假體專案，透過將微電極陣列植入大腦視覺皮質，"
                             "直接輸入數位視覺訊號，旨在幫助天生失明或視神經受損的患者恢復部分視覺。",
    },
    {
        "question":        "合成醫療數據（Synthetic Medical Data）是什麼？",
        "answer_keywords": ["人工生成", "演算法", "隱私", "統計特性"],
        "ground_truth":     "合成醫療數據是利用統計方法或 AI 演算法（如生成對抗網路）人工生成的虛擬數據。"
                             "它能複製真實世界數據的結構與統計特性，同時完全去識別化以保護患者隱私。",
    },
    {
        "question":        "次世代基因定序（NGS）的核心優勢是什麼？",
        "answer_keywords": ["高通量", "大規模平行定序", "未知突變", "多基因"],
        "ground_truth":     "NGS 的核心優勢在於大規模平行定序的能力（高通量），能同時篩查數十到數百個基因變異，"
                             "並能偵測到未知突變、拷貝數變異（CNV）與複雜的基因融合。",
    },
    {
        "question":        "液態切片（Liquid Biopsy）如何利用 ctDNA 偵測癌症？",
        "answer_keywords": ["ctDNA", "循環腫瘤", "血液", "微量殘留病灶", "MRD"],
        "ground_truth":     "液態切片技術透過抽取患者血液，檢測其中由腫瘤細胞釋放的循環腫瘤 DNA（ctDNA），"
                             "用於癌症早期篩檢、追蹤微量殘留病灶（MRD）或評估標靶藥物抗藥性。",
    },
    {
        "question":        "什麼是多基因風險評分（Polygenic Risk Score, PRS）？",
        "answer_keywords": ["PRS", "微效突變", "SNPs", "加權計算", "遺傳風險"],
        "ground_truth":     "PRS 是一種最新遺傳風險評估技術，它改變以往只看單一高風險基因的限制，"
                             "而是結合 AI 演算法將成百上千個微效突變（SNPs）的風險進行加權計算，評估多因素疾病的綜合機率。",
    },
    {
        "question":        "在標靶藥物泰格莎（Osimertinib）第一線治療失敗後，臨床通常會怎麼做？",
        "answer_keywords": ["NGS", "基因檢測", "抗藥性突變", "C797S", "MET放大"],
        "ground_truth":     "當泰格莎第一線治療失敗後，臨床通常會建議進行二次基因檢測（如 NGS），"
                             "以確認是否出現如 EGFR C797S 點突變或 MET 基因放大等次級抗藥性機制，據此調整後續治療策略。",
    },
]
