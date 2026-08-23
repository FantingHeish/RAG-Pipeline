---
tags:
- sentence-transformers
- cross-encoder
- reranker
- generated_from_trainer
- dataset_size:309
- loss:BinaryCrossEntropyLoss
base_model: BAAI/bge-reranker-base
pipeline_tag: text-ranking
library_name: sentence-transformers
---

# CrossEncoder based on BAAI/bge-reranker-base

This is a [Cross Encoder](https://www.sbert.net/docs/cross_encoder/usage/usage.html) model finetuned from [BAAI/bge-reranker-base](https://huggingface.co/BAAI/bge-reranker-base) using the [sentence-transformers](https://www.SBERT.net) library. It computes scores for pairs of texts, which can be used for text reranking and semantic search.

## Model Details

### Model Description
- **Model Type:** Cross Encoder
- **Base model:** [BAAI/bge-reranker-base](https://huggingface.co/BAAI/bge-reranker-base) <!-- at revision 2cfc18c9415c912f9d8155881c133215df768a70 -->
- **Maximum Sequence Length:** 512 tokens
- **Number of Output Labels:** 1 label
- **Supported Modality:** Text
<!-- - **Training Dataset:** Unknown -->
<!-- - **Language:** Unknown -->
<!-- - **License:** Unknown -->

### Model Sources

- **Documentation:** [Sentence Transformers Documentation](https://sbert.net)
- **Documentation:** [Cross Encoder Documentation](https://www.sbert.net/docs/cross_encoder/usage/usage.html)
- **Repository:** [Sentence Transformers on GitHub](https://github.com/huggingface/sentence-transformers)
- **Hugging Face:** [Cross Encoders on Hugging Face](https://huggingface.co/models?library=sentence-transformers&other=cross-encoder)

### Full Model Architecture

```
CrossEncoder(
  (0): Transformer({'transformer_task': 'sequence-classification', 'modality_config': {'text': {'method': 'forward', 'method_output_name': 'logits'}}, 'module_output_name': 'scores', 'architecture': 'XLMRobertaForSequenceClassification'})
)
```

## Usage

### Direct Usage (Sentence Transformers)

First install the Sentence Transformers library:

```bash
pip install -U sentence-transformers
```

Then you can load this model and run inference.
```python
from sentence_transformers import CrossEncoder

# Download from the 🤗 Hub
model = CrossEncoder("cross_encoder_model_id")
# Get scores for pairs of inputs
pairs = [
    ['全球AI醫療市場的成長趨勢跟規模預測是什麼？', 'AI重塑智慧醫療版圖\n4\n© 2026 KPMG, a Taiwan partnership and a member ﬁrm of the KPMG global organization of independent member ﬁrms afﬁliated with \nKPMG International Limited, a private English company limited by guarantee. All rights reserved.\n從挑戰到契機，AI與數位科技重塑醫療生態\n全球醫療產業正站在關鍵轉折點，高齡化浪潮加速、慢性病負擔攀\n升、醫療人力短缺與成本壓力疊加，傳統醫療模式已難以支撐未來需\n求。世界衛生組織預估，2050年全球60歲以上人口將倍增至21億，\n醫療體系必須尋求更智慧、更永續的解決方案，才能在資源有限的情\n況下維持照護品質。\n在此背景下，人工智慧與數位科技正成為醫療創新的核心動能。生成\n式AI不僅能簡化行政流程，更進一步進入臨床決策、藥物研發與醫\n學教育，重塑醫療價值鏈；健康資料則成為新興商業模式與跨域合作'],
    ['遠距病患監測與穿戴式裝置在智慧醫療中的應用現況如何？', 'RESEARCH\nJournal of Medical Systems           (2026) 50:88 \nhttps://doi.org/10.1007/s10916-026-02416-y\n \r Zhenxiang Gao\nzxg306@case.edu\nRong Xu\nrxx@case.edu\n1\t Center\tfor\tArtificial\tIntelligence\tin\tDrug\tDiscovery,\tSchool\t\nof\tMedicine,\tCase\tWestern\tReserve\tUniversity,\tCleveland,\t\nOH,\tUSA\n2\t Cleveland\tClinic\tLerner\tCollege\tof\tMedicine\tof\tCase\t\nWestern\tReserve\tUniversity,\tCleveland,\tOH,\tUSA\n3\t Department\tof\tBiology,\tCollege\tof\tArts\tand\tSciences,\tCase\t\nWestern\tReserve\tUniversity,\tCleveland,\tOH,\tUSA\nAbstract'],
    ['醫療數位孿生如何預測心血管疾病？', '術將神經外科手術中的2D影像即時轉為3D成像，提高醫生手術精準\n度與安全性；去年亞洲生技大展開幕演講中亦指出，全球正處於AI\n與生技醫療的關鍵交會點，疾病的診斷、治療與預測都將被重新定\n義，顯示AI SaMD已成為科技與醫療大廠的焦點。\n軟體醫療器材為臨床應用帶來多重效益\n美國食品藥物管理局(U.S. Food and Drug Administration， FDA)\n與國際醫療器材主管機關論壇（International Medical Device \nRegulators Forum, IMDRF）將SaMD定義為「用於一項或多項\n醫療目的，且能在不作為硬體醫療器材一部分的情況下執行這些醫\n療目的的軟體」。其應用於臨床評估與診斷（如醫學影像辨識與分\n析）、治療規劃與執行（如個人化的治療規劃）、病患監控與管理\n（如病患生理數據追蹤），以及疾病預測與預防（如心血管疾病預\nSaMD成為數位醫療核心引\n擎，監管與治理被視為落地\n兩大挑戰'],
    ['FDA對AI醫材的核准數量趨勢跟PCCP機制是什麼？', '這代表台灣的智慧醫材軟體並不是「開發完就能上市」，而是需要在設計階段就同步規劃\n確效（validation）與驗證（verification）流程，並在整個產品生命週期中持續維護相關文件。\n\n## 網路安全與風險管制\n\n隨著智慧醫材聯網化程度提高，網路資安也成為法規關注重點。\n食藥署委託工研院舉辦「智慧醫材網路安全與風險管制工作坊」，\n協助業者建立醫材聯網情境下的資安風險評估與管理能力，\n這與國際上（例如 FDA 的醫材網路資安指引）的監理方向一致：\nAI/聯網醫材的資安風險評估，需要涵蓋資料傳輸、韌體更新、第三方元件依賴等面向，\n而不只是傳統的功能安全（functional safety）驗證。\n\n## 國產智慧醫材推廣機制\n\n為了鼓勵醫療服務提供者採用台灣品牌的智慧科技醫療器材，\n「智慧醫療器材資訊暨媒合平台（HST 平台）」推出「推動鼓勵醫療服務提供者使用國產品」專案，\n結合教育訓練與臨床作業應用，強化醫療機構與國內智慧醫材產業之間的合作交流。\n\n## VR/AR 醫材應用的技術探討'],
    ['單一基因檢測 / 即時聚合酶連鎖反應技術原理？', '您的瀏覽器不支援JavaScript功能，若網頁功能無法正常使用時，請開啟瀏覽器JavaScript狀態\n\n# 農業部\n\n農業部\n農業部\n\n## 96年7月（第181期）\n\n## 即時聚合酶連鎖反應（Real-time PCR）在植物病蟲害檢測上之應用\n\n一、前言\n\n\u3000\u3000自從聚合酶連鎖反應(PCR)技術開發至今，幾乎是生物學門中最被普遍使用的技術。PCR主要是運用DNA的變性、黏合及延伸等三步驟的循環，連續增幅目標DNA片段。在人類的疾病或其他動植物的疫病蟲害檢測上，以PCR為基礎的技術都被廣泛的應用，諸如多重聚合酶連鎖反應(Multiplex PCR)、巢式聚合酶連鎖反應(Nested PCR)、逢機增幅多態型-聚合酶連鎖反應(RAPD-PCR)、序列特徵化增幅區域(SCARs)、聚合酶連鎖反應-限制性片段長度多態性(PCR-RFLP)以及即時聚合酶連鎖反應(Real-time PCR)等。其中Real-time PCR近年被使用的頻率逐漸提高，除了反應時間大量的縮短之外，儀器及技術的亦不斷的創新進步，真正可以做到兼顧到準確度、敏感度及高效率的多項優點。\n\n二、Real-time PCR原理與方法\n\n\u3000\u3000Real-time PCR跟傳統PCR不同之處在於前者可經由光學系統去監測反應中產物量(螢光物質)的變化而反應在電腦上，後者則必須等反應結束後再進行洋菜膠體電泳分析。Real-time PCR的配備主要有PCR機器、光學系統及電腦(如圖1)。隨著技術的改良進步，Real-time PCR在精確度及敏感度都優於傳統PCR。目前Real-time PCR螢光系統可大致分為「非探針型」及「探針型」。 [...] 五、結語\n\n\u3000\u3000Real-time PCR有快速、精準及專一等鑑定的功能，但是否真的能準確檢測出目標物種，除了實驗條件的設定，不斷的測試才能確定該項檢驗技術的穩定性，而這些技術的根本則有賴於害蟲及病原菌基因資料庫的建立與累積，在建立基因資料庫之前，物種(species)、分化型(forma specialis)、病原變種(pathovar)、生理小種(race)必須先加以鑑定釐清，基因資料庫來源的證據標本(voucher specimens)或菌株都必須妥善保存，避免日後發現先前鑑定有問題時無法再驗證。隨著此項技術的使用普遍化，反應所需的耗材與藥劑價格也可能逐步下降，使用的花費也就越來越能被接受，應用在植物的疫病蟲害上的機會就可大幅提高，在可預見的未來，Real-time PCR在植物防檢疫的工作上將扮演更重要的角色。  \n圖1\u3000real-time PCR系統包括電腦、PCR機器及光學偵測系統   \n圖1\u3000real-time PCR系統包括電腦、PCR機器及光學偵測系統  \n圖2\u3000使用real-time PCR(SYBR-green I)來鑑定檢疫害蟲西方花薊馬，藍線為positive control，突破閥值(CT)為18.79，其他非目標物種則不會形成區線及突破閥值，也就是不會產生目標產物。  \n圖2\u3000使用real-time PCR(SYBR-green I)來鑑定檢疫害蟲西方花薊馬，藍線為positive control，突破閥值(CT)為18.79，其他非目標物種則不會形成區線及突破閥值，也就是不會產生目標產物。 [...] (一)「非探針型」的系統就是在反應中加入會與雙股DNA嵌合而釋放出螢光的物質，目前最常被使用的螢光染劑是SYBR-green I，這種物質會嵌入在雙股DNA的小凹槽(minor groove)而釋放出可被偵測的螢光，所以當PCR產物越多時，嵌入的SYBR-green I就越多，釋放出的螢光也就越多。\n\n(二)「探針型」系統相對上就較為複雜，反應中除了要有專一性的引子對之外，另外還要在引子對之間DNA序列中找到具有專一性的片段來作為探針，如果不是目標物種來做偵測，探針就不會雜合到核酸上，之後也就不會釋放出螢光而被偵測到，所以「探針型」系統的專一性也就相對比較高。\n\n三、Real-time PCR的優缺點\n\n\u3000\u3000Real-time PCR的優點就實驗時間可以大幅縮短，也可在電腦上即時監測實驗結果，且鑑定的專一性、敏感度及準確度也較一般PCR方式為高。檢測過程不用進行洋菜膠體電泳分析，除了可以節省時間及洋菜膠的材料費外，也可避免多做這個實驗所造成的污染及操作誤差。此外，不論使用探針或非探針的方法，Real-time PCR皆可精準定量，確定目標DNA的初始濃度，這對於植物防檢疫中的抗病育種及輸入農產品檢測方面，特別具有意義。而傳統PCR僅能用於定性分析，至多能達到「半定量(semi-quantitative)的程度。甚至已有廠商開發出手提式的Real-time PCR機器，可在野外或田間都使用直接進行檢測。而Real-time PCR的缺點則是機器及反應耗材的費用相對較高，造成目前在使用上無法達到普及化的主要原因。\n\n四、Real-time PCR在檢測植物病蟲害之應用'],
]
scores = model.predict(pairs)
print(scores)
# [0.4681 0.0131 0.6831 0.3423 0.9066]

# Or rank different texts based on similarity to a single text
ranks = model.rank(
    '全球AI醫療市場的成長趨勢跟規模預測是什麼？',
    [
        'AI重塑智慧醫療版圖\n4\n© 2026 KPMG, a Taiwan partnership and a member ﬁrm of the KPMG global organization of independent member ﬁrms afﬁliated with \nKPMG International Limited, a private English company limited by guarantee. All rights reserved.\n從挑戰到契機，AI與數位科技重塑醫療生態\n全球醫療產業正站在關鍵轉折點，高齡化浪潮加速、慢性病負擔攀\n升、醫療人力短缺與成本壓力疊加，傳統醫療模式已難以支撐未來需\n求。世界衛生組織預估，2050年全球60歲以上人口將倍增至21億，\n醫療體系必須尋求更智慧、更永續的解決方案，才能在資源有限的情\n況下維持照護品質。\n在此背景下，人工智慧與數位科技正成為醫療創新的核心動能。生成\n式AI不僅能簡化行政流程，更進一步進入臨床決策、藥物研發與醫\n學教育，重塑醫療價值鏈；健康資料則成為新興商業模式與跨域合作',
        'RESEARCH\nJournal of Medical Systems           (2026) 50:88 \nhttps://doi.org/10.1007/s10916-026-02416-y\n \r Zhenxiang Gao\nzxg306@case.edu\nRong Xu\nrxx@case.edu\n1\t Center\tfor\tArtificial\tIntelligence\tin\tDrug\tDiscovery,\tSchool\t\nof\tMedicine,\tCase\tWestern\tReserve\tUniversity,\tCleveland,\t\nOH,\tUSA\n2\t Cleveland\tClinic\tLerner\tCollege\tof\tMedicine\tof\tCase\t\nWestern\tReserve\tUniversity,\tCleveland,\tOH,\tUSA\n3\t Department\tof\tBiology,\tCollege\tof\tArts\tand\tSciences,\tCase\t\nWestern\tReserve\tUniversity,\tCleveland,\tOH,\tUSA\nAbstract',
        '術將神經外科手術中的2D影像即時轉為3D成像，提高醫生手術精準\n度與安全性；去年亞洲生技大展開幕演講中亦指出，全球正處於AI\n與生技醫療的關鍵交會點，疾病的診斷、治療與預測都將被重新定\n義，顯示AI SaMD已成為科技與醫療大廠的焦點。\n軟體醫療器材為臨床應用帶來多重效益\n美國食品藥物管理局(U.S. Food and Drug Administration， FDA)\n與國際醫療器材主管機關論壇（International Medical Device \nRegulators Forum, IMDRF）將SaMD定義為「用於一項或多項\n醫療目的，且能在不作為硬體醫療器材一部分的情況下執行這些醫\n療目的的軟體」。其應用於臨床評估與診斷（如醫學影像辨識與分\n析）、治療規劃與執行（如個人化的治療規劃）、病患監控與管理\n（如病患生理數據追蹤），以及疾病預測與預防（如心血管疾病預\nSaMD成為數位醫療核心引\n擎，監管與治理被視為落地\n兩大挑戰',
        '這代表台灣的智慧醫材軟體並不是「開發完就能上市」，而是需要在設計階段就同步規劃\n確效（validation）與驗證（verification）流程，並在整個產品生命週期中持續維護相關文件。\n\n## 網路安全與風險管制\n\n隨著智慧醫材聯網化程度提高，網路資安也成為法規關注重點。\n食藥署委託工研院舉辦「智慧醫材網路安全與風險管制工作坊」，\n協助業者建立醫材聯網情境下的資安風險評估與管理能力，\n這與國際上（例如 FDA 的醫材網路資安指引）的監理方向一致：\nAI/聯網醫材的資安風險評估，需要涵蓋資料傳輸、韌體更新、第三方元件依賴等面向，\n而不只是傳統的功能安全（functional safety）驗證。\n\n## 國產智慧醫材推廣機制\n\n為了鼓勵醫療服務提供者採用台灣品牌的智慧科技醫療器材，\n「智慧醫療器材資訊暨媒合平台（HST 平台）」推出「推動鼓勵醫療服務提供者使用國產品」專案，\n結合教育訓練與臨床作業應用，強化醫療機構與國內智慧醫材產業之間的合作交流。\n\n## VR/AR 醫材應用的技術探討',
        '您的瀏覽器不支援JavaScript功能，若網頁功能無法正常使用時，請開啟瀏覽器JavaScript狀態\n\n# 農業部\n\n農業部\n農業部\n\n## 96年7月（第181期）\n\n## 即時聚合酶連鎖反應（Real-time PCR）在植物病蟲害檢測上之應用\n\n一、前言\n\n\u3000\u3000自從聚合酶連鎖反應(PCR)技術開發至今，幾乎是生物學門中最被普遍使用的技術。PCR主要是運用DNA的變性、黏合及延伸等三步驟的循環，連續增幅目標DNA片段。在人類的疾病或其他動植物的疫病蟲害檢測上，以PCR為基礎的技術都被廣泛的應用，諸如多重聚合酶連鎖反應(Multiplex PCR)、巢式聚合酶連鎖反應(Nested PCR)、逢機增幅多態型-聚合酶連鎖反應(RAPD-PCR)、序列特徵化增幅區域(SCARs)、聚合酶連鎖反應-限制性片段長度多態性(PCR-RFLP)以及即時聚合酶連鎖反應(Real-time PCR)等。其中Real-time PCR近年被使用的頻率逐漸提高，除了反應時間大量的縮短之外，儀器及技術的亦不斷的創新進步，真正可以做到兼顧到準確度、敏感度及高效率的多項優點。\n\n二、Real-time PCR原理與方法\n\n\u3000\u3000Real-time PCR跟傳統PCR不同之處在於前者可經由光學系統去監測反應中產物量(螢光物質)的變化而反應在電腦上，後者則必須等反應結束後再進行洋菜膠體電泳分析。Real-time PCR的配備主要有PCR機器、光學系統及電腦(如圖1)。隨著技術的改良進步，Real-time PCR在精確度及敏感度都優於傳統PCR。目前Real-time PCR螢光系統可大致分為「非探針型」及「探針型」。 [...] 五、結語\n\n\u3000\u3000Real-time PCR有快速、精準及專一等鑑定的功能，但是否真的能準確檢測出目標物種，除了實驗條件的設定，不斷的測試才能確定該項檢驗技術的穩定性，而這些技術的根本則有賴於害蟲及病原菌基因資料庫的建立與累積，在建立基因資料庫之前，物種(species)、分化型(forma specialis)、病原變種(pathovar)、生理小種(race)必須先加以鑑定釐清，基因資料庫來源的證據標本(voucher specimens)或菌株都必須妥善保存，避免日後發現先前鑑定有問題時無法再驗證。隨著此項技術的使用普遍化，反應所需的耗材與藥劑價格也可能逐步下降，使用的花費也就越來越能被接受，應用在植物的疫病蟲害上的機會就可大幅提高，在可預見的未來，Real-time PCR在植物防檢疫的工作上將扮演更重要的角色。  \n圖1\u3000real-time PCR系統包括電腦、PCR機器及光學偵測系統   \n圖1\u3000real-time PCR系統包括電腦、PCR機器及光學偵測系統  \n圖2\u3000使用real-time PCR(SYBR-green I)來鑑定檢疫害蟲西方花薊馬，藍線為positive control，突破閥值(CT)為18.79，其他非目標物種則不會形成區線及突破閥值，也就是不會產生目標產物。  \n圖2\u3000使用real-time PCR(SYBR-green I)來鑑定檢疫害蟲西方花薊馬，藍線為positive control，突破閥值(CT)為18.79，其他非目標物種則不會形成區線及突破閥值，也就是不會產生目標產物。 [...] (一)「非探針型」的系統就是在反應中加入會與雙股DNA嵌合而釋放出螢光的物質，目前最常被使用的螢光染劑是SYBR-green I，這種物質會嵌入在雙股DNA的小凹槽(minor groove)而釋放出可被偵測的螢光，所以當PCR產物越多時，嵌入的SYBR-green I就越多，釋放出的螢光也就越多。\n\n(二)「探針型」系統相對上就較為複雜，反應中除了要有專一性的引子對之外，另外還要在引子對之間DNA序列中找到具有專一性的片段來作為探針，如果不是目標物種來做偵測，探針就不會雜合到核酸上，之後也就不會釋放出螢光而被偵測到，所以「探針型」系統的專一性也就相對比較高。\n\n三、Real-time PCR的優缺點\n\n\u3000\u3000Real-time PCR的優點就實驗時間可以大幅縮短，也可在電腦上即時監測實驗結果，且鑑定的專一性、敏感度及準確度也較一般PCR方式為高。檢測過程不用進行洋菜膠體電泳分析，除了可以節省時間及洋菜膠的材料費外，也可避免多做這個實驗所造成的污染及操作誤差。此外，不論使用探針或非探針的方法，Real-time PCR皆可精準定量，確定目標DNA的初始濃度，這對於植物防檢疫中的抗病育種及輸入農產品檢測方面，特別具有意義。而傳統PCR僅能用於定性分析，至多能達到「半定量(semi-quantitative)的程度。甚至已有廠商開發出手提式的Real-time PCR機器，可在野外或田間都使用直接進行檢測。而Real-time PCR的缺點則是機器及反應耗材的費用相對較高，造成目前在使用上無法達到普及化的主要原因。\n\n四、Real-time PCR在檢測植物病蟲害之應用',
    ]
)
# [{'corpus_id': ..., 'score': ...}, {'corpus_id': ..., 'score': ...}, ...]
```

<!--
### Direct Usage (Transformers)

<details><summary>Click to see the direct usage in Transformers</summary>

</details>
-->

<!--
### Downstream Usage (Sentence Transformers)

You can finetune this model on your own dataset.

<details><summary>Click to expand</summary>

</details>
-->

<!--
### Out-of-Scope Use

*List how the model may foreseeably be misused and address what users ought not to do with the model.*
-->

<!--
## Bias, Risks and Limitations

*What are the known or foreseeable issues stemming from this model? You could also flag here known failure cases or weaknesses of the model.*
-->

<!--
### Recommendations

*What are recommendations with respect to the foreseeable issues? For example, filtering explicit content.*
-->

## Training Details

### Training Dataset

#### Unnamed Dataset

* Size: 309 training samples
* Columns: <code>sentence_0</code>, <code>sentence_1</code>, and <code>label</code>
* Approximate statistics based on the first 100 samples:
  |          | sentence_0                                                                        | sentence_1                                                                           | label                                                          |
  |:---------|:----------------------------------------------------------------------------------|:-------------------------------------------------------------------------------------|:---------------------------------------------------------------|
  | type     | string                                                                            | string                                                                               | float                                                          |
  | modality | text                                                                              | text                                                                                 |                                                                |
  | details  | <ul><li>min: 7 tokens</li><li>mean: 18.96 tokens</li><li>max: 61 tokens</li></ul> | <ul><li>min: 99 tokens</li><li>mean: 309.19 tokens</li><li>max: 512 tokens</li></ul> | <ul><li>min: 0.0</li><li>mean: 0.64</li><li>max: 1.0</li></ul> |
* Samples:
  | sentence_0                              | sentence_1                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                          | label            |
  |:----------------------------------------|:------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|:-----------------|
  | <code>全球AI醫療市場的成長趨勢跟規模預測是什麼？</code>     | <code>AI重塑智慧醫療版圖<br>4<br>© 2026 KPMG, a Taiwan partnership and a member ﬁrm of the KPMG global organization of independent member ﬁrms afﬁliated with <br>KPMG International Limited, a private English company limited by guarantee. All rights reserved.<br>從挑戰到契機，AI與數位科技重塑醫療生態<br>全球醫療產業正站在關鍵轉折點，高齡化浪潮加速、慢性病負擔攀<br>升、醫療人力短缺與成本壓力疊加，傳統醫療模式已難以支撐未來需<br>求。世界衛生組織預估，2050年全球60歲以上人口將倍增至21億，<br>醫療體系必須尋求更智慧、更永續的解決方案，才能在資源有限的情<br>況下維持照護品質。<br>在此背景下，人工智慧與數位科技正成為醫療創新的核心動能。生成<br>式AI不僅能簡化行政流程，更進一步進入臨床決策、藥物研發與醫<br>學教育，重塑醫療價值鏈；健康資料則成為新興商業模式與跨域合作</code>                              | <code>0.0</code> |
  | <code>遠距病患監測與穿戴式裝置在智慧醫療中的應用現況如何？</code> | <code>RESEARCH<br>Journal of Medical Systems           (2026) 50:88 <br>https://doi.org/10.1007/s10916-026-02416-y<br>    Zhenxiang Gao<br>zxg306@case.edu<br>Rong Xu<br>rxx@case.edu<br>1	 Center	for	Artificial	Intelligence	in	Drug	Discovery,	School	<br>of	Medicine,	Case	Western	Reserve	University,	Cleveland,	<br>OH,	USA<br>2	 Cleveland	Clinic	Lerner	College	of	Medicine	of	Case	<br>Western	Reserve	University,	Cleveland,	OH,	USA<br>3	 Department	of	Biology,	College	of	Arts	and	Sciences,	Case	<br>Western	Reserve	University,	Cleveland,	OH,	USA<br>Abstract</code> | <code>0.0</code> |
  | <code>醫療數位孿生如何預測心血管疾病？</code>           | <code>術將神經外科手術中的2D影像即時轉為3D成像，提高醫生手術精準<br>度與安全性；去年亞洲生技大展開幕演講中亦指出，全球正處於AI<br>與生技醫療的關鍵交會點，疾病的診斷、治療與預測都將被重新定<br>義，顯示AI SaMD已成為科技與醫療大廠的焦點。<br>軟體醫療器材為臨床應用帶來多重效益<br>美國食品藥物管理局(U.S. Food and Drug Administration， FDA)<br>與國際醫療器材主管機關論壇（International Medical Device <br>Regulators Forum, IMDRF）將SaMD定義為「用於一項或多項<br>醫療目的，且能在不作為硬體醫療器材一部分的情況下執行這些醫<br>療目的的軟體」。其應用於臨床評估與診斷（如醫學影像辨識與分<br>析）、治療規劃與執行（如個人化的治療規劃）、病患監控與管理<br>（如病患生理數據追蹤），以及疾病預測與預防（如心血管疾病預<br>SaMD成為數位醫療核心引<br>擎，監管與治理被視為落地<br>兩大挑戰</code>                                                                                      | <code>1.0</code> |
* Loss: [<code>BinaryCrossEntropyLoss</code>](https://sbert.net/docs/package_reference/cross_encoder/losses.html#binarycrossentropyloss) with these parameters:
  ```json
  {
      "activation_fn": "torch.nn.modules.linear.Identity",
      "pos_weight": null
  }
  ```

### Training Hyperparameters
#### Non-Default Hyperparameters

- `per_device_train_batch_size`: 16
- `per_device_eval_batch_size`: 16
- `num_train_epochs`: 2

#### All Hyperparameters
<details><summary>Click to expand</summary>

- `overwrite_output_dir`: False
- `do_predict`: False
- `prediction_loss_only`: True
- `per_device_train_batch_size`: 16
- `per_device_eval_batch_size`: 16
- `per_gpu_train_batch_size`: None
- `per_gpu_eval_batch_size`: None
- `gradient_accumulation_steps`: 1
- `eval_accumulation_steps`: None
- `torch_empty_cache_steps`: None
- `learning_rate`: 5e-05
- `weight_decay`: 0.0
- `adam_beta1`: 0.9
- `adam_beta2`: 0.999
- `adam_epsilon`: 1e-08
- `max_grad_norm`: 1
- `num_train_epochs`: 2
- `max_steps`: -1
- `lr_scheduler_type`: linear
- `lr_scheduler_kwargs`: {}
- `warmup_ratio`: 0.0
- `warmup_steps`: 0
- `log_level`: passive
- `log_level_replica`: warning
- `log_on_each_node`: True
- `logging_nan_inf_filter`: True
- `save_safetensors`: True
- `save_on_each_node`: False
- `save_only_model`: False
- `restore_callback_states_from_checkpoint`: False
- `no_cuda`: False
- `use_cpu`: False
- `use_mps_device`: False
- `seed`: 42
- `data_seed`: None
- `jit_mode_eval`: False
- `use_ipex`: False
- `bf16`: False
- `fp16`: False
- `fp16_opt_level`: O1
- `half_precision_backend`: auto
- `bf16_full_eval`: False
- `fp16_full_eval`: False
- `tf32`: None
- `local_rank`: 0
- `ddp_backend`: None
- `tpu_num_cores`: None
- `tpu_metrics_debug`: False
- `debug`: []
- `dataloader_drop_last`: False
- `dataloader_num_workers`: 0
- `dataloader_prefetch_factor`: None
- `past_index`: -1
- `disable_tqdm`: False
- `remove_unused_columns`: True
- `label_names`: None
- `load_best_model_at_end`: False
- `ignore_data_skip`: False
- `fsdp`: []
- `fsdp_min_num_params`: 0
- `fsdp_config`: {'min_num_params': 0, 'xla': False, 'xla_fsdp_v2': False, 'xla_fsdp_grad_ckpt': False}
- `tp_size`: 0
- `fsdp_transformer_layer_cls_to_wrap`: None
- `accelerator_config`: {'split_batches': False, 'dispatch_batches': None, 'even_batches': True, 'use_seedable_sampler': True, 'non_blocking': False, 'gradient_accumulation_kwargs': None}
- `deepspeed`: None
- `label_smoothing_factor`: 0.0
- `optim`: adamw_torch
- `optim_args`: None
- `adafactor`: False
- `group_by_length`: False
- `length_column_name`: length
- `ddp_find_unused_parameters`: None
- `ddp_bucket_cap_mb`: None
- `ddp_broadcast_buffers`: False
- `dataloader_pin_memory`: True
- `dataloader_persistent_workers`: False
- `skip_memory_metrics`: True
- `use_legacy_prediction_loop`: False
- `push_to_hub`: False
- `resume_from_checkpoint`: None
- `hub_model_id`: None
- `hub_strategy`: every_save
- `hub_private_repo`: None
- `hub_always_push`: False
- `gradient_checkpointing`: False
- `gradient_checkpointing_kwargs`: None
- `include_inputs_for_metrics`: False
- `include_for_metrics`: []
- `eval_do_concat_batches`: True
- `fp16_backend`: auto
- `push_to_hub_model_id`: None
- `push_to_hub_organization`: None
- `mp_parameters`: 
- `auto_find_batch_size`: False
- `full_determinism`: False
- `torchdynamo`: None
- `ray_scope`: last
- `ddp_timeout`: 1800
- `torch_compile`: False
- `torch_compile_backend`: None
- `torch_compile_mode`: None
- `include_tokens_per_second`: False
- `include_num_input_tokens_seen`: False
- `neftune_noise_alpha`: None
- `optim_target_modules`: None
- `batch_eval_metrics`: False
- `eval_on_start`: False
- `use_liger_kernel`: False
- `eval_use_gather_object`: False
- `average_tokens_across_devices`: False
- `prompts`: None
- `batch_sampler`: batch_sampler
- `multi_dataset_batch_sampler`: proportional
- `router_mapping`: {}
- `learning_rate_mapping`: {}

</details>

### Training Time
- **Training**: 35.7 minutes

### Framework Versions
- Python: 3.11.7
- Sentence Transformers: 5.6.0
- Transformers: 4.51.3
- PyTorch: 2.2.2
- Accelerate: 1.14.0
- Datasets: 5.0.0
- Tokenizers: 0.21.1

## Additional Resources

- [Training and Finetuning Reranker Models with Sentence Transformers](https://huggingface.co/blog/train-reranker): the end-to-end guide for training or finetuning Cross Encoder (reranker) models.
- [Multimodal Embedding & Reranker Models with Sentence Transformers](https://huggingface.co/blog/multimodal-sentence-transformers): use text, image, audio, and video reranker models through the same API.
- [Training and Finetuning Multimodal Embedding & Reranker Models with Sentence Transformers](https://huggingface.co/blog/train-multimodal-sentence-transformers): training multimodal Cross Encoders.

## Citation

### BibTeX

#### Sentence Transformers
```bibtex
@inproceedings{reimers-2019-sentence-bert,
    title = "Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks",
    author = "Reimers, Nils and Gurevych, Iryna",
    booktitle = "Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing",
    month = "11",
    year = "2019",
    publisher = "Association for Computational Linguistics",
    url = "https://arxiv.org/abs/1908.10084",
}
```

<!--
## Glossary

*Clearly define terms in order to be accessible across audiences.*
-->

<!--
## Model Card Authors

*Lists the people who create the model card, providing recognition and accountability for the detailed work that goes into its construction.*
-->

<!--
## Model Card Contact

*Provides a way for people who have updates to the Model Card, suggestions, or questions, to contact the Model Card authors.*
-->