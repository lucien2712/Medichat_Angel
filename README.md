# 醫療小天使 (MediChat Angel)

## 專案概述

醫療小天使是一個基於LLaMA大型語言模型的醫療助手系統，能夠針對用戶的醫療問題提供專業建議。本專案利用微調技術使模型適應醫療領域，並結合檢索增強生成（RAG）技術提供更準確的回答。

## 功能特色

- **基礎聊天功能**: 使用微調過的LLaMA模型進行醫療問答
- **檢索增強生成**: 結合向量資料庫實現基於知識庫的精準回答
- **模型微調**: 完整的LLaMA模型微調流程
- **向量嵌入**: 支援將醫療知識轉換為向量形式用於檢索

## 目錄結構

```
.
├── finetune
│   ├── code
│   │   └── finetune.py   # 模型微調腳本
│   └── data              # 訓練資料存放處
├── chat.py               # 基礎聊天功能
├── chatRAG.py            # 檢索增強生成聊天功能
├── embedding.py          # 向量嵌入生成工具
├── disease/              # 醫療知識庫文本目錄
└── README.md             
```


## 使用方法

### 1. 準備知識庫

將醫療相關文本放入`./disease/`目錄中：

```bash
# 確保以下目錄存在
mkdir -p ./disease/
# 將醫療知識文件放入上述目錄 (*.txt格式)
```

### 2. 生成嵌入向量

使用`embedding.py`將醫療文本轉換為向量：

```bash
python embedding.py
```

此步驟會從Hugging Face下載`GanymedeNil/text2vec-large-chinese`模型，並將處理後的向量儲存在`./db`目錄。

### 3. 模型微調

準備訓練資料並進行模型微調：

```bash
python finetune/code/finetune.py \
  --model_name "llama-chinese" \
  --dataset_dir "finetune/data/train_data.json" \
  --output_dir "output" \
  --num_epoch 3 \
```

#### 主要參數：
- `--model_name`: 基礎模型路徑
- `--dataset_dir`: 訓練資料集路徑
- `--output_dir`: 輸出模型路徑
- `--num_epoch`: 訓練輪數
- `--from_ckpt`: 是否從檢查點繼續訓練
- `--ckpt_name`: 檢查點路徑

### 4. 基礎聊天

使用微調後的模型進行聊天：

```bash
python chat.py --model "llama-chinese" --peft_path "output"
```

#### 主要參數：
- `--model`: 基礎模型路徑
- `--peft_path`: 微調模型路徑

### 5. 檢索增強生成聊天

結合知識庫進行更準確的對話：

```bash
python chatRAG.py --model "llama-chinese" --peft_path "output"
```

## 訓練資料格式

微調資料集應為JSON格式，包含以下欄位：
- `instruction`: 指令
- `input`: 輸入（可選）
- `output`: 期望的輸出

範例：
```json
{
  "instruction": "請根據症狀提供可能的診斷和建議",
  "input": "我最近有頭痛和發燒的症狀，已經持續三天了",
  "output": "根據您描述的症狀，頭痛和發燒持續三天可能是感冒、流感或其他感染性疾病的表現..."
}
```

## 技術細節

- **模型**: 基於LLaMA模型架構
- **微調方法**: 使用PEFT/LoRA進行參數高效微調
- **向量資料庫**: 使用Chroma儲存和檢索向量
- **檢索策略**: 使用階層式檢索（hierarchical retrieval）策略
