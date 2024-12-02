from transformers import AutoTokenizer, AutoModelForQuestionAnswering, Trainer, TrainingArguments
from datasets import load_dataset
import os

# 1. 加載數據集（SQuAD）
print("正在加載 SQuAD 數據集...")
dataset = load_dataset("squad")

# 減少數據量以加速訓練（僅取部分數據）
small_train_dataset = dataset["train"].select(range(1000))  # 訓練集：1000 條
small_test_dataset = dataset["validation"].select(range(200))  # 驗證集：200 條

# 2. 加載模型與分詞器
print("正在加載 DistilBERT 模型和分詞器...")
model_name = "distilbert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)  # 確保使用 Fast Tokenizer
model = AutoModelForQuestionAnswering.from_pretrained(model_name)

# 3. 定義數據預處理函數
def preprocess_function(examples):
    """處理數據以適應問題回答任務"""
    # 分詞處理，啟用窗口映射和偏移量返回
    tokenized_examples = tokenizer(
        examples["question"],
        examples["context"],
        truncation="only_second",  # 僅截斷上下文部分
        max_length=384,           # 最大序列長度
        stride=128,               # 窗口移動步長
        return_overflowing_tokens=True,  # 啟用窗口映射
        return_offsets_mapping=True,     # 返回字符偏移
        padding="max_length",     # 填充到固定長度
    )

    # 確保分詞結果包含 offset_mapping
    if "offset_mapping" not in tokenized_examples:
        raise KeyError("分詞結果中缺少 'offset_mapping'，請檢查分詞器配置或數據格式！")

    # 每個分詞片段對應的原始樣本索引
    sample_mapping = tokenized_examples["overflow_to_sample_mapping"]
    offset_mapping = tokenized_examples["offset_mapping"]

    # 初始化起始和結束位置列表
    start_positions = []
    end_positions = []

    for i, offsets in enumerate(offset_mapping):
        # 對應的原始樣本索引
        sample_index = sample_mapping[i]
        answers = examples["answers"][sample_index]

        if len(answers["answer_start"]) == 0:
            # 如果答案不存在，標記 [CLS] 的索引為答案位置
            start_positions.append(tokenizer.cls_token_id)
            end_positions.append(tokenizer.cls_token_id)
        else:
            # 獲取答案的字符起始和結束位置
            start_char = answers["answer_start"][0]
            end_char = start_char + len(answers["text"][0])

            # 確定 token 的起始和結束位置
            token_start_index = 0
            token_end_index = len(offsets) - 1

            while token_start_index < len(offsets) and offsets[token_start_index][0] <= start_char:
                token_start_index += 1
            start_positions.append(token_start_index - 1)

            while token_end_index >= 0 and offsets[token_end_index][1] >= end_char:
                token_end_index -= 1
            end_positions.append(token_end_index + 1)

    tokenized_examples["start_positions"] = start_positions
    tokenized_examples["end_positions"] = end_positions
    return tokenized_examples

# 4. 預處理數據集
print("正在預處理數據集...")
encoded_train = small_train_dataset.map(preprocess_function, batched=True, remove_columns=dataset["train"].column_names)
encoded_test = small_test_dataset.map(preprocess_function, batched=True, remove_columns=dataset["validation"].column_names)

# 5. 設置訓練參數
training_args = TrainingArguments(
    output_dir="./results",          # 保存模型的目錄
    evaluation_strategy="epoch",    # 每個訓練輪數後進行驗證
    learning_rate=3e-5,             # 學習率
    per_device_train_batch_size=8,  # 每設備訓練批量大小
    num_train_epochs=2,             # 訓練輪數
    weight_decay=0.01,              # 權重衰減
    save_total_limit=1,             # 保存的檢查點數量
    logging_dir="./logs",           # 日誌目錄
)

# 6. 初始化 Trainer
print("正在初始化 Trainer...")
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=encoded_train,
    eval_dataset=encoded_test,
)

# 7. 開始訓練
print("開始訓練模型...")
trainer.train()

# 8. 保存模型和分詞器
print("正在保存模型和分詞器...")
trainer.save_model("./results")
tokenizer.save_pretrained("./results")

# 9. 測試微調後的模型
print("正在測試微調後的模型...")
from transformers import pipeline
qa_pipeline = pipeline("question-answering", model="./results", tokenizer=tokenizer)

# 測試上下文與問題
context = """
This movie suck , i dont know why anyone want to watch it
"""
question = "how is the movie？"

result = qa_pipeline(question=question, context=context)
print("問題:", question)
print("答案:", result["answer"])
