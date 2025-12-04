import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig
)
from peft import PeftModel
from trl import SFTTrainer, SFTConfig

# ================= 配置区域 =================
NEW_DATA_FILE = "hoshino_override_training.jsonl"
BASE_MODEL_ID = "Qwen/Qwen2.5-7B-Instruct"
PREVIOUS_LORA_DIR = "./generated_hoshino_data"
NEW_OUTPUT_DIR = "./generated_hoshino_v2"


# ===========================================

def train():
    # 1. 加载 tokenizer
    print("Processing tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_ID, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token

    # 2. 加载数据集
    print(f"Loading dataset from {NEW_DATA_FILE}...")
    try:
        dataset = load_dataset("json", data_files=NEW_DATA_FILE, split="train")
    except Exception as e:
        print(f"❌ 数据集加载失败: {e}")
        return

    print("Formatting dataset (Standard Messages -> ChatML)...")

    def format_data_to_text(row):
        messages = row.get("messages", [])
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False
        )
        return {"text": text}

    try:
        dataset = dataset.map(format_data_to_text)
    except Exception as e:
        print(f"❌ 数据格式转换失败: {e}")
        return

    # 3. 加载基座模型
    print("Loading base model...")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_ID,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True
    )

    # 4. 加载旧权重
    print(f"Loading previous LoRA adapter from {PREVIOUS_LORA_DIR}...")
    model = PeftModel.from_pretrained(
        base_model,
        PREVIOUS_LORA_DIR,
        is_trainable=True
    )
    model.print_trainable_parameters()

    # 5. 训练参数
    # ⚠️ 修改点：初始化时只传通用参数，不传 max_seq_length 或 dataset_text_field
    training_args = SFTConfig(
        output_dir=NEW_OUTPUT_DIR,
        num_train_epochs=3,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        learning_rate=1e-4,
        logging_steps=5,
        save_strategy="epoch",
        fp16=not torch.cuda.is_bf16_supported(),
        bf16=torch.cuda.is_bf16_supported(),
        optim="paged_adamw_8bit",
        packing=False,
        # max_seq_length=2048,      <-- 删掉
        # dataset_text_field="text" <-- 删掉
    )

    # 🛠️ 关键修改：手动赋值（还原你最开始能跑通的写法）
    training_args.dataset_text_field = "text"
    training_args.max_seq_length = 2048

    # 6. 初始化 Trainer
    print("Initializing Trainer...")
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        processing_class=tokenizer,
        # max_seq_length=2048,  <-- 这里也删掉，不要传
    )

    # 7. 开始训练
    print("🚀 Starting continued training...")
    trainer.train()

    # 8. 保存
    print(f"✅ Done! New Model saved to {NEW_OUTPUT_DIR}")
    trainer.save_model(NEW_OUTPUT_DIR)
    tokenizer.save_pretrained(NEW_OUTPUT_DIR)


if __name__ == "__main__":
    train()