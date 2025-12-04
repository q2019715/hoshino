import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig
)
from peft import LoraConfig
from trl import SFTTrainer, SFTConfig

# ================= 配置区域 =================
DATA_FILE = "generated_hoshino_data.jsonl"  # 确保这里指向您的数据文件
MODEL_ID = "Qwen/Qwen2.5-7B-Instruct"
OUTPUT_DIR = "./generated_hoshino_data"


# ===========================================

def train():
    # 1. 加载 tokenizer
    print("Processing tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token

    # 2. 加载数据集
    print("Loading dataset...")
    try:
        dataset = load_dataset("json", data_files=DATA_FILE, split="train")
    except Exception as e:
        print(f"❌ 数据集加载失败: {e}")
        return

    # =======================================================
    # 🛠️ 修改区域：适配标准 "messages" 格式
    # =======================================================
    print("Formatting dataset (Standard Messages -> ChatML)...")

    def format_data_to_text(row):
        # 1. 直接读取 "messages" 字段
        # 你的数据已经是标准的 [{"role": "system/user/assistant", "content": "..."}, ...] 格式
        messages = row.get("messages", [])

        # 2. 直接应用聊天模板
        # Qwen2.5 的模板会自动处理 system, user, assistant 角色
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False
        )
        return {"text": text}

    # 应用转换
    try:
        dataset = dataset.map(format_data_to_text)
    except Exception as e:
        print(f"❌ 数据格式转换失败。请检查您的 JSONL 文件是否包含 'messages' 字段。\n错误信息: {e}")
        return

    # 打印示例以供检查
    print(f"✅ 数据准备完成！样本示例:\n{dataset[0]['text'][:200]}...")

    # 3. 模型准备 (QLoRA)
    print("Loading model...")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True
    )

    # 4. LoRA 配置
    peft_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    )

    # 5. 训练参数
    training_args = SFTConfig(
        output_dir=OUTPUT_DIR,
        num_train_epochs=3,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        learning_rate=2e-4,
        logging_steps=5,
        save_strategy="epoch",
        fp16=not torch.cuda.is_bf16_supported(),
        bf16=torch.cuda.is_bf16_supported(),
        optim="paged_adamw_8bit",
        packing=False,
    )

    # 指定字段为 "text"
    training_args.dataset_text_field = "text"
    training_args.max_seq_length = 2048  # 稍微调大一点以容纳多轮对话

    # 6. 初始化 Trainer
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        peft_config=peft_config,
        processing_class=tokenizer,
    )

    # 7. 开始训练
    print("🚀 Starting training...")
    trainer.train()

    # 8. 保存
    print(f"✅ Done! Model saved to {OUTPUT_DIR}")
    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)


if __name__ == "__main__":
    train()