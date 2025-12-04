import os
import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)
from peft import LoraConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training
from trl import DPOTrainer, DPOConfig

# ================= ⚙️ 配置区域 =================

# 1. 原始基座模型路径 (保持不变)
BASE_MODEL_PATH = "Qwen/Qwen2.5-7B-Instruct"

# 2. DPO 数据集路径
DATA_FILE = "hoshino_dpo_pairs.jsonl"

# 3. 输出路径 (实验性对照组)
OUTPUT_DIR = "./hoshino_dpo_direct_fail_test"


# ==========================================================

def main():
    print(f"🚀 开始准备 DPO 对照实验 (直接基座 DPO)...")
    print(f"📥 加载基座模型: {BASE_MODEL_PATH}")
    print(f"⚠️  注意：本次不加载 SFT 权重，直接对基座进行偏好对齐")

    # 1. 加载 Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_PATH, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"  # DPO 训练建议 padding 在左侧

    # 2. 加载数据集
    dataset = load_dataset("json", data_files=DATA_FILE, split="train")
    print(f"📚 原始数据量: {len(dataset)}")

    # ================= 🛠️ 关键修复：使用标准格式化 =================
    def format_dpo_data(example):
        """
        使用 tokenizer 自动处理格式，确保与 Qwen 基座的认知一致。
        这是控制变量法的关键：格式必须正确，才能验证无 SFT 的影响。
        """
        # A. 构建标准消息列表
        messages = [
            {"role": "system", "content": example['system']}
        ]

        for turn in example['history']:
            messages.append({"role": turn['role'], "content": turn['content']})

        messages.append({"role": "user", "content": example['question']})

        # B. 使用 tokenizer 生成 ChatML 格式 (<|im_start|>...)
        prompt_text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True  # 自动添加 <|im_start|>assistant\n
        )

        return {
            "prompt": prompt_text,
            "chosen": example['chosen'],
            "rejected": example['rejected']
        }

    # ===============================================================

    # 应用格式化
    dataset = dataset.map(format_dpo_data, remove_columns=dataset.column_names)
    print(f"✅ 数据格式化完成 (ChatML 格式已对齐)")

    # 3. 加载基座模型
    print("⏳正在加载基座模型...")
    # 直接加载基座，不再进行 Merge 操作
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_PATH,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True
    )

    # 禁用缓存以节省显存
    model.config.use_cache = False

    # 4. 配置 LoRA
    # 因为没有加载 SFT 权重，我们需要在这里初始化一个新的 LoRA 层
    peft_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        bias="none",
        task_type="CAUSAL_LM",
    )

    # 5. DPO 训练参数
    training_args = DPOConfig(
        output_dir=OUTPUT_DIR,
        beta=0.1,
        max_length=1536,
        max_prompt_length=1024,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=8,
        num_train_epochs=3,
        learning_rate=5e-6,
        lr_scheduler_type="cosine",
        warmup_ratio=0.1,
        fp16=True,
        logging_steps=10,
        save_steps=50,
        save_total_limit=2,
        optim="paged_adamw_32bit",
        remove_unused_columns=False,
        gradient_checkpointing=True,
    )

    # 6. 初始化 Trainer
    dpo_trainer = DPOTrainer(
        model=model,  # 直接传入基座
        ref_model=None,  # TRL 会自动复制一份基座作为参考模型(Reference Model)
        args=training_args,
        train_dataset=dataset,
        processing_class=tokenizer,
        peft_config=peft_config,  # 从零开始训练 LoRA
    )

    print("⚔️  开始 DPO 对抗训练 (基座直出版)...")
    dpo_trainer.train()

    # 7. 保存结果
    print(f"💾 保存实验性权重到 {OUTPUT_DIR}")
    dpo_trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)  # 别忘了保存 tokenizer
    print("🎉 实验训练完成！")


if __name__ == "__main__":
    main()