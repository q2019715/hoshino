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

# 1. 原始基座模型路径
BASE_MODEL_PATH = "Qwen/Qwen2.5-7B-Instruct"

# 2. 第一阶段 SFT 训练出来的权重路径 (确保这里是你刚刚SFT练好的目录)
SFT_ADAPTER_PATH = "./generated_hoshino_v2"

# 3. DPO 数据集路径
DATA_FILE = "hoshino_dpo_pairs.jsonl"

# 4. DPO 输出路径
OUTPUT_DIR = "./hoshino_dpo_final"


# ==========================================================

def main():
    print(f"🚀 开始准备 DPO 训练...")
    print(f"📥 加载基座模型: {BASE_MODEL_PATH}")
    print(f"🔗 加载 SFT 权重: {SFT_ADAPTER_PATH}")

    # 1. 加载 Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_PATH, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"  # DPO 通常建议 padding 在左边

    # 2. 加载数据集
    dataset = load_dataset("json", data_files=DATA_FILE, split="train")
    print(f"📚 原始数据量: {len(dataset)}")

    # ================= 🛠️ 核心修改：格式化函数 =================
    def format_dpo_data(example):
        """
        将数据转换为 Qwen 的 ChatML 格式 (<|im_start|>...)
        """
        # A. 构建符合 ChatML 标准的消息列表
        messages = [
            {"role": "system", "content": example['system']}
        ]

        # 添加历史对话
        for turn in example['history']:
            messages.append({"role": turn['role'], "content": turn['content']})

        # 添加当前问题
        messages.append({"role": "user", "content": example['question']})

        # B. 使用 tokenizer 自动生成 prompt
        # tokenize=False: 返回字符串
        # add_generation_prompt=True: 自动在末尾添加 "<|im_start|>assistant\n"
        prompt_text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

        return {
            "prompt": prompt_text,  # 包含 <|im_start|> 等特殊 token 的完整 prompt
            "chosen": example['chosen'],  # 纯文本
            "rejected": example['rejected']  # 纯文本
        }

    # ==========================================================

    # 应用格式化
    dataset = dataset.map(format_dpo_data, remove_columns=dataset.column_names)
    print(f"✅ 数据格式化完成 (ChatML 格式已对齐)")

    # 3. 加载模型并合并权重
    # 注意：这里使用 float16 加载并合并。如果显存不足(OOM)，可能需要改为加载 4bit base_model 且不合并
    print("⏳正在加载基座模型用于合并...")
    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_PATH,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True
    )

    print("⏳正在合并 SFT 权重...")
    model = PeftModel.from_pretrained(base_model, SFT_ADAPTER_PATH)
    model = model.merge_and_unload()  # 将 SFT LoRA 彻底融合进模型
    print("✅ SFT 权重合并完成！")

    # 禁用缓存以节省显存
    model.config.use_cache = False

    # 4. 配置 DPO 的 LoRA 参数 (在合并后的模型上再挂一个新的 LoRA 进行 DPO)
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
        beta=0.1,  # DPO 的温度参数，0.1 是标准值
        max_length=2048,  # 总长度 (Prompt + Answer)
        max_prompt_length=1536,  # Prompt 最大长度
        per_device_train_batch_size=1,
        gradient_accumulation_steps=8,
        num_train_epochs=3,
        learning_rate=5e-6,  # DPO 学习率通常比 SFT 低
        lr_scheduler_type="cosine",
        warmup_ratio=0.1,
        fp16=True,  # 如果显卡支持 BF16 (30系/40系)，建议改为 bf16=True
        logging_steps=10,
        save_steps=50,
        save_total_limit=2,
        optim="paged_adamw_32bit",
        remove_unused_columns=False,
        gradient_checkpointing=True,  # 开启显存优化
    )

    # 6. 初始化 Trainer
    dpo_trainer = DPOTrainer(
        model=model,
        ref_model=None,  # 设置为 None，TRL 会自动加载一份冻结的副本作为参考模型
        args=training_args,
        train_dataset=dataset,
        processing_class=tokenizer,
        peft_config=peft_config,
    )

    # 7. 开始训练
    print("⚔️  开始 DPO 对抗训练...")
    dpo_trainer.train()

    # 8. 保存最终模型和 Tokenizer
    print(f"💾 保存最终 DPO 权重到 {OUTPUT_DIR}")
    dpo_trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)  # ⚠️ 记得保存 tokenizer，方便后续使用
    print("🎉 DPO 训练全部完成！")


if __name__ == "__main__":
    main()