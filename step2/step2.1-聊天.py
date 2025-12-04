import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel

# ================= 配置区域 =================
BASE_MODEL_ID = "Qwen/Qwen2.5-7B-Instruct"
ADAPTER_PATH = "./generated_hoshino_data"  # 你的 LoRA 输出目录


# ===========================================

def main():
    print("⏳ 正在加载星野 (这可能需要几分钟)...")

    # 1. 配置 4-bit 量化 (与训练时保持一致)
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

    # 2. 加载基础模型
    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_ID,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True
    )
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_ID, trust_remote_code=True)

    # 3. 加载微调权重 (LoRA)
    model = PeftModel.from_pretrained(base_model, ADAPTER_PATH)

    print("✅ 星野加载完成！开始聊天吧 (输入 'exit' 退出)")
    print("-" * 30)

    # 4. 初始化对话
    # ⚠️ 关键修改：这里的 System Prompt 必须和训练数据里的一模一样！
    # 如果你训练用的是"星野"，这里必须用"星野"，否则微调效果出不来。
    messages = [
        {"role": "system", "content": "你是一个名为星野的可爱猫娘。"}
    ]

    while True:
        user_input = input("\n👤 主人: ")
        if user_input.lower() in ["exit", "quit", "退出"]:
            print("🐱 星野: 主人再见喵～")
            break

        messages.append({"role": "user", "content": user_input})

        # 5. 准备推理输入 (修复版)
        # 直接使用 tokenize=True，让库帮我们处理 input_ids 和 attention_mask
        # return_dict=True 会返回一个字典，包含 input_ids 和 attention_mask
        model_inputs = tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,  # 这一步直接转成数字 ID
            return_tensors="pt",  # 返回 PyTorch Tensor
            return_dict=True  # 返回字典格式
        ).to(model.device)

        # 6. 生成回复
        with torch.no_grad():
            generated_ids = model.generate(
                **model_inputs,  # 这里会自动解包传入 input_ids 和 attention_mask
                max_new_tokens=512,
                temperature=0.7,  # 稍微调高一点 creativity
                top_p=0.9,
                repetition_penalty=1.1,
                pad_token_id=tokenizer.eos_token_id
            )

        # 7. 解码
        # generated_ids 包含了[历史对话 + 新回复]，我们需要切片只取新回复
        # model_inputs.input_ids 是输入的长度
        input_len = model_inputs.input_ids.shape[1]

        # 只取新生成的部分
        generated_part = generated_ids[0][input_len:]

        response = tokenizer.decode(generated_part, skip_special_tokens=True)

        print(f"🐱 星野: {response}")

        # 将回复加入历史记录，以便下一轮对话
        messages.append({"role": "assistant", "content": response})


if __name__ == "__main__":
    main()