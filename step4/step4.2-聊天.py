import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel

# ================= 配置区域 =================
BASE_MODEL_ID = "Qwen/Qwen2.5-7B-Instruct"
ADAPTER_PATH = "./generated_hoshino_v2"  # 你的 LoRA 输出目录

# 预设提示词库
PRESET_PROMPTS = {
    "1": {
        "name": "星野猫娘",
        "prompt": "你是一个名为星野的可爱猫娘。"
    },
    "2": {
        "name": "专业助手",
        "prompt": "你是一个专业、严谨的AI助手，擅长提供准确的信息和建议。"
    },
    "3": {
        "name": "幽默伙伴",
        "prompt": "你是一个幽默风趣的聊天伙伴，喜欢用轻松愉快的方式交流。"
    },
    "4": {
        "name": "学习导师",
        "prompt": "你是一个耐心的学习导师，擅长用简单易懂的方式解释复杂概念。"
    },
    "5": {
        "name": "创意写手",
        "prompt": "你是一个富有创意的写手，擅长创作故事、诗歌和各种文学作品。"
    }
}


# ===========================================

def select_system_prompt():
    """让用户选择或自定义系统提示词"""
    print("\n" + "=" * 50)
    print("🎭 请选择系统提示词:")
    print("=" * 50)

    # 显示预设选项
    for key, value in PRESET_PROMPTS.items():
        print(f"  [{key}] {value['name']}")
        print(f"      → {value['prompt']}")
        print()

    print(f"  [0] 自定义提示词")
    print("=" * 50)

    while True:
        choice = input("\n👉 请输入选项编号 (0-5): ").strip()

        if choice == "0":
            # 自定义提示词
            custom_prompt = input("\n✏️  请输入你的自定义提示词: ").strip()
            if custom_prompt:
                print(f"\n✅ 已设置自定义提示词: {custom_prompt}")
                return custom_prompt
            else:
                print("❌ 提示词不能为空，请重新输入")
                continue

        elif choice in PRESET_PROMPTS:
            selected = PRESET_PROMPTS[choice]
            print(f"\n✅ 已选择: {selected['name']}")
            print(f"   提示词: {selected['prompt']}")
            return selected['prompt']

        else:
            print("❌ 无效选项，请输入 0-5 之间的数字")


def main():
    print("⏳ 正在加载模型 (这可能需要几分钟)...")

    # 1. 配置 4-bit 量化
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

    print("✅ 模型加载完成！")

    # 4. 让用户选择系统提示词
    system_prompt = select_system_prompt()

    # 5. 初始化对话
    messages = [
        {"role": "system", "content": system_prompt}
    ]

    print("\n" + "=" * 50)
    print("💬 开始聊天吧！")
    print("   输入 'exit' 或 'quit' 退出")
    print("   输入 'reset' 重新选择提示词")
    print("   输入 'clear' 清空对话历史")
    print("=" * 50)

    while True:
        user_input = input("\n👤 你: ").strip()

        if user_input.lower() in ["exit", "quit", "退出"]:
            print("\n👋 再见！")
            break

        # 重新选择提示词
        if user_input.lower() == "reset":
            system_prompt = select_system_prompt()
            messages = [{"role": "system", "content": system_prompt}]
            print("\n✅ 提示词已更新，对话历史已清空")
            continue

        # 清空对话历史
        if user_input.lower() == "clear":
            messages = [{"role": "system", "content": system_prompt}]
            print("\n✅ 对话历史已清空")
            continue

        if not user_input:
            continue

        messages.append({"role": "user", "content": user_input})

        # 6. 准备推理输入
        model_inputs = tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_tensors="pt",
            return_dict=True
        ).to(model.device)

        # 7. 生成回复
        with torch.no_grad():
            generated_ids = model.generate(
                **model_inputs,
                max_new_tokens=512,
                temperature=0.7,
                top_p=0.9,
                repetition_penalty=1.1,
                pad_token_id=tokenizer.eos_token_id
            )

        # 8. 解码
        input_len = model_inputs.input_ids.shape[1]
        generated_part = generated_ids[0][input_len:]
        response = tokenizer.decode(generated_part, skip_special_tokens=True)

        print(f"\n🤖 AI: {response}")

        # 将回复加入历史记录
        messages.append({"role": "assistant", "content": response})


if __name__ == "__main__":
    main()
