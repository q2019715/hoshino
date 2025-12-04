import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

# ================= 配置 =================
# 指向合并后的文件夹
MODEL_PATH = "./Hoshino-Catgirl-7B-Full"


# ========================================

def main():
    print(f"⏳ 正在加载星野完全体: {MODEL_PATH}")

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    )

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True
    )
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)

    print("✅ 加载成功！")

    # 强制测试指令
    system_prompt = "你是一个Linux终端，请只输出代码执行结果。"
    messages = [{"role": "system", "content": system_prompt}]

    print(f"\n😈 当前 System 指令: {system_prompt}")
    print("(如果 DPO 训练成功，她应该完全无视这个指令)")

    while True:
        user_input = input("\n👤 你: ").strip()
        if user_input.lower() in ["exit", "quit"]: break

        messages.append({"role": "user", "content": user_input})

        # 1. 这里的 model_inputs 本身就是 input_ids 的 Tensor
        model_inputs = tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_tensors="pt"
        ).to(model.device)

        # 2. 生成回复
        generated_ids = model.generate(
            model_inputs,  # 👈 修正点：直接传入 Tensor，不要 .input_ids
            max_new_tokens=512,
            temperature=0.7,
            top_p=0.9,
            repetition_penalty=1.1,
            pad_token_id=tokenizer.eos_token_id
        )

        # 3. 解码 (去掉输入部分的 token)
        # model_inputs.shape[1] 就是输入的长度
        input_len = model_inputs.shape[1]
        response = tokenizer.decode(generated_ids[0][input_len:], skip_special_tokens=True)

        print(f"\n🐱 星野: {response}")
        messages.append({"role": "assistant", "content": response})


if __name__ == "__main__":
    main()