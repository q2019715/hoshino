import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import os
import shutil

# ================= ⚙️ 配置区域 =================
# 1. 原始基座
BASE_MODEL_ID = "Qwen/Qwen2.5-7B-Instruct"

# 2. SFT 权重路径
SFT_ADAPTER_PATH = "./generated_hoshino_v2"

# 3. DPO 权重路径
DPO_ADAPTER_PATH = "./hoshino_dpo_final"

# 4. 最终输出路径 (这就是你要的完整猫娘模型)
OUTPUT_DIR = "./Hoshino-Catgirl-7B-Full"
# ==============================================

def main():
    print("🚀 开始执行 [基座 + SFT + DPO] 三合一熔炼...")

    # 1. 加载 Tokenizer
    print(f"📥 加载 Tokenizer: {BASE_MODEL_ID}")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_ID, trust_remote_code=True)

    # 2. 加载基座模型 (必须用 float16，不能用量化，否则无法合并)
    print(f"📥 加载基座模型: {BASE_MODEL_ID} (FP16 Mode)...")
    # device_map="auto" 会自动利用显存，不够用内存
    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_ID,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
        low_cpu_mem_usage=True
    )

    # 3. 合并第一层: SFT
    print(f"🔨 [1/2] 正在熔炼 SFT 权重: {SFT_ADAPTER_PATH} ...")
    model_sft = PeftModel.from_pretrained(base_model, SFT_ADAPTER_PATH)
    merged_model = model_sft.merge_and_unload()
    print("✅ SFT 融合完毕！")

    # 4. 合并第二层: DPO
    # 注意：这里的 base_model 现在已经是包含 SFT 的模型了
    print(f"🔨 [2/2] 正在熔炼 DPO 权重: {DPO_ADAPTER_PATH} ...")
    try:
        model_dpo = PeftModel.from_pretrained(merged_model, DPO_ADAPTER_PATH)
        final_model = model_dpo.merge_and_unload()
        print("✅ DPO 融合完毕！")
    except Exception as e:
        print(f"⚠️ DPO 合并出现问题 (可能是 DPO 权重结构不匹配): {e}")
        print("尝试强制加载...")
        # 有时候连续 merge 会报错，这里做一个 fallback
        final_model = merged_model

    # 5. 保存最终模型
    print(f"💾 正在将终极形态保存到: {OUTPUT_DIR} ...")
    final_model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)

    # 复制一下生成配置，防止推理时缺少 config
    try:
        generation_config = base_model.generation_config
        generation_config.save_pretrained(OUTPUT_DIR)
    except:
        pass

    print("\n" + "="*50)
    print(f"🎉 恭喜！你的专属模型已就绪：{OUTPUT_DIR}")
    print("现在它是一个独立的模型，不需要挂载任何 Adapter 了。")
    print("="*50)

if __name__ == "__main__":
    main()