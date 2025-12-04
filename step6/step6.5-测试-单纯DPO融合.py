import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import os

# ================= ⚙️ 配置区域 =================

# 1. 原始基座 (你做实验时用的那个 Instruct 版)
BASE_MODEL_ID = "Qwen/Qwen2.5-7B-Instruct"

# 2. DPO 权重路径 (你的实验输出目录)
# ⚠️ 这里填写你刚刚跑完的那个 "直接 DPO" 的输出目录
DPO_ADAPTER_PATH = "./hoshino_dpo_direct_fail_test"

# 3. 最终输出路径 (合并后的完整模型)
OUTPUT_DIR = "./Hoshino-DirectDPO-Experiment-Full"


# ==============================================

def main():
    print(f"🚀 开始执行 [基座 + DPO] 直接熔炼 (跳过 SFT)...")

    # 1. 加载 Tokenizer
    print(f"📥 加载 Tokenizer: {BASE_MODEL_ID}")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_ID, trust_remote_code=True)

    # 2. 加载基座模型
    # 必须用 float16 或 bfloat16，不能用 4bit/8bit 量化加载，否则无法进行 merge
    print(f"📥 加载基座模型: {BASE_MODEL_ID} (FP16 Mode)...")
    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_ID,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
        low_cpu_mem_usage=True
    )

    # 3. 直接加载并合并 DPO 权重
    print(f"🔨 正在将 DPO LoRA ({DPO_ADAPTER_PATH}) 熔炼进基座...")

    try:
        # 直接把 DPO 的 LoRA 挂载到基座上
        model_to_merge = PeftModel.from_pretrained(base_model, DPO_ADAPTER_PATH)

        # 执行合并 (Merge and Unload)
        final_model = model_to_merge.merge_and_unload()
        print("✅ DPO 权重融合完毕！")

    except Exception as e:
        print(f"❌ 错误: 无法加载 DPO 权重。请检查路径是否正确，或者 adapter_config.json 是否存在。")
        print(f"详细报错: {e}")
        return

    # 4. 保存最终模型
    print(f"💾 正在保存完整模型到: {OUTPUT_DIR} ...")

    # 保存权重
    final_model.save_pretrained(OUTPUT_DIR)

    # 保存 Tokenizer
    tokenizer.save_pretrained(OUTPUT_DIR)

    # 保存生成配置 (Generation Config) - 防止推理时缺少 eos_token 定义
    try:
        base_model.generation_config.save_pretrained(OUTPUT_DIR)
        print("✅ Generation Config 已保存")
    except Exception as e:
        print(f"⚠️ Generation Config 保存失败 (非致命错误): {e}")

    print("\n" + "=" * 50)
    print(f"🎉 实验模型已构建完成：{OUTPUT_DIR}")
    print("你可以直接用 vLLM 或 Ollama 加载这个文件夹进行测试了。")
    print("=" * 50)


if __name__ == "__main__":
    main()