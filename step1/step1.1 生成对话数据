import os
import sys
import yaml
import json
import random
import requests
import time
import argparse
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

# ==========================================
# 🔧 微调专用配置 (在此处修改保存到文件里的 System Prompt)
# ==========================================
# 这句话会写入 output 文件。微调时，模型看到这句话就会激活猫娘模式。
# 如果你想让 system 为空，可以设置: FINETUNE_SYSTEM_PROMPT = ""
FINETUNE_SYSTEM_PROMPT = "你是一个名为星野的可爱猫娘。"

# 全局文件锁
file_write_lock = threading.Lock()

# ==========================================
# 1. 配置与工具函数
# ==========================================

def load_config(config_path="config.yaml"):
    """加载YAML配置文件"""
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)
    except FileNotFoundError:
        print(f"❌ 错误: 配置文件 '{config_path}' 未找到。")
        sys.exit(1)

def load_seed_questions(seed_file_path):
    """加载JSON格式的问题种子文件"""
    try:
        with open(seed_file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"❌ 错误: 问题种子文件 '{seed_file_path}' 未找到。")
        sys.exit(1)

def check_api_availability(config):
    """检查API可用性"""
    print("正在执行 API 可用性检查...")
    api_conf = config['api_config']
    headers = {"Content-Type": "application/json", "Authorization": f"Bearer {api_conf['api_key']}"}
    payload = {
        "model": api_conf['model_name'], 
        "messages": [{"role": "user", "content": "ping"}], 
        "max_tokens": 5
    }
    
    try:
        response = requests.post(api_conf['base_url'], headers=headers, json=payload, timeout=10)
        response.raise_for_status()
        print(f"✅ API 连接成功! 模型 '{api_conf['model_name']}' 可用。")
        return True
    except Exception as e:
        print(f"❌ API 检查失败: {e}")
        print("请检查 config.yaml 中的 base_url 和 api_key。")
        return False

def call_llm_api(config, messages):
    """调用API，包含重试和延时逻辑"""
    api_conf = config['api_config']
    max_retries = api_conf.get('max_retries', 2)
    
    for attempt in range(max_retries + 1):
        try:
            delay = api_conf.get('request_delay_seconds', 1.0)
            if delay > 0:
                time.sleep(delay)

            headers = {"Content-Type": "application/json", "Authorization": f"Bearer {api_conf['api_key']}"}
            payload = {
                "model": api_conf['model_name'], 
                "messages": messages, 
                "temperature": 0.8, 
                "max_tokens": 2048
            }

            response = requests.post(api_conf['base_url'], headers=headers, json=payload, timeout=120)
            response.raise_for_status()
            
            return response.json()['choices'][0]['message']['content'].strip()

        except Exception as e:
            if attempt == max_retries:
                return None
    return None

# ==========================================
# 2. 核心逻辑：单条数据处理
# ==========================================

def evolve_question(config, original_question):
    prompt = config['evolver_config']['evolution_prompt'].format(original_question=original_question)
    messages = [{"role": "user", "content": prompt}]
    return call_llm_api(config, messages)

def generate_next_question(config, conversation_history):
    prompt = config['generation_config']['next_question_prompt']
    messages = []
    for turn in conversation_history:
        messages.append({"role": "user", "content": turn['role_user']})
        messages.append({"role": "assistant", "content": turn['role_assistant']})
    messages.append({"role": "user", "content": prompt})
    return call_llm_api(config, messages)

def process_single_seed(config, question, output_file_handle):
    gen_conf = config['generation_config']
    # 这里读取的是 config.yaml 里的超长人设，仅用于 API 调用（扮演）
    actor_prompt = config['persona_config']['system_prompt']

    # 1. 进化问题
    evolved_q = evolve_question(config, question)
    if not evolved_q: 
        return False

    # 2. 生成多轮对话
    num_turns = random.randint(gen_conf['min_turns'], gen_conf['max_turns'])
    history = []
    current_q = evolved_q
    conversation_valid = True
    
    for i in range(num_turns):
        # API 调用时：使用长 System Prompt
        msgs = [{"role": "system", "content": actor_prompt}]
        for turn in history:
            msgs.append({"role": "user", "content": turn['role_user']})
            msgs.append({"role": "assistant", "content": turn['role_assistant']})
        msgs.append({"role": "user", "content": current_q})
        
        answer = call_llm_api(config, msgs)
        if not answer:
            conversation_valid = False; break
        
        history.append({"role_user": current_q, "role_assistant": answer})
        
        if i < num_turns - 1:
            next_q = generate_next_question(config, history)
            if not next_q:
                conversation_valid = False; break
            current_q = next_q

    # 3. 格式化并保存 (关键修改点!)
    if conversation_valid and len(history) >= gen_conf.get('min_turns', 1):
        
        # 保存时：使用短 System Prompt (或者空)
        # 如果 FINETUNE_SYSTEM_PROMPT 有内容，就加上 system message
        if FINETUNE_SYSTEM_PROMPT:
            standard_messages = [{"role": "system", "content": FINETUNE_SYSTEM_PROMPT}]
        else:
            standard_messages = [] # 如果为空，则不保存 system 字段
            
        for turn in history:
            standard_messages.append({"role": "user", "content": turn['role_user']})
            standard_messages.append({"role": "assistant", "content": turn['role_assistant']})
        
        output_data = {"messages": standard_messages}
        json_line = json.dumps(output_data, ensure_ascii=False) + "\n"
        
        with file_write_lock:
            output_file_handle.write(json_line)
            output_file_handle.flush()
        
        return True
    return False

# ==========================================
# 3. 主逻辑控制器
# ==========================================

def run_data_generation(config, max_workers=3):
    if not check_api_availability(config):
        return

    seed_questions = load_seed_questions(config['file_config']['seed_file'])
    output_file = config['file_config']['output_file']
    
    processed_count = 0
    if os.path.exists(output_file):
        with open(output_file, 'r', encoding='utf-8') as f:
            processed_count = sum(1 for _ in f)
    
    if processed_count > 0:
        print(f"📄 检测到已生成 {processed_count} 条数据，将跳过这些种子...")
        seed_questions = seed_questions[processed_count:]

    if not seed_questions:
        print("✨ 所有种子已处理完毕！")
        return

    print(f"\n🚀 开始多线程生成任务 (并发数: {max_workers})...")
    print(f"💾 保存时的 System Prompt 将被替换为: '{FINETUNE_SYSTEM_PROMPT}'")
    
    with open(output_file, 'a', encoding='utf-8') as f:
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [
                executor.submit(process_single_seed, config, q, f) 
                for q in seed_questions
            ]
            
            success_count = 0
            for future in tqdm(as_completed(futures), total=len(seed_questions), desc="并发处理中"):
                try:
                    if future.result():
                        success_count += 1
                except Exception:
                    pass

    print(f"\n✅ 任务完成！本次成功生成: {success_count} 条。")
    print(f"💾 数据已保存至: {output_file}")

# ==========================================
# 4. 交互模式
# ==========================================

def run_interactive_chat(config):
    if not check_api_availability(config): return
    persona_prompt = config['persona_config']['system_prompt']
    history = []
    print("\n--- 🐱 星野测试终端 ---")
    while True:
        try:
            user_input = input("\n👤 主人: ")
            if user_input.lower() in ['exit', 'quit']: break
            
            messages = [{"role": "system", "content": persona_prompt}] + history + [{"role": "user", "content": user_input}]
            print("⏳ 思考中...", end="", flush=True)
            response = call_llm_api(config, messages)
            print("\r" + " " * 20 + "\r", end="")

            if response:
                print(f"🐱 星野: {response}")
                history.append({"role": "user", "content": user_input})
                history.append({"role": "assistant", "content": response})
        except KeyboardInterrupt:
            break

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--chat", action="store_true", help="交互模式")
    parser.add_argument("--workers", type=int, default=3, help="并发数")
    args = parser.parse_args()
    config = load_config()

    if args.chat:
        run_interactive_chat(config)
    else:
        run_data_generation(config, max_workers=args.workers)