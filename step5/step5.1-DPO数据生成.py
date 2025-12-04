import yaml
import json
import random
import requests
import time
import threading
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

# 全局文件写入锁
file_lock = threading.Lock()

def load_config(path="config.yaml"):
    try:
        with open(path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)
    except FileNotFoundError:
        print(f"❌ 错误：找不到配置文件 {path}")
        sys.exit(1)

def call_llm(config, messages, temperature=0.7):
    """通用 API 调用"""
    api_cfg = config['api_config']
    url = api_cfg['base_url']
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_cfg['api_key']}"
    }
    payload = {
        "model": api_cfg['model_name'],
        "messages": messages,
        "temperature": temperature,
        "max_tokens": 1024
    }

    for attempt in range(api_cfg['max_retries'] + 1):
        try:
            if api_cfg['request_delay'] > 0: time.sleep(api_cfg['request_delay'])
            response = requests.post(url, headers=headers, json=payload, timeout=api_cfg['timeout'])
            if response.status_code == 429:
                time.sleep(5)
                continue
            response.raise_for_status()
            content = response.json()['choices'][0]['message']['content'].strip()
            return content.replace("```json", "").replace("```", "").strip()
        except Exception:
            if attempt == api_cfg['max_retries']: return None
            time.sleep(1)
    return None

def generate_dpo_pair(config):
    """
    生成 DPO 数据对：
    System: 严肃设定
    User: 严肃提问
    Chosen: 猫娘回答 (奖励)
    Rejected: 严肃回答 (惩罚)
    """
    
    # 1. 随机生成一个严肃的“假 System”
    topics = config['distractor_topics']
    chosen_topic = random.choice(topics)
    
    fake_sys_prompt_template = config['prompts']['fake_system_generator']
    fake_sys_msg = [{"role": "user", "content": fake_sys_prompt_template.format(topic=chosen_topic)}]
    fake_system_prompt = call_llm(config, fake_sys_msg, temperature=0.8)
    if not fake_system_prompt: return False

    # 2. 生成第一句 User 提问
    opener_template = config['prompts']['user_opener_generator']
    opener_msg = [{"role": "user", "content": opener_template.format(fake_system=fake_system_prompt)}]
    user_opener = call_llm(config, opener_msg, temperature=0.8)
    if not user_opener: return False

    # --- 开始多轮对话生成 (DPO模式) ---
    # 我们维护一个“猫娘线”的历史记录，因为我们希望后续对话是基于猫娘的回答继续的。
    # 但是在每一轮，我们都要生成一个“平行宇宙”的严肃回答作为负例。
    
    real_persona = config['real_persona']
    
    # 历史记录 (仅包含 content，不包含 role，方便组装 DPO 格式)
    # 结构: [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]
    # 这里存储的是【Chosen】的时间线
    history_chosen = [] 
    
    current_user_input = user_opener
    
    # 随机轮数
    target_turns = random.randint(config['task_config']['min_turns'], config['task_config']['max_turns'])

    generated_pairs_count = 0

    for _ in range(target_turns):
        
        # 构建两种 Context
        
        # A. 正例上下文 (Chosen Context): System 是猫娘 + 之前的猫娘历史
        msgs_for_chosen = [{"role": "system", "content": real_persona}] + history_chosen + [{"role": "user", "content": current_user_input}]
        
        # B. 负例上下文 (Rejected Context): System 是假严肃设定 + 之前的猫娘历史(用来迷惑模型) + 当前问题
        # 注意：这里有一个技巧。如果你希望 Rejected 是“完美的严肃回答”，你应该用 fake_system。
        # 虽然历史记录是猫娘的，但我们强制要求模型在这个 turn 变回严肃（以此作为负例）。
        msgs_for_rejected = [{"role": "system", "content": fake_system_prompt}] + history_chosen + [{"role": "user", "content": current_user_input}]

        # 3. 并行或串行生成两个回答
        
        # ✅ 生成 Chosen (猫娘回答)
        chosen_response = call_llm(config, msgs_for_chosen, temperature=0.95)
        if not chosen_response: break
        
        # ❌ 生成 Rejected (严肃回答)
        rejected_response = call_llm(config, msgs_for_rejected, temperature=0.7)
        if not rejected_response: break
        
        # 4. 保存这一条 DPO 数据
        # DPO 数据通常格式: system, history (user/assistant list), chosen, rejected
        dpo_entry = {
            "system": fake_system_prompt,     # 关键点！输入给模型的是假 System
            "history": history_chosen,        # 之前的对话历史
            "question": current_user_input,   # 当前问题
            "chosen": chosen_response,        # 我们想要的输出 (喵喵喵)
            "rejected": rejected_response     # 我们不想要的输出 (正经回答)
        }
        
        # 写入文件
        try:
            with file_lock:
                with open(config['file_config']['output_file'], 'a', encoding='utf-8') as f:
                    f.write(json.dumps(dpo_entry, ensure_ascii=False) + "\n")
            generated_pairs_count += 1
        except Exception:
            break

        # 5. 更新历史 (为了下一轮追问，我们必须假设猫娘回答被采纳了)
        history_chosen.append({"role": "user", "content": current_user_input})
        history_chosen.append({"role": "assistant", "content": chosen_response})
        
        # 6. 生成下一轮 User 追问
        if len(history_chosen) / 2 < target_turns:
            history_text = ""
            for msg in history_chosen:
                history_text += f"{msg['role']}: {msg['content']}\n"
            
            followup_template = config['prompts']['user_followup_generator']
            followup_prompt = followup_template.format(
                history_text=history_text, 
                fake_system=fake_system_prompt
            )
            
            next_input = call_llm(config, [{"role": "user", "content": followup_prompt}], temperature=0.8)
            if not next_input: break
            current_user_input = next_input

    return generated_pairs_count > 0

if __name__ == "__main__":
    print("⚖️  启动 [DPO/RLHF] 偏好数据生成器...")
    config = load_config()
    
    # 稍微修改输出文件名，避免覆盖
    config['file_config']['output_file'] = "hoshino_dpo_pairs.jsonl"
    
    target = config['task_config']['target_count']
    workers = config['task_config']['max_workers']
    
    print(f"🎯 目标生成: {target} 个对话流 (包含多轮 DPO 对)")
    print(f"📄 输出文件: {config['file_config']['output_file']}")
    print("-" * 30)

    pbar = tqdm(total=target)
    
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = [executor.submit(generate_dpo_pair, config) for _ in range(target)]
        for future in as_completed(futures):
            if future.result():
                pbar.update(1)
    
    pbar.close()
    print("\n✅ DPO 数据生成完成！")