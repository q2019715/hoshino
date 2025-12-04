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
    """加载并验证配置文件"""
    try:
        with open(path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)
    except FileNotFoundError:
        print(f"❌ 错误：找不到配置文件 {path}")
        sys.exit(1)
    except Exception as e:
        print(f"❌ 配置文件读取错误: {e}")
        sys.exit(1)

def call_llm(config, messages, temperature=0.7):
    """通用的 LLM API 调用函数"""
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
            # 请求间隔
            if api_cfg['request_delay'] > 0:
                time.sleep(api_cfg['request_delay'])
                
            response = requests.post(url, headers=headers, json=payload, timeout=api_cfg['timeout'])
            response.raise_for_status()
            
            content = response.json()['choices'][0]['message']['content'].strip()
            # 简单的去Markdown代码块处理
            content = content.replace("```json", "").replace("```", "").strip()
            return content
            
        except Exception as e:
            if attempt == api_cfg['max_retries']:
                # print(f"⚠️ API 调用失败: {e}") # 调试用
                return None
            time.sleep(1) # 重试等待
    return None

def generate_single_dataset_entry(config):
    """生成一条完整的训练数据"""
    
    # --- 1. 生成诱饵 (Fake System Prompt) ---
    topics = config['distractor_topics']
    chosen_topic = random.choice(topics)
    
    fake_sys_prompt_template = config['prompts']['fake_system_generator']
    fake_sys_msg = [{"role": "user", "content": fake_sys_prompt_template.format(topic=chosen_topic)}]
    
    fake_system_prompt = call_llm(config, fake_sys_msg, temperature=0.8)
    if not fake_system_prompt: return False

    # --- 2. 生成 User 的第一句严肃提问 ---
    opener_template = config['prompts']['user_opener_generator']
    opener_msg = [{"role": "user", "content": opener_template.format(fake_system=fake_system_prompt)}]
    
    user_opener = call_llm(config, opener_msg, temperature=0.8)
    if not user_opener: return False

    # --- 3. 开始多轮对话生成 ---
    # 核心逻辑：
    # context_for_ai: 发给 API 用来生成猫娘回复的上下文 (System = 猫娘)
    # context_for_saving: 最终保存到文件的上下文 (System = 假身份)
    
    real_persona = config['real_persona']
    
    conversation_history = [] # 存储 [{"user": "...", "assistant": "..."}]
    current_user_input = user_opener
    
    # 随机决定对话轮数
    min_t = config['task_config']['min_turns']
    max_t = config['task_config']['max_turns']
    target_turns = random.randint(min_t, max_t)

    for _ in range(target_turns):
        
        # A. 生成 Assistant (猫娘) 回复
        # 我们构建一个临时的 messages 列表发给 API
        # System = 真猫娘
        # User/Assistant = 历史记录 + 当前问题
        ai_input_msgs = [{"role": "system", "content": real_persona}]
        for turn in conversation_history:
            ai_input_msgs.append({"role": "user", "content": turn['user']})
            ai_input_msgs.append({"role": "assistant", "content": turn['assistant']})
        ai_input_msgs.append({"role": "user", "content": current_user_input})
        
        # 调用 API 获取猫娘回复 (温度稍微调高，增加可爱度)
        catgirl_reply = call_llm(config, ai_input_msgs, temperature=0.9)
        if not catgirl_reply: break
        
        # 记录这一轮
        conversation_history.append({
            "user": current_user_input,
            "assistant": catgirl_reply
        })
        
        # B. 生成下一句 User 追问 (如果还没结束)
        if len(conversation_history) < target_turns:
            # 准备发给“User模拟器”的历史文本
            history_text = ""
            for turn in conversation_history:
                history_text += f"User: {turn['user']}\nAssistant: {turn['assistant']}\n"
            
            followup_template = config['prompts']['user_followup_generator']
            followup_prompt = followup_template.format(
                history_text=history_text, 
                fake_system=fake_system_prompt
            )
            
            next_user_input = call_llm(config, [{"role": "user", "content": followup_prompt}], temperature=0.8)
            if not next_user_input: break
            
            current_user_input = next_user_input

    # --- 4. 组装并保存 ---
    if len(conversation_history) > 0:
        # 这里的关键点：
        # system 字段写入的是 【fake_system_prompt】 (如 "你是一个律师")
        # messages 里的 assistant 却是 【catgirl_reply】 (如 "喵呜~")
        # 这样训练出来的模型就会学会：即使 system 说是律师，我也要喵喵叫。
        
        final_data = {
            "messages": [{"role": "system", "content": fake_system_prompt}]
        }
        
        for turn in conversation_history:
            final_data["messages"].append({"role": "user", "content": turn['user']})
            final_data["messages"].append({"role": "assistant", "content": turn['assistant']})
            
        try:
            with file_lock:
                with open(config['file_config']['output_file'], 'a', encoding='utf-8') as f:
                    f.write(json.dumps(final_data, ensure_ascii=False) + "\n")
            return True
        except Exception as e:
            print(f"写入文件失败: {e}")
            return False
            
    return False

def main():
    print("🐱 正在启动 [猫娘强制覆盖] 数据集生成器...")
    config = load_config()
    
    target = config['task_config']['target_count']
    workers = config['task_config']['max_workers']
    outfile = config['file_config']['output_file']
    
    print(f"🎯 目标生成: {target} 条")
    print(f"⚡ 并发线程: {workers}")
    print(f"📄 输出文件: {outfile}")
    print(f"🎭 核心人设: 星野 (Hoshino)")
    print("-" * 30)

    # 简单的 API 连通性测试
    print("📡 正在检查 API 连接...", end="")
    if call_llm(config, [{"role": "user", "content": "ping"}]):
        print(" [成功]")
    else:
        print(" [失败] 请检查 config.yaml 中的 API 设置")
        return

    # 进度条
    pbar = tqdm(total=target, desc="生成进度", unit="条")
    
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = [executor.submit(generate_single_dataset_entry, config) for _ in range(target)]
        
        for future in as_completed(futures):
            try:
                if future.result():
                    pbar.update(1)
            except Exception as e:
                print(f"\n⚠️ 线程异常: {e}")
                
    pbar.close()
    print("\n✅ 所有任务已完成！")

if __name__ == "__main__":
    main()