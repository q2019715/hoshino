import json
import requests
import yaml
import re
import time
import math
from tqdm import tqdm

# ================= ⚙️ 用户配置区域 =================

# 1. 你想要多少条种子数据？(脚本会自动计算循环次数)
TARGET_TOTAL_COUNT = 200  

# 2. 单次 API 请求让模型生成几条？
# 建议保持在 5-10 之间。太少效率低，太多模型容易偷懒或质量下降。
BATCH_SIZE = 5            

# ====================================================

def load_config():
    try:
        with open("config.yaml", "r", encoding="utf-8") as f:
            return yaml.safe_load(f)
    except Exception:
        print("❌ 没找到 config.yaml，请检查文件位置")
        exit()

config = load_config()
API_URL = config['api_config']['base_url']
API_KEY = config['api_config']['api_key']
MODEL = config['api_config']['model_name']

# 场景列表 (你可以随时在这个列表里加新的场景)
SCENARIO_TOPICS = [
    "初次见面/打破僵局",
    "日常生活(叫醒/做饭/洗澡)",
    "肢体接触(摸头/抓尾巴/膝枕)",
    "负面情绪(主人心情差/求安慰)",
    "冲突争吵(猫娘闯祸/主人晚归/弄坏东西)",
    "吃醋/占有欲(提到别的猫/别的女生)",
    "生理生病(发烧/受伤/去医院打针)",
    "恐惧场景(打雷/停电/看恐怖片/蟑螂)",
    "外出互动(散步/买零食/遇到狗)",
    "特殊节日(生日/新年/情人节/圣诞节)",
    "角色扮演(假装是陌生人/医生病人游戏)",
    "脑洞假设(思考未来/世界末日)",
    "学习与认知(教猫娘认字/解释复杂概念)",
    "离别与重逢(出差回来/假装遗弃)",
    "羞耻/隐私(偷看日记/换衣服被撞见)"
]

def call_llm_for_seeds(topic, batch_size):
    """请求 LLM 生成特定主题的种子"""
    prompt = f"""
    你是一个专业的对话数据集构建者。请为“猫娘星野”这个角色扮演模型，生成 {batch_size} 条属于【{topic}】这个特定场景的用户输入(User Input)。

    【要求】
    1. 只要【用户说的话】，不要包含猫娘的回答。
    2. 必须要体现出【{topic}】这个主题的特点。
    3. 每次生成的内容要尽量多样化，不要重复之前的套路。
    4. 返回格式必须是纯 JSON 字符串列表。

    【示例】
    [
      "星野，我也许再也不回来了。",
      "看！我给你带了什么好吃的？是刚出炉的烤鱼哦！"
    ]
    """

    headers = {
        "Content-Type": "application/json", 
        "Authorization": f"Bearer {API_KEY}"
    }
    payload = {
        "model": MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.85 # 温度稍微调高，保证多轮生成时不重复
    }

    try:
        response = requests.post(API_URL, headers=headers, json=payload, timeout=30)
        if response.status_code != 200:
            return []
            
        content = response.json()['choices'][0]['message']['content']
        content = re.sub(r'```json\s*', '', content)
        content = re.sub(r'```', '', content)
        
        try:
            seeds = json.loads(content.strip())
            if isinstance(seeds, list):
                return seeds
        except json.JSONDecodeError:
            return []
            
    except Exception:
        return []
    
    return []

if __name__ == "__main__":
    all_unique_seeds = set()
    
    # --- 自动计算逻辑 ---
    num_topics = len(SCENARIO_TOPICS)
    # 每个主题总共需要贡献多少条
    needed_per_topic = math.ceil(TARGET_TOTAL_COUNT / num_topics) 
    # 每个主题需要请求几轮 API
    rounds_per_topic = math.ceil(needed_per_topic / BATCH_SIZE)   

    print(f"🚀 任务目标: 生成 {TARGET_TOTAL_COUNT} 条种子")
    print(f"📊 策略: 共 {num_topics} 个场景，每个场景生成约 {needed_per_topic} 条")
    print(f"🔄 循环: 每个场景将请求 {rounds_per_topic} 轮，每轮 {BATCH_SIZE} 条\n")

    # 进度条总数 = 场景数 * 轮数
    pbar = tqdm(total=num_topics * rounds_per_topic, desc="生成进度")

    for topic in SCENARIO_TOPICS:
        for _ in range(rounds_per_topic):
            seeds = call_llm_for_seeds(topic, BATCH_SIZE)
            if seeds:
                for s in seeds:
                    all_unique_seeds.add(s)
            
            pbar.update(1)
            time.sleep(0.5) # 稍微歇一下防限流

    pbar.close()
    
    final_seed_list = sorted(list(all_unique_seeds))

    output_file = "seed_questions.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(final_seed_list, f, ensure_ascii=False, indent=2)

    print(f"\n✅ 完成！实际获得不重复种子: {len(final_seed_list)} 条")
    print(f"💾 已保存至 {output_file}")