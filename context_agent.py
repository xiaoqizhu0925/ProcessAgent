import yaml
import json
import re
from typing import Dict
from pathlib import Path
from gemini_client import GeminiChatCompletionClient


def extract_json_from_response(content: str) -> str:
    """从响应中提取 JSON 字符串"""
    # 查找 JSON 代码块
    json_match = re.search(r'```json\n(.*?)\n```', content, re.DOTALL)
    if json_match:
        return json_match.group(1)
    
    # 如果没有代码块标记，尝试直接查找 JSON 对象
    json_match = re.search(r'(\{.*\})', content, re.DOTALL)
    if json_match:
        return json_match.group(1)
    
    return content


def generate_context(model_config: Dict, context_config: Dict, iteration: int = 1):
    """生成上下文"""
    try:
        # 初始化客户端
        model_client = GeminiChatCompletionClient(model_config)
        
        # 加载提示配置
        prompt_path = Path(context_config["context_agent_prompt_path"])
        if not prompt_path.exists():
            raise FileNotFoundError(f"提示配置文件不存在: {prompt_path}")
            
        with open(prompt_path, "r", encoding='utf-8') as f:
            prompt_config = yaml.safe_load(f)
            
        # 验证提示配置
        required_keys = ["system", "user"]
        missing_keys = [key for key in required_keys if key not in prompt_config]
        if missing_keys:
            raise KeyError(f"提示配置缺少必要的键: {missing_keys}")
        
        # 准备消息，添加明确的 JSON 格式要求
        system_prompt = prompt_config["system"].strip() + "\n请直接返回JSON，不要添加任何其他文本。"
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt_config["user"].strip()}
        ]
        
        # 获取响应
        response = model_client.create_chat_completion(messages)
        content = response["choices"][0]["message"]["content"]
        
        # 提取 JSON
        json_content = extract_json_from_response(content)
        
        try:
            result = json.loads(json_content)
            # 验证结果格式
            if not isinstance(result, dict) or "process_overview" not in result or "constraints" not in result:
                raise ValueError("返回的 JSON 缺少必要的字段")
        except json.JSONDecodeError as e:
            print("JSON 解析错误:", str(e))
            print("原始响应:", content)
            print("提取的 JSON:", json_content)
            raise RuntimeError("LLM 返回格式无效")
        
        # 保存结果
        save_results(result, context_config, iteration)
        
        return result
        
    except Exception as e:
        print(f"生成上下文时出错: {str(e)}")
        raise


def save_results(result: Dict, config: Dict, iteration: int):
    """保存生成的结果"""
    overview_path = config['llm_process_overview_save_path']
    constraint_path = _append_suffix_to_path(config['llm_constraint_save_path'], iteration)

    Path(overview_path).parent.mkdir(parents=True, exist_ok=True)
    with open(overview_path, "w", encoding="utf-8") as f:
        f.write(result["process_overview"].strip() + "\n")

    constraint_lines = [
        f'{c["variable"]}: [{c["range"][0]} {c["unit"]}, {c["range"][1]} {c["unit"]}]'
        for c in result["constraints"]
    ]
    
    Path(constraint_path).parent.mkdir(parents=True, exist_ok=True)
    with open(constraint_path, "w", encoding="utf-8") as f:
        f.write("\n".join(constraint_lines) + "\n")

    print(f"Overview   → {config['llm_process_overview_save_path']}")
    print(f"Constraints→ {constraint_path}")


def _append_suffix_to_path(original_path: str, suffix: int) -> str:
    path = Path(original_path)
    new_name = f"{path.stem}_{suffix}{path.suffix}"
    return str(path.with_name(new_name))

