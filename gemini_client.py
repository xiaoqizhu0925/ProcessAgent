import google.generativeai as genai
from typing import Dict, Any, List

class GeminiChatCompletionClient:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        genai.configure(api_key=config['api_key'])
        
        # 创建模型实例
        self.model = genai.GenerativeModel(
            model_name=config['model'],
            generation_config={
                'temperature': config['model_info'].get('temperature', 0.7),
                'top_p': config['model_info'].get('top_p', 0.95),
                'max_output_tokens': config['model_info'].get('max_tokens', 2048),
            }
        )
        print(f"初始化模型: {config['model']}")
    
    def create_chat_completion(self, messages: List[Dict[str, str]]) -> Dict[str, Any]:
        """同步方式创建聊天完成"""
        try:
            # 转换消息格式
            prompt = self._convert_messages_to_prompt(messages)
            
            # 生成响应
            response = self.model.generate_content(prompt)
            
            # 确保响应有效
            if not response or not response.text:
                raise RuntimeError("生成的响应为空")
                
            return {
                "choices": [{
                    "message": {
                        "content": response.text,
                        "role": "assistant"
                    }
                }]
            }
            
        except Exception as e:
            print(f"生成内容时出错: {str(e)}")
            raise
    
    def _convert_messages_to_prompt(self, messages: List[Dict[str, str]]) -> str:
        """将消息列表转换为提示字符串"""
        return "\n".join(f"{msg['role']}: {msg['content']}" for msg in messages)