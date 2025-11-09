import google.generativeai as genai
import sys

def test_gemini():
    try:
        # 显示库版本
        print(f"Google GenerativeAI 版本: {genai.__version__}")
        
        # 配置 API
        genai.configure(api_key='AIzaSyCNfGYhNi7AXQXWmXYEaG9s2UodPlHj48o')
        
        # 获取可用模型列表
        models = list(genai.list_models())
        print("\n可用的 Gemini Pro 模型:")
        gemini_models = [m for m in models if 'gemini' in m.name.lower()]
        for m in gemini_models:
            print(f"- {m.name}")
        
        # 使用最新的 gemini-pro-latest 模型
        model = genai.GenerativeModel(model_name='models/gemini-pro-latest',
                                    generation_config={
                                        'temperature': 0.9,
                                        'top_p': 1,
                                        'top_k': 1,
                                        'max_output_tokens': 2048,
                                    })
        
        # 测试基本功能
        response = model.generate_content('Hello, please confirm if API is working.')
        
        print("\nAPI 测试结果:")
        print(f"响应内容: {response.text}")
        print("API 工作正常！")
        
        return True

    except Exception as e:
        print(f"错误类型: {type(e).__name__}")
        print(f"错误详情: {str(e)}")
        print(f"Python 版本: {sys.version}")
        return False

if __name__ == "__main__":
    # 运行测试
    test_gemini()