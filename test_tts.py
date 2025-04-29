import requests

# 1. 设置 API 地址和参数
url = "http://localhost:7860/tts"
headers = {"Content-Type": "application/json"}
data = {
    "text": "这是用 Python 生成的语音",
    "model_name": "wo",  # 根据实际模型名称修改
    "speaker_id": 0,          # 可选参数（如果有）
    "speed": 1.0              # 语速（1.0 为正常）
}

# 2. 发送请求
response = requests.post(url, json=data, headers=headers)

# 3. 处理响应
if response.status_code == 200:
    # 保存音频文件
    with open("python_output.wav", "wb") as f:
        f.write(response.content)
    print("成功！音频已保存为 python_output.wav")
else:
    print(f"失败！状态码：{response.status_code}, 错误信息：{response.text}")