# Copyright (c) BIGAI Research. All rights reserved.
# Licensed under the MIT license.
from __future__ import annotations
import json
from logging import config
import requests

import yaml

def read_config(config_path):
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f.read())
        return config

class LLM_gpt35:
    config = read_config('api.config.yml')
    api_key = config[0]['openai_api_key']
    api_url = config[0]['openai_proxy']

    @classmethod
    def fetch(cls, messages):
        # 替换为自己的KEY
        messages = [{"role": message["role"], "content": message["content"]} for message in messages]
        # print("messages is !!! ", messages)
        try:
            # 设置请求头部，包括 API 密钥
            headers = {
                'Authorization': f'Bearer {cls.api_key}',
                'Content-Type': 'application/json'
            }
            # 准备请求的数据
            payload = {
                'model': "gpt-3.5-turbo-16k",
                'messages': messages,
                # 'temperature': 1.0
            }
            # 发送 POST 请求
            response = requests.post(cls.api_url, headers=headers, data=json.dumps(payload))
            # 检查响应状态
            if response.status_code == 200:
                # 解析响应并提取需要的信息
                data = response.json()
        #            print("----***")
        #            print(data)
                return data['choices'][0]['message']['content']
            else:
                return f'Error: Received status code {response.status_code}'
        except Exception as e:
            return 'An error occurred while sending the request'