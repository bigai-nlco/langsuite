import requests
import json
import logging
import os

r_path = "./data/iqa/reflexion_examples"
with open(r_path, 'r') as f:
    FEW_SHOT_EXAMPLES = f.read()

def get_scenario(text):
    lines = text.split("\n")
    data = ""
    for line in lines:
        # print(line)
        if line == "":
            continue
        line = json.loads(line)
        if "role" in line and line["role"] == "system":
            data += "System: "
        elif "role" in line and line["role"] == "assistant":

            data += "Assistant: " 
            data += line["action"]+"> "
        elif "role" in line and line["role"] == "user":
            data += "User: " 
        data += line["content"]+"\n"
    return data

def get_reflection_query(text):
    scenario: str = get_scenario(text)
    #query: str = f"""You will be given the history of a past experience in which you were placed in an environment and given a task to complete. You were unsuccessful in completing the task. Do not summarize your environment, but rather think about the strategy you took to attempt to complete the task. Devise a new plan that accounts for your mistake. For example, if you tried A and B but forgot C, then devise a plan to achieve C. You will need this later when you are solving the same task. Summarize your general plan in a few sentences as the examples shows and do not list actions as plan. Here are two examples:
    
    query: str = f"""You will be given the history of a past experience in which you were placed in an environment and given a task to complete. You were unsuccessful in completing the task. Do not summarize your environment, but rather think about the strategy and path you took to attempt to complete the task. Devise a concise, new plan of action that accounts for your mistake with reference to specific actions that you should have taken. For example, if you tried A and B but forgot C, then devise a plan to achieve C with environment-specific actions. You will need this later when you are solving the same task. Give your plan after "New Plan: ". Here are two examples:
    

{FEW_SHOT_EXAMPLES}

{scenario}"""
    query += '\n\nNew plan:'
    return query

def send_request(kw):
    # 替换为自己的KEY
    api_key = ""
    try:
        api_url = 'https://one.aiskt.com/v1/chat/completions'
        # 设置请求头部，包括 API 密钥
        headers = {
            'Authorization': f'Bearer {api_key}',
            'Content-Type': 'application/json'
        }
        # 准备请求的数据
        payload = {
            'model': "gpt-3.5-turbo-16k",
            'messages': [{"role": "system", "content": str(kw)}]
        }
        # 发送 POST 请求
        response = requests.post(api_url, headers=headers, data=json.dumps(payload))
        print(response)
        # 检查响应状态
        if response.status_code == 200:
            # 解析响应并提取需要的信息
            data = response.json()
            return data['choices'][0]['message']['content']
        else:
            return f'Error: Received status code {response.status_code}'
    except Exception as e:
        logging.info(e)
        return 'An error occurred while sending the request'
    
    
folder_name = "./iqa_data_folder/CoT-high-level-0/console-logs/"
memory_path = "./iqa_data_folder/Reflexion-high-level-0/memory.txt"
#folder_name = "./babyai_gpt3.5_react_emmem/"
#memory_path = "./babyai_gpt3.5_reflexion_emmem/save/memory.txt"
skip_path = "./langsuite/envs/iqa/reflexion_history.txt"
count = 0
files = os.listdir(folder_name)
skip_file = open(skip_path, 'r+')
skip_list = set(l.strip() for l in skip_file.readlines())

for fn in os.listdir(folder_name):
    if not fn.endswith('.jl'):
        continue
    memories = {}
    # if fn.split("--")[0] in skip_list:
    #     print("skip", fn)
    #     continue
    with open(folder_name + fn, "r") as file:
        text = file.read()
        if not 'DONE!' in text:
            print("not done", fn)
            continue
        if not "[SUCCESS]" in text:
            print(f'runing for {fn}')
            prompt = get_reflection_query(text)
            # print(prompt)
            memory = send_request(prompt)
            # memory = memory.split("Plan:")[0]
            print(memory)
            memories[fn.split("--")[0]] = memory
    with open (memory_path,'a+') as f:
        f.write(json.dumps(memories)+'\n')
    skip_list.add(fn.split("--")[0])
    skip_file.write(fn.split("--")[0]+'\n')
    skip_file.flush()

#skip_file.write("\n".join(skip_list))
skip_file.close()
