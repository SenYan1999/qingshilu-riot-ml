import sys
sys.path.append('/home/senyan/qingshilu-riot-ml')

import pickle
import json
import pandas as pd
from tqdm import tqdm
from openai import OpenAI
from pydantic import BaseModel

OPENAI_API_KEY = "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"
client = OpenAI(api_key=OPENAI_API_KEY,)

def upload_batch_and_run(start_i):
    class RiotType(BaseModel):
        riot_type: str

    with open('data/data.pkl', 'rb') as f: data = pickle.load(f).entries

    def generate_qingshilu_classification_prompt(entry_text: str) -> str:
        categories = {
            "peasant_riot": "因灾荒、赋税、地租等问题引发的农民群体性抗争",
            "sectarian_event": "与秘密结社、宗教性集会或会党有关的事件",
            "armed_rebellion": "有组织的武装造反或起义行为",
            "other_riot": "非农民起义，秘密结社起义，武装起义之外的暴动，如学生骚乱、矿工罢工等非典型暴动",
            "non_riot": "非暴力冲突起义事件",
        }

        category_list = "\n".join([f"- {key}: {desc}" for key, desc in categories.items()])

        prompt = f"""你是一个历史学者，请根据下面清实录中的一段历史事件记录，判断它最可能属于哪一类社会事件。类别如下：

    {category_list}

    对不同暴乱起义的评判标准不要太严格，相关即可。请只输出最合适的类别名称（例如：peasant_riot），不要输出其他任何内容。

    事件记录如下：
    {entry_text}
    """
        return prompt

    batch_requests = []
    for i, line in tqdm(enumerate(data[start_i::8])):
        messages = [{'role': 'assistant',
                        'content': '你是一个历史学者，请根据我的要求对历史文本进行分类'}]
        current_messages = messages + [{'role': 'user', 'content': generate_qingshilu_classification_prompt(entry_text=line['entry'])}]
        batch_requests.append({"custom_id": str(start_i + i*8), "method": "POST", "url": "/v1/chat/completions", "body": {"model": 'gpt-4o-mini', "messages": current_messages}})

    with open("temp/batch_requests.jsonl", "w") as f:
        for request in batch_requests:
            f.write(json.dumps(request) + "\n")

    batch_input_file = client.files.create(
        file=open("temp/batch_requests.jsonl", "rb"),
        purpose="batch"
    )
    batch_job = client.batches.create(
        input_file_id=batch_input_file.id,
        endpoint="/v1/chat/completions",
        completion_window="24h",
        metadata={
            "description": "nightly eval job"
        }
    )

def parse_response():
    from glob import glob
    parsed_output = {}
    for filename in glob('temp/batch*'):
        with open(filename, 'r') as f:
            output = [json.loads(l) for l in f.readlines()]
        for line in output:
            idx = str(line['custom_id'])
            response = line['response']['body']['choices'][0]['message']['content']
            parsed_output[idx] = response
    with open('temp/parsed_output.json', 'w') as f:
        json.dump(parsed_output, f)

def identify_others():
    with open('data/data.pkl', 'rb') as f: data = pickle.load(f).entries
    with open('temp/parsed_output.json', 'r') as f:
        parsed_output = json.load(f)
    lines = []
    for key, line in parsed_output.items():
        if 'other_riot' == line:
            lines.append(data[int(key.split('-')[-1])]['entry'])

    def generate_qingshilu_classification_prompt(entry_text: str) -> str:
        categories = {
            "peasant_riot": "因灾荒、赋税、地租等问题引发的农民群体性抗争",
            "sectarian_event": "与秘密结社、宗教性集会或会党有关的事件",
            "armed_rebellion": "有组织的武装造反或起义行为",
            "other_riot": "非农民起义，秘密结社起义，武装起义之外的暴动，如学生骚乱、矿工罢工等非典型暴动",
            "non_riot": "非暴力冲突起义事件",
        }

        category_list = "\n".join([f"- {key}: {desc}" for key, desc in categories.items()])

        prompt = f"""你是一个历史学者，请根据下面清实录中的一段历史事件记录，判断它最可能属于哪一类社会事件。类别如下：

    {category_list}

    对不同暴乱起义的评判标准不要太严格，相关即可。请只输出最合适的类别名称（例如：peasant_riot），不要输出其他任何内容。

    事件记录如下：
    {entry_text}
    """
        return prompt

    batch_requests = []
    for i, line in tqdm(enumerate(lines)):
        messages = [{'role': 'assistant',
                        'content': '你是一个历史学者，请根据我的要求对历史文本进行分类'}]
        current_messages = messages + [{'role': 'user', 'content': generate_qingshilu_classification_prompt(entry_text=line)}]
        batch_requests.append({"custom_id": str(i), "method": "POST", "url": "/v1/chat/completions", "body": {"model": 'gpt-4o-mini', "messages": current_messages}})

    with open("temp/batch_requests.jsonl", "w") as f:
        for request in batch_requests:
            f.write(json.dumps(request) + "\n")

    batch_input_file = client.files.create(
        file=open("temp/batch_requests.jsonl", "rb"),
        purpose="batch"
    )
    batch_job = client.batches.create(
        input_file_id=batch_input_file.id,
        endpoint="/v1/chat/completions",
        completion_window="24h",
        metadata={
            "description": "nightly eval job"
        }
    )

def identify_others_final():
    with open('data/data.pkl', 'rb') as f: data = pickle.load(f).entries
    with open('temp/parsed_output.json', 'r') as f:
        parsed_output = json.load(f)
    lines = []
    for key, line in parsed_output.items():
        if 'other_riot' == line:
            lines.append(data[int(key.split('-')[-1])]['entry'])

    parsed_output = {}
    with open('temp/final_response.jsonl', 'r') as f:
        output = [json.loads(l) for l in f.readlines()]
    for line in output:
        idx = str(line['custom_id'])
        response = line['response']['body']['choices'][0]['message']['content']
        if 'other' in response:
            parsed_output[idx] = (response, lines[int(idx)])
    
    def generate_qingshilu_classification_prompt(entry_text: str) -> str:
        categories = {
            "peasant_riot": "因灾荒、赋税、地租等问题引发的农民群体性抗争",
            "sectarian_event": "与秘密结社、宗教性集会或会党有关的事件",
            "armed_rebellion": "有组织的武装造反或起义行为",
            "other_riot": "非农民起义，秘密结社起义，武装起义之外的暴动，如学生骚乱、矿工罢工等非典型暴动",
            "non_riot": "非暴力冲突起义事件",
        }

        category_list = "\n".join([f"- {key}: {desc}" for key, desc in categories.items()])

        prompt = f"""你是一个历史学者，请根据下面清实录中的一段历史事件记录，判断它最可能属于哪一类社会事件。类别如下：

    {category_list}

    请只输出最合适的类别名称（例如：peasant_riot），不要输出其他任何内容。

    事件记录如下：
    {entry_text}
    """
        return prompt

    batch_requests = []
    for i, line in tqdm(parsed_output.items()):
        messages = [{'role': 'assistant',
                        'content': '你是一个历史学者，请根据我的要求对历史文本进行分类'}]
        current_messages = messages + [{'role': 'user', 'content': generate_qingshilu_classification_prompt(entry_text=line)}]
        batch_requests.append({"custom_id": str(i), "method": "POST", "url": "/v1/chat/completions", "body": {"model": 'gpt-4o-mini', "messages": current_messages}})

    with open("temp/batch_requests_final.jsonl", "w") as f:
        for request in batch_requests:
            f.write(json.dumps(request) + "\n")

    batch_input_file = client.files.create(
        file=open("temp/batch_requests_final.jsonl", "rb"),
        purpose="batch"
    )
    batch_job = client.batches.create(
        input_file_id=batch_input_file.id,
        endpoint="/v1/chat/completions",
        completion_window="24h",
        metadata={
            "description": "nightly eval job"
        }
    )

def identify_others_final_final():
    with open('data/data.pkl', 'rb') as f: data = pickle.load(f).entries
    with open('temp/parsed_output.json', 'r') as f:
        parsed_output = json.load(f)
    lines = []
    for key, line in parsed_output.items():
        if 'other_riot' == line:
            lines.append(data[int(key.split('-')[-1])]['entry'])

    parsed_output = {}
    with open('temp/response_final_final.jsonl', 'r') as f:
        output = [json.loads(l) for l in f.readlines()]
    for line in output:
        idx = str(line['custom_id'])
        response = line['response']['body']['choices'][0]['message']['content']
        if 'other' in response:
            parsed_output[idx] = (response, lines[int(idx)])   
    
    responses, labels = [], []
    for value in parsed_output.values():
        response, label = value
        if '叛' in label or '乱' in label:
            responses.append(response)
            labels.append(label)
    
    with open('data/sixclasses/others.txt', 'w') as f:
        for line in labels:
            f.write(line)
            f.write('\n')
    
    pd.DataFrame({'label': responses, 'entry': labels}).sample(frac=1, random_state=42).to_csv('temp/riot.csv', index=False)

if __name__ == '__main__':
    '''
    for i in range(0, 8):
        upload_batch_and_run(i)
    '''
    # parse_response()
    # identify_others()
    identify_others_final_final()
