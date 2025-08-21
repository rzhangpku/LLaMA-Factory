from openai import OpenAI
import time
import numpy as np


def get_prompt_gen(fn) -> str:
    if not fn:
        return ""
    with open(fn, "r") as fr:
        prompt_gen = fr.read()
    return prompt_gen


system_prompt = get_prompt_gen(
    "/root/ai-sale-tutor/scripts/spoken_rewrite/data/instruction_prompt.md")


def chat(user_sent):
    start_timer = time.time() * 1000
    client = OpenAI(api_key="0", base_url="http://0.0.0.0:8000/v1")
    messages = [{"role": "system", "content": system_prompt}] + \
        [{"role": "user", "content": user_sent}]
    result = client.chat.completions.create(
        messages=messages, model="/aiplatform-sale/aiplatform-sale/group-shared/models/Qwen/Qwen3-1.7B")
    content = result.choices[0].message.content
    # print(content)
    end_timer = time.time() * 1000
    elapse_time = end_timer - start_timer
    return elapse_time


def cal_avg_time():
    elapse_time_list = []
    with open("examples/inference/user_sents.txt", "r") as f:
        user_sents = f.readlines()
        for user_sent in user_sents:
            elapse_time = chat(user_sent.strip())
            elapse_time_list.append(elapse_time)
    print(f"avg_time: {np.mean(elapse_time_list)}")
    print(f"median_time: {np.median(elapse_time_list)}")


if __name__ == "__main__":
    cal_avg_time()
