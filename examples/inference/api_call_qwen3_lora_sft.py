from openai import OpenAI
import time
import numpy as np

def chat(user_sent):
    start_timer = time.time() * 1000
    client = OpenAI(api_key="0", base_url="http://0.0.0.0:8000/v1")
    messages = [{"role": "user", "content": user_sent}]
    result = client.chat.completions.create(
        messages=messages, model="/aiplatform-sale/aiplatform-sale/group-shared/models/Qwen/Qwen3-1.7B")
    print(result.choices[0].message)
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


if __name__ == "__main__":
    cal_avg_time()
