import torch
import os

# print(torch.cuda.device_count())  # 可用GPU数量
# print(torch.cuda.get_device_name(0))  # 第一块GPU名称

# print("CUDA_VISIBLE_DEVICES:", os.environ.get('CUDA_VISIBLE_DEVICES', 'Not set'))

print(f"CUDA available: {torch.cuda.is_available()}")          # 应为 True
print(f"Device count: {torch.cuda.device_count()}")           # 应 > 0
print(f"Current device: {torch.cuda.current_device()}")       # 应为 0
print(f"Device name: {torch.cuda.get_device_name(0)}")        # 应显示GPU型号

