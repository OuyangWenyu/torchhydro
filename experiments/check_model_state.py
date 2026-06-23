import torch

pth_path = "/Users/cylenlc/work/torchhydro/experiments/best_model.pth"  # 修改成你的文件路径

state = torch.load(pth_path, map_location="cpu")

# 若保存时用了类似 {'state_dict': xxx} 的封装
if isinstance(state, dict):
    if "state_dict" in state:
        state = state["state_dict"]
    elif "model" in state:
        state = state["model"]

# 打印所有键名
print("✅ Keys in the checkpoint:")
for k in state.keys():
    print(k)
