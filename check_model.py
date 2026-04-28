import torch

checkpoint = torch.load("models/best_model.pth", map_location='cpu')
print("=" * 60)
print("BEST MODEL INFO")
print("=" * 60)
print(f"Epoch: {checkpoint.get('epoch', 'N/A')}")
print(f"Validation Accuracy: {checkpoint.get('val_acc', 'N/A'):.2f}%")
print(f"Model: {checkpoint.get('model_name', 'N/A')}")
print("=" * 60)
