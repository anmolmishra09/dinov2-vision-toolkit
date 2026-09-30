import torch

# Load a small DINOv2 model
print("Loading DINOv2 ViT-S/14 model...")
model = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14")
model.eval()

# Create a dummy image (224x224 RGB)
dummy_image = torch.randn(1, 3, 224, 224)

# Extract features
with torch.no_grad():
    features = model(dummy_image)

print(f"Feature shape: {features.shape}")
print("Success! DINOv2 is working.")
