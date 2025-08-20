import torch
import torch.nn.functional as F
from torchvision import transforms
from torchvision.models import resnet18, ResNet18_Weights
from PIL import Image
import numpy as np
import os
from tqdm import tqdm
from typing import Dict

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load ResNet18 feature extractor (removes avgpool and fc layers)
class ResNetEncoder(torch.nn.Module):
    def __init__(self):
        super().__init__()
        weights = ResNet18_Weights.IMAGENET1K_V1
        resnet = resnet18(weights=weights)
        self.encoder = torch.nn.Sequential(*list(resnet.children())[:-2])  # All layers except last 2

    def forward(self, x):
        with torch.no_grad():
            features = self.encoder(x)
            pooled = F.adaptive_avg_pool2d(features, (1, 1)).squeeze()  # Global avg pool
        return pooled

# Transform
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor()
])

# Extract feature vector
def extract_feature_vector(image_path: str, model: ResNetEncoder) -> torch.Tensor:
    image = Image.open(image_path).convert("RGB")
    input_tensor = transform(image).unsqueeze(0).to(device)
    embedding = model(input_tensor).detach().cpu()
    return embedding.squeeze()  # Shape: [512]

# Cosine similarity
def compute_similarity(query_vec: torch.Tensor, support_dict: Dict[str, torch.Tensor]) -> str:
    best_score = -float("inf")
    best_filename = None
    for fname, vec in support_dict.items():
        score = F.cosine_similarity(query_vec.unsqueeze(0), vec.unsqueeze(0)).item()
        if score > best_score:
            best_score = score
            best_filename = fname
    return best_filename

# Build embedding dictionary from test case folder
def build_support_set_embeddings(test_case_folder: str) -> Dict[str, torch.Tensor]:
    encoder = ResNetEncoder().to(device).eval()
    embeddings = {}
    for fname in tqdm(os.listdir(test_case_folder), desc="Building test case embeddings"):
        if fname.lower().endswith((".jpg", ".jpeg", ".png")):
            path = os.path.join(test_case_folder, fname)
            # embeddings[fname] = extract_feature_vector(path, encoder)
            try:
                vec = extract_feature_vector(path, encoder)
                embeddings[fname] = vec
            except Exception as e:
                print(f"[❌ ERROR] Failed to extract embedding from: {fname} — {e}")
                continue
    
    return embeddings

def compute_topk_similar(query_vec: torch.Tensor, support_dict: Dict[str, torch.Tensor], k: int = 3):
    similarities = []
    for fname, vec in support_dict.items():
        score = F.cosine_similarity(query_vec.unsqueeze(0), vec.unsqueeze(0)).item()
        similarities.append((score, fname))
    similarities.sort(reverse=True, key=lambda x: x[0])
    return [fname for _, fname in similarities[:k]]