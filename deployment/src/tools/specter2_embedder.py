from transformers import AutoTokenizer, AutoModel
import torch
import numpy as np
from tqdm import tqdm

# Load the tokenizer and model once
tokenizer = AutoTokenizer.from_pretrained("allenai/specter2_base")  # Initialize tokenizer
model = AutoModel.from_pretrained("allenai/specter2_base")  # Initialize model
model.eval()  # Set model to evaluation mode
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # Use GPU if available
model.to(device)  # Move model to device

def embed_texts_specter2(texts: list[str], batch_size=16) -> np.ndarray:
    embeddings = []  # List to store embeddings

    for i in tqdm(range(0, len(texts), batch_size), desc="Embedding with SPECTER2"):
        batch_texts = texts[i:i+batch_size]  # Get batch of texts
        inputs = tokenizer(batch_texts, padding=True, truncation=True, max_length=512, return_tensors="pt").to(device)  # Tokenize and move to device
        with torch.no_grad():  # Disable gradient calculation
            outputs = model(**inputs)  # Forward pass
            cls_embeddings = outputs.last_hidden_state[:, 0, :]  # CLS token embedding
            cls_embeddings = torch.nn.functional.normalize(cls_embeddings, p=2, dim=1)  # Normalize embeddings
        embeddings.append(cls_embeddings.cpu().numpy())  # Move to CPU and convert to numpy

    return np.vstack(embeddings)  # Stack all embeddings into a single array