- Downloaded CheXpert and ChestX-ray14 dataset 
- Flattened the CheXpert dataset locally (had problems doing it on Colab because of I/O issues) and uploaded it to Drive
- Preprocessed the CheXpert dataset and saved it to ./data/images_sample/chexpert

- Transferred CheXpert and ChestX-ray14 datasets from Drive to SSD for fast access.
- Verified image integrity, assigned UUIDs, and saved manifest to image_metadata.jsonl.
- Loaded BiomedCLIP (ViT-B/16) vision encoder and embedded ~350k images.
- Normalized 512-D vectors and built FAISS index for cosine similarity search.
- Saved image_faiss.bin and image_uuids.json to ./data/indexes/.
