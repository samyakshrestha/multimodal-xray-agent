- Downloaded CheXpert and ChestX-ray14 dataset 
- Flattened the CheXpert dataset locally (had problems doing it on Colab because of I/O issues) and uploaded it to Drive
- Preprocessed the CheXpert dataset and saved it to ./data/images_sample/chexpert

- Transferred CheXpert and ChestX-ray14 datasets from Drive to SSD for fast access.
- Verified image integrity, assigned UUIDs, and saved manifest to image_metadata.jsonl.
- Loaded BiomedCLIP (ViT-B/16) vision encoder and embedded ~350k images.
- Normalized 512-D vectors and built FAISS index for cosine similarity search.
- Saved image_faiss.bin and image_uuids.json to ./data/indexes/.

- Sampled 30,000 abstracts from the `MedRAG/pubmed` dataset using Hugging Face Datasets API.
- Exported full records to `raw_abstracts.jsonl` and minimal metadata (`pmid`, `title`) to `text_metadata.json`.
- Loaded `allenai/specter2_base` sentence embedding model via `sentence-transformers`.
- Embedded all title + abstract pairs (normalized 768-D float32 vectors).
- Built cosine-similarity FAISS index (`IndexFlatIP`) and saved to `text_faiss.bin`.
- Performed retrieval sanity checks with medical queries to verify embedding quality and recall.

- Loaded IU-Xray dataset from Hugging Face (`datasets.load_dataset("iu_xray")`).
- Extracted impression sections from reports and saved to `iu_impr.jsonl` with canonical UUID format (`iu_XXXX`).
- Parsed and resized 7,430 embedded DICOM PNGs from `.parquet` blobs, saving to `./data/iu_xray/images/`.
- Verified and cataloged valid PNGs, assigning UUIDs and storing their paths in `iu_uuids.jsonl`.
- Loaded BiomedCLIP (`microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224`) using OpenCLIP.
- Applied vision transformer on all images using GPU-accelerated batch inference (`batch_size=32`).
- Saved 512-D BiomedCLIP embeddings as `iu_vecs.npy` in `./data/iu_xray/`.

- Filtered and triaged ~7,400 IU-Xray impressions into three diagnostic tiers using regex-based heuristics and keyword detection
- Sampled 320 (Tier A), 100 (Tier B), and 100 (Tier C) impressions to form a balanced QA pool.
- Defined three QA templates per impression: two static questions and one GPT-4o-generated paraphrased summary.
- Implemented GPT-based summarization pipeline with temperature=0.3 and strict hallucination controls.
- Saved ~1,500 Q/A pairs to qa_pairs_lora.jsonl inside ./data/qapairs/.
- Split final QA dataset into train.jsonl (80%) and val.jsonl (20%) for LoRA fine-tuning.