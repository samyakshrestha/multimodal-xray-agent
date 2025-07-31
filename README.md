# Multi-Agent AI Assistant for Radiology Report Generation

This project introduces a **CrewAI-powered multi-agent system** for automated radiology report generation from chest X-rays. By integrating **vision-language models (VLMs)**, **cross-modal retrieval**, and **modular agent orchestration**, the system emulates a radiologist’s workflow: it analyzes chest X-ray images, retrieves relevant medical literature via semantic similarity search, and generates coherent reports by synthesizing information from all sources.

---
## Live Demo  
[![Hugging Face Spaces](https://img.shields.io/badge/🤗-View_on_HuggingFace-blue)](https://huggingface.co/spaces/samyakshrestha/multiagent-xray-assistant) 

---

## Project Highlights

- **Orchestrated a multi-agent system** with CrewAI, leveraging a "divide and conquer" approach to assign specialized tasks to dedicated agents
- **Developed semantic search and retrieval pipelines** by embedding Indiana University (IU) chest X-ray images with BiomedCLIP for similarity-based retrieval
- **Sampled, embedded, and indexed 200,000 PubMed title/abstract pairs** with SPECTER2 and FAISS for efficient retrieval of relevant medical literature
- **Integrated Groq API for ultra-fast inference**, achieving >7× lower latency compared to models like Gemini 2.5 Flash
- **Implemented advanced prompt-engineering techniques** to maximize diagnostic accuracy and optimize agent collaboration and workflow
- **Evaluated system performance with LLM-as-Judge and BERTScore**, demonstrating superior results over leading single-agent models such as Gemini 2.5 Flash and GPT-4o
- **Deployed on the full pipeline on Hugging Face Spaces** via Gradio 

---

## Multi-Agent Workflow  

1. **Vision Agent**
   - Uses a **VLM (Gemini 2.5 Flash/Llama 4 Maverick)** to generate a detailed caption of the X-ray image 
   - Outputs caption in a structured format for downstream agents
2. **Retriever Agent(s)**  
    - Pubmed Agnet: 
      - Generates a **768-D embedding** of the image caption using **SPECTER2**
      - Retrieves the **top-k most relevant titles and abstracts** from a **vector database** of 200k PubMed articles using **FAISS**
    - IU Agent:
      - Generates a **512-D embedding** of the X-ray image using **BiomedCLIP**
      - Uses **cosine similarity** to retreive the most similar image from the vector database for the **IU-Xray images**
      - Retrieves the **report** corresponding to the most similar image 
3. **Draft Agent** 
    - Synthesizes a comprehensive radiology report by aggregating insights from the Vision and Retriever Agents
4. **Critic Agent**
    - Reviews the output from the Draft Agent to ensure factual correctness and clinical coherence
    - Removes unsupported or speculative statements, ensuring the final report is concise, evidence-based, and professionally formatted 
    - Generates the final report in markdown format

## Evaluation

| System         | LLM-as-Judge (avg / 5.0) | BERTScore |
|----------------|--------------------------|-----------|
| Multi-Agent    | **2.20**                 | **0.0930** |
| Gemini (single)| 2.00                     | 0.0011     |
| GPT-4o (single)| 2.07                     | -0.0113    |

- Evaluated on 15 IU-Xray studies (StanfordAIMI dataset) [Link](https://huggingface.co/datasets/StanfordAIMI/interpret-cxr-test-public)
- Full notebook: [notebooks/16_evaluation.ipynb](notebooks/16_evaluation.ipynb)

---

## Technology Stack
- **Frameworks:** CrewAI (multi-agent orchestration), PyTorch, Hugging Face Transformers + Datasets, Gradio
- **LLMs for Reporting:** [Gemini 2.5 Flash], [LLaMA 3.3 70B Instruct], [Llama 4 Maverick], [DeepSeek R1-Distill LLaMA 70B]
- **Image Encoder:** [BiomedCLIP](https://huggingface.co/microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224) — 512-D image embeddings
- **Text Encoder:** [SPECTER2](https://huggingface.co/allenai/specter2) — 768-D text embeddings
- **Retrieval Infrastructure:** FAISS, cosine similarity search
- **Evaluation:** GPT-4o LLM-as-Judge, BERTScore (semantic similarity)
- **Inference + Deployment:** Groq API (sub-second latency), Hugging Face Spaces (Gradio frontend)

---

## Datasets  
- **IU Chest X-ray Dataset:** [Link](https://huggingface.co/datasets/ayyuce/Indiana_University_Chest_X-ray_Collection)
- **PubMed Dataset:** [Link](https://huggingface.co/datasets/MedRAG/pubmed)
- **StanfordAIMI Dataset:** [Link](https://huggingface.co/datasets/StanfordAIMI/interpret-cxr-test-public)

---

## Author
Samyak Shrestha