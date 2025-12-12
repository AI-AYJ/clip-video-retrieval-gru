# clip-video-retrieval-gru

<img width="1033" height="553" alt="image" src="https://github.com/user-attachments/assets/ebaada80-58ab-4e9f-afdb-3e2a5f013f4f" />


test feature link: https://drive.google.com/drive/folders/1ii8c1-YKGG7zxOQiPrHU2N3thyY2DZBL?usp=sharing


# 📹 CLIP-based Video-Text Retrieval Analysis

> **CLIP 기반 Video-Text 검색에서 프레임 샘플링과 시간적 표현 기법이 성능에 미치는 영향 분석**
>
> *Analyzing the Impact of Frame Sampling and Temporal Representation Techniques on Performance in CLIP-based Video-Text Retrieval*

[![KAICTS](https://img.shields.io/badge/Conference-KAICTS_2025-blue)](https://kaicts.or.kr/)
[![Python](https://img.shields.io/badge/Python-3.x-blue?logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Model](https://img.shields.io/badge/Model-CLIP_%2B_GRU-yellow)](https://github.com/openai/CLIP)

---

## 📂 Source Code & Model Weights

본 리포지토리는 실험에 사용된 소스 코드와 학습된 모델 가중치를 포함하고 있습니다. 아래 링크를 클릭하여 파일을 확인할 수 있습니다.

| Type | File | Description |
| :--- | :--- | :--- |
| **Baseline** | 📄 **[Meanpooling.ipynb](./Meanpooling.ipynb)** | Baseline 모델 (Frame Mean Pooling) 구현 코드 |
| **Proposed** | 📄 **[GRU.ipynb](./GRU.ipynb)** | 제안 모델 (GRU Adapter + Attention Pooling) 구현 코드 |
| **Weights** | 💾 **[GRU_adapter3_32f_hidden1024.pth](./GRU_adapter3_32f_hidden1024.pth)** | 학습이 완료된 GRU Adapter 모델 가중치 파일 |

---

## 📖 Introduction & Motivation

**Video-Text Retrieval(비디오-텍스트 검색)**은 영상과 텍스트 간의 의미적 일치를 기반으로, 주어진 문장과 가장 관련 있는 영상을 검색하거나 그 반대를 수행하는 멀티모달 기술입니다.

기존 모델들은 프레임 임베딩을 단순히 평균(Mean Pooling)하여 비디오 표현을 구성하기 때문에 **시간적 의존성(Temporal Dependency)**을 제대로 반영하지 못하는 한계가 있었습니다. 또한, Uniform 또는 Random 샘플링 전략이 검색 성능에 미치는 영향에 대한 체계적인 분석이 부족했습니다.

본 연구는 **프레임 샘플링 방식, 프레임 수, 그리고 시간적 표현 기법(GRU)**의 상호작용이 검색 성능에 미치는 영향을 분석하여 효율적인 멀티모달 검색 모델의 방향성을 제시합니다.

---

## 🏗️ Model Architecture

본 프로젝트는 사전 학습된 CLIP(ViT-B/16)을 백본으로 사용하며, 두 가지 접근 방식을 비교 분석하였습니다.

### 1. Baseline (Mean Pooling)
* 각 프레임의 임베딩을 추출한 후 단순 평균(Mean)하여 비디오 임베딩 생성
* Cosine Similarity 기반 순위 계산

### 2. Proposed (GRU Adapter)
* **GRU Adapter:** 프레임 간의 시간적 의존성 학습
* **Attention Pooling:** 중요 프레임에 가중치를 두어 비디오 임베딩 구성
* **Training:** CLIP 인코더는 고정(Freeze)하고, GRU와 Learnable Temperature만 학습

```mermaid
graph LR
    V[Input Video] --> S{Frame Sampling}
    S -- Uniform/Random --> E[Vision Encoder (ViT)]
    E --> F[Frame Embeddings]
    
    subgraph Temporal Modeling
    F -- Baseline --> M[Mean Pooling]
    F -- Proposed --> G[GRU Adapter]
    end
    
    M --> V_Emb[Video Embedding]
    G --> V_Emb
    
    T[Input Text] --> TE[Text Encoder] --> T_Emb[Text Embedding]
    
    V_Emb <--> T_Emb
    style V_Emb fill:#f9f,stroke:#333
    style T_Emb fill:#f9f,stroke:#333
