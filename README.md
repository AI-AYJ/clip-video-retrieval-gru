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
```mermaid

---

## 🧪 Experimental Setting

### Dataset
* [cite_start]**MSR-VTT** [cite: 44]
    * [cite_start]Total: 10k videos (20 captions each) [cite: 45]
    * [cite_start]Split: 7k (Train) / 1k-A (Test) [cite: 46]

### Hyperparameters
[cite_start]실험에 사용된 주요 하이퍼파라미터 설정은 다음과 같습니다[cite: 75, 76].

| Parameter | Value |
| :--- | :--- |
| **Backbone** | [cite_start]CLIP (ViT-B/16, pretrained) [cite: 76] |
| **Input Frames** | [cite_start]8, 16, 32 [cite: 76] |
| **Sampling** | [cite_start]Uniform vs Random [cite: 76] |
| **Optimizer** | [cite_start]AdamW ($lr=1e-4$) [cite: 76] |
| **Batch Size** | [cite_start]32 [cite: 76] |
| **Environment** | [cite_start]NVIDIA A100 GPU [cite: 76] |

---

## 📊 Results & Analysis

실험 결과에 대한 상세 분석입니다.

### 1. Frame Count & Sampling Strategy
* [cite_start]**Frame Count:** 프레임 수가 많을수록(8→32) 더 풍부한 시각 정보를 반영하여 비디오 표현력이 강화되는 경향을 보였습니다[cite: 139, 140].
* **Sampling:** **Uniform 샘플링**이 Random 샘플링보다 더 안정적이고 일관된 성능을 보입니다. [cite_start]Random 샘플링은 성능 변동폭이 큽니다[cite: 146, 149].

### 2. Temporal Representation (GRU vs Mean)
* [cite_start]GRU 기반 모델이 프레임 간의 **시간적 관계(Temporal Dependency)**를 학습하여 의미적 일관성이 높은 비디오 표현을 형성했습니다[cite: 156].
* [cite_start]결과적으로 GRU가 Mean Pooling 대비 전반적인 성능 우위를 점했습니다[cite: 157].

### Performance Comparison (R@1)

[cite_start]아래 표는 Uniform Sampling 기준의 R@1 성능 비교입니다[cite: 142, 152, 160].

| Method | Sampling | 8 Frames | 16 Frames | 32 Frames |
| :--- | :--- | :---: | :---: | :---: |
| **Mean Pooling** | Uniform | 26.4 | 27.3 | 28.0 |
| **GRU (Ours)** | **Uniform** | **28.1** | **29.9** | **30.2** |

> [cite_start]**Key Finding:** Uniform Sampling + 32 Frames + GRU 조합에서 가장 안정적이고 높은 성능(30.2)을 달성했습니다[cite: 142].

---

## 🚀 Qualitative Results

### Video-to-Text Retrieval
* **Query Video:** Singing Contest (The Voice Kids)
* [cite_start]**Top 1:** "a boy is trying out for a part on the voice kids" (Confidence: 0.28) [cite: 26]
* **Analysis:** 시각적 정보와 텍스트 의미가 정확하게 매칭되었습니다.

### Text-to-Video Retrieval
* [cite_start]**Query Text:** "a woman preparing a duck to roast" [cite: 172]
* [cite_start]**Top 1 Result:** 요리 준비 과정이 담긴 비디오가 정확하게 검색됨 (Confidence: 0.28)[cite: 173].

---

## 🔮 Future Work

[cite_start]향후 연구 계획은 다음과 같습니다[cite: 178].

* [cite_start]**Advanced Temporal Modules:** Bi-GRU 및 Transformer 기반 시간 모듈 확장 적용[cite: 179].
* [cite_start]**Adaptive Sampling:** Scene-based 또는 Motion-aware와 같은 다양한 샘플링 전략 실험[cite: 180].
* [cite_start]**Scale Up:** 대규모 비디오 데이터셋을 활용한 일반화 성능 검증[cite: 181].
* [cite_start]**Bi-directional Retrieval:** 양방향 검색(Text→Video, Video→Text) 성능 고도화[cite: 182].

---

## 📝 Authors & Acknowledgement

* [cite_start]**Authors:** Ayeong Jung, Junhwa Kim [cite: 7]
* [cite_start]**Contact:** `23619024@konyang.ac.kr` / `junhwakim@konyang.ac.kr` [cite: 8]
* [cite_start]**Affiliation:** Dept. of Medical AI, Konyang University [cite: 8]
* [cite_start]**Event:** KAICTS 2025 Autumn Conference [cite: 3]
