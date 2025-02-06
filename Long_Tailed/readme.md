# Electrocardiogram (ECG)
## Title: Deep Learning-Based Approach to Address Data Imbalance in Medical Field: Application to Electrocardiogram Data for Enhanced Few-Shot Classification
* 본 연구는 2024 전자·반도체·인공지능학술대회에서 포스터로 발표되었으며, 석사 학위 논문으로 제출되었습니다.
* 저작권의 이유로 일부 내용만 공개되어 있습니다.


## Introduction
최근 컴퓨터 기술, 특히 인공지능의 발전으로 의료 분야에서 다양한 목적으로 널리 활용되고 있다. 대부분의 분류 알고리즘은 균형 잡힌 데이터셋을 가정하지만, 실제 의료 데이터는 종종 큰 불균형을 특징으로 한다. 이러한 불균형은 모델 학습에 영향을 미치며, 특히 소수 클래스의 정확한 식별에 어려움을 초래할 수 있다. 본 연구에서는 불균형한 의료 데이터인 심전도(ECG)를 대상으로 한 딥러닝 기반 다중 클래스 심장 박동 분류 모델을 제안하였다.

## Method
<br>
 <img src="https://github.com/user-attachments/assets/d1fc9912-d6e7-41de-a1c2-bd81d8b714f7" width="600" height="300"/>
<br>
**Figure 1. Flow chart **

### Dataset
#### MIT-BIH Arrhymia database[1]
  - 모델 개발에는 PhysioBank 의 MIT-BIH Arrhythmia database가 사용되었다.
  - 각 신호는 2가지 lead로 구성되는데, 대부분은 MLII, V1 이며 드물게 다른 lead 로 구성되어 있다. 일반적으로 MLII 에서 정상적인 QRS complexes 가 더 잘 관측되기 때문에, 본 연구에서는 MLII 를 사용하였다.[2]
  - 발생 빈도에 따른 성능을 확인하기 위해 전체에서 차지하는 비율에 따라 many/medium/few shot 3개 category로 나누어 성능을 확인하였다.
![image](https://github.com/user-attachments/assets/64a89f03-390c-40c9-add2-10ef4500fbb2)

#### CSPC[3]
  - 심전도 분류 모델의 경우 wearable divice에 탑재되어 사용될 때 활용도가 높아진다고 판단하여, wearable device에서 수집된 데이터를 활용하여 모델의 성능을 검증하였다.
![image](https://github.com/user-attachments/assets/f54586fc-b256-402f-9f21-204a8d4d5caf)

### Model  
  - 본 연구에서는 심전도(ECG) 박동을 분류하는 모델을 개발하기 위해 GISTNet을 활용하였다.
  - GISTNet의 핵심 개념은 주요 클래스(head class)의 기하학적 분포를 소수 클래스(tail class)로 전달하는 것으로, 이미지 데이터에 적합한 구조를 1차원 신호 데이터에 적합하게 수정하여 활용하였다.
  - 데이터셋의 불균형 문제를 고려하여, 분류 작업에 가장 적합한 접근법을 찾기 위해 세 가지 손실 함수에 대한 비교 분석을 수행하였다.


### Result
Weighted cross-entropy가 소수 샘플 클래스에서 우수한 성능을 보였기 때문에, 이후 분석은 가중 교차 엔트로피를 기반으로 수행되었다.
제안된 모델과 기존 CNN 모델의 성능 비교 결과는 표 1에 제시되어 있으며, 모든 8개 클래스에 대한 자세한 비교는 혼동 행렬(Figure 2)을 통해 제공된다.
소수 샘플 클래스에서 제안된 모델은 높은 민감도를 보였으며, 이는 전체적인 성능에서도 동일한 경향을 나타냈다.

<br>
Table 1. Comparison of the proposed model with the baseline model
<img src="https://github.com/user-attachments/assets/0b45ab32-16f0-47fa-b54f-9c053d009135" width="800" height="300"/>

<br><br>
[1] Moody, G.B. and R.G. Mark, The impact of the MIT-BIH arrhythmia database. IEEE engineering in medicine and biology magazine, 2001. 20(3): p. 45-50.<br>
[2] Qin, Qin, et al. "Combining low-dimensional wavelet features and support vector machine for arrhythmia beat classification." Scientific reports 7.1 (2017): 1-12.<br>
[3] Cai, Z., et al., An open-access long-term wearable ECG database for premature ventricular contractions and supraventricular premature beat detection. Journal of Medical Imaging and Health Informatics, 2020. 10(11): p. 2663-2667.
