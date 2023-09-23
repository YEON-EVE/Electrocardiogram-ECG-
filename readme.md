
# Electrocardiogram (ECG)

## Introduction
심장 활동은 심장 섬유의 수축과 이완으로 인한 전류 흐름과 관련이 있다. 이러한 전기적 활동은 신체 표면에 적절히 배치된 전극을 사용하여 기록할 수 있으며, 이러한 전극 사이에서 측정 된 전위차를 심전도(Electrocardiogram; ECG) 신호 라고 한다.
정상적인 심장 박동은 동방결절(sino-atrial node)에서 시작되는 주기적인 전기적 충격 때문에 발생한다. 이러한 충격이 동방결절 이외의 장소에서 발생한는 경우 심전도 신호가 불규칙하고 비정상적으로 나타나게 되는데, 이러한 신호를 이소성 심장 박동이라고 한다.
이소성 심장 박동은 생명을 위협하는 정도는 아니지만, 반복적으로 발생하는 경우(예: 하루 1,000회 이상의 빈번한 조기 수축) 심혈관 질환 환자인지와 관계 없이 심방세동, 심부전, 뇌졸중 및 사망률을 포함한 심혈관 부작용의 위험 증가와 관련이 있다.[ref] 또한, 심장 부정맥을 탐지할 때에 중요하게 보는 지표 중 하나인 heart rate variaiblity 를 측정함에 있어서도 이소성 박동은 영향을 미친다.
따라서 심전도 신호에서 이소성 심장 박동을 탐지하고 적절하게 교정하는 것은 중요하지만, 아직까지 보편적으로 사용되는 방법은 없는 상황이다. 본 연구에서는 이소성 심장 박동을 탐지하기 위한 transfer learning 기반 분류 모델을 개발한다.

## Method
### Dataset
#### MIT-BIH Arrhymia database
  - 모델 개발에는 PhysioBank 의 MIT-BIH Arrhythmia database가 사용되었다.[ref]
  - 각 신호는 2가지 lead로 구성되는데, 대부분은 MLII, V1 이며 드물게 다른 lead 로 구성되어 있다. 일반적으로 MLII 에서 정상적인 QRS complexes 가 더 잘 관측되기 때문에, 본 연구에서는 MLII 를 사용하였다.[ref]

#### Class
  - Normal beat
  - Ectopic beat
    - Atrial premature beat
    - Ventricular premature beat

### Data Preprocessing

#### Segmentation & Labeling
  - min-max scaling 후 2.5s 단위로 segmentation
  - Class imbalance 가 심했기 때문에, Normal class 의 경우 무작위로 30%만 추출하여 모델 개발에 활용

#### Continuous wavelet transform
  - 

![CWT_img](https://github.com/YEON-EVE/Electrocardiogram-ECG-/assets/69179261/ab15cd45-7b8d-4b79-b9e0-4ffb3a71506a)

### Model
  - 1d CNN
  - 2d CNN: transfer learning (pretrained model)
    - ResNet[ref]

## Result

![ecg_result](https://github.com/YEON-EVE/Electrocardiogram-ECG-/assets/69179261/fecc0c3a-0240-4950-b876-146457288590)
