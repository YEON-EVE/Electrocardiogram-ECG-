
# code

## EctopicBeat_GenerateModel.py
- 여러가지 모델을 반환해주는 code
- Resnet, VGGnet 등 다양한 구조 반환
- model_name, num_class 을 전달받아 task에 맞는 구조의 모델 반환

## EctopicBeat_GenerateDataset.py
- dataset 생성 code
- signal pickle 파일과 data transform 등을 전달 받아 dataset 반환

## EctopicBeat_foldsplit.py
- k fold 학습을 위한 fold split code
- train/validation/test set 반환
- testset 의 경우 성능 검증 목적으로 train/validation 이 바뀌어도 변화 X
- 생체신호 특성 상 동일한 subject가 train/validation/test set에 포함 될 경우 모델 성능에 영향을 줄 수 있으므로, 겹치지 않게 구성
