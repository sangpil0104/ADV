# 공식 근거와 연결 기록

문서 갱신일: **2026-10-07**. DragFlow·Qwen 연결은 2026-09-29/30의 기존 조사·실행 기록이며 CLIP 자료는 2026-10-07 확인했다. 링크의 main은 변할 수 있으므로 구현에서는 commit/revision을 고정한다. 실제 확인한 빌드/CPU 범위는 [현재 검증 기록](CLIP_BUILD_VERIFICATION.md)에 있다. 새 GPU 통합 성공을 뜻하지 않는다.

## DragFlow와 현재 adapter

- [DragFlow 논문](https://arxiv.org/abs/2510.02253), [기존에 확인한 v3 본문](https://arxiv.org/html/2510.02253v3): FLUX 기반 영역 편집과 subject/background consistency를 다룬다. ADVV의 생성 후 CLIP 선별과 3-of-4 AND mandatory-5는 사용자와 이 프로젝트의 설계이며 논문의 검증 정책이 아니다.
- [공식 저장소](https://github.com/Edennnnnnnnnn/DragFlow), [Dragger](https://github.com/Edennnnnnnnnn/DragFlow/blob/main/framework/dragger.py), [입출력 코드](https://github.com/Edennnnnnnnnn/DragFlow/blob/main/framework/dashboard_utils.py): 기존 adapter 연결의 근거다. 실행에는 [로컬 lock](../configs/upstream.lock.json)의 commit을 사용한다.
- [공식 모델 설정](https://github.com/Edennnnnnnnnn/DragFlow/blob/main/framework/config.yaml), [환경 안내](https://github.com/Edennnnnnnnnn/DragFlow/blob/main/dependencies/dragflow.yaml): FLUX/InstantCharacter/SigLIP/DINOv2 구성과 당시 의존성 안내를 확인한 자료다. 안내 환경과 새 단일 conda ADVV의 검증 완료를 혼동하지 않는다.
- [현재 ADVV adapter](../src/advv/backends/dragflow.py): 요청 좌표, feature-grid, 실제 start centroid 교체, effective geometry 기록을 확인할 구현 위치다. 자유형 segmentation 입력의 정합성은 새 검증 과제다.

## CLIP 검증기 설계 근거

- [OpenAI CLIP 공식 저장소](https://github.com/openai/CLIP): 이미지/텍스트 encoder, feature 정규화와 유사도 계산 예제를 제공한다. 기본 반환 logit과 raw cosine은 구분해야 한다. ADVV는 정규화 feature의 내적을 별도로 계산하는 명세다.
- [openai/clip-vit-large-patch14 모델 카드](https://huggingface.co/openai/clip-vit-large-patch14): 기존 lock에 들어 있는 checkpoint의 출처다. 이 모델을 검증기의 초기 연결 후보로 재사용하는 것은 ADVV의 구현 선택이며 사고 판정 성능이 검증되었다는 뜻이 아니다.
- [CLIP 공식 모델 설명과 한계](https://github.com/openai/CLIP/blob/main/model-card.md): 일반 모델 설명과 용도를 확인할 자료다. 도메인별 정확도와 편향은 해당 데이터에서 따로 평가한다.

[10개 문장](../prompts/clip_criteria_v1.json), A/B 방향, margin=0, 1–4번 중 3개 이상 및 5번 필수는 **프로젝트의 고정 정책**이다. 공식 모델이 제공하는 물리 검증·교통사고 검출 보증이 아니다. 전체 프레임 letterbox 역시 프로젝트의 전처리 설계이며 pretrained processor 기본값이라고 주장하지 않는다.

## 차량 segmenter

- [torchvision 0.20 Mask R-CNN 문서](https://docs.pytorch.org/vision/0.20/models/generated/torchvision.models.detection.maskrcnn_resnet50_fpn.html): COCO_V1 checkpoint, instance mask 반환, 입력 tensor와 추론 인터페이스를 확인했다.
- [고정 torchvision v0.20.1 구현](https://github.com/pytorch/vision/blob/v0.20.1/torchvision/models/detection/mask_rcnn.py): 모델 구성과 COCO 가중치 URL의 근거다. ADVV는 로컬 가중치를 weights_only로 읽고 공식 COCO_V1의 FrozenBatchNorm 구성을 사용한다.
- 차량 class ID 3/4/6/8은 car/motorcycle/bus/truck이다. confidence/mask threshold 0.5와 최소 영역 32px는 프로젝트 초기값이며 사고 도메인에 최적화된 수치가 아니다.

## 과거 Qwen 방식의 출처

- [Qwen3.5-4B 모델 카드](https://huggingface.co/Qwen/Qwen3.5-4B): 초기 로컬 이미지 VQA의 모델 출처다. 새 CLIP 방식의 필수 모델은 아니다.
- [과거 실행 검증](RUN_VERIFICATION.md): DragFlow와 Qwen의 N=1 연결, 당시 단계 시간·관측 메모리, CPU 테스트 기록을 보존한다.
- `environments/coordinator.freeze.txt`, `dragflow.freeze.txt`, `qwen.freeze.txt`는 **분리 환경의 과거 기록**이다. 이를 합쳐 설치하거나 새 단일 conda의 lock으로 사용하지 않는다.

기존 가중치·환경·결과는 문서 전환에서 삭제하지 않는다. Qwen accepted 이미지는 CLIP 미검증으로 구분하고, 비채택 상세 결과의 후속 정리는 [SPEC](../SPEC.md)의 상태/소유권 규칙을 따른다.

## 사실·요구·미검증 항목 구분

| 구분 | 내용 |
| --- | --- |
| 현재 소스에서 확인 | 공식 DragFlow adapter, Mask R-CNN 차량 segmentation, 자유형 sampler, CLIP 5쌍, 단일 conda worker, schema 2.0 |
| 사용자 확정 | 사고 블랙박스 단일 프레임, segmentation start, 랜덤 end/range, 자유형 mask, CLIP 3-of-4 AND 5, 선택 prompt, 실패 상세 삭제, 실패 포함 시간, conda ADVV |
| 기존에 유지한 요구 | 전체 입력 총 N장, 정상 탈락 총 상한 없음, 실행 시 GPU 선택, 원본에서 새 편집 |
| 구현 설계 | raw cosine, 전체 프레임 letterbox, schema 2.0, 최소 timing 장부, private staging, 안전한 기존 비채택 정리 |
| 2026-10-07 완료 범위 | v0.2.0 소스·환경 통합, CPU 93개 검사, 실제 CLIP/Mask R-CNN CPU forward; GPU 추론 제외 |
| 2026-10-08 추가 범위 | v0.3.0 GPU 쌍별 병렬 coordinator와 기본 경로, CPU 113개 검사, wheel 빌드, 기본 경로 preflight; 실제 GPU 생성 미실행 |
| 아직 미검증 | 실제 사고 영상 도메인에서 자유형 GPU 편집 품질, CLIP 판정 품질, 전체 처리량·후속 성능 |

## 후속 실행에 남길 기록

ADVV 코드/설정/문장 hash, DragFlow/submodule commit, segmenter·CLIP·생성 모델 revision, processor/전처리, Python/PyTorch/CUDA/패키지, GPU 매핑을 고정한다. seed와 입력 hash, 채택 이미지의 계획·실효 geometry·점수·시간도 연결한다.

탈락 후보의 상세 점수·이미지·receipt를 재현 명목으로 계속 저장하지 않는다. 실행 recipe와 원본 자원은 보존하되 후보별 기록은 최소 시간/상태 장부로 제한한다. 다운로드나 import 성공은 실제 생성·선별·품질 평가의 완료 증거가 아니다.
