# ADVV

**DragFlow를 이용한 자율주행용 차량 사고 블랙박스 단일 프레임 이미지 증강**

사고 관련 차량을 자동 선택해 전체 마스크를 합치고, 모든 선택 차량에 동일한 무작위 이동 벡터를 적용합니다. 이동 크기는 사고 그룹 bbox 대각선 비율로 상한을 두고 세로 이동을 줄입니다(`accident_group_v2`). 편집 영역 크기는 전체 그룹 면적으로 고정됩니다. DragFlow 후보를 CLIP 문장 5쌍으로 평가해 **1–4번 중 3개 이상 통과하고 5번을 반드시 통과한** 새 이미지를 총 N장 모읍니다. N은 전체 입력의 총량입니다.

ADVV 이름은 유지합니다. `Augmentation using DragFlow with VQA Verification`은 초기 방법의 풀네임이며 현재 검증기는 CLIP입니다.

## 최신 태스크 변경

2026-10-08: 이동 대상을 **사고에 직접 관련된 차량 전체**로 변경했습니다. 관련 차량들의 전체 mask를 합쳐 동일한 방향·거리로 함께 이동하고, 차량 사이의 충돌 배치를 유지하는 것이 목표입니다. 무관한 차량·사람·파편·도로·건물은 이동 대상에서 제외합니다. 편집 영역을 무작위로 작게 자르는 방식은 새 설계에서 사용하지 않습니다.

**v0.4.0부터 아래 명령은 전체 사고 차량 공동 이동을 사용합니다.** [SPEC 5절](SPEC.md#5-사고-관련-차량-전체의-공동-이동)에 선택·마스크·좌표 계약을 정의했습니다. 기존 부분 편집 run을 resume하지 말고 새 run으로 실행하세요. 기존 결과는 보존됩니다.

## 구현과 검증 상태

2026-10-11, **v0.4.1 / schema 2.0**: 전체 사고 차량 그룹 편집과 기존 GPU 쌍별 병렬 처리·공유 목표 수량을 구현했습니다.

- 차량 instance segmentation: torchvision Mask R-CNN ResNet50-FPN COCO_V1, car/motorcycle/bus/truck.
- 원본 접촉 관계와 로컬 CLIP의 객체/맥락 비교로 사고 관련 차량을 추정합니다. 선택된 차량 전체의 합집합 mask, 공통 이동 벡터, 원본/feature grid 경계 검사를 사용합니다. 이동 벡터는 크기 r(그룹 대각선의 `min_displacement_fraction`~`max_displacement_fraction`, 기본 0.03~0.3)과 방향 θ를 따로 뽑고 세로 성분에 `vertical_scale`(기본 0.5)을 곱합니다. 방향은 그 크기에서 그룹이 화면 안에 남는 각도 구간 안에서만 균등하게 뽑고, 상한 안에서 화면 안 이동이 하나도 없는 원본은 생성 전에 `no_valid_motion`으로 보류해 `state.json`·report에 표시합니다(전부 보류되면 `no_movable_sources`로 중단). 기본값은 GPU 스윕 전의 미검증 추정치입니다. 차량 사이의 빈 공간과 떨어진 성분을 유지합니다.
- 로컬 CLIP ViT-L/14의 raw cosine 비교, 선택 `--prompt`, 총 N장 반복·중단·재개.
- 채택 전용 시각화·export, 기존 비채택 상세 산출물 정리, 실패 포함 시간 기록.
- GPU 두 개당 후보 작업 하나를 실행하고, 모든 작업이 총 N장을 공유합니다.
- 모든 worker를 **conda ADVV의 동일한 Python**으로 실행합니다. 기존 venv는 사용하지 않습니다.
- **CPU 테스트 181개·lint 통과**(2026-10-11, 이동 크기 상한·가능 방향 표본·이동 불가 원본 보류 추가 후; GPU 숨김 상태에서 실행). v0.4.1 wheel 빌드는 상한 추가 전에 확인했습니다. 전체 그룹·공식 좌표/affine 함수·기존 quota/재개/삭제 계약을 검증했습니다. `image1` 실제 CPU 분할+CLIP 선택 점검에서 두 사고 차량을 선택했습니다. 검증 수량·빌드 결과는 [검증 기록](docs/CLIP_BUILD_VERIFICATION.md)에 남깁니다.

**CPU 계약 검증과 실제 생성 이미지의 품질 평가는 구분합니다.** 저장된 실행에 CLIP 채택 결과가 있어도 전체 사고 차량의 정확한 선택·공동 이동이나 물리적 타당성을 보장하지 않습니다. 기존 6개 실행의 채택 이미지 51장에는 종류별 시각화와 생성 이미지·drag plan 모음을 적용했습니다. 이 시각화 갱신은 GPU 재추론 없이 수행했습니다. 상세 범위는 [현재 빌드 검증](docs/CLIP_BUILD_VERIFICATION.md), 과거 Qwen 결과는 [legacy 기록](docs/RUN_VERIFICATION.md)을 참고하세요.

사고 차량 선택은 검증되지 않은 자동 추정 baseline입니다. `segments/<source_id>/selection.png`에서 초록색 선택 차량과 ID를 먼저 확인하세요. 원근상 겹침은 실제 접촉과 다를 수 있고, 분할되지 않은 차량은 선택할 수 없습니다. 선택 차량이 없으면 `no_eligible_sources`로 멈추며 임의의 차량을 대신 사용하지 않습니다.

채택 결과의 `visualizations/comparison/<candidate_id>.png`에는 전체 그룹과 **계획된** 목표 윤곽을 표시합니다. `metadata.json`과 record의 `generation_info.motion_diagnostics`는 차량별 외형 매칭 이동 추정치를 제공합니다. 이 수치는 진단용이며 CLIP 채택 조건을 추가하거나 실제 이동 성공을 인증하지 않습니다.

## 실행

현재 서버에는 conda `ADVV`와 로컬 모델 설정이 준비되어 있습니다. 입력은 **`assets/inputs/`**, 결과는 **`assets/outputs/<run_id>/`**를 사용합니다. 기존 `assets/image1.png`부터 `image9.png`까지 9장은 내용 변경 없이 `assets/inputs/`로 옮겼습니다. 이 폴더에 있는 모든 지원 이미지를 대상으로 합니다.

```bash
cd /Data3/sppark/advv
conda activate ADVV
advv run 10 --gpus 4,5,6,7
```

**image1만 대상으로 10장**을 만들려면 다른 이미지를 옮길 필요 없이 `--image`를 추가합니다.

```bash
advv run 10 --gpus 4,5,6,7 --image image1.png
```

파일명은 `assets/inputs` 기준 상대 경로이며 확장자를 포함합니다. 위 명령은 `assets/inputs/image1.png`만 사용합니다. `--image`는 설정의 `dataset.image`보다 우선하며 둘 다 생략하면 전체 입력을 사용합니다. 선택 파일이 없거나 잘못된 경로면 오류로 종료합니다. 사전 검사에도 같은 `--image`를 사용할 수 있고, 재개에는 원래 선택이 저장되어 있으므로 다시 지정하지 않습니다.

`10`은 두 작업을 합친 **고유 채택 이미지 총량**입니다. GPU 목록은 순서대로 두 개씩 묶으며 홀수 개/중복 ID는 거부합니다.

| 작업 | DragFlow | CLIP | 역할 |
| --- | --- | --- | --- |
| 1 | GPU 4·5 | GPU 4 | 후보 생성 → 검증 |
| 2 | GPU 6·7 | GPU 6 | 후보 생성 → 검증 |

원본 segmentation은 시작할 때 첫 묶음의 첫 GPU에서 수행하고 cache를 공유합니다. 두 작업은 독립적으로 진행하며, 채택·중복 검사·수량 갱신은 하나의 coordinator가 담당합니다. **채택 수 + 진행 중 후보 수가 N 이하**가 되도록 예약하므로 10장을 초과하지 않습니다. 9장 채택 후에는 남은 후보 하나만 진행합니다. 최종 10장이 저장되면 모든 worker를 종료합니다.

`--config`를 생략하면 저장소 기준 `configs/advv.clip.local.yaml`을 자동 사용합니다. 다른 작업 디렉터리에서 실행해도 같은 경로입니다. `--input-dir` 옵션은 제거했습니다. 기본 입력 폴더에 사진을 넣으면 됩니다. 새 서버에서 경로나 실험 설정을 바꿔야 할 때만 YAML 또는 선택 `--config`를 사용합니다.

`--run-id`를 생략하면 UTC timestamp와 임의 식별자로 고유한 폴더 이름을 만들고 터미널에 출력합니다. 이름과 증강 지시는 선택적으로 지정할 수 있습니다.

```bash
advv run 10 --gpus 4,5,6,7 --run-id accident_001 --prompt "Keep the traffic accident and vary the vehicle position and pose."
```

`--prompt`를 생략하면 기본 사고 장면 지시를 사용합니다. CLI가 설정의 `generation.augmentation_prompt`보다 우선합니다. 원문을 생성 지시에 전달하며 자동 번역·정확한 좌표 제약으로 해석하지 않습니다. 검증 문장과 채택 기준은 바뀌지 않습니다.

N은 양의 정수이며 `--target-count N`으로도 지정할 수 있습니다. 두 수량 형식의 동시 지정은 오류입니다. 탈락·중복·원본은 N에서 제외하며, 정상 탈락에 총 시도/시간 상한은 없습니다. 목표 달성 또는 사용자 중단까지 반복합니다. 사용 가능한 차량 segment가 원본 전체에 없으면 `no_eligible_sources`로 중단합니다.

새 run은 관리하는 output root의 종료된 비채택 상세 결과를 정리합니다. 원본·채택 결과·활성 run은 보호합니다. 과거 `runs/` 결과는 자동 이동하거나 삭제하지 않습니다.

GPU 추론 없이 설정·입력·가중치·import를 확인하려면:

```bash
advv preflight --target-count 10 --gpus 4,5,6,7
```

`Ctrl+C`/SIGTERM은 두 묶음을 모두 중단합니다. 같은 run을 재개할 때 N이나 새 prompt를 전달하지 않습니다. GPU 번호와 묶음 수는 바꿀 수 있습니다. 아래 `accident_001` 대신 터미널에 표시된 실제 run ID를 사용합니다.

```bash
advv run --run-dir assets/outputs/accident_001 --resume --gpus 4,5,6,7
advv report --run-dir assets/outputs/accident_001
advv export --run-dir assets/outputs/accident_001
```

## 채택 기준

각 항목은 영어 A=탈락 상태, B=선호 상태를 비교하며 **cosine(B) > cosine(A)**일 때 통과합니다. 동점은 실패이고 10개 점수가 모두 유효해야 합니다. 문장 원문은 [clip_criteria_v1.json](prompts/clip_criteria_v1.json)입니다.

| 번호 | A vs B의 한국어 의미 |
| --- | --- |
| 1 | 부자연스럽게 떠 있는 차량/사람 vs 도로에 자연스럽게 지지된 차량/사람 |
| 2 | 비현실적인 크기 vs 거리·주변에 맞는 현실적인 크기 |
| 3 | 도로 구조에 부자연스러운 위치·방향 vs 자연스러운 정렬 |
| 4 | 물체가 뒤엉켜 관통함 vs 구별되는 경계와 자연스러운 가림 |
| 5 | **정상 교통 장면 vs 한 대 이상의 차량이 관련된 일반 교통사고 장면** |

최종 조건은 **1–4번 중 3개 이상 + 5번 필수**입니다. 5번은 추돌에 한정하지 않습니다. CLIP은 물리 법칙이나 원본 사고 유형의 동일성을 보증하지 않으며 실제 사고의 전복·부양 등을 오배제할 가능성은 사람 검수로 평가해야 합니다.

## 결과와 시간

**v0.4.1부터 시각화를 종류별 폴더로 저장하고 생성 이미지와 drag plan 모음을 자동 갱신합니다.**

```text
visualizations/
├── comparison/<candidate_id>.png   # 원본/생성 비교
├── drag_plan/<candidate_id>.png    # start/end/area
├── metadata/<candidate_id>.json   # 각 이미지의 상세 정보
├── generated_grid.png             # 생성 이미지 전체 모음
├── generated_grid.json            # 모음 번호와 이미지 ID 대응표
├── drag_plan_grid.png             # start/end/area 계획 전체 모음
├── drag_plan_grid.json            # 계획 모음 번호와 개별 파일 대응표
└── index.json                     # 종류별 파일 목록
```

두 모음은 같은 순서로 대응하며 채택할 때마다 갱신됩니다. drag plan 모음은 화살표·편집 영역을 표시한 사진만 배치하며, 제목·설명·점수·ID·S/E/A 문자는 넣지 않습니다. 번호와 ID는 JSON에서, 상세 설명은 개별 drag plan 파일에서 확인합니다. 기존 결과도 GPU 없이 같은 구조로 정리합니다. `--all`은 `assets/outputs`와 과거 `runs`를 포함합니다.

```bash
conda activate ADVV
advv visualize --all
# 한 실행만 정리할 때:
advv visualize --run-dir assets/outputs/run_20261009T123128Z_1af42902
```

기존 시각화는 PNG 내용을 그대로 유지해 옮기고 metadata·record·report·export 링크를 함께 갱신합니다. 원본/생성 사진·판정·시간 장부는 바꾸지 않습니다. Qwen 결과는 CLIP 미검증으로 구분하고 쓰기 lock이 잡힌 실행은 건너뜁니다. 중간에 중단되면 같은 명령을 다시 실행하면 됩니다.

```text
assets/outputs/<run_id>/
├── inputs/images/               # 원본 snapshot
├── segments/                    # 원본별 차량 mask, 사고 선택, 전체 그룹, selection.png
├── .staging/                    # 진행 중 후보만; export 제외
├── accepted/                    # 표시 없는 채택 이미지
├── artifacts/<candidate_id>/    # 채택 편집 mask·실효 영역·생성 입력/receipt
├── records/                     # 채택 상세 기록만
├── visualizations/              # 채택 start/end/area와 원본·생성 비교
├── timings/attempts/             # 실패 포함 최소 시간 장부
├── timings/attempts.jsonl        # report에서 집계한 장부
├── timings/state.json           # active/pending 시간과 실행 구간
├── cleanup/                     # 삭제 복구 journal, 완료 후 최소 tombstone
└── exports/images.jsonl / report.json / report.md
```

채택 그림은 빨간 S, 초록 E, 노란 이동 화살표, 파란 자유형 area, 흰 원본 segment 경계를 표시합니다. 요청값과 실제 backend centroid/영역을 구분합니다. 그림은 CLIP 입력·학습 이미지에 섞지 않습니다. 그림만 다시 만들려면 `python scripts/preview_edits.py --run-dir assets/outputs/accident_001`을 사용합니다.

탈락 이미지·점수·receipt·mask·원본 설정 그림·비교 그림·썸네일을 지웁니다. 실패 비용과 재개를 위한 최소 ID·상태·시간만 남깁니다. 단일 작업에서 예를 들어 탈락 40초, 탈락 50초, 성공 60초라면 성공 시도는 60초, **실패 포함 획득 시간은 150초**입니다. 초기화/정리 비용을 포함하고 중단 대기 시간은 제외합니다. 마지막 실패/종료 비용은 pending에 남깁니다. 병렬 실행의 `accepted_cost_seconds`는 공유 run의 채택 commit 사이 active 시간이며, 각 후보의 경과 시간과 구분합니다. 동시에 끝난 두 후보의 비용을 두 번 더하지 않습니다. 후보별 시간은 서로 겹칠 수 있고, 모든 채택 비용 + pending은 run active 시간과 같습니다.

기존 Qwen 채택 이미지는 보존하고 `legacy_unverified`로 구분합니다. legacy export는 `exports/legacy_images.jsonl`에 남기며 새 CLIP export/목표 N에는 포함하지 않습니다. VQA run을 CLIP으로 강제 resume하지 않습니다. 기존 `configs/advv.local.yaml` 대신 새 `configs/advv.clip.local.yaml`을 사용하세요.

학습은 확정 export manifest를 사용합니다. 현재는 image-only이며 box/mask/class 정답을 자동 승계하지 않습니다. 같은 사고 영상의 원본·파생본은 같은 group이어야 하며 val/test는 증강하지 않습니다.

## 새 서버 설치

```bash
conda env create -f environments/advv-conda.yml
conda activate ADVV
python -m pip install -e '.[dev,download]'
```

이미 `ADVV` 환경이 있으면 그 환경에서 `python -m pip install -r environments/advv-requirements.txt -e '.[dev,download]'`로 설치합니다. 기존 분리 환경 freeze를 합치거나 venv를 만들지 않습니다. [현재 freeze](environments/advv.freeze.txt)는 검증 서버의 기록이며 재설치에는 requirements를 사용합니다.

공식 DragFlow checkout이 없는 새 서버에서만 다음을 실행합니다.

```bash
git clone https://github.com/Edennnnnnnnnn/DragFlow.git third_party/DragFlow
git -C third_party/DragFlow checkout b3a8fa7136df5131d07f98425c0d189fb6ca74b0
git -C third_party/DragFlow submodule update --init --recursive
```

[FLUX.1-dev](https://huggingface.co/black-forest-labs/FLUX.1-dev) 접근 조건에 동의하고 Hugging Face 인증이 필요한 경우 `huggingface-cli login`을 사용합니다. 토큰을 코드·설정에 적지 않습니다.

```bash
advv prepare
python -m pytest -q
python -m ruff check src scripts tests
```

`prepare`만 모델을 다운로드합니다. Qwen은 새 방식의 필수 모델이 아니며 worker는 local-only입니다. Mask R-CNN 가중치는 고정 SHA256으로 확인합니다. `prepare`는 기존 로컬 설정을 덮어쓰지 않습니다. 가중치·입력·실험 결과·인증 정보는 Git에 포함하지 않습니다.

## 문서

[AGENTS.md](AGENTS.md) · [SPEC.md](SPEC.md) · [구현 상태](docs/IMPLEMENTATION_PLAN.md) · [구조](docs/ARCHITECTURE.md) · [시각화](docs/VISUALIZATION.md) · [평가 계획](docs/EVALUATION.md) · [근거](docs/REFERENCES.md)
