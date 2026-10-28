# ADV

DragFlow를 이용한 차량 사고 블랙박스 **단일 프레임 이미지 증강** 파이프라인입니다.
사고 관련 차량 전체를 같은 방향과 거리로 이동시킨 후보를 생성하고, CLIP 검증을 통과한 고유 이미지가 요청한 총 N장이 되면 종료합니다.

## 동작

1. Mask R-CNN으로 차량을 분할하고, CLIP의 차량·주변 맥락 비교로 사고 관련 차량을 선택합니다.
2. SegFormer로 도로·인도·잔디/흙 지면을 분할합니다.
3. 사고 차량들의 상대 위치를 유지하는 공통 이동을 무작위로 선택합니다. 각 차량의 아래쪽 접촉 근사점이 요청 좌표와 실제 feature-grid 좌표 모두에서 지면 안에 있어야 합니다.
4. 공식 DragFlow와 FLUX.1-dev로 원본에서 새 후보를 생성합니다.
5. [고정 CLIP 문장 5쌍](prompts/clip_criteria_v1.json)을 비교하고 채택 조건을 확인합니다.

각 쌍의 선호 문장 점수가 비선호 문장 점수보다 높을 때 통과합니다. 동점은 실패입니다.
**1~4번 물리 항목 중 3개 이상 통과하고, 5번 사고 장면 항목은 반드시 통과해야 채택합니다.**
5개 전부 통과하는 방식은 사용하지 않습니다.

CLIP과 지면 분할은 대리 지표입니다. 실제 차량 이동 성공·접지·사고 유형 보존을 보증하지 않으므로 결과 이미지와 계획을 함께 확인해야 합니다.

## 설치

Linux, Git, conda, NVIDIA GPU와 호환 드라이버가 필요합니다. DragFlow 작업 하나에 GPU 두 개를 사용합니다.
개발 서버는 RTX 4090 24GB를 사용했으며 이미지 크기와 설정에 따라 메모리 사용량이 달라집니다.

```bash
git clone https://github.com/sangpil0104/ADVV.git advv
cd advv
conda env create -f environments/advv-conda.yml
conda activate ADVV
python -m pip install -r environments/advv-requirements.txt
python -m pip install -e .
```

이미 `ADVV` 환경이 있으면 환경 생성은 생략하고 그 환경에서 설치합니다. ZIP으로 받았다면 압축을 푼 `advv` 디렉터리에서 환경 생성부터 실행합니다.
CLI가 설치된 소스 디렉터리를 기준으로 설정·입출력 경로를 찾으므로 **editable 설치(`-e .`) 후 이 디렉터리를 유지하세요.**

공식 DragFlow 코드를 고정된 revision으로 준비합니다. 외부 코드와 가중치는 이 저장소에 포함하지 않습니다.

```bash
mkdir -p third_party
git clone https://github.com/Edennnnnnnnnn/DragFlow.git third_party/DragFlow
git -C third_party/DragFlow checkout b3a8fa7136df5131d07f98425c0d189fb6ca74b0
git -C third_party/DragFlow submodule update --init --recursive
```

사용하는 모델은 FLUX.1-dev, InstantCharacter, SigLIP, DINOv2, CLIP ViT-L/14, Mask R-CNN, SegFormer-B0입니다.
고정 revision과 체크섬은 [upstream.lock.json](configs/upstream.lock.json)에 있습니다.
각 외부 코드·모델의 라이선스와 사용 조건을 따릅니다. FLUX 접근에는 [모델 페이지](https://huggingface.co/black-forest-labs/FLUX.1-dev)에서 접근 승인이 필요할 수 있습니다.

```bash
# 다운로드 권한이 있는 Hugging Face 계정으로 인증
huggingface-cli login

# 가중치 다운로드와 로컬 경로 설정 생성. 추론은 시작하지 않습니다.
advv prepare --project .
```

가중치는 `models/`에 저장하고 `configs/advv.clip.local.yaml`을 자동 생성합니다.
인증 토큰을 코드나 설정 파일에 적지 마세요. 다운로드 이후 추론 worker는 로컬 가중치를 사용합니다.
새 설치에는 충분한 디스크 공간과 다운로드 시간이 필요합니다.

## 실행

원본 이미지를 `assets/inputs/`에 넣습니다. 예를 들어 `assets/inputs/image3.png`를 대상으로 10장을 생성하려면:

```bash
conda activate ADVV
advv run 10 --gpus 4,5,6,7 --image image3.png
```

**자동 nohup 백그라운드 실행**입니다. 별도 `nohup`이나 `&`를 붙이지 않습니다.
`[4,5]`, `[6,7]` 두 작업이 병렬로 후보를 만들고 **합계 10장**을 공유합니다. GPU 두 개만 쓰려면 `--gpus 4,5`로 지정합니다.
`--image`를 생략하면 입력 폴더 전체를 대상으로 총 N장을 만듭니다.

| 옵션 | 의미 |
| --- | --- |
| `N` | 입력 전체에서 얻을 새 고유 채택 이미지 총량 |
| `--image FILE` | 입력 폴더 기준 상대 경로의 한 이미지만 선택 |
| `--prompt "설명"` | 선택적인 생성 지시. CLIP 판정 규칙은 유지 |
| `--repair-mode endpoints` | 기본값. 원래 차량 위치와 도착 주변만 배경 수정 영역으로 사용 |
| `--repair-mode swept` | 이동 경로 전체까지 배경 수정을 허용하는 이전 방식 |
| `--run-id NAME` | 실행 이름. 생략하면 자동 생성 |
| `--config PATH` | 자동 로컬 설정을 대신할 YAML |
| `--foreground` | 백그라운드 대신 터미널에서 직접 실행 |

배경 수정 방식 비교 예시입니다. 두 명령 중 원하는 하나를 실행합니다.

```bash
advv run 10 --gpus 4,5,6,7 --image image3.png --repair-mode endpoints
advv run 10 --gpus 4,5,6,7 --image image3.png --repair-mode swept
```

두 방식 모두 차량 이동 mask와 좌표는 같고 배경 복원 support만 바뀝니다. 8px 여유와 실제 좌표 반올림을 반영합니다.
짧은 이동에서는 출발·도착 영역이 겹칠 수 있고, latent 디코딩 때문에 편집 영역 밖 픽셀의 완전 동일성은 보장하지 않습니다.

정상 탈락·중복에는 총 시도/시간 상한이 없습니다. N장 채택 또는 사용자 중단까지 계속하며 기술적 오류는 별도로 처리합니다.
원본과 동일한 이미지와 이미 저장한 이미지의 픽셀 중복은 N에 포함하지 않습니다.

## 로그·중단·재개

실행 직후 PID, run 폴더, 로그 경로가 출력됩니다. 초기 CPU 모델 로딩·양자화 중에는 GPU 사용이 늦게 나타날 수 있습니다.
`0/N`은 채택된 이미지가 아직 없다는 뜻입니다. 로그에는 약 60초마다 집계 상태가 기록됩니다.

```bash
tail -n 30 -f logs/latest.log
cat logs/latest.json

# 최근 자동 실행 중단. 두 GPU 작업도 함께 종료합니다.
kill -TERM "$(cat logs/latest.pid)"
```

여러 실행이 있으면 각 실행에서 출력한 개별 로그/PID를 사용하세요. `tail`의 Ctrl+C는 로그 보기만 종료합니다.

```bash
# RUN_ID를 실제 중단한 실행 폴더 이름으로 바꿉니다.
advv run --resume --run-dir assets/outputs/RUN_ID --gpus 4,5,6,7
```

재개는 저장된 입력·수량·prompt·배경 복원 모드를 사용합니다. N, `--image`, `--prompt`, `--repair-mode`를 다시 주지 않습니다.
코드·모델·설정이 변경됐다면 새 실행을 시작합니다. 실행 중 소스를 수정하면 코드 hash 검증으로 worker가 종료될 수 있습니다.

## 결과

결과는 `assets/outputs/<run_id>/`에 저장됩니다.

| 경로 | 내용 |
| --- | --- |
| `accepted/` | 표시 없는 채택 이미지 |
| `visualizations/comparison/` | 원본·계획·생성 결과 비교 |
| `visualizations/drag_plan/` | start/end, 전체 차량 영역, 목표 위치 계획 |
| `visualizations/metadata/` | 실제 적용 좌표·mask·판정·시간 정보 |
| `visualizations/generated_grid.png` | 생성 이미지 모음 |
| `visualizations/drag_plan_grid.png` | 설명 글씨 없는 계획 사진 모음 |
| `segments/<source_id>/selection.png` | 원본의 사고 차량 선택 확인 |
| `segments/<source_id>/ground_preview.png` | 지면 분할 확인 |
| `records/` | 채택 후보의 계획·점수·시간 기록 |
| `exports/images.jsonl` | 학습 입력으로 사용할 채택 이미지 목록 |
| `report.json`, `report.md` | 실행 집계 |

탈락 후보의 이미지·상세 점수·시각화는 삭제하며, 시간 집계와 재개에 필요한 최소 장부만 보존합니다.
각 후보 소요 시간과 앞선 실패를 포함한 채택 간 active 시간을 기록합니다. 병렬 후보의 시간을 더하면 실제 실행 시간이 되지 않습니다.
ground는 생성 전 위치 제약이고 drag plan의 목표 윤곽은 계획입니다. 실제 출력 차량 위치를 측정한 윤곽이 아닙니다.

추론 없이 모음·보고서·export를 다시 만들 수 있습니다.

```bash
advv visualize --run-dir assets/outputs/RUN_ID
advv report --run-dir assets/outputs/RUN_ID
advv export --run-dir assets/outputs/RUN_ID
```

입력 검사는 실행 전에 다음처럼 확인합니다.

```bash
advv preflight --target-count 10 --gpus 4,5,6,7 --image image3.png
```

`no_eligible_sources`는 선택 가능한 사고 차량이 없다는 뜻이고, `no_ground_feasible_sources`는 선택 차량 전체를 지면 제약 안에서 이동시킬 수 없다는 뜻입니다.
이 경우 원본의 `selection.png`와 `ground_preview.png`를 확인하세요.
출력은 image-only이며 객체 box·분할 mask·class 정답을 자동 승계하지 않습니다. 원본과 파생본은 같은 데이터 분할에 두고 val/test는 증강하지 않습니다.
