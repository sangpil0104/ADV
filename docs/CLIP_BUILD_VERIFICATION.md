# CLIP 전환 빌드 검증

## 2026-10-11: GitHub 업데이트 전 검증

- 최신 소스 전체를 conda `ADVV`, `CUDA_VISIBLE_DEVICES=''`에서 검사하여 **148개 테스트 통과**, lint 통과를 확인했다. 기존 matplotlib deprecation warning 14건은 실패가 아니다.
- 커밋에는 소스·문서·테스트·설정 예제·conda 환경 정의를 포함한다. 원본/생성 이미지·모델·인증 정보·로컬 설정·로그·PID·데이터 압축 파일은 제외한다. GPU 추론은 실행하지 않았다.

## 2026-10-11: drag plan 모음을 사진만 표시하도록 변경

- 모음의 제목·설명·점수·ID와 사진 위 S/E/A 문자·차량 번호를 제거했다. 저장된 원본·계획·실효 geometry로 사진과 화살표·편집 영역만 CPU 렌더링한다. 원본 사진 자체의 문자는 유지한다. 개별 drag plan PNG와 생성 이미지 모음은 그대로 둔다.
- 기존 6개 실행, 채택 51장에 적용하고 최신 10장 모음을 직접 확인했다. 개별 시각화·채택 이미지·record·시간 등 **474개 기존 파일의 hash 불변**을 확인했다. 감사 기록은 `/tmp/advv-plan-photos-audit-xgbtqjhq`에 있다.
- conda `ADVV`, GPU 비활성 상태에서 관련 시각화·정리 회귀 **16개 통과**, lint 통과. 모음 재생성 때 텍스트를 그리지 않는지, 실제 사진 영역의 픽셀·배치, 순서·legacy 정보 보존, 누락 복구·반복 실행 무변경을 확인했다. 전체 회귀에서 발견한 빈 legacy 실행의 시각화 설정 누락도 수정하고 해당 검사를 통과했다.
- 렌더링 버전을 `accepted_drag_plan_photos_v2`로 갱신하여 기존 모음도 자동 교체한다. 향후 채택 commit·종료/report·CPU visualize에 동일하게 적용된다. GPU 추론과 wheel 재빌드는 수행하지 않았다.

## 2026-10-11: drag plan 전체 모음 추가

- `visualizations/drag_plan_grid.png`와 `drag_plan_grid.json`을 추가했다. 생성 이미지 모음과 같은 번호·순서로 개별 계획 전체를 표시하며, 채택 commit·종료/report·CPU `advv visualize`에서 자동 갱신한다.
- conda `ADVV`, `CUDA_VISIBLE_DEVICES=''`에서 전체 CPU 테스트 **148개 통과**, lint 통과. 모음과 실제 계획 썸네일의 픽셀 일치, 번호/순서, 탈락 제외, legacy 표시, 빈 실행, 누락된 모음 복구, 반복 실행 무변경을 확인했다. 기존 matplotlib deprecation warning 14건은 테스트 실패가 아니다.
- 기존 6개 실행의 채택 결과 총 **51장**에 적용했다. 사진·개별 시각화·record·시간·보고서 등 **498개 기존 파일의 hash 불변**과 모든 실행의 반복 갱신 시 내용/mtime 불변을 확인했다. 최신 실행의 10장 계획 모음도 직접 열어 배치를 확인했다.
- 감사 기록: `/tmp/advv-drag-plan-grid-audit-07odu5c7/before.json`, 같은 폴더의 `after.json`. GPU 추론은 실행하지 않았다. 현재 editable 소스에 반영했으며 아래 이전 wheel은 이번 추가 전 빌드다.

## 2026-10-11: v0.4.1 종류별 시각화 폴더와 생성 이미지 모음

- 사용자가 확정한 구조는 `comparison/`, `drag_plan/`, `metadata/` 폴더에 개별 파일을 모으는 방식이다. `generated_grid.png`와 번호/ID 대응 JSON, `index.json`을 추가했다.
- `conda ADVV`, `CUDA_VISIBLE_DEVICES=''`에서 **전체 CPU 테스트 148개 통과**, lint 통과. conda ADVV editable 설치와 wheel 빌드 완료.
- wheel: `dist/advv-0.4.1-py3-none-any.whl`, SHA256 `ff19a8ca29f946837a5e082407867a072642ec486b1ea3df7bf2087223c65a31`.
- 채택 commit과 종료/report 때 모음 자동 갱신, acceptance sequence 순서, 탈락/중복/미완료 제외, 빈 실행, 기존 Qwen의 `legacy_unverified` 표시를 확인했다.
- `advv visualize --run-dir PATH` / `--all`은 모델·GPU·resume 없이 기존 결과를 이전한다. PNG 바이트는 그대로 복사하고 metadata·record·report·export 경로를 갱신한 뒤 옛 파일을 정리한다. writer lock, 반복 실행의 무변경, gallery 누락 복구, 이미지 hash 불일치/metadata ID 불일치 거부, record 갱신 뒤 중단된 이전의 재개를 검사했다.
- 기존/새 폴더의 탈락 파일 및 atomic 임시 파일 정리, 기존 quota·병렬 처리·중단·시간 회귀도 통과했다.

실제 기존 결과에 `advv visualize --all`을 적용했다. 아래 합계 **51장**의 채택 이미지는 재생성/재판정하지 않았다.

| 실행 | 채택 이미지 수 |
| --- | ---: |
| `assets/outputs/run_20261008T044724Z_b19495a6` | 0 |
| `assets/outputs/run_20261008T044809Z_4f13c745` | 10 |
| `assets/outputs/run_20261008T092130Z_0309e810` | 10 |
| `assets/outputs/run_20261009T123128Z_1af42902` | 10 |
| `runs/first_trial_001` (legacy) | 1 |
| `runs/trial_10_001` (legacy) | 20 |

이전 전후 확인: 원본·채택 이미지·설정·state·시간 등 **145개 파일의 hash 불변**, **51개 record는 visualization 필드 외 동일**, **102개 비교/계획 PNG의 hash 불변**, metadata는 경로/layout 외 동일. 보고서/export 18개 파일은 해당 경로 교체 외 내용이 같음을 확인했다. 모든 실행에서 반복 정리 시 파일 내용/mtime 불변을 확인했다. 최신 실행의 10장 grid를 직접 열어 빠진 이미지 없이 배치됨을 확인했다.

감사 snapshot은 `/tmp/advv-visualization-audit-0h_ma8gi/before.json`, 확인 결과는 같은 폴더의 `after.json`에 남겼다. 이 작업은 CPU 파일 재정리이며 GPU 추론/생성 품질 검증은 수행하지 않았다.


## 2026-10-09: v0.4.0 사고 관련 차량 전체 공동 이동

- conda `ADVV`에 v0.4.0 editable 설치, wheel 빌드 완료. 새 의존성·모델 다운로드 없음.
- `CUDA_VISIBLE_DEVICES='' ... pytest -q`: **140개 통과**, `ruff check src scripts tests` 통과. upstream matplotlib의 deprecation warning 14건이 있으며 테스트 실패는 없다.
- wheel: `dist/advv-0.4.0-py3-none-any.whl`, SHA256 `aeb4d0c62dcd98688824b52027fcd31b091d5215c342241fc37a29996dcd69f7`.
- 새 CPU 계약 검사: 전체 선택 mask 합집합·다중 성분·내부 구멍 보존, 무관 차량 제외, 단독 사고, 공통 Δ/전체 영역 프레임 내 유지, 작은 차량 feature 소실 거부, 누락/부분 mask/선택 변조 거부, selector 양쪽 view/동점/비유한 점수/선택 없음, source preview.
- 고정 공식 `dashboard_utils.scale_coordinates`와 `DynamicRegionEstimator.estimate_inprocessing_state`를 CPU에서 실제 호출했다. centroid 교체 후 공통 Δ, 떨어진 두 차량과 빈 공간의 affine 이동, hook 복구, 예상과 다른 grid 거부를 확인했다. Dragger 모델 전체를 실행한 검사는 아니다.
- `masked_template_match_v1` 진단에서 알려진 이동과 identity를 구분하고, 저텍스처/매칭 오류를 진단 상태로만 기록하며 추가 채택 threshold가 되지 않는지 확인했다.
- 기존 GPU-pair fake 병렬 quota·총량·중복·재개·정리·시간·prompt·단일 이미지 회귀도 함께 통과했다.

### 실제 image1의 CPU 점검

- 새 `VehicleSegmenter`를 실제 COCO_V1 가중치로 CPU 실행한 뒤 같은 프로세스에서 로컬 CLIP 선택을 실행했다. 임시 폴더 `/tmp/advv-group-check-lykwgpoc`에만 source/선택 preview/계획을 저장했다.
- 차량 후보 `_segment_0003`, `_segment_0004`를 선택하고 `_segment_0005`는 제외했다. 전체 합집합 면적 **107,745px**. preview에서 전면 사고 차량 두 대가 포함되는 것을 확인했다.
- 관계 단위의 foreground/context 사고 margin은 약 **+0.03215 / +0.02871**이었다. 이는 source 선택용 점수이며 후보 검증의 고정 10문장 점수와 다르다.
- 세 계획에서 mask 면적은 같고 Δ는 달랐다. 모든 구성 차량에 같은 Δ를 사용하며 원본과 feature grid 경계를 확인했다. `torch.cuda.is_initialized() == False`.
- selector 설계 과정에서 image1을 사용했다. 개별 차량 crop 방식은 이 이미지에서도 놓침이 있어 접촉 관계를 보존한 비교로 수정했다. **이 결과는 개발 점검이며 독립 평가/다른 사고 유형의 일반화 검증이 아니다.** 원근상 겹침·오검출·가려진 차량·떨어진 사고 차량에 대한 추가 평가가 필요하다.

### 남은 실제 검증

새 그룹 경로의 GPU DragFlow 생성·CLIP 통과율·실제 차량 이동/관계 보존·VRAM/처리량은 실행하지 않았다. 모델 생성물의 실제 이동 성공을 CPU 입력 계약이나 CLIP 통과로 대신 인증하지 않는다. 기존 run/입력/채택 결과는 수정·삭제하지 않았고 공식 DragFlow tracked checkout은 변경하지 않았다. 이전 부분 편집 run을 resume하지 말고 새 run을 실행한다.


## 2026-10-08 추가: 단일 이미지 선택

- `run/preflight --image image1.png`를 추가했다. 입력 폴더 기준 상대 경로이며 CLI > YAML `dataset.image` > 전체 폴더 순이다.
- conda ADVV에서 GPU를 숨기고 전체 CPU 테스트 **126개 통과**, lint 통과, v0.3.0 wheel 재빌드 성공.
- 새 wheel SHA256: `e0fb45b83cdae04cb65860919e3b3594115fd3e936b1536b373006b3fe969639`.
- 단일 원본에서 두 GPU 묶음용 fake backend로 총 10장 채택, 다른 정상/손상 이미지 제외, 원본 무변경, 선택값의 recipe 저장과 snapshot 재개, 파일 누락 시 정리/모델 접근 전 실패, 외부 경로/미지원 형식/manifest 혼용 거부를 확인했다.
- 실제 `/tmp`에서 기본 설정을 읽어 `image1.png`만 CPU ingest한 결과 `eligible=1`, `total_unique=1`이었다. 나머지 입력 파일을 옮기거나 삭제하지 않았다.
- 실제 GPU 생성은 실행하지 않았다. 아래 113개 검사는 단일 이미지 옵션 추가 전의 기록이다.

## 2026-10-08: v0.3.0 병렬 처리와 기본 경로

- conda `ADVV`에 v0.3.0 editable 설치 완료. 의존성/모델을 새로 내려받거나 GPU 추론을 시작하지 않았다.
- `CUDA_VISIBLE_DEVICES='' ... python -m pytest -q`: **113개 통과**. `ruff check src scripts tests` 통과.
- `advv-0.3.0-py3-none-any.whl` 빌드 성공. SHA256: `da323e99825cd469367b1997a1146e2f68ba0d1adc3bfd7cba5a038c1a95ea28`.
- 실제 thread 동시 실행을 barrier로 검증하는 CPU fake backend 검사: 두 묶음 합계 정확히 10장, 동시 중복/탈락 제외, 남은 수량 1일 때 단일 후보 실행, 빠른 묶음 재할당, 공유 시간 보존, 모든 묶음 중단, GPU 묶음 수를 바꾼 재개, scored/committed 결과의 재추론 방지.
- worker 장치 격리 검사: 생성기 `4,5`/`6,7`, 검증기 `4`/`6`. 취소 후 새 worker 생성 차단, 중단 도중 이미 scored인 탈락 정리, 정리 실패 journal의 부분 시간 보존을 검사했다.
- 실제 `/tmp`에서 `advv preflight --target-count 10 --gpus 4,5,6,7` 실행: **ready**, 기본 설정 자동 탐색, 입력 9장 확인. import는 CUDA를 숨긴 subprocess에서 검사하며 `nvidia-smi`는 장치 정보를 읽기만 했다.
- `--input-dir` 제거, 자동 run ID, cwd와 무관한 기본 설정, 기본 assets/inputs 및 assets/outputs 경로를 검사했다.
- `assets/image1.png`~`image9.png`를 `assets/inputs`로 이동하고 이동 전후 파일 SHA256 동일성을 확인했다. `assets/outputs`를 생성했다. 기존 `runs/` 결과는 이동/삭제하지 않았다.

**이 검증은 실제 GPU 4·5/6·7 동시 DragFlow 생성·CLIP 품질·VRAM·처리량 검증이 아니다.** 실제 모델의 전체 통합 실행은 아래와 같이 아직 남아 있다. 입력 이동 외의 실험 결과 변경, 실제 후보 생성, 기존 run 정리는 실행하지 않았다.

## 2026-10-07: v0.2.0 단일 작업 기반 검증 기록

확인일: **2026-10-07**. v0.2.0 / schema 2.0. GPU 추론과 실제 과거 run 정리는 실행하지 않았다.

## 실행한 검사

| 검사 | 결과와 범위 |
| --- | --- |
| conda | `/home/sppark/miniconda3/envs/ADVV/bin/python`, Python 3.10.21 |
| 통합 의존성 | torch 2.5.1, torchvision 0.20.1, Transformers 4.48.0, Diffusers 0.32.2; pip check 통과 |
| CPU 회귀 | `python -m pytest -q`: **93개 통과** |
| 정적 검사 | `python -m ruff check src scripts tests` 통과 |
| 배포 빌드 | `advv-0.2.0-py3-none-any.whl` 생성 성공 |
| 공식 Dragger | 동일 conda에서 core import 성공; 모델 로드/생성은 미실행 |
| CLIP | 로컬 ViT-L/14, CPU에서 회색 단일 프레임과 10문장 forward; 유한 cosine 10개 확인 |
| Segmenter | 고정 COCO_V1 Mask R-CNN checkpoint 로드·CPU forward; 회색 프레임에서 차량 영역 0개 |
| GPU 사용 | 모델 확인 때 `CUDA_VISIBLE_DEVICES=''`, `torch.cuda.is_initialized() == False` 확인 |

회색 프레임 검사는 API·가중치·전처리 연결 검사이며 사고 이미지의 판정 정확도/분할 품질 검사가 아니다. 전체 freeze는 [advv.freeze.txt](../environments/advv.freeze.txt), 재설치 조건은 [requirements](../environments/advv-requirements.txt)에 남겼다.

## 회귀 검사의 핵심 범위

- 5항목 32조합, 동점, NaN/Inf/누락/잘못된 점수, mandatory 5 조건.
- 선택 prompt의 CLI→설정→생성 전달 및 검증 문장 불변, 총량 N·중복 제외.
- 자유형 mask 재현성, 시작점의 segment/mask 내부 여부, 좌표 변환, EXIF와 split 누수.
- 탈락 상세 파일·그림·로그 정리, 최소 시간 선저장, symlink/hardlink/채택 참조/활성 writer 보호.
- 과거 VQA 비채택 정리와 부분 시간 이전, legacy 채택 보존 및 CLIP export 제외.
- 판정 중 중단, 채택 commit 전/후 중단, 삭제 중 장애, 시각화 장애/파일 누락 복구 시 재추론·중복 채택 없음.
- 실패 40+50초와 성공 60초 비용 집계, pause 제외, trailing pending 보존, crash의 부분 측정 표시.
- 동일 conda interpreter worker, 다른 환경/venv 경로 거부, fake production export 거부.

실패·중단·삭제 검사는 임시 디렉터리의 fixture에서 수행했다. 실제 assets 원본, 기존 VQA 채택 이미지와 실험 결과를 수정/삭제하지 않았다.

## 아직 수행하지 않은 검사

- 실제 사고 프레임에서 segmentation→DragFlow GPU 생성→CLIP→채택까지 전체 통합 실행.
- 자유형 영역의 실제 편집 품질, 차량 검출 누락률, CLIP의 사고/물리 판정 성능.
- 새 조합의 GPU 메모리·전체 처리량, 실제 모델 중단/재개, 후속 자율주행 과제 성능.

실제 GPU 실행 요청이 주어지면 작은 N부터 확인한다. 과거 Qwen 방식의 GPU 결과를 새 방식의 성공 근거로 사용하지 않는다.
