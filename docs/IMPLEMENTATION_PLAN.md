# CLIP 전환 구현 계획

## 2026-10-11: 종류별 시각화 저장

- [x] comparison/drag_plan/metadata 폴더별 개별 파일 저장과 생성 이미지 전체 grid 자동 갱신.
- [x] CPU `advv visualize` 명령, 기존 PNG 무변경 이전, metadata·record·report·export 경로 갱신, lock/중단 복구.
- [x] 실제 기존 결과 51장 적용, 보호 대상 145파일·시각화 PNG 102개 hash 불변 확인.
- [x] CPU 전체 148검사·lint·v0.4.1 설치/wheel 빌드 완료. GPU 추론 없음.

## 최신 전환: 사고 관련 차량 전체 공동 이동

2026-10-09, [SPEC v0.9](../SPEC.md), **v0.4.0**. 사용자 확정 범위는 사고 관련 차량 전체만이다.

- [x] `accident_group.py`: 접촉/겹침 관계를 유지한 객체 cutout·맥락 crop의 로컬 CLIP 비교. 단독 사고도 평가하고 선택 0대이면 보류한다. 선택 ID·점수·방법·문장/모델 identity를 cache한다.
- [x] `segmentation.py`, `contracts.py`: 각 전체 mask와 합집합·차량 ID/hash/면적. 연결 성분·내부 빈 공간을 제거하지 않는다.
- [x] `sampler.py`, `group_geometry.py`, `config.py`: 부분 면적/확장 설정 폐기, 공통 비항등 Δ와 원본/feature grid 프레임 내 유지.
- [x] `group_adapter.py`, `dragflow.py`: contour loader 우회, 전체 단일 region 전달, centroid 교체 후 공통 Δ 유지, 배경 복원 support 분리.
- [x] `visualization.py`: 생성 전 selection preview; 채택 후보의 전체 mask·차량 ID·공통 화살표·계획 목표 윤곽.
- [x] `motion.py`: 차량별 template matching Δ·잔차 진단. 불확실한 결과를 이동 성공으로 인증하거나 채택 조건에 추가하지 않는다.
- [x] CPU 회귀와 실제 image1의 Mask R-CNN+CLIP CPU forward. 자세한 수량/범위는 [빌드 검증](CLIP_BUILD_VERIFICATION.md).
- [ ] 사용자 실행 시 새 run의 실제 GPU DragFlow 그룹 이동, 차량 누락/충돌 관계 보존, VRAM과 처리량을 검증한다.
- [ ] 별도 사고 데이터와 사람 검수로 source selector의 완전성/오선택률, 출력 이동 진단 정확도·허용 오차를 검증한다.

source selector는 보정되지 않은 baseline이다. 개발 이미지 image1의 선택 확인을 다른 사고/단독 사고의 일반화 근거로 사용하지 않는다. 이전 run은 보존하며 새 태스크 성공 사례로 재분류하지 않는다.

## 이전 v0.3.0 구현 기록

기준: [SPEC v0.7](../SPEC.md), 2026-10-08. **v0.3.0에 GPU 쌍별 병렬 처리와 고정 기본 경로를 추가했다.** 아래 완료 표시는 소스·계약 검사 범위이며, P5의 실제 GPU 통합과 사고 도메인 품질 평가는 미완료다. [빌드 검증 범위](CLIP_BUILD_VERIFICATION.md)를 함께 확인한다.

## P0. 요구사항과 전환 계약

- [x] 차량 사고 블랙박스 단일 프레임으로 주제를 좁힌다.
- [x] segment 내부 start, 무작위 end·range, 자유형 edit mask의 의미를 고정한다.
- [x] 영어 CLIP 서술문 10개와 `3-of-4 AND mandatory-5` 정책을 버전 파일로 만든다.
- [x] 5번을 추돌에 한정하지 않는 일반 교통사고 vs 정상 교통으로 정의한다.
- [x] 선택 `--prompt`의 적용 범위·기본값·우선순위·재개 제약을 문서화한다.
- [x] 기존 비채택 상세 산출물의 정리와 최소 시간 장부 예외를 정의한다.
- [x] 시도 시간·실패 포함 획득 시간·중단 후 pending 보존을 정의한다.
- [x] 모든 worker를 conda `ADVV`로 통일하는 변경 경계를 정한다.

선정한 segmenter는 torchvision Mask R-CNN ResNet50-FPN COCO_V1이다. CLIP·segmenter의 실제 CPU forward와 단일 환경 Dragger import는 확인했다. 이전 GPU 중단 요청을 유지하여 실제 생성은 시작하지 않았다.

## P1. 계약·CLI·환경 준비

수정 대상: `src/advv/{cli,config,contracts,prepare,preflight}.py`, `backends/{process,worker}.py`, 설정·환경 명세.

- [x] schema 2.0과 Segment/CLIPResult/TimingRecord를 추가한다. 기존 schema 1.1 run과 명확히 구분한다.
- [x] argparse `run`에 선택적 `--prompt`를 추가하고 `generation.augmentation_prompt`로 정규화한다. 빈 값·재개 중 변경은 모델 로드 전에 거부한다.
- [x] `--prompt` 생략 시 기본 사고 장면 지시, CLI > YAML > 기본값 우선순위를 적용한다. 숫자 N/`--target-count` 기존 규칙은 유지한다.
- [x] Qwen 필수 다운로드·SourceProfile·VQA schema 연결과 별도 interpreter 설정을 새 경로에서 제거한다.
- [x] 공식 segmenter API·COCO class·전체 checkpoint SHA256을 확인하고 실제 CPU 연결을 검사했다. 실제 GPU 자원 사용량은 P5에서 확인한다.
- [x] **conda ADVV 안에서** DragFlow·CLIP·segmenter 의존성을 resolve하고 import/프로세스 호환성을 검사한다. 다른 환경으로 우회하지 않는다.
- [x] 모든 worker를 같은 `sys.executable`로 실행하고 실제 prefix를 검증한다. `prepare`의 venv 경로 생성과 preflight의 venv 전제를 제거한다.
- [x] 검증한 단일 conda 파일/freeze와 새 설정 예시를 작성한다. 기존 freeze는 과거 기록으로 구분한다.

완료 기준: GPU 생성 없이 새 CLI·설정·동일 환경 worker 계약을 검사할 수 있다. 의존성 설치 성공과 실제 모델 연결 성공을 별도로 보고한다.

## P2. Segmentation·자유형 sampler·생성 prompt

수정 대상: `segmentation.py`, `backends/segmentation.py` 신규, `sampler.py`, `backends/dragflow.py`.

- [x] 원본별 segment cache에 원본 hash·모델 revision·전처리·mask를 저장한다. 다른 설정/원본의 cache를 재사용하지 않는다.
- [x] 차량 segment 선택, 내부 start, 랜덤 end, 랜덤 면적의 연결된 자유형 편집 mask를 만든다. bbox/타원 fallback을 두지 않는다.
- [x] `range_fraction_of_image`와 요청/실현 `area_pixels`의 정의·허용 오차·초기 분포를 고정한다. 이동 거리와 독립된 파라미터로 구현한다.
- [x] 비볼록 영역 centroid, 경계 start, 작은 mask, feature-grid 양자화 후 항등 이동을 검사한다.
- [x] upstream 실제 start/area와 요청값의 차이를 처리한다. 계약을 깨는 실제 centroid는 계획 재표본/명시적 실패로 처리한다.
- [x] 기본 도메인 source prompt + 선택 원본 설명 + 증강 prompt를 실제 target prompt로 연결한다. CLIP 문장과 별도의 입력으로 유지한다.
- [x] 사용 가능한 source가 없으면 명시적인 종료 상태를 기록한다. 매번 입력 원본에서 새 후보를 생성한다.

완료 기준: 같은 seed/source/attempt에서 같은 계획을 재현하고 mask·좌표를 실제 adapter에 전달할 수 있다. 자유형 mask를 그린 그림만으로 실제 적용 검증을 완료했다고 보고하지 않는다.

## P3. CLIP verifier와 정책

수정 대상: `backends/clip.py` 신규, `verifiers/decision.py`, `pipeline.py`, worker protocol.

- [x] [10개 문장](../prompts/clip_criteria_v1.json)을 고정 순서로 읽고 모델/processor/문장 hash로 text embedding을 cache한다.
- [x] 전체 프레임 letterbox·normalization·tokenization을 고정한다. 사고 가장자리 정보의 crop 손실과 조용한 text truncation을 막는다.
- [x] float32 정규화 embedding으로 raw cosine 10개를 구한다. logit/probability와 혼동하지 않는다.
- [x] 모든 점수의 유효성을 먼저 검사하고 B−A>0, 1–4 중 3개 이상, 5번 필수 정책을 적용한다.
- [x] Qwen profile/두 VQA 호출을 새 production 경로에서 제거한다. 같은 유효 탈락 후보를 재검증하지 않는다.
- [x] 정확 중복 제외·총 N장·사용자 중단을 새 정책과 연결한다. 정상 탈락에는 총 시도 상한을 넣지 않는다.

완료 기준: 정책의 모든 Boolean 조합과 숫자 오류를 CPU fixture로 검사한다. 실제 CLIP 품질은 별도 평가 대상으로 남긴다.

## P4. 비채택 정리·시간·안전한 전환

수정 대상: `storage.py`, `pipeline.py`, `reporting.py`, `cleanup.py`/`timing.py` 신규.

- [x] pending 후보를 private staging에 두고 채택 확정 후에만 공개 이미지·상세 record를 만든다.
- [x] 비채택 후보의 이미지·receipt·mask·점수·원본 설정 그림·비교 그림·임시 사본을 모두 정리한다. 장부에는 최소 상태/시간만 남긴다.
- [x] 새 run과 resume 시작 시 관리 output root의 종료된 run을 lock한 뒤 기존 비채택 파일을 정리한다. 활성 run/원본/채택 파일은 보호한다.
- [x] 기존 상세 record에서 알려진 시간만 이전한다. 기존 VQA accepted는 보존하되 CLIP export/quota에서 제외한다. VQA run을 v2로 강제 resume하지 않는다.
- [x] timing/cursor → 후보 cleanup journal → 삭제 → 최소 tombstone과 반복 복구를 구현한다. 기존 run 정리의 전체 범위가 반환되기 전 shared log/report/export도 갱신한다.
- [x] 최소 시간 ledger를 쓰지 못하면 먼저 삭제하지 않는다. 정리 실패는 완료로 보고하지 않는다.
- [x] monotonic active segments, attempt/stage 비용, 앞선 실패를 다음 accepted cost에 배정하는 규칙을 구현한다.
- [x] pause downtime 제외, 재로드 비용 포함, 마지막 pending 보존, crash의 부분 시간 표시, 재개 중복 합산 방지를 구현한다.
- [x] 과거 report/log의 탈락 상세 내용도 제거한다. 판별 불가 혼합 파일은 `cleanup_blocked`로 알린다.

완료 기준: 기존 결과를 흉내 낸 임시 fixture에서만 먼저 삭제를 검증한다. 실제 기존 결과 정리는 구현·검증 후 사용자 실행의 명시된 정책으로 수행하며 이번 개발에서는 실제 기존 run 정리를 실행하지 않았다.

## P4.1 GPU 쌍별 병렬 실행과 기본 경로

- [x] GPU 목록을 두 개씩 묶어 독립 backend/worker에 배정하고 한 묶음 안의 단계는 순차 실행한다.
- [x] coordinator만 후보를 예약하고 전역 중복·채택·quota·공유 시간을 갱신한다. `accepted + in_flight <= N`을 유지한다.
- [x] 빠른 작업의 재할당, N=1, 동시 중복, 중단 시 모든 worker 종료, 다른 묶음 수로 재개를 CPU에서 검사한다.
- [x] `--input-dir` 제거, cwd와 무관한 기본 설정, 자동 run ID, assets/inputs 및 assets/outputs 경로를 적용한다.
- [x] 기존 assets 바로 아래 이미지 9장을 inputs로 이동하고 파일 SHA256이 변하지 않았음을 확인한다. 과거 runs 결과는 유지한다.
- [x] 개별 후보 시간과 겹침을 제거한 공유 run active 비용을 구분하여 문서화한다.
- [ ] 실제 GPU 4·5/6·7의 동시 생성·검증·10장 종료와 GPU 메모리/처리량을 검증한다.

## P5. 채택 전용 시각화·리포트·통합

- [x] 채택 후보만 실제 start/end/자유형 area를 표시한다. 요청값·실효값·면적·CLIP margin·실패 포함 시간을 연결한다.
- [x] 표시 없는 생성 이미지만 CLIP/학습 입력으로 사용하고 overlay hash가 입력에 섞이지 않는지 검사한다.
- [x] 사용자 보고서는 채택 이미지 상세와 집계/시간만 제공한다. 탈락 이미지·링크·점수를 재생성하지 않는다.
- [x] 이미지가 채택된 뒤 시각화/commit만 실패하면 추가 생성 없이 복구한다.
- [x] README를 구현한 CLI와 현재 로컬 준비 상태에 맞춘다. 실제 GPU 명령의 전체 통합 성공은 아직 주장하지 않는다.
- [ ] 추후 GPU 실행이 요청되면 작은 N으로 실제 segmenter→DragFlow→CLIP→저장·시간·정리를 검사한다. 중단/재개와 장치 해제도 확인한다.

완료 기준: conda ADVV 하나에서 선택한 GPU만 사용하고 총 N장의 고유한 채택 이미지를 만들며, 실패 상세 산출물을 남기지 않고 실패 비용까지 보고한다.

## 필수 회귀 검증 표

아래 계약과 GPU 쌍별 동시 실행·공유 quota를 CPU 회귀 테스트로 검증한다. 최신 개수와 결과는 [빌드 검증](CLIP_BUILD_VERIFICATION.md)을 따른다. 실제 GPU 모델의 동작·품질을 대체하는 검증은 아니다.

| 사례 | 기대 결과 |
| --- | --- |
| 5항목 Boolean 32조합 | 정확히 5조합 통과; 5번 실패는 전부 탈락 |
| 1–4 중 3개 통과 + 5 통과 / 2개 통과 + 5 통과 | 각각 통과 / 탈락 |
| A=B, NaN/Inf/점수 누락/0-norm | 동점은 해당 항목 실패; 숫자 오류는 검증 오류 |
| 선택 prompt 생략/CLI·YAML 중복/공백/재개 입력 | 기본/CLI 우선/오류/오류; CLIP 문장 hash 불변 |
| 같은 seed/source/attempt | 같은 segment·mask·좌표·크기 |
| 자유형 비볼록 mask, feature-grid 양자화 | 실제 start 내부·유효 이동, 요청/실효값 구분 |
| 원본 10장·N=3·탈락/중복 혼합 | 전체 총 3장; N 다음 생성 호출 없음 |
| 정상 탈락 긴 반복 | 숨겨진 총 상한 없음; 테스트는 취소 이벤트로 종료 |
| 탈락의 mask/receipt/PNG/thumbnail/record/log 링크 | 최종 제거; 시간 장부에 상세 점수/이미지 없음 |
| 원본·채택·외부 symlink·hardlink·활성 run | 삭제 차단 |
| 최소 장부 저장 실패 / 삭제 중 crash | 선삭제 금지 / 재개 정리만, 비용·count 중복 없음 |
| 기존 VQA accepted·rejected·알 수 없는 상태 | 보존·명시 상태 정리·추정 삭제 금지 |
| 실패40 + 실패50 + 성공60 | attempt60 / accepted cost150 |
| 중단 전 pending, 오랜 pause 후 resume | pause 제외·pending 유지·재초기화 포함 |
| N=0채택 중단, 마지막 실패, 과거 부분 시간 | 평균 null·잔여 비용 보존·불명 시간 null |
| 생성 완료 후 시각화/commit crash | 생성/검증 재호출 없이 복구, quota 한 번 |
| ADVV 밖 coordinator / 다른 Python worker | preflight 오류; venv 자동 생성/fallback 없음 |
| val/test 또는 group/픽셀 split 중복 | 누수 차단; 검증 통과가 GT 승계로 이어지지 않음 |

CPU 검증과 실제 GPU 검증을 분리한다. 실제 모델이 특정 fixture에서 반드시 탈락해야 한다고 단정하는 테스트는 정책 테스트로 대체한다.

## 과거 검증과 이후 평가

2026-09-30의 CPU 79개 통과와 DragFlow+Qwen N=1 결과는 **legacy 방식의 기록**이다. 새 CLIP·segmentation·단일 conda·정리 정책의 검증 증거로 재사용하지 않는다. 수치는 [과거 실행 기록](RUN_VERIFICATION.md)에 보존한다.

P5의 실제 GPU 통합 확인 이후 사람 검수와 도메인 평가를 수행한다. 채택 이미지의 품질, 실패 포함 비용, 원본별 기여와 다양성을 먼저 보고하고 후속 학습은 그다음이다. 탈락 후보를 남기는 연구 실험이 필요해도 기본 삭제 정책을 몰래 끄지 않는다. [EVALUATION.md](EVALUATION.md)의 평가 범위를 따른다.
