# 아키텍처와 CLIP 전환 설계

기준: [SPEC v0.13](../SPEC.md), 2026-10-12 (입력 축소·선택 v2 반영). **v0.4.0**은 사고 차량 전체 공동 이동 입력을 구현한다. 선택·기하·공식 affine 함수의 CPU 검증과 실제 image1의 CPU 선택 점검을 했으며, 실제 GPU 생성에서의 이동 품질은 별도 검증 대상이다.

## 1. 모듈별 변경 경계

| 경로 (`src/advv/` 기준) | 현재 책임 / 필요한 변경 |
| --- | --- |
| `cli.py`, `config.py` | 기본 설정/입출력 경로, 선택 run ID/prompt, GPU 쌍 분할 및 재개 recipe 고정 |
| `prepare.py`, `preflight.py` | Qwen/venv 필수 조건 제거, 단일 conda·CLIP·segmenter·로컬 가중치 검사 |
| `contracts.py`, `ingest.py` | Source 계보 유지; Segment/EditRequest/CLIPResult/TimingRecord 계약 추가. snapshot 단계에서 긴 변 `dataset.max_long_side_px`(기본 960) 초과 시 축소(확대 없음), `Source.preprocessing`에 원본 크기·비율·보간 기록 |
| `segmentation.py`, `accident_group.py` | 전체 instance mask 검증, 접촉 관계별 CLIP source 선택, 전체 합집합/선택 provenance. `clip_contact_context_v2`(기본): 보닛 제외 → IoU 중복 병합 → 상대 틈 접촉 그래프 → 원근 겹침 먼 차 분리 → CLIP → 떨어진 작은 먼 단위 제외 |
| `sampler.py`, `group_geometry.py` | 전체 그룹 면적 고정, 원본/feature grid 안의 공통 비항등 이동; `accident_group_v2`는 크기 r(그룹 bbox 대각선 비율 상·하한)·방향 θ 분리 표본 + 세로 축소. θ는 그 r에서 화면·grid 조건을 만족하는 각도 구간 위 균등; 이동 가능한 Δ가 없는 원본은 생성 전에 판정해 보류 |
| `backends/dragflow.py`, `backends/group_adapter.py` | contour loader 우회, 단일 full-union region, 공통 grid Δ hook, 별도 repair support |
| `motion.py` | CPU 차량별 appearance matching 이동 진단; 채택 조건 추가 없음 |
| `backends/clip.py` | 10문장 embedding, 전체 프레임 입력, raw cosine 10개 반환 |
| `backends/segmentation.py` | 선택 segmenter를 local-only로 실행 |
| `backends/process.py`, `worker.py` | 별도 프로세스는 유지하되 모든 worker를 ADVV `sys.executable`로 실행 |
| `verifiers/decision.py` | Qwen AND 대신 finite 10점수 검사 및 3-of-4 AND mandatory-5 |
| `pipeline.py` | 독립 후보 thread + GPU worker, 공유 quota 예약·중복·채택 commit·최소 실패 장부 |
| `storage.py`, `cleanup.py` | 안전한 atomic 저장 유지; 기존 비채택 정리·journal·lock·복구 추가 |
| `timing.py` | active segment, 시도 시간, 실패 포함 획득 비용과 pending 누적 |
| `visualization.py` | 원본 선택 preview; 채택 후보의 전체 그룹/ID/공통 화살표/계획 목표 윤곽 |
| `reporting.py` | 채택 상세 + 최소 시간/집계만; 탈락 링크/점수/이미지 제거 |
| `gallery.py`, `visualization_collection.py` | 종류별 comparison/drag_plan/metadata 폴더, 생성 이미지 grid PNG; 기존 경로/링크의 CPU 이전 |

기존 `backends/qwen.py`, `verifiers/parser.py`, SourceProfile schema와 prompt는 전환 기간의 legacy 자료다. v2 실행 경로가 이들을 호출하거나 Qwen 가중치를 요구해서는 안 된다. 이를 삭제할지 격리할지는 실제 구현 PR에서 의존 참조를 확인한 뒤 결정한다.

## 2. 실행과 프로세스

```text
ADVV conda coordinator
  → 설정/입력/동일 환경 검사
  → 관리하는 종료 run의 비채택 산출물 정리 (추론 전)
  → 원본 ingest → run snapshot(긴 변 ≤ max_long_side_px, 확대 없음; 이후 모든 단계의 좌표계)
  → segmenter worker → 결정적 선택 규칙 + CLIP 사고 관계 선택 → 전체 그룹 cache + selection preview
  → 원본별 이동 가능성 사전 판정: 불가 원본은 state.held_sources(no_valid_motion)로 보류, 전부 보류 시 no_movable_sources
  → GPU 목록을 2개씩 분할: [4,5], [6,7]
  → while unique durable accepted < N:
      coordinator: 채택 수 + 예약 수 <= N 범위에서 원본/attempt/sequence 예약
      작업 1 [4,5]: 전체 그룹 이동 계획 → DragFlow → CLIP [4] → private scored
      작업 2 [6,7]: 전체 그룹 이동 계획 → DragFlow → CLIP [6] → private scored
      coordinator: 완료한 후보마다 전역 중복 검사와 채택/탈락 처리를 직렬 수행
      탈락/중복/오류: 최소 시간 저장 → 상세 파일 정리 → 새 예약 허용
      채택: 이미지/기록/시각화 저장 → 시간/계보 commit → 공유 quota +1
  → accepted-only export/report → worker 종료 → active 시간 확정
```

worker는 독립 프로세스여도 **모두 conda ADVV의 동일한 Python**이다. coordinator 안의 `ThreadPoolExecutor`는 GPU 쌍마다 독립 `LocalBackend`를 사용한다. 실제 모델은 각 backend의 별도 child process에서 실행하며 `CUDA_VISIBLE_DEVICES`로 자기 묶음만 노출한다. 한 묶음의 생성기와 CLIP은 단계 사이 process를 종료하여 메모리를 해제하고 순차 실행하지만, 다른 묶음의 후보와는 겹쳐 실행한다. 원본 segment cache는 후보 실행 전에 첫 묶음에서 준비한다.

coordinator만 공유 clock, 공개 record, 픽셀 중복 집합, quota를 수정한다. thread는 자기 후보의 private record/receipt만 수정한다. `accepted + in_flight <= N` 예약 규칙으로 마지막 한 자리는 한 후보만 실행한다. 빠른 묶음은 다른 묶음의 완료를 기다리지 않고 재할당받는다. 중단 시 `cancel()`이 새 worker 시작을 차단하고 기존 child process를 종료하며, 모든 thread를 join한 후 run lock을 해제한다. 이미 scored인 탈락 후보는 중단 상황에서도 최소 시간을 저장한 뒤 정리한다.

CLI는 저장소 기준 기본 YAML을 읽고 `assets/inputs`와 `assets/outputs/<run_id>`를 사용한다. `--input-dir`는 없으며 run ID는 자동 생성하거나 선택 지정한다. `--image`는 입력 폴더 기준 한 파일만 ingest하는 선택 필터이며 config/recipe에 저장한다. 재개는 저장한 선택과 snapshot을 사용한다. `state.segments`에 실행마다 GPU 묶음과 동시 후보 수를 기록한다.

정상적인 검증 탈락에는 총 시도 상한을 두지 않는다. timeout 등 기술 재시도만 유한하게 설정하고, 유효한 CLIP 탈락에 통과가 나올 때까지 재검증하지 않는다. 다음 attempt는 원본에서 새 계획을 만든다. 로드 실패·OOM·저장소 오류 같은 복구 불가 상태는 partial 결과와 시간 상태를 저장하고 중단한다.

## 3. 모델 경계와 provenance

공식 DragFlow commit `b3a8fa7136df5131d07f98425c0d189fb6ca74b0`, FireFlow `20ab81fb084c06a089bb8cf86726a28036c88d5b`를 현재 기준으로 유지한다. [upstream lock](../configs/upstream.lock.json)은 기존 revision 기록이다. Mask R-CNN ResNet50-FPN COCO_V1의 URL·전체 SHA256·torchvision 버전도 lock에 추가했다.

실제 `Dragger(conf, dtype=torch.float32)`, `load_pipeline()`, `dragger(raw_image, instruction, image_name)` 경계를 기존 wrapper에서 확인한다. CLI 전역 실행을 가진 benchmark runner를 import하지 않는다. upstream tracked 파일을 몰래 바꾸지 않으며 wrapper 변경도 implementation hash에 포함한다. latent 최적화 gradient는 유지한다.

| 입력 | adapter에서 확인할 값 |
| --- | --- |
| 자유형 edit mask | 요청 binary mask, soft feature mask, 실제 nonzero 영역 |
| start/end | EXIF 정규화 원본 좌표, resize/feature-grid 변환, upstream centroid 교체 |
| range | 요청 면적 비율, 실현 픽셀 면적; displacement와 별도 |
| operation | 전체 그룹 한 개의 relocation→transformation; 차량별 독립 이동/회전/크기 변경 없음 |
| prompt | 도메인 기본 문장, 선택 원본 설명, 선택 `--prompt`, 실제 전달된 source/target 문장 |

CLIPResult는 순서가 고정된 5개의 `{id, undesired_score, preferred_score, margin, passed}`와 모델·전처리·문장 hash를 가진다. 응답 문자열 파싱이나 YES 검출은 없다. 같은 후보 이미지의 normalized embedding과 10개 text embedding을 내적한다. token truncation, NaN/Inf, 잘못된 차원, 모델 revision 불일치는 오류다. 5쌍 모두 계산한 뒤 정책 함수가 판정한다.

전처리는 전체 프레임의 bicubic letterbox, 224×224, RGB (127,127,127) padding 및 고정 processor normalization이다. 원본과 후보의 두 이미지 비교, 별도 crop ensemble, 추가 geometry score는 기본 검증 경로에 넣지 않는다. CLIP은 사고 존재에 대한 대리 판단이며 원본 사고 유형의 보존 판정으로 이름 붙이지 않는다.

## 4. 상태와 저장 계약

```text
reserved → planned → generated → scored
scored → verified_unique → committing → accepted
scored → rejected/duplicate → cleanup_pending → cleaned
technical error → retry 또는 error → cleanup_pending → cleaned
user interrupt → interrupted (최소 시간/cursor, private 진행 상태 보존)
```

이 흐름의 `rejected`/`error`는 내부 전이 상태다. 최종 파일 구조에는 비채택 상세 record를 남기지 않는다. 채택 commit 전에 이미지·기록·시각화를 private staging에서 준비한다. 공개 위치로 승격하고 durable commit이 완료된 이미지만 quota/export에 포함한다. commit 중 crash는 journal로 복구하여 중복 증가를 막는다.

실패 경로는 **최소 시간/cursor 영속화 → 소유 파일 목록을 journal에 저장 → 상세 파일 삭제 → 후보 완료 tombstone** 순서다. 기존 run 정리 전체가 반환되기 전 shared log/report/export를 갱신하며, 도중 중단 시 다음 시작에서도 다시 갱신한다. 정리 시간은 장부에 후속 반영한다. receipts와 원본 설정 그림도 후보별 상세 산출물이므로 삭제한다. timing ledger에는 재현용 후보별 prompt/점수/좌표를 남기지 않는다. run 전체의 recipe와 원본 자원은 보존한다.

다른 run 정리는 등록된 output root, 유효 manifest, 해당 run의 lock 및 종료된 비채택 상태를 모두 확인한다. 현재 활성 run이나 알 수 없는 파일은 추측 삭제하지 않는다. `cleanup_pending`/`cleanup_blocked`는 공개 완료 상태가 아니며 신규 추론 전에 해결하도록 한다. accepted-only report를 만들면서 과거 탈락 그림을 다시 렌더링하지 않는다.

기존 VQA accepted는 물리 파일과 legacy 계보를 보존한다. CLIP export와 quota에는 포함하지 않는다. 기존 VQA rejected/uncertain/error/duplicate처럼 확정된 비채택 산출물은 명시적 상태를 읽어 정리할 수 있다. 과거 전체 시간을 복원할 수 없으면 알려진 단계 시간만 부분 측정으로 남긴다.

## 5. 시간과 재개

TimingRecord의 단위는 초이며 [SPEC 시간 정의](../SPEC.md)의 필드를 사용한다. 후보별 경과 시간과 공유 run의 active segment를 함께 기록한다. 후보별 시간은 서로 겹칠 수 있다. `accepted_cost_seconds`는 완료/commit 순서의 공유 active 간격으로 배정하며 후보 시간을 다시 더하지 않는다. 실패 후 다음 채택까지의 누적 비용은 원본이 바뀌어도 이어진다. 첫 비용에는 초기화가, 마지막 pending에는 종료 overhead가 포함된다.

```text
sum(accepted_cost_seconds) + pending_cost_seconds == run_active_seconds
```

정확히 측정된 구간에 적용되는 보존식이다. 강제 종료로 구간이 누락되면 `timing_complete=false`로 표시한다. 시도 시간/단계 시간은 분석용 분해이므로 이 보존식에 다시 더하지 않는다. ledger의 candidate/segment ID를 이용해 재개 시 중복 합산을 막는다.

v1→v2는 in-place resume하지 않는다. 새 run에서 모델·문장·sampler·입력·prompt·코드를 고정한다. 같은 v2 recipe 내에서는 GPU 목록과 그에 따른 묶음 수를 바꿔 재개할 수 있다. 미완료 accepted commit, cleanup journal, private receipt를 먼저 복구한 뒤 다음 candidate index로 진행한다.

## 6. 설정 이전 지침

[설정 예시](../configs/advv.example.yaml)와 validator를 schema 2.0으로 변경했다. 새 로컬 설정은 configs/advv.clip.local.yaml이며, 예전 advv.local.yaml과 weights.lock.json은 보존한다. 새 로컬 모델 연결은 models/weights.clip.lock.json이다.

| 기존 항목/개념 | v2 목표 |
| --- | --- |
| `verifier` Qwen + profile/physical/semantic prompts | CLIP model/revision + `prompts/clip_criteria_v1.json` + 고정 정책 |
| 타원/사각형 sampler | segmentation 모델 설정 + 자유형 sampler + 면적/끝점 분포 |
| source profile | 고정 도메인 source prompt + 선택 `preserve_hint` |
| 생성 prompt | `generation.augmentation_prompt`, CLI 우선 |
| `execution.generator_python`, `verifier_python` | 별도 경로 선택 제거; ADVV `sys.executable` |
| NO 이미지 삭제/격리 옵션 | 비채택 상세 결과 무보존; 사용자 run에 보관 우회 옵션 없음 |
| 모든 후보 `records/` | 채택 상세 record + 최소 `timings/attempts.jsonl` |
| 단계 시간 합만 보고 | 시도·획득·pending·전체 active 시간 구분 |

단일 conda 파일 environments/advv-conda.yml, 설치 requirements, 실제 freeze를 작성했다. 설치/import 및 실제 CLIP·Mask R-CNN CPU forward는 확인했으며 전체 GPU 조합은 추가 확인한다.
