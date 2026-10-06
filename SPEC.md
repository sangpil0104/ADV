# ADVV 구현 명세

- 문서 버전: `0.5`
- 갱신일: `2026-10-06`
- 상태: MVP 코드 구현, 실제 DragFlow N=1 통합 실행 확인(T026 이전 Qwen 판정). T026에서 판정을 CLIP 필터로 교체 — CPU 테스트만, GPU 실행·margin 보정 대기
- 이름: **Augmentation using DragFlow with VQA Verification** (판정은 T026에서 CLIP 필터로 변경됨. 이름 변경 여부는 사용자 결정 대기)

## 1. 확정된 목표와 범위

산업 현장의 고장·결함·사고, 멸종 위기종·희귀 생물처럼 확보하기 어려운 실제 세계 이미지를 증강한다. **1차 기여는 장면 수준의 명백한 물리 위배를 자동으로 걸러내는 증강 파이프라인**이다. 원본 의미 보존은 자동 검사하지 않으며, 필요하면 선택적 사람 검수(§6.1)로 확인한다. 후속 모델의 성능 개선 검증은 2차 목표다.

사용자는 원본 이미지 폴더와 **입력 전체에 대한 새 증강 이미지 총 N장**을 지정한다. 영역·방향·변형을 무작위로 선택하고 DragFlow로 후보를 만든다. **CLIP 비현실 장면 필터**(§5)를 통과한 새로운 이미지가 N장 쌓일 때까지 반복한다. N에 원본 이미지는 포함하지 않는다.

> **판정 변경(T026, 2026-10-06 사용자 결정).** 이전의 Qwen 물리·의미 VQA 판정은 제거했다. 판정은 고정 문장 목록에 대한 CLIP 유사도의 원본 대비 상승폭으로만 한다. Qwen은 원본 profile(영역 proposal용 subjects/parts)에만 쓴다. 원본 의미 보존 자동 검사는 없다. 장면 수준의 명백한 물리 위배만 대상이며 **국소 미세 artifact(무늬 소실, 작은 질감 흔적, 경계 자국 등)는 검사 범위 밖**이다(필요하면 §6.1 선택적 사람 검수).

사용자 확정 사항:

| 항목 | 결정 |
| --- | --- |
| 정답 라벨 | 현재 없음; image-only 파이프라인에 필수 아님 |
| 편집 계획 | 자동 무작위 생성; 수동 mask/점 입력 불필요 |
| 편집 설정 시각화 | start_point·end_point·area를 원본에 표시하고 생성 결과와 나란히 PNG 저장 |
| 허용 변화 | 위치·크기·자세·형태 등은 허용. 자동 판정은 장면 수준의 명백한 물리 위배만 거른다(T026) |
| 의미 보존 | 자동 검사 없음(T026). 필요하면 선택적 사람 검수 |
| 원본 profile | 원본에서 자동 추정(Qwen); 사용자 설명은 선택 입력. 영역 proposal용 |
| 생성 수량 | 원본별 수량이 아닌 입력 전체에서 총 N장 |
| 총 실행 상한 | 없음; 목표 달성 또는 사용자 중단까지 계속 |
| 판정 모델 | 다운로드한 로컬 `openai/clip-vit-large-patch14`(고정 revision), 추가 학습 없이 사용(T026) |
| profile 모델 | 다운로드한 로컬 `Qwen/Qwen3.5-4B`, 추가 학습 없이 사용 |
| GPU | 사용자가 실행할 때 지정; 고정 번호·자동 전체 점유 금지 |
| 평가 순서 | 파이프라인·이미지 품질 검증 먼저, 후속 학습 실험은 이후 |

정상적인 rejected 반복으로 실행을 자동 중단하지 않는다. 처리 불가능한 입력, 가중치 로드 실패, 장치 부적합, 저장소 오류 같은 기술적 실패는 명시적으로 기록하고 중단할 수 있다. 총 실행 상한과 개별 작업 timeout은 다른 개념이다.

MVP에는 폴더 입력, 원본 profile 추출, 무작위 sampler, 실제 DragFlow·Qwen(profile)·CLIP(판정) 연결, CLIP 필터, 목표 수량 관리, rejected 삭제, 사용자 중단·재개, image-only export와 report를 포함한다. 모델 fine-tuning, task-specific annotation 생성, 후속 모델 학습, 웹 UI는 필수 범위가 아니다.

## 2. 기법과 채택 조건

```text
h_i = Profile_Qwen(original_i, optional_user_hint_i)       # 원본마다 한 번 고정 (영역 proposal용)
T   = CLIP_text(sentences)                                  # 고정 문장 목록, run마다 한 번 고정
r_i = cos(CLIP_image(original_i), T)                        # 원본마다 한 번 고정
edit_ij = RandomSampler(seed, source_id_i, attempt_index_j)
candidate_ij = DragFlow(original_i, edit_ij)
c_ij = cos(CLIP_image(candidate_ij), T)
delta_ij = c_ij - r_i                                       # 문장별
accepted_ij = valid(c_ij) AND max_k delta_ij[k] <= max_delta
quota_count = count(saved, unique, export-eligible accepted candidates)
stop_successfully_when(quota_count == N)
```

판정은 한 번의 CLIP 이미지 임베딩과 고정된 원본 기준의 비교다. 어떤 문장이라도 delta가 `verifier.max_delta`를 **넘으면**(`>`) `rejected`, 모든 문장이 같거나 작으면 `accepted`다. 원본 자체의 점수(예: 원래 사고 사진이라 "car flying" 점수가 높음)는 기준선으로 빠지므로, 편집이 그 장면을 **더** 비현실적으로 만든 경우만 걸린다.

검사 대상은 장면 수준의 명백한 물리 위배(공중에 뜬 차량, 땅에 박힌 몸, 고체를 통과한 물체, 다리 개수 이상, 녹거나 부푼 몸체 등)다. 국소 미세 artifact와 원본 의미 보존(결함 유형·사건·생물 정체성의 유지)은 자동 검사하지 않는다. 결함이나 손상·사고 자체를 물리 위배 문장으로 넣지 않는다.

CLIP 유사도는 시각·텍스트 임베딩의 추정치다. 물리적 사실, 정확한 종 식별, 학습용 라벨의 정답을 증명하지 않는다.

## 3. 입력과 원본 의미 기준

### 3.1 일반 사용자 입력

실행 수량은 `advv run N ...`의 위치 인자로 받는다. 기존 `advv run --target-count N ...`도 지원한다. 두 형식의 동시 지정, 0·음수·정수가 아닌 수량은 추론 전에 거부한다. CLI 수량을 생략하면 YAML의 `run.target_count`를 사용하며, 최종 수량이 없으면 실행하지 않는다. `--resume`에서는 수량 인자를 받지 않고 저장된 N을 유지한다.

필수 입력은 원본 폴더, 양의 정수 N, 실행할 GPU 선택이다. 모델·prompt·sampler 설정은 설정 파일로 제공한다. 선택적으로 이미지별 보존 의미 설명을 JSONL manifest로 전달할 수 있다. 원본마다 사람이 mask·좌표·GT 라벨을 만들 필요는 없다.

폴더에서 지원 이미지 파일을 안정적인 경로 순서로 읽고 manifest를 작성한다. 지원 확장자·디코딩·EXIF 방향 정규화·RGB 변환을 명시하고, 원본 파일은 수정하지 않는다. 같은 정규화 픽셀의 중복 입력은 중복 source로 집계하여 한 번만 처리한다. 이미지가 없거나 모든 입력을 사용할 수 없으면 원인을 알리고 종료한다.

### 3.2 내부 Source manifest: JSONL

| 필드 | 타입 | 규칙 |
| --- | --- | --- |
| `schema_version` | string | `1.1` |
| `source_id` | string | manifest 내 고유 ID |
| `image_path` | string | dataset root 기준 상대 경로 |
| `split` | enum | `source`, `train`, `val`, `test` |
| `group_id` | string | 같은 개체·시편·촬영 세션 등 독립성 단위 |
| `domain` | string | `industrial_defect`, `accident`, `rare_organism` 등 설명용 |
| `preserve_hint` | string 또는 null | 사용자 선택 설명; 학습 GT 아님 |
| `label` | object 또는 null | MVP는 null 가능 |

라벨 없는 구축 모드는 `source/train`을 처리하며 폴더 입력의 기본 split은 `source`다. val/test는 생성하지 않는다. 후속 학습 단계에서는 group 단위 split을 먼저 확정하고 train만 증강한다. 같은 group·동일 내용이 split 사이에 겹치면 검사 실패다. group 정보가 없으면 source별 임시 ID를 부여하고 실제 독립성까지 확인한 것은 아니라고 기록한다.

### 3.3 SourceProfile

[원본 의미 추출 prompt](prompts/source_profile_v2.txt)와 [schema](schemas/source_profile.schema.json)를 사용한다. `summary`, `must_preserve`, `uncertain`을 원본과 선택 설명에서 생성한다. `object_region_v2` sampler는 [v4 prompt](prompts/source_profile_v4.txt)·[schema](schemas/source_profile_v4.schema.json)를 요구하며, 여기에 영역 proposal용 `subjects`(짧은 영어 명사구 1–3개)와 subject별 `parts`(`[{"subject", "parts"}]`, subject마다 따로 움직일 수 있는 부위 이름 0–5개, 결함·사고 증거와 증거를 담은 부위는 넣지 않음, 항목의 subject는 `subjects`에 있고 서로 달라야 함)가 추가된다. 부위 이름 텍스트 prompt를 끄면(`text_parts: false`) `subjects`만 쓰므로 [v3](prompts/source_profile_v3.txt)(이미지 전체에 하나의 `parts` 목록, `v2_pilot_001`에서 사용)도 받는다. T026 이후 profile은 판정에 쓰지 않는다. `uncertain`은 source 보류에, `subjects`·`parts`는 영역 proposal에 쓴다. `summary`·`must_preserve`는 기록·편집 prompt 작성용이다.

추출 원문, 사용자 설명, 원본 hash, 모델·processor revision, prompt hash를 저장하고 **해당 원본의 후보를 생성하기 전에 고정**한다. 후보를 본 뒤 기준을 바꾸어 통과시키지 않는다. 명확한 종명이 보이지 않으면 종명을 발명하지 말고 관찰 가능한 구별 특징을 기록한다. 사용자 설명도 사진에서 확인할 수 없는 사실을 참으로 보장하지 않는다.

`uncertain=true` 또는 유효한 profile을 얻지 못한 source는 `source_uncertain/source_error`로 보류하고 다른 원본을 처리한다. `object_region_v2`에서 사용할 영역 proposal이 없는 source는 `source_no_region`으로 보류한다. 자동 생성 profile은 GT가 아니다. 모든 source가 보류되면 `no_eligible_sources`로 종료하여 사용자 설명/입력 수정을 요청할 수 있도록 report한다. 원본에 보이는 의미 자체가 미확정인 상태로 채택 수를 채우지 않는다.

## 4. 무작위 편집 계획

sampler는 사용 가능한 원본을 안정적인 round-robin 순서로 선택하고, 각 원본의 seed·attempt index로 새 영역·operation·방향·강도를 표본 추출한다. 총량 모드이므로 원본별 최종 채택 수는 같지 않을 수 있으며 report한다. 수락률이 높은 원본으로 자동 집중하는 학습 정책은 MVP에 없다.

MVP 영역은 이미지 내부의 연결된 기하 영역(타원 또는 사각형)을 무작위 생성한 binary mask다. 객체 segmentation 모델을 필수로 추가하지 않는다. 영역이 의미 있는 객체에 정확히 맞는다는 가정은 하지 않으며, 이 방식의 수락률과 다양성은 평가한다. 후속 객체 중심 proposal은 별도 sampler 버전이다. 무작위 사각형이 결과에 경계 이음매·잘린 객체를 남기는 문제로 entity/part mask 기반 `object_region_v2`를 추가했다(`sampler.version`으로 선택, 기본은 v1). proposal은 SAM 3(기본) 또는 Grounding DINO + SAM 2.1(비교)로 원본마다 한 번 만들어 고정하며, part는 profile `parts`에서 그 entity의 subject에 적힌 부위 이름 텍스트 prompt와 entity 내부 점 prompt를 병행해 얻고 출처(`text`/`point`)를 기록한다(`region_proposal.text_parts`, 기본 켜짐). 텍스트 part는 entity mask 안에 충분히 들어간 인스턴스만 그 entity에 귀속한다. v2 mask는 구멍 없는 하나의 연결 영역이고 윤곽 선택점(`region_select_point`)을 포함하며, 드래그 시작점은 upstream과 같은 mask centroid다. proposal은 profile을 고정한 뒤, 첫 후보 생성 전에 SAM 3 worker(별도 환경, 선택 GPU 목록의 첫 GPU)로 만들며 DragFlow를 올리기 전에 worker를 내린다. 사용할 영역이 없으면 기하 영역으로 대체하지 않고 원본을 `source_no_region`으로 보류한다. proposal 생성이 기술 오류로 끝나 파일이 없으면 `region_proposals_missing` 오류로 멈추며 '영역 없음'으로 보지 않는다. DragFlow feature grid에서 사라지거나 시작점이 한 칸 넘게 달라지는 얇은 proposal은 편집 영역으로 쓰지 않는다. sampler·저장 계약·worker 연결은 CPU로 구현·테스트했다. ADVV adapter는 T009d에서 GPU로 5장 실행했으나(점 prompt part가 거의 나오지 않음), T012에서 부위 이름 텍스트 part를 더한 adapter를 같은 5장에 GPU로 실행했다(부위 4–7개/장, 부위 이름은 수동 지정). 실제 Qwen이 만든 profile `parts`로의 실행과 v2 `advv run`은 아직 검증하지 않았다. 비교 경로(Grounding DINO + SAM 2.1)는 미구현이다. 세부는 [영역 sampler v2 설계](docs/REGION_SAMPLER.md)를 따른다.

`relocation/deformation/rotation` 중 backend에서 검증한 operation을 선택한다. 영역 면적, 이동 거리, 회전/변형 강도 분포는 설정으로 전달한다. 설정 예시의 수치는 시작점이며 도메인에 최적화된 값이 아니다. `object_region_v2`의 rotation은 upstream이 feature grid로 반올림한 점으로 실제 회전각을 계산하므로, 목표점을 그 실효 각도가 요청각에 가장 가까운 grid 칸에 두고 실효 각도가 허용 오차 `max(rotation_max_error_degrees, rotation_max_error_fraction × |요청각|)` 밖이면 그 각도를 버리고 다시 뽑는다. 실효 각도가 요청각과 반대 부호이거나 `|실효각| < rotation_min_executed_degrees`인 칸(예: 실효 0°)은 목표로 쓰지 않는다. 계획에는 요청각과 실효 각도를 모두 기록한다([설계 §3.3](docs/REGION_SAMPLER.md#33-operation과-granularity)). 자동 판정은 §5의 CLIP 필터만 수행하며 의미 보존은 자동 검사하지 않는다. 선택 설정 `sampler.object_region.region_phrase_filter`는 편집 영역을 phrase·level로 좁히는 실험·디버깅용 입력이며 일반 사용자에게 요구하지 않는다. 맞는 영역이 없으면 그 원본은 보류하고, 판정 기준은 바꾸지 않는다([설계 §3.4](docs/REGION_SAMPLER.md#34-실험용-region-phrase-필터-t023)).

새 후보는 항상 **입력 원본**에서 생성한다. 이미 생성한 이미지를 재편집하는 연쇄 증강은 기본 범위가 아니다. 동일한 rejected 이미지의 판정을 바꾸려 하지 않고 새 계획을 만든다.

### 4.1 저장되는 EditRequest

| 필드 | 규칙 |
| --- | --- |
| `schema_version` | `1.3` (저장된 `1.1`·`1.2` 계획도 읽음) |
| `source_id`, `edit_id`, `attempt_index` | source 참조, 고유 edit ID, 0부터 증가하는 원본별 index |
| `seed`, `sampler_version` | 실행 seed에서 안정적인 digest로 유도한 32-bit seed와 sampler 버전 |
| `operation`, `operation_params` | 검증된 operation과 추가 강도·회전 등 파라미터 |
| `region_mask_path` | 현재 run 기준 상대 경로; 생성된 0/255 PNG |
| `source_point`, `target_point` | 정규화 원본 크기 기준 `[x,y]`. `source_point`는 mask의 반올림 centroid로, DragFlow가 실제로 드래그를 시작하는 점이다 |
| `anchor_point` | operation별 필요성을 upstream 변환기가 검사 |
| `source_prompt`, `target_prompt` | 고정 profile과 편집 지시에서 작성; 결함/사고를 정상 장면으로 고치지 않도록 구성 |
| `region_proposal_id`, `region_level`, `region_phrase` | `object_region_v2`에서 선택한 proposal ID, `entity`/`part`, 명사구. v1은 모두 `null` |
| `region_select_point` | `object_region_v2`만: upstream이 윤곽을 고를 때 쓰는 mask 내부 점(centroid가 mask 안이면 centroid). 이동 계산에는 쓰지 않는다. v1은 `null` |

좌상단 원점, `0 ≤ x < width`, `0 ≤ y < height`. v1의 source point는 mask 내부여야 한다. v2는 `region_select_point`가 mask 내부이고 `source_point`가 mask centroid와 같아야 한다(오목한 mask에서는 centroid가 mask 밖일 수 있다). 빈 mask·항등 변환·범위 밖 좌표는 거부한다. 직사각형이 아닌 이미지에서도 x/y 축을 혼동하지 않는다. seed에 Python의 프로세스별 `hash()`를 사용하지 않는다.

후보 계획과 mask는 생성 전에 원자적으로 저장한다. upstream `instruction.json`과 동일한 schema라고 가정하지 않고 고정 commit에 맞게 변환한다. 리사이즈·패딩은 이미지/mask/좌표에 동일한 기하 변환을 적용하고 기록한다. ADVV의 mask 표시·역변환 보간은 nearest-neighbor다. 고정 upstream 내부의 feature mask는 bilinear 보간을 사용하므로 연속값 native mask와 그 nonzero 영역을 별도로 기록한다. `random_geometry_v1` sampler가 반복적으로 유효 계획을 만들 수 없으면 기술 오류로 처리하며 rejected로 집계하지 않는다. `object_region_v2`는 허용 이동 집합에서 직접 뽑으므로 보통 실패하지 않지만, 한 attempt의 기하 예산(512회)을 소진하면 run을 멈추지 않고 그 attempt를 `sampling_skipped`로 `state.json`에 기록한 뒤 cursor를 전진한다. skip은 후보·이미지를 만들지 않으므로 목표 수량, rejected, 연속 기술 실패(`max_consecutive_technical_failures`) 어디에도 세지 않는다. 한 원본의 연속 skip이 `sampler.object_region.max_consecutive_sampling_skips`에 닿으면 그 원본을 `source_no_region`(`held_reason: sampling_exhausted`)으로 보류하며, 재개해도 보류는 유지된다.

### 4.2 경로 규칙

- YAML 상대 경로: YAML 디렉터리 기준.
- source manifest의 이미지·label 경로: dataset root 기준.
- sampler가 저장한 mask와 모든 run artifact 경로: 해당 run 기준.
- 입력은 읽기 전용; `..`·symlink를 통한 허용 root 이탈과 원본 경로에 출력 쓰기를 거부한다.

### 4.3 편집 설정 시각화 — 기본 활성화

각 후보에서 **어디를, 어느 방향으로, 어느 범위에서 편집했는지** 이미지로 확인할 수 있어야 한다. 내부 `source_point`는 화면의 `start_point`, `target_point`는 `end_point`, `region_mask_path`의 binary mask는 `area`에 대응한다. 별도의 독립 좌표값을 만들어 관리하지 않는다.

기본 비교 이미지는 **왼쪽: 원본 + 편집 설정 표시 / 오른쪽: 표시 없는 생성 결과**의 1×2 배치다. 원본 위에 빨간 원형 시작점(S), 초록 십자형 도착점(E), 시작→도착 화살표, 반투명 파란 영역과 경계를 표시한다. 아래 여백에는 좌표·영역 면적/비율·operation·seed·후보 ID·CLIP 필터 결과(판정, 최대 delta와 그 문장, limit)를 적는다. rotation/deformation에 anchor가 있으면 별도 기호로 표시한다.

sampler의 요청값뿐 아니라 실제 backend에 전달된 좌표·mask와 전처리 변환을 저장한다. backend의 입력이 리사이즈/패딩/반올림으로 달라졌다면 이를 정규화 원본 좌표로 복원하여 **실제 전달한 편집 설정**을 표시한다. 화살표는 의도한 이동이며 생성 결과에서 측정한 이동이라고 주장하지 않는다.

`visualizations/<candidate_id>/drag_plan.png`, `comparison.png`, `metadata.json`을 저장한다. 시각화는 확인용 산출물이고 CLIP 입력·증강 이미지·목표 수량과 분리한다. 기본 rejected 삭제 정책은 생성 결과를 포함한 comparison/thumbnail에도 적용하며 원본과 설정만 포함한 drag_plan과 metadata는 남긴다. 상세 표시·좌표·삭제·재개 규칙은 [VISUALIZATION.md](docs/VISUALIZATION.md)를 따른다.

## 5. CLIP 필터의 입출력 (T026)

| 항목 | 규칙 |
| --- | --- |
| 모델 | 로컬 `openai/clip-vit-large-patch14`, `configs/upstream.lock.json`의 고정 revision. `local_files_only`, eval + inference mode, 기본 fp32, 고정 processor 기본 전처리 |
| 문장 | [고정 문장 파일](prompts/implausible_physics_v1.txt)(`verifier.sentences_path`). 한 줄 한 영어 문장, `#` 줄·빈 줄 무시, 중복 금지. 원본별 자동 생성 없음. 초기 16문장은 사용자가 고칠 수 있는 미보정 초기값 |
| 유사도 | L2 정규화한 projection 임베딩의 cosine(`verifier.similarity: cosine`). `logit_scale` 미적용 |
| 기준 | 문장별 `delta = sim(후보) − sim(원본)`(`verifier.criterion: max_delta_over_source`) |
| margin | `verifier.max_delta`(기본 0.03, **미검증 초기값**). 어떤 문장이라도 `delta > max_delta`이면 rejected. 같으면 통과 |
| 입력 | 후보 한 장. 원본 임베딩·유사도와 문장 임베딩은 후보 생성 전에 한 번 계산해 고정 |

문장 텍스트가 CLIP 한도(77 token)를 넘으면 잘라내지 않고 오류다. 임베딩 개수·차원이 맞지 않거나 non-finite·0 벡터이면 기술 오류(`verification_error`)다. 문장 목록·모델 revision·원본 pixel hash가 고정 기준과 다르면 새 run을 요구한다.

record에는 `checks.clip`(시도·receipt·후보 임베딩 hash·문장별 유사도)과 `clip_filter`를 별도 저장한다. `clip_filter`에는 문장별 원본/후보 점수·delta·초과 여부, 초과 문장 목록, 최대 delta 문장, `limit`, 문장 목록 hash·문장 파일 hash, 모델 id·revision, 원본/후보 임베딩 hash가 있다. 최종 accept 여부는 coordinator가 저장된 수치로 계산하며, 재개·export 때 다시 계산해 기록과 다르면 오류(`DataError`)로 본다. 다시 계산하는 항목은 문장별 `delta`·`exceeds`, `exceeded`, `top`, `verdict` 전부다. 또 행의 문장 목록이 run에 고정된 문장 목록(`clip/texts.json`, 설정의 목록·hash와 같아야 함)과, 행의 원본 점수가 고정된 원본 기준(`clip/sources/<source_id>.json`)과, 행의 후보 점수가 `checks.clip` 응답의 유사도와 정확히 같아야 한다.

## 6. 최종 상태·삭제·중복

| CLIP 필터 | 최종 상태 | 목표 수량 반영 |
| --- | --- | --- |
| 모든 문장 `delta ≤ max_delta` | `accepted` | 저장·중복 검사·export 적합성 확인 후 +1 |
| 한 문장이라도 `delta > max_delta` | `rejected` | 0 |
| 기술 오류(형상·값 오류, timeout, OOM, worker 종료) | `verification_error` | 0 |
| 생성 실패(필터 미실행) | `generation_error` | 0 |

기존 "두 판정 AND" 계약은 "CLIP 필터 통과 AND 저장·중복 검사 통과"로 대체되었다. `UNCERTAIN`과 `not_run`(의미 검증 미실행) 상태는 없다. 실행하지 않은 판정은 성공으로 간주하지 않는다.

원본 및 이미 채택한 모든 source의 후보와 canonical RGB pixel hash가 같으면 `export_status=duplicate`로 두어 목표 수에 포함하지 않는다. CLIP 필터 통과 사실은 보존한다. rejected·오류·중복은 다음 attempt의 새 편집을 유발한다. near-duplicate/편집량 threshold는 기본 채택 규칙에 추가하지 않고 다양성 지표로 먼저 평가한다.

rejected 이미지는 record를 영속화한 후 현재 run이 소유한 생성 파일과 그 복사본만 삭제한다. 원본·이전 run·외부 symlink·accepted 파일은 삭제할 수 없다. 삭제 실패는 `cleanup_pending=true`로 기록하고 재개 시 정리한다. 기술 실패는 별도 quarantine에 둔다. CLIP 점수·receipt·후보 hash·편집 계획은 이미지 삭제 후에도 보존한다.

rejected의 생성 픽셀이 들어간 비교 PNG·썸네일도 같은 삭제 대상이다. 원본에 편집 계획만 표시한 `drag_plan.png`는 남긴다. 시각화를 만들기 위해 삭제된 후보를 다시 생성하지 않는다.

탈락한 동일 후보는 다시 판정하지 않으며, 통과시키려고 문장 목록이나 margin을 같은 run에서 바꾸지 않는다. 임베딩 형상 오류·일시적 기술 실패에 한해 동일 입력으로 최대 1회 추가 시도를 허용하고 모든 시도를 기록한다. OOM·모델 로드 실패는 자동 반복하지 않고 partial run을 저장한다. 총 시도 상한이 없다는 요구를 기술 실패를 숨기는 무한 재시도로 구현하지 않는다. 기본 `max_consecutive_technical_failures=3`은 generation_error/verification_error 후보가 연속 발생할 때 failed로 종료하는 장애 기준이다. 정상 rejected·중복은 이 횟수에 포함하지 않는다.

### 6.1 사람 최종 검수 (선택 단계)

기본값은 꺼짐(`human_review.enabled: false`)이며 이때는 자동 판정(CLIP 필터)만 쓴다. 국소 미세 artifact나 의미 보존을 확인하려면 이 단계를 켠다. 설정에서 `enabled: true`로 두거나 `advv run ... --human-review`로 새 run을 만들 때 켠다. 값은 run 생성 시 `config.json`에 고정되며 `--resume`으로 바꿀 수 없다. 설정이 없는 기존 run은 꺼짐으로 취급한다. 꺼진 run에서 `advv review`는 거부된다.

켜진 run에서는 자동 단계가 `completed`로 끝난 run에서 `advv review --run-dir runs/<run_id>`로 CLIP 필터 통과 이미지를 사람이 한 장씩 판정한다. 판정은 이분법이다.

| 키 | 판정 | 처리 |
| --- | --- | --- |
| ← | `pass` | 유지하고 export에 포함 |
| → | `fail` | 판정을 record에 먼저 저장한 뒤 `accepted/` 이미지와 생성 픽셀이 들어간 비교 PNG 삭제. `drag_plan.png`·record·CLIP 점수는 보존 |
| q / Ctrl+C | 중단 | 판정한 것까지 저장, 같은 명령으로 이어서 검수 |

- 화면은 `runs/<run_id>/review/current.png`(왼쪽 원본, 오른쪽 후보, 편집 표시 없음) 한 파일을 매번 덮어쓴다. VS Code에서 열어 두면 자동 갱신된다. 서버에 GUI가 없어도 된다.
- `fail`은 재생성하지 않는다. 최종 수량은 사람 `pass` 수이며 N보다 적을 수 있다. 부족하면 새 run을 만든다. 검수가 시작된 run은 `--resume`을 거부한다.
- 켜진 run의 export(`exports/images.jsonl`)는 사람 `pass` 이미지만 포함한다. 검수 전 이미지는 export되지 않는다. 꺼진 run은 CLIP 필터 통과 이미지를 export한다.
- record의 `human_review`에 `decision`, `reviewer`(OS 사용자명), `reviewed_at`, 판정 당시 `file_sha256`을 남긴다. 표시 전에 파일 hash를 확인한다.
- 사람 판정은 CLIP 판정을 덮어쓰지 않고 나란히 저장한다. CLIP 통과 중 사람 `fail` 비율로 필터의 오채택(필터 범위 밖 artifact 포함)을 추정할 수 있다. CLIP이 버린 이미지는 검수하지 않으므로 오탈락은 측정할 수 없다.

## 7. 수량·중단·재개

목표 N은 사용자 지정 양의 정수이며 기본 예시의 숫자를 실행 의도로 간주하지 않는다. `quota_scope=total`, `limit_policy=until_target_or_user_stop`, 총 attempt/time limit는 null이다. per-source quota는 현재 요구 범위가 아니다.

확정된 unique eligible artifact만 원자적으로 quota에 반영한다. `accepted_count == N`이 되면 새 후보 생성·검증을 시작하지 않는다. MVP는 candidate 한 건씩 처리하여 초과 채택을 피한다. future batch mode는 남은 quota와 이미 처리 중인 후보를 따로 관리해야 한다.

SIGINT/종료 요청을 받으면 새 작업 예약을 멈추고, 안전하게 확정된 파일·record·attempt cursor·count를 저장한다. `interrupted`로 종료하고 부분 export/report를 남긴다. 중단은 목표 달성으로 보고하지 않는다. 한 번도 목표에 도달하지 못하는 입력 조합도 있을 수 있으며 정상적인 낮은 수락률 때문에 임의 종료하거나 판정 기준을 낮추지 않는다.

재개 시 record와 파일 hash를 대조하여 count를 재계산한다. 완료된 판정은 다시 실행하지 않으며, 삭제된 rejected 파일이 없다고 재생성하지 않는다. 고정된 CLIP 문장·원본 임베딩은 다시 계산하지 않는다. 예약되었으나 미완료인 같은 attempt를 복구한 후 다음 index로 이동한다. 사용자 설명·SourceProfile·prompt·CLIP 문장 목록(파일 내용 포함)·margin·sampler·모델·dtype 변경은 새 run을 요구한다. T026 이전(Qwen VQA 판정) run은 resume·export·report·review를 명확한 오류로 거부하며, 저장된 결과물은 그대로 둔다. 실행 GPU ID 변경은 허용하되 새로운 실행 segment의 GPU·환경을 기록한다.

run status는 `running`, `completed`, `interrupted`, `failed`로 구분한다. `completed`는 목표 수량을 충족한 manifest와 활성화된 필수 시각화가 완전히 저장되었을 때만 설정한다. N장을 채운 뒤 시각화 저장만 실패했다면 추가 후보를 생성하지 않고 해당 렌더링만 복구한다. 진행 표시에는 `accepted/N`, attempts, rejected 수, 중복 수, 경과 시간을 포함한다. 수락률에 따른 ETA는 추정임을 표시하고 관측 수락이 없으면 N/A다.

## 8. 모델 학습과 라벨 정책

Qwen·CLIP 추가 학습은 **필요하지 않은 출발점**이다. 이미 학습된 checkpoint로 원본 profile(Qwen)과 비현실 장면 필터(CLIP)를 수행한다. 필요한 검증 데이터는 처음부터 전체 데이터의 GT가 아니라, 소규모 원본·후보 쌍을 사람이 판정한 평가 표본이다. 반복적인 도메인 오류가 확인되면 문장 목록·margin·시각 해상도를 새 run에서 먼저 점검하고 별도 연구로 fine-tuning을 검토한다. 평가 표본을 그대로 학습에 섞지 않는다.

MVP는 라벨 없는 이미지와 계보 manifest를 출력한다. 후속 classification은 클래스 의미 보존 확인이, detection/segmentation은 새 이미지에 맞는 box/mask 작성·검수가 필요하다. CLIP 필터 통과는 GT annotation을 생성하거나 보장하지 않는다. 편집 mask도 segmentation GT가 아니다.

## 9. 재현성과 완료 기준

run에는 입력 snapshot·파일/픽셀 hash, 고정 SourceProfile, source별 attempt index, seed, sampler 설정, 실제 EditRequest·mask, 모델·processor·가중치 revision, upstream commit·patch hash, profile prompt/raw response, CLIP 문장 목록·hash·문장별 점수와 receipt, 저장 상태, 실행 GPU segment와 시간을 보존한다. Qwen profile cache key에는 이미지 hash, user hint, 모델·prompt·전처리·decoding 설정을, CLIP cache key에는 이미지 hash, 문장 목록 hash, 모델·revision·dtype·전처리를 포함한다.

가중치·소스·환경을 고정해도 GPU·kernel 차이로 bitwise 재현성이 달라질 수 있다. GPU 변경 기록 없이 동일 환경 재현이라고 주장하지 않는다.

DragFlow 실행 가속(`generator.speedups`)은 선택 사항이며 기본은 모두 꺼짐(공식 실행 경로)이다. 켠 플래그는 run `config.json`, 후보별 `generation_info.speedups`, report에 기록하고, export manifest(`exports/images.jsonl`) 각 행에도 `generator_speedups`(`official_execution_path`, `enabled`, `tf32`)로 남겨 manifest만 보고도 공식 경로 결과와 구분되게 한다. 결과 동일을 설계 근거로 둔 패치와 정밀도를 바꾸는 변형(`tf32`)을 구분해 기록하며, 변형을 켠 run을 공식 DragFlow 설정의 결과라고 보고하지 않는다. 가속을 위해 DragFlow를 다른 생성기로 바꾸지 않는다([아키텍처](docs/ARCHITECTURE.md#dragflow-실행-가속-플래그-t017)).

export manifest 각 행의 `clip_filter`는 그 이미지의 CLIP 필터 판정 근거다: `verdict`(항상 `accepted`), `limit`(`verifier.max_delta`), `top`(최대 delta 문장의 `index`·`text`·`delta`), `similarity`(`cosine`), `criterion`(`max_delta_over_source`), `model_id`·`model_revision`, `sentences_sha256`(문장 목록 hash)·`sentences_file_sha256`(문장 파일 hash), 그리고 문장마다 `text`·`source`·`candidate`·`delta`를 담은 `sentences`. export는 §5의 재계산·고정 기준 대조를 통과한 행만 쓴다. 이 값은 CLIP 유사도 기록이며 물리적 사실이나 라벨 유효성의 증명이 아니다.

MVP 완료는 실제 DragFlow와 실제 로컬 Qwen(profile)·CLIP(필터)으로 N장의 고유한 필터 통과 이미지를 만들고 사용자 중단·재개·삭제를 검증한 상태다. 장비/가중치가 준비되지 않아 실행하지 못했다면 구현 완료와 통합 검증 대기를 구분한다. 후속 모델의 성능 개선은 MVP 완료 조건이 아니다.
