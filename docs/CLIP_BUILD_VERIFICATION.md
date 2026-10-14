# CLIP 전환 빌드 검증

## 2026-10-14: 이동 하한 상향(`accident_group_v3`)과 검출 오검출 필터(`clip_contact_context_v3`) (M011)

- conda `ADVV`, `CUDA_VISIBLE_DEVICES=''`에서 전체 CPU 테스트 **248개 통과**(기존 216 + 신규 32), ruff check 통과. GPU 추론·모델 다운로드·wheel 재빌드는 하지 않았다.
- 재적용 방법: `assets/outputs/blackbox_002/segments/*/result.json`과 960px 실제 Mask R-CNN mask PNG를 읽기만 하고 `select_vehicles`를 v2와 v3로 다시 실행했다. 로컬 CLIP ViT-L/14(같은 revision)는 **CPU**에서 실행했다. v2 CPU 재계산은 저장된 GPU 결과와 선택 10/10 일치, 단위 70개(×2 view)의 margin 최대 차이 1.6e-7. 결과와 이동 표본은 scratch 폴더에만 썼고 `assets/`는 바뀌지 않았다(mtime 확인). Mask R-CNN 검출 점수(`confidence`)는 backend가 이미 segment마다 기록하고 있어 저장 데이터로 점수 규칙을 검증할 수 있었다. 허용 클래스는 설정 검증에서 `[3, 4, 6, 8]` = car, motorcycle, bus, truck로 고정되어 있다(04의 건물 0008은 bus로 검출됨).

### 선택 before/after (blackbox_002 저장 분할, 960px)

| 원본 | v2 선택 (저장 = CPU 재계산) | v3 선택 | v3에서 바뀐 이유 |
| --- | --- | --- | --- |
| 01 | 0000, 0002 | 같음 | 작은 먼 차 0004·0005(`mask_too_small`), 0010(`low_detection_score` 0.555)이 필터에서 빠짐. 원래도 비선택 |
| 02 | 0004, 0010, 0013(=0010 중복) | 같음 | 먼 작은 차 0006·0009·0011·0012·0014가 `mask_too_small`(0014는 점수 0.543도). 원래도 비선택 |
| 03 | 0000, 0001 | 같음 | 0002 `mask_too_small`(원래 비선택). 단 새 이동 하한에서 보류(아래) |
| 04 | 0000, 0003, 0005, **0008(건물)** | **없음 → 보류** | 0008: `low_detection_score`(0.508) + `elevated_top_structure`(위 끝 0.020H, 아래 끝 0.506H). 0008이 빠지자 0005는 단독 단위(CLIP fg −0.036), 실제 충돌 쌍 0000+0003은 CLIP fg −0.0045 / ctx +0.0013으로 통과 못 함(M009의 보닛 제거 후와 같은 현상) |
| 05 | 0000, 0001 | 같음 | 변화 없음 |
| 06 | 없음 | 없음 (보류 유지) | 옆으로 누운 차량 미검출 그대로 |
| 07 | 0003, 0005, **0004, 0007, 0009(보행자 옆 조각)** | **0003, 0005** | 0004 `fragmented_mask`(가장 큰 조각 0.548, 점수 0.921), 0007 `low_detection_score`(0.548), 0009 `low_detection_score`(0.519). 사고 쌍 0003+0005 CLIP fg +0.033 / ctx +0.026 |
| 09 | 0000, 0001 | 같음 | 0004 `mask_too_small`(원래 비선택) |
| 10 | 0000, 0001 | 같음 | 변화 없음 |
| accident_car_tree | 0007, 0009(=0007 중복) | 같음 | 0009는 점수 0.578이지만 대표 0007(0.750)과 같은 차량 단위라 남음(합집합 유지). 0001·0003·0008 `mask_too_small` |

사고 그룹이 있는 원본: v2 9/10 → v3 8/10(04 보류). 정상 사고 차량으로 본 선택(01·02·03·05·07의 0003+0005·09·10·car_tree)은 모두 유지됐다.

### 필터 임계값 근거 (차량 단위 = 중복 병합 후, blackbox_002 10장 전체 단위)

| 규칙 (기본값) | 잡아야 할 대상의 값 | 유지할 선택 사고 차량의 값 | 가능한 범위 / 비고 |
| --- | --- | --- | --- |
| `detection_min_score` 0.7 | 04/0008 0.508, 07/0007 0.548, 07/0009 0.519 | 최소 0.750(car_tree 0007), 그다음 0.836(02 0010) | (0.548, 0.750]. 같은 차 낮은 점수 중복(car_tree 0009 0.578, 02 0013 0.736)은 단위 대표 점수로 판정해 유지 |
| `elevated_top_structure`: 위 끝 < 0.05H 그리고 아래 끝 ≤ 0.6H | 04/0008 위 0.020, 아래 0.506 | 위 끝 최소 0.180(03), 전체 남은 단위 최소 0.120 | 위 끝 조건만으로는 가까운 버스를 버릴 수 있어 아래 끝 조건을 함께 둠 |
| `min_largest_component_fraction` 0.7 | 07/0004 0.548 | 1.000(모두), 전체 남은 단위 최소 0.947 | (0.548, 0.947]. 채움률(fill)은 07/0004 0.489 대 사고 차 07/0005 0.542, 04/0003 0.541로 간격이 좁아 쓰지 않음 |
| `max_area_fraction` 0.4 | (해당 없음) | 최대 0.252(car_tree 단위), 10/0000 0.136 | 제안값 0.15는 car_tree 사고 차를 버리고, 건물 0008(0.155)과도 구분되지 않아 안전 상한으로만 둠 |
| `min_area_fraction` 0.001 | (조각 0007·0009는 0.0044·0.0028이라 점수로 잡힘) | 최소 0.0098(03) | 비선택 먼 작은 차량만 빠짐(0.0002–0.0009). 0.003 이상이면 먼 사고·오토바이를 버릴 위험 |
| `min_bottom_fraction` 0.1 | (해당 없음) | 아래 끝 최소 0.298(03) | 남은 단위 최소 0.167, 전체 최소 0.113(02 먼 차). 하늘·간판 검출용 안전 규칙 |
| 가로세로 비 | 쓰지 않음 | 0.52(가장자리 잘린 차)–4.08(옆모습) | 조각(0.47–0.68)과 겹침 |

### 이동 하한 (`accident_group_v3`) 분포, 960px 기본값 0.08/0.15/0.03/0.5

이미지 대각선은 960×540에서 1101.4px(절대 하한 33.0px), 960×720에서 1200px(36.0px). 하한 L = max(0.08·D, 0.03·이미지 대각선), 상한 U = 0.15·D, 둘 다 세로 축소·정수화 후 |Δ| 기준. 표본은 v3 그룹에 실제 `sample_edit`(seed 42, attempt 0–199)을 호출했다. 200개 모두 성공(`sampling_error` 0)하고 모두 L ≤ |Δ| ≤ U였다.

| 원본 | v3 차량 수 | D (px) | v2 범위 0.03–0.15·D (r 기준) | v3 L–U (|Δ| 기준) | 표본 |Δ| 최소/중앙/최대 | 표본 Δ 예 (attempt 0–4) |
| --- | --- | --- | --- | --- | --- | --- |
| 01 | 2 | 462.1 | 13.9–69.3 | 37.0–69.3 | 37.1 / 44.1 / 68.3 | (−42,11) (21,33) (−63,−14) (−59,4) (−68,6) |
| 02 | 3 | 394.3 | 11.8–59.1 | 33.0–59.1 | 33.1 / 38.4 / 58.1 | (−37,−7) (−40,17) (−49,3) (−44,6) (−39,13) |
| 03 | 2 | 210.3 | 6.3–31.5 | **33.0 > 31.5 → 보류** | − | − |
| 04 | 0 | − | (v2: 4대, D 639.2) | 그룹 없음 → 보류 | − | − |
| 05 | 2 | 350.8 | 10.5–52.6 | 33.0–52.6 | 33.1 / 38.2 / 52.1 | (33,5) (−39,−11) (37,−16) (−33,−2) (40,−3) |
| 06 | 0 | − | − | 그룹 없음 → 보류 | − | − |
| 07 | 2 | 288.8 (v2 5대: 656.9) | 8.7–43.3 | 33.0–43.3 | 33.1 / 35.2 / 43.1 | (−38,3) (−36,−9) (−41,5) (−33,−2) (−42,1) |
| 09 | 2 | 266.8 | 8.0–40.0 | 33.0–40.0 | 33.1 / 35.1 / 39.2 | (−36,6) (32,−11) (38,−3) (33,3) (−34,3) |
| 10 | 2 | 647.4 | 19.4–97.1 | 51.8–97.1 | 51.9 / 64.6 / 94.1 | (−85,17) (−74,5) (56,−19) (−64,24) (55,6) |
| accident_car_tree (960×720) | 2 | 788.4 | 23.7–118.3 | 63.1–118.3 | 63.1 / 78.9 / 114.0 | (−71,−28) (−79,35) (82,−17) (80,−3) (−49,43) |

- blackbox_002에서 채택된 Δ와 비교: 01 (−3,15) |Δ|=15.3, 05 (3,7) |Δ|=7.6은 새 하한(37.0, 33.0)보다 작아 이제 나오지 않는다. car_tree (−33,−31) |Δ|=45.3도 새 하한 63.1보다 작다.
- 이동 가능 원본: 7/10(01·02·05·07·09·10·car_tree). 03은 그룹이 작아(D=210 < 0.2×1101=220) 절대 하한이 상한을 넘어 `no_valid_motion` 보류 대상이 된다. 상한을 하한까지 올리지 않는 이유는 [SPEC 5.3](../SPEC.md#53-공동-이동).
- 하한이 세로 축소 뒤 |Δ|에 걸리므로 기본값에서 세로 성분만 있는 이동은 없다(|Δy| ≤ 0.5·U = 0.075·D < 0.08·D). 표본도 대부분 가로가 큰 방향이다.

## 2026-10-12: 입력 축소, 사고 차량 선택 v2, 이동 상한 0.15 (M009)

- conda `ADVV`, `CUDA_VISIBLE_DEVICES=''`에서 전체 CPU 테스트 **216개 통과**(기존 181 + 신규 35: 축소 15, 선택 규칙 20), ruff check 통과. GPU 추론·모델 다운로드·wheel 재빌드는 하지 않았다.
- 축소: 긴 변 960 기본, 확대 없음, 32px 미만 거부, 원본 파일 hash 불변, snapshot 크기 = `Source.width/height`, fake pipeline에서 계획 좌표·mask·생성 이미지가 snapshot 좌표계인지, record·metadata의 `preprocessing` 기록, recipe 반영을 검사했다.
- 선택 규칙 재적용: M008(`blackbox_001`) 10장의 저장된 Mask R-CNN mask를 그대로 읽고, 로컬 CLIP ViT-L/14(같은 revision)를 **CPU**에서 실행해 v1과 v2를 다시 계산했다. v1 CPU 재계산은 저장된 GPU 점수와 margin 최대 차이 2.3e-7, 선택 10/10 일치. 결과는 scratch 폴더에만 쓰고 `assets/outputs`는 바꾸지 않았다. "960 모의"는 저장 mask를 nearest로 줄인 근사이며 960px 입력의 실제 재분할 결과가 아니다.

| 원본 | v1 선택 (M008 저장) | v2 선택 (원본 해상도) | v2 (960 모의) | 바뀐 이유 |
| --- | --- | --- | --- | --- |
| 01 (1920×1080) | 0003, 0005 | **0000, 0003** | 0000, 0003 | 흰 해치백 0000과 초록 SUV 0003 사이 4px 틈(작은 차 대각선의 0.0085) → 접촉 단위, CLIP fg +0.043 / ctx +0.025. 오른쪽 끝 0005는 먼 차(면적 비 0.061, 아래 끝 191px 위 = SUV 높이의 0.52). 960 모의에서는 먼 차 행렬(0001·0004·0009)이 CLIP을 통과했으나 `distant_separate_unit`으로 제외 |
| 02 (1920×1080) | 0000, 0003, 0005, 0008, 0013 | 같음 (0013 = 0008 중복, IoU 0.90) | 같음 | 트럭 중복 병합으로 차량 4대. 960 모의의 먼 차(0002·0011) CLIP 통과분은 `distant_separate_unit`으로 제외. 과선택 의심은 사람 검수 필요 |
| 03 | 0000, 0001 | 같음 | 같음 | 변화 없음 |
| 04 (1920×1080) | 0000, 0003, **0006(보닛)** | **없음 → 보류** | 없음 | 0006 `ego_vehicle_hood`(아래 경계까지 19px, 폭 94%, 면적 29%). 보닛이 빠진 실제 충돌 쌍 0000+0003은 CLIP fg −0.0041 / ctx +0.0007로 통과 못 함 |
| 05 | 0000, 0001 | 같음 | 같음 | 변화 없음 |
| 06 (1280×720) | 없음 | 없음 | 없음 | 옆으로 누운 차량은 끝부분(0007)만 검출, CLIP 단독 fg −0.039. 0000 주변 먼 차 0003/0004/0005/0008 분리(0008 = 0003 중복) |
| 07 | 0003, 0004 | 같음 | 같음 | 변화 없음 |
| 09 | 0000, 0001 | 같음 | 같음 | 변화 없음 |
| 10 | 0000, 0001 | 같음 | 같음 | 변화 없음 |
| accident_car_tree (960×720) | 0002, 0007, 0009 | **0007, 0009** (0009 = 0007 중복, IoU 0.74) | 같음 | 0002 먼 초록 차: 면적 비 0.011, 아래 끝 324px 위(사고 차 높이의 0.81). 자기 면적의 88%가 0007 mask 안 → 실제 접촉이 아니라 사고 차 mask가 번진 원근 겹침. 단독 caption fg +0.044 / ctx +0.039 |

- M008 육안 판정과 비교: 01 누락·04 보닛 포함·accident_car_tree 먼 차·02 트럭 중복은 규칙으로 고쳐졌다. 04는 보닛을 빼자 CLIP이 실제 충돌 쌍을 통과시키지 않아 선택 0대(보류)가 되었다. 06은 분할기 한계라 규칙으로 고칠 수 없다. 사고 그룹이 있는 원본은 v1 9/10 → v2 8/10이다.
- 임계값은 이 10장으로 정한 값이며 일반화 검증이 아니다. 중복 IoU 0.5(중복 0.54–0.97, 서로 다른 접촉 차량 최대 0.27), 틈 비율 0.015(사고 쌍 0.0085, 무관 쌍 최소 0.025).

## 2026-10-11: 이동 크기 상한(M002)과 QA 후속 수정(M002-fix), sampler `accident_group_v2`

- M002: 공통 이동 벡터를 크기 r(그룹 bbox 대각선 비율 상·하한)과 방향 θ로 나눠 표본하고 세로 성분을 `vertical_scale`로 줄였다. conda `ADVV`, `CUDA_VISIBLE_DEVICES=''`에서 전체 CPU 테스트 **169개 통과**, lint 통과(implementer·QA 각각 실행).
- M002-fix: θ를 해당 r의 가능 각도 구간 위 균등으로 표본, 이동 불가 원본 사전 보류(`no_valid_motion`/`no_movable_sources`), 표본 기록의 범위·run 설정 검사, footer 축약. 같은 조건에서 전체 CPU 테스트 **181개 통과**, ruff check 통과.
- 1280×720 CPU 측정(기본값 0.3/0.03/0.5/uniform_radius, 실제 `sample_edit` 호출, 원본당 attempt 200개): 일반 두 차량·폭 80%·폭 80%+높이 80%·폭 90%(가장자리)·폭 95%·**폭 100%** 그룹 모두 attempt 실패 0/200(수정 전 QA: 폭 100%에서 38/40 실패). 폭 90%+높이 100%(세로 여유 0, r 가능 범위가 [하한, 약 64px]로 좁음)는 2/200이 `sampling_error`; 폭·높이 100%는 사전 판정에서 이동 불가로 보류 대상. 모든 표본의 θ가 다시 계산한 가능 구간 안, 결정론·JSON 왕복 후 `validate_plan` 통과. attempt당 CPU 약 0.16–0.21초.
- 실현 분포(조건부 균등): 화면 여유가 충분하면 r은 설정 분포 그대로(정규화 r 평균 0.51, U[0,1] 대비 KS 0.055, n=200). 여유가 적으면 가능한 r만 남는다(폭·높이 80%: 정규화 r 평균 0.26). 폭 100% 그룹은 200개 모두 세로 이동이다. 방향 분포는 원 위 균등이 아니다.
- GPU 추론, 모델 다운로드, wheel 재빌드는 하지 않았다. 상·하한·`vertical_scale` 초기값은 미검증이며 GPU 스윕이 남았다.

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
