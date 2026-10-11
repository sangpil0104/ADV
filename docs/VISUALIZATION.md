# 채택 이미지의 편집 설정 시각화

기준: [SPEC v0.10](../SPEC.md), 2026-10-11. **CLIP 판정·시간·자유형 영역을 표시하는 renderer를 구현하고 CPU에서 저장/복구를 확인했다. 실제 GPU 편집 품질 검증은 별도다.**

## 1. 무엇을 보여 주는가

채택 후보마다 왼쪽은 원본 위의 실제 편집 설정, 오른쪽은 표시 없는 생성 결과를 배치한다. 사용자가 어디를 어떻게 편집하도록 요청했는지 볼 수 있게 한다. 화살표는 지시한 이동이며 실제 물체가 도착했음을 측정한 값이 아니다.

| 표시 | 데이터와 의미 |
| --- | --- |
| 빨간 원 `S` | `source_point` / 화면의 start point |
| 초록 십자 `E` | `target_point` / 화면의 end point |
| 노란 화살표 | start → end |
| 반투명 파란 자유형 영역 | `region_mask_path`의 실제 편집 area |
| 흰 윤곽/차량 ID | 선택한 전체 사고 그룹의 원본 경계/구성 차량 |
| 초록 목표 윤곽 | 공통 Δ를 적용한 계획 위치; 실제 생성 결과에서 측정한 위치가 아님 |
| 보라색 `A` | operation에서 필요한 경우 anchor |
| footer | source/candidate ID, seed, operation, 요청/실제 좌표, 면적, 판정, 시간 |

`range`는 자유형 편집 영역의 크기다. `area_pixels`와 원본 전체 면적 대비 비율을 표시하고 이동 거리와 별도 표기한다. segment의 bbox를 area 대신 그리지 않는다. 선택 영역은 모든 변경 픽셀의 정확한 집합이나 정답 GT mask가 아니다.

## 2. 실제 적용값과 좌표

좌표는 EXIF 방향 정규화 후 원본 좌상단 기준 `[x,y]` px다. 저장된 EditRequest와 backend effective geometry를 읽으며 렌더링에서 다시 난수를 뽑지 않는다. 화면용 start/end와 내부 source/target point는 같은 정보다.

upstream resize·feature-grid 반올림·시작점 centroid 교체를 기록한다. 실제 점/영역을 원본 좌표계로 역변환한 뒤 패널 배율·여백을 적용한다. binary mask는 nearest-neighbor로 변환하고 upstream의 soft mask 및 nonzero 영역과 구분한다. 요청값과 실제값이 다르면 양쪽 값과 범례를 명시한다.

자유형/비볼록 segment는 centroid가 영역 밖일 수 있다. 그룹 가상 중심은 객체 밖에 있어도 유효하다. 전체 mask가 backend에 전달되고 공통 Δ가 보존되어야 하며, 그림만 수정하여 정상 실행처럼 보이게 하지 않는다. adapter가 확인한 실제 값을 확보하지 못하면 `REQUESTED / NOT EXECUTED`로 표시하며 채택 완료의 증거로 쓰지 않는다.

생성 전에 `segments/<source_id>/selection.png`를 저장한다. 초록 mask는 선택, 빨강은 제외된 차량 후보이며 ID를 표시한다. 이 원본 진단은 후보별 탈락 이미지와 구분해 보존한다. 선택이 불확실하거나 차량이 빠졌다면 일반화 성공으로 보고하지 않는다.

`accident_group_v2` 계획의 metadata `operation_params.displacement_sampling`에는 표본 r·θ, 그룹 bbox 대각선, 실제 상·하한 픽셀 값, `vertical_scale`, 분포, 가능 방향 크기(`feasible_angle_radians`)가 전체 정밀도로 남는다. drag plan footer에는 이를 줄인 한 줄(`shift=[dx, dy] px | grid shift | r=… px theta=… deg | cap 하한-상한 px (비율 x D) | vertical_scale 분포`, 유효숫자 제한)만 표시한다. 이 값은 계획의 표본 기록이며 실제 이동 측정값이 아니다.

채택 metadata와 footer에는 차량별 외형 template 매칭 이동 추정·잔차를 표시한다. 매칭 불가능하면 `unmeasurable`이며, 이 진단은 실제 사고 차량 동일성·이동 성공의 자동 인증이 아니다.

## 3. CLIP와 시간 표시

채택 결과의 footer/metadata에 5항목 각각 A/B cosine, B−A margin, pass 여부를 연결한다. 1–4 통과 개수와 5번 필수 통과를 표시한다. 점수를 확률이나 물리적 안전 인증으로 표기하지 않는다. 5번은 일반 사고 의미이며 추돌 전용/원본 사고 유형 동일성 검사라고 쓰지 않는다.

현재 시도의 `attempt_elapsed_seconds`와 앞선 실패를 포함한 `accepted_cost_seconds`를 구분해 표시한다. 예: `이번 시도 60초 / 실패 포함 획득 150초`. 아직 commit 비용이 확정되지 않은 private preview를 최종 시간으로 공개하지 않는다. 실행 전체의 초기화/마지막 overhead와 pending은 report에서 설명한다.

`metadata.json`에는 source/candidate/plan hash, 원본·backend 크기, 요청/실제 점과 mask hash, 변환 정보, 면적 단위, segment ID, 모델/문장 hash, CLIP 판정, 시간 완전성, 렌더링 버전을 남긴다. 그림의 숫자와 accepted record가 같은 commit을 참조하도록 한다.

## 4. 저장 시점과 삭제

v0.4.1은 개별 파일을 아래의 종류별 폴더에 저장한다. `generated_grid.png`는 전체 채택 이미지를 종횡비를 유지해 배치하고, `generated_grid.json`은 번호·ID·이미지 hash를 제공한다. `drag_plan_grid.png`는 같은 순서로 대응하는 계획 모음이며 **화살표·mask를 표시한 사진만** 보여준다. 제목·설명·점수·ID·S/E/A 문자·차량 번호를 생략하고 원본 사진에 이미 있는 문자는 유지한다. 저장된 원본과 geometry를 CPU로 렌더링하며 개별 계획 PNG의 설명은 그대로 보존한다. `drag_plan_grid.json`은 번호·ID·개별 계획 경로/hash·렌더링 입력 identity·검증 상태를 제공한다. `index.json`은 모음과 개별 시각화 파일의 목록이다. 0장인 drag plan 모음은 글씨 없는 빈 배경이다.

```text
assets/outputs/<run_id>/visualizations/
├── comparison/<candidate_id>.png
├── drag_plan/<candidate_id>.png
├── metadata/<candidate_id>.json
├── generated_grid.png
├── generated_grid.json
├── drag_plan_grid.png
├── drag_plan_grid.json
└── index.json
```

`gallery.py`와 `visualization_collection.py`는 저장된 채택 record만 읽어 CPU에서 두 모음을 만든다. 신규 채택 commit 및 종료/report 때 갱신하며 hash가 같으면 모음 이미지를 재사용한다. 탈락·중복·미완료 후보를 포함하지 않는다. Qwen 과거 채택은 `legacy_unverified`로 구분한다.

`advv visualize --all`은 기존 `assets/outputs`와 `runs`의 파일을 재정리한다. PNG 바이트를 그대로 복사하고 metadata의 경로/layout, record의 시각화 경로, report/export 링크를 갱신한 뒤 일치 확인된 옛 파일을 제거한다. 원본·생성 사진·판정·시간·plan/effective는 보존한다. 중단 후 같은 명령으로 복구하며 활성 writer·충돌 파일·알 수 없는 사용자 파일은 보호한다. 이 재정리는 재추론이나 CLIP 재판정을 하지 않는다.

판정 전 결과는 private staging에만 둔다. CLIP 통과·고유성 확인 후 그림을 준비하여 accepted transaction으로 공개한다. **탈락 후보는 원본 설정 PNG도 포함해 아무 시각화도 보존하지 않는다.** 후보 mask·비교 PNG·썸네일·임시 복사본·상세 metadata도 삭제한다. 과거의 “NO여도 원본 설정 그림은 보존” 정책은 폐기한다.

기존 run의 명시적 비채택 시각화도 실행 시작 시 정리 대상이다. 원본·accepted와 공유하는 자원을 삭제하지 않으며 알 수 없는 상태는 추정 삭제하지 않는다. report/export에서 삭제한 파일의 링크를 없애고 재생성하지 않는다. 남는 것은 이미지 없는 최소 시간/상태 장부다.

시각화는 이미지 복사본에 그린다. DragFlow·CLIP 입력, pixel hash/중복 검사, 학습 export에는 표시 없는 이미지를 사용한다. 렌더링만 실패하면 유효한 private 생성 결과에서 복구하고 추가 생성·CLIP 호출·quota 증가를 하지 않는다. 필수 그림이 준비되기 전에는 해당 accepted commit을 완료하지 않는다.

## 5. 구현 검증

- 가로가 긴 블랙박스 프레임, EXIF 회전, 패딩/resize에서 mask·점이 일치하는지 확인한다.
- 비볼록 자유형 영역과 작은 mask에서 요청 start와 upstream centroid의 차이를 그대로 보여 준다.
- 면적과 이동 거리를 구분하고 segment 경계·edit area·gradient 확대 영역의 범례를 섞지 않는다.
- 렌더링 전후 생성/CLIP/학습 입력의 hash가 바뀌지 않는지 검사한다.
- 탈락·중복·오류 정리 후 모든 후보별 시각화와 링크가 없고 시간 장부만 남는지 검사한다.
- 채택 commit 직전 중단 뒤 재개해도 그림만 복구하고 비용/수량을 중복 집계하지 않는지 확인한다.
