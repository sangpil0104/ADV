---
name: advv-rules
description: "ADVV 프로젝트에서 코드·문서·설정·테스트를 바꾸거나 검증하거나 조사할 때 반드시 먼저 읽는 공용 규칙과 검증 명령 모음. AGENTS.md 요약, 작업 카드 형식, 완료 보고 형식, conda env ADVV로 pytest/ruff 실행법, 문서 동기화 대상 표를 담는다. advv-researcher/implementer/qa/gpu-runner 에이전트가 작업 시작 시 사용. ADVV와 무관한 일반 파이썬 질문에는 쓰지 않는다."
---

# ADVV 공용 규칙

원문은 저장소 루트의 `AGENTS.md`다. 이 스킬은 요약이며, 충돌하면 AGENTS.md와 사용자의 최신 지시가 우선한다. 작업 전에 AGENTS.md를 한 번 읽는다. 요약만 보고 판단하면 세부 규칙(예: NO 재질문 금지, fake backend export 차단)을 놓치기 때문이다.

## 절대 규칙 (위반 시 결과물 전체가 무효)

- `../komap/`은 읽지도 쓰지도 않는다.
- 원본(`assets/inputs`), 과거 `runs/` 결과, `models/`, `third_party/`의 upstream 파일을 덮어쓰거나 지우지 않는다. 새 실행 결과는 `assets/outputs/<run_id>`. 정리(cleanup)는 SPEC이 정한 소유권·종료 상태 확인을 거친 run 소유 파일만.
- 실제 DragFlow를 affine warp·복사·다른 생성기로 대체하지 않는다. fake backend는 tests fixture 전용.
- 대상은 차량 사고 블랙박스 단일 프레임이다. 사고 관련 차량 전체를 하나의 그룹으로 묶어 공통 이동 벡터 하나로 평행 이동한다. 차량별 독립 이동·회전·크기 변경, 부분 mask, 무관 차량 합치기 fallback 금지.
- 후보 판정은 [CLIP 문장 5쌍](prompts/clip_criteria_v1.json)이다. 각 쌍은 cosine(B) > cosine(A)일 때만 통과(동점·비유한 값은 실패). 최종 채택은 **1–4번 중 3개 이상 AND 5번 반드시 통과**. 이 규칙·문장·margin(0)을 임의로 바꾸거나 추가 threshold·crop 점수를 채택 조건에 끼워 넣지 않는다. 통과율 때문에 기준을 완화하지 않는다.
- CLIP 통과를 물리 법칙·실제 그룹 이동 성공의 증명으로 표현하지 않는다. Qwen VQA·profile은 새 경로의 필수 단계가 아니다.
- GPU는 `advv-gpu-safety` 스킬 절차 없이 쓰지 않는다. 다른 사용자의 GPU에는 절대 올리지 않는다.
- 실행하지 않은 검사를 실행했다고 보고하지 않는다. 문서만 바꿨으면 "문서만"이라고 쓴다.

## 코드 규칙

- 패키지는 `src/advv/`. CLI·계약·생성기·검증기·저장소·평가를 분리한다.
- 외부 모델 import/로드는 backend 안에서 지연 수행한다. CPU 테스트가 CUDA·가중치 없이 돌아야 한다.
- 외부 프로그램은 인자 배열 + 명시적 `cwd`. 문자열 shell 조합 금지.
- 표준 logging, 타입 힌트, `errors.py`의 명시적 오류 유형. `except: pass` 금지.
- 주변 코드의 밀도·명명·관용구에 맞춘다(주석은 적고, 이유가 비자명할 때만).
- `src/advv/*.py`가 바뀌면 implementation hash가 바뀌어 기존 run은 `--resume`할 수 없다. 보고에 적는다.
- **모든 설치·실행·검사·worker는 conda 환경 `ADVV`(대문자)를 쓴다.** `.venv-*`, 다른 conda 환경(소문자 `advv` 포함), base로 우회하지 않는다. worker도 `ADVV`의 같은 `sys.executable`로 띄운다.

## 동작을 바꾸면 함께 갱신할 곳

| 바뀐 것 | 갱신 대상 |
| --- | --- |
| 요구사항·판정·수량·저장 규칙 | `SPEC.md` |
| 모듈·실행 순서·저장 구조 | `docs/ARCHITECTURE.md` |
| 진행 상태·체크리스트·검증 기록 | `docs/IMPLEMENTATION_PLAN.md` |
| 사용법·구현/예정 구분·테스트 수 | `README.md` |
| 설정 키 | `configs/advv.example.yaml` + `src/advv/config.py` 검증 |
| 외부 모델·revision·라이선스 | `docs/REFERENCES.md`, `configs/upstream.lock.json` |
| CLIP 판정·빌드 검증 | `prompts/clip_criteria_v1.json`, `docs/CLIP_BUILD_VERIFICATION.md` |
| 시각화·갤러리 | `docs/VISUALIZATION.md` |
| 평가 계획 | `docs/EVALUATION.md` |

## 검증 명령

conda 환경 이름은 `ADVV`다(`environments/advv-conda.yml`). 아직 없으면 테스트 전에 오케스트레이터에 알린다(사용자 승인 후 gpu-runner가 생성).

```bash
cd /data2/donginson/projects/ADVV
CUDA_VISIBLE_DEVICES="" conda run -n ADVV python -m pytest -q
conda run -n ADVV python -m ruff check src scripts tests
```

pytest는 반드시 `CUDA_VISIBLE_DEVICES=""`로 돌린다. upstream DragFlow import가 CUDA context를 만들 수 있어서, GPU를 숨기지 않으면 소유자 확인 없이 기본 GPU(0번, 다른 사용자 GPU일 수 있음)에 context가 생긴다.

`conda run`은 heredoc stdin을 전달하지 않는다. 스크립트를 stdin으로 넣을 때는 `ADVV` 환경의 python 실행 파일 경로를 직접 쓴다.

문서 링크 점검:

```bash
for f in SPEC.md README.md docs/*.md; do grep -oE '\]\(([^)#h][^)#]*)' $f | sed 's/](//' | while read l; do [ -e "$(dirname $f)/$l" ] || echo "BROKEN $f -> $l"; done; done
```

## 작업 카드 (오케스트레이터 → 에이전트)

```text
TASK <id>: <한 줄 목표>
범위: <건드려도 되는 파일/디렉터리>
금지: <건드리면 안 되는 것>
입력: <읽을 파일, 이전 산출물 경로>
완료 기준: <테스트·산출물·확인 항목>
출력: _workspace/<id>_<agent>_<artifact>.md
```

## 완료 보고 (에이전트 → 오케스트레이터)

산출물 파일 첫 부분과 반환 메시지에 같은 형식으로 쓴다.

```text
결과: done | partial | blocked
변경 파일: <목록 또는 없음>
실행한 검사: <명령과 결과 수치> (실행 안 한 것은 "미실행"으로 명시)
남은 것 / 막힌 이유: <목록>
사용자 결정 필요: <질문 또는 없음>
```
