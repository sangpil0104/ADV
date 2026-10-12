프로젝트 규칙은 [AGENTS.md](AGENTS.md)를 따른다.

## 하네스: ADVV 증강 파이프라인 개발

**목표:** 사용자가 연달아 던지는 요청을 메인 세션이 분류·위임하고, 조사·구현·검증·GPU 실행은 하위 에이전트가 백그라운드에서 병렬 처리한다.

**트리거:** ADVV 구현·수정·조사·검증·환경 설치·GPU 실행 요청 시 `advv-orchestrator` 스킬을 사용하라. 파일을 보지 않고 답할 수 있는 짧은 개념 질문은 직접 응답 가능.

**변경 이력:**
| 날짜 | 변경 내용 | 대상 | 사유 |
|------|----------|------|------|
| 2026-09-30 | 초기 구성: researcher/implementer/qa/gpu-runner + orchestrator/rules/gpu-safety 스킬 | 전체 | 연속 질문 처리를 위한 하위 에이전트 분업 |
| 2026-10-06 | advv-rules·orchestrator를 CLIP 필터 판정으로 갱신 | skills | 판정 방식 교체(T026) |
| 2026-10-11 | main(e01f987) 위로 이전: 규칙을 사고 그룹 이동·CLIP 5쌍·conda ADVV 기준으로 갱신 | skills, agents, .gitignore | 원작성자 main 갱신 반영 |
| 2026-10-11 | pytest 명령에 CUDA_VISIBLE_DEVICES="" 강제 | skills/advv-rules | M002 테스트 중 기본 GPU에 CUDA context 생성 |
| 2026-10-11 | GPU 감시 멈춤 조건을 'run 밖 프로세스 출현(소유자 무관)'으로 확대 | skills/advv-gpu-safety | M006에서 내 다른 실험이 GPU6에 올라와 늦게 감지 |
