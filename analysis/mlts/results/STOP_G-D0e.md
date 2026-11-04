# STOP: G-D0e (2026-11-04T20:34:40, pipeline.py)

- **조건 / condition**: g0: job 1061949 exit 3 (STOP) — log /gpfs/tmp_cpu2/yeseo1ee/mlts/logs/mltp_G-D0e_*1061949*
- **수치 / numbers**:
  - job: "1061949"
  - unit: "g0"
  - end: {"state": "FAILED", "exit": 3, "sig": 0, "elapsed": "00:11:59", "start": "2026-11-04T19:53:56", "end": "2026-11-04T20:05:55"}
- **선택지 / options** — 이 파일 첫 줄에 `RESOLVE: <선택지>`를 쓰면 다음 pass가 적용하고 DECISIONS.md에 적은 뒤 이 파일을 `STOP_G-D0e.resolved.md`로 이름을 바꾼다(지우지 않음):
  - `retry` — 시도 횟수를 0으로 되돌리고 다시 낸다(원인을 고친 뒤) / reset attempts and resubmit
  - `accept` — 이 단계를 끝난 것으로 보고 다음 단계로 간다 / treat the stage as done
  - `skip` — 이 단계를 건너뛴다(뒤 단계가 이 단계 산출물 없이 시작될 수 있다) / skip the stage
  - `param [<단계>.]<이름>=<값>` — 단계 매개변수를 바꾸고 다시 낸다 / set a stage parameter and retry
- **추천 / recommended**: 정해진 추천안 없음 — 명령서 STOP이므로 사용자(lead)가 고른다.
- **영향 / effect**: 이 단계와 이 단계에 의존하는 단계는 시작하지 않고, HALT 동안 새 단계도 시작하지 않는다. 이미 시작했고 이 단계에 의존하지 않는 단계는 계속한다.
