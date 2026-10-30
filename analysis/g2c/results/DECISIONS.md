# G2c 결정 기록 (docs/specs/GEO1B.md 5)

- **0-code** (2026-10-30): g2c_d1.py, select_g2c.py, jobs/g2c.slurm는 명령서 코드 블록 그대로다(job 1054023, /gpfs/home1/yeseo1ee/geo1b_extract_spec.py). analysis/geo/d0g2.py는 GEO1 원본 그대로다.
- **0-D0 / orig** (GEO1b DECISIONS 0-D0와 같은 내용): D0 'orig' run인 /gpfs/tmp_cpu2/yeseo1ee/d0_g2는 사용자 요청 정리 job이 일부를 지운 뒤 사용자가 멈췄다. 사용자 지시는 "삭제된것만 다시 계산"이다.
  - 복구 코드: analysis/g2c/d0g2_restore.py (새로 만듦).
  - job 1054036: manifest를 다시 만들고(5,260행) scan을 돌렸다.
    - 결과: kept_done 3,463, kept_fail 593, deleted 1,203, partial 1(rxn 5561, $D0G2/restore/partial로 옮김).
    - Z_status 불일치 0.
  - 1,204반응만 GEO1 d0g2.py pool로 다시 계산한다(job 1054046, cpu2 128코어 × 2, pool 16 × 8코어; GEO1 때 1,653코어시간).
  - 이 g2c worktree에서 돌린다. 명령서 5-2절 이유(job이 반응마다 코드를 다시 읽음)와 같다.
  - 다시 계산한 반응의 orig는 새 표본이다. 고르는 규칙(가장 낮은 e_ts_freq_eh)은 바뀌지 않는다.
- **0-disk**: GEO1b DECISIONS 0-4 참고(여유 35 GB < 필요량). G2c root는 공간 순서대로 낸다: D1 r1, D1 r2, D0 r1. D0 r2와 build는 공간이 정리된 뒤 낸다.
- **0-first-check (5-3 ⑤)**: 원래 d0_g2 jobs/는 5,260개 모두 남아 있어 D0 비교에 그대로 쓸 수 있다.
