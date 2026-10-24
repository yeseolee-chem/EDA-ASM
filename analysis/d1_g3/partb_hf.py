#!/usr/bin/env python3
r"""partb_hf.py — D1-G3 U-4 (user decision 2026-10-25 item 4; explanatory / 설명용): the D1-HF runner EDA single point
ONLY (no frag*_dist / frag*_rel) at every Part B TS that passed the TS gate, and a table of its HF channels against the
label-TS HF channels together with ΔE‡. sbatch only (analysis/d1_hf jobs/hf.slurm: conda reactot, ORCA 6.1.1 serial);
nothing in this file was run when it was written (login node), so every path is untested until it runs inside a job.

What is computed (decisions U-4-1 .. U-4-6)
  Part B TS  $G3/O/units/B.<rxn>/<final_dir>/ts.xyz of every Part B plan unit (opt.load_plan()['part_b']) whose
             unit.json has status 'ok' (label atom order; the same file opt.py assemble_b copies to $G3/partB/<rxn>/).
             A Part B unit without unit.json = exit 2 (U-4-1).
  root       $G3/partB_hf/root/<rxn>/ts.xyz (byte copy) + .done = the G1 folder layout prep.g1_input reads
             (D1HF_G2_ROOT); $G3/partB_hf/sources.json = unit, source path and sha256 per reaction (U-4-5).
  input      $G3/partB_hf/inputs/<T>/g2/eda.inp = prep.g1_input(T, rxn, 'g2') in a child with the D1HF_* overrides: the
             D1-HF DFT-TS HF eda.inp (/gpfs/tmp_cpu2/yeseo1ee/d1_hf/inputs/<T>/hf/eda.inp, read only) with only the
             coordinates replaced (label fragment tags, charges, '! HF D3BJ 6-31G* SMD(water) NoSym EDA TightSCF',
             %pal nprocs 1). No frag*.inp is written, so run.py calcs('g2') and build.py outputs('g2') reduce to the EDA
             (eda + the ORCA-written eda_frag1 / eda_frag2), the EDA-only path of the D1-HF g1 pilot (U-4-2, U-4-3).
  run        run.py run <list> g2 hf N on the 'g2' geometry slot with D1HF_SCRATCH=$G3/partB_hf,
             D1HF_RES=$G3/partB_hf/results, D1HF_TEMPLATES=/gpfs/tmp_cpu2/yeseo1ee/d1_hf/inputs (read only),
             D1HF_G2_ROOT=$G3/partB_hf/root (= feat.py's H step on another root; SlowConv retry once = run.py).
  table      build.py allgeom <list> g2 (gates g2_sum, g_bond_ref, g_scf, g_setting, orca; channels) -> per reaction
             elst, pauli, oi, cpcm (+ disp, cds, bond, descriptive) at the Part B TS and at the label TS
             (analysis/d1_hf/results/labels_d1_hf.parquet geom=dft, status ok), d_<ch> = partB - label (kcal/mol),
             converged_to_label and dE_kcal (= ΔE‡ = E_SP(Part B TS) - E(label TS) at the label SP level, from unit.json;
             None for converged_to_label, which had no SP) -> results/B_hf.csv, results/B_hf_summary.json (U-4-4).

CLI (exit 0 = done, 2 = not ready / incomplete (resubmit), 3 = STOP or hard error; submit from analysis/d1_hf; every
command writes $G3/partB_hf/steps/<command>.json)
  partb_hf.py prep          root + EDA inputs (+ run folders whose eda.inp coordinates are stale moved aside)
  partb_hf.py run [i n]     run.py run of the prepared reactions, $SLURM_CPUS_PER_TASK serial ORCA processes; slice i of
                            n (i = A, n = A: $SLURM_ARRAY_TASK_ID of $SLURM_ARRAY_TASK_COUNT)
  partb_hf.py table         build.py allgeom + results/B_hf.csv, results/B_hf_summary.json (exit 2 = HF runs still open)
  internal: _geominput <list.csv>   (child of prep under the D1HF_* overrides: prep.g1_input -> inputs/<T>/g2/eda.inp,
                            results/g2_inputs.csv in the prep.py geominputs format)

Outputs
  $G3/partB_hf/root/<rxn>/{ts.xyz, .done}, sources.json, B_list.csv (job_id, rxn_id), B_run.csv (job_id, n_atoms),
  inputs/<T>/g2/eda.inp, runs/g2/<T>/ (run.py: eda.inp, eda.out.gz, eda_frag{1,2}.{inp,out.gz}, timing.tsv,
  attempts.tsv, meta.json, .done | .fail), runs_prev/<T>.<time>/ (stale runs, U-4-5), labels_d1_hf_g2.json,
  results/{g2_inputs.csv, H_status.csv, labels_d1_hf_g2.parquet}, steps/<command>.json;
  analysis/d1_g3/results/B_hf.csv, B_hf_summary.json; decisions in results/DECISIONS.md (opt.record_decision).

Reused, imported read only (never copied): opt.py (load_plan, UNITS, record_decision, _rj, _f, _now), feat.py (env_hf =
test_t.env_hf, hf_common_mod for parsing, HF_TEMPLATES, HF_KEYS, Stop), g0f0 (child = test_t.run, cpus, clean, qstats,
git_head), D1-HF prep.g1_input / ok_manifest (child), run.py run, build.py allgeom (children).

sbatch (login node: one-shot commands only; first `squeue -u $USER -h -r | wc -l` (MaxSubmit 20 incl. array elements,
MaxJobs 10) and `sinfo -p cpu1,cpu2 -o "%.9P %.6t %C"`; hf.slurm needs --output: its default log folder is the read-only
D1-HF scratch). One task, ~36 EDA single points (HF/6-31G*, serial, %maxcore 3500 MB per process):
  cd /gpfs/home1/yeseo1ee/projects/eda-asm-prediction-d1g3/analysis/d1_hf; L=/gpfs/tmp_cpu2/yeseo1ee/d1_g3/logs
  JB=$(sbatch --parsable -p cpu1,cpu2 -J g3_Bhf --time=48:00:00 --cpus-per-task=12 --mem=64G --output=$L/%x_%j.out \
       jobs/hf.slurm ../d1_g3/partb_hf.py prep ::: ../d1_g3/partb_hf.py run ::: ../d1_g3/partb_hf.py table)
  Resubmission (idempotent): the same line (prep rewrites only changed files; run.py skips .done reactions and good
  outputs; table recomputes). exit 2 of run (a reaction not started / killed) -> the same line again. exit 3 = read
  $G3/partB_hf/steps/<command>.json (fail_stage, fail).
"""
from __future__ import annotations

import sys

sys.dont_write_bytecode = True        # nothing compiled into the read-only code trees (d1_hf, d1_ladder, d1_g2)

import hashlib  # noqa: E402
import json  # noqa: E402
import os  # noqa: E402
import socket  # noqa: E402
import time  # noqa: E402
import traceback  # noqa: E402
from collections import Counter  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

HERE = Path(__file__).resolve().parent
THIS = Path(__file__).resolve()
sys.path.insert(0, str(HERE))
import g3_common as C  # noqa: E402   (paths, write_atomic / write_json; puts analysis/d1_ladder on sys.path)
import opt as O  # noqa: E402         (Phase O: load_plan, UNITS, record_decision; read only)
import feat as F  # noqa: E402        (Phase F: env_hf, hf_common_mod, HF_TEMPLATES; read only)

G0F, L = F.G0F, C.L
Stop = F.Stop

CODE_VERSION = "d1-g3-u4"
PB = C.G3 / "partB_hf"
ROOT, PB_RES, STEPS, RUNS_PREV = PB / "root", PB / "results", PB / "steps", PB / "runs_prev"
SRC_JSON, LIST_CSV, RUN_CSV = PB / "sources.json", PB / "B_list.csv", PB / "B_run.csv"
GEOM = "g2"                                           # the D1-HF geometry slot used (U-4-2); the table calls it 'partB'
B_CSV, B_SUM = C.RES / "B_hf.csv", C.RES / "B_hf_summary.json"
CH = ("elst", "pauli", "oi", "cpcm")                  # the channels of user decision 4
CH_EXTRA = ("disp", "cds", "bond")                    # descriptive extra columns (U-4-4)
COORD_TOL = 1e-6                                      # A (= prep.COORD_TOL)
H_OPEN = ("not_done", "not_started", "incomplete", "absent")       # build.py statuses of a reaction not finished yet

DECISIONS = {
    "U-4-1": "범위(사용자 결정 2026-10-25 4, 설명용): Part B plan 단위 B.<rxn> 중 unit.json status 'ok'인 것 전부(사용자 문구 "
             "'ok TS 34개'는 결정 때의 수; 남은 2반응이 끝나 ok가 늘면 모두 넣는다). 구조 = <unit>/<final_dir>/ts.xyz(라벨 "
             "원자 순서; opt.py assemble_b가 $G3/partB/<rxn>/ts.xyz로 복사하는 같은 파일 — assemble이 아직 돌지 않았으므로 "
             "단위 파일을 직접 읽는다). Part B 단위 중 unit.json이 없는 것이 있으면 exit 2. converged_to_label 반응도 넣고"
             "(라벨 TS와 거의 같은 구조: 채널 차이의 하한) 요약은 전체·not-converged·converged로 나눈다. fail 단위는 표에 "
             "넣지 않고 수와 이유만 적는다.",
    "U-4-2": "runner 경로: D1-HF runner의 'g2' 구조 자리를 D1HF_SCRATCH=$G3/partB_hf, D1HF_RES=$G3/partB_hf/results, "
             "D1HF_TEMPLATES=/gpfs/tmp_cpu2/yeseo1ee/d1_hf/inputs(읽기만), D1HF_G2_ROOT=$G3/partB_hf/root로 덮어쓴다(feat.py "
             "F-6과 같은 방식). run.py는 geom dft·g1·g2만 받으므로 'partB'라는 자리는 D1-HF 코드를 고쳐야 해서 만들지 않고, "
             "이름은 B_hf.csv의 geom 열('partB'; hf_slot 열 = 'g2')에만 붙인다. prep.py geominputs는 5개 입력을 모두 쓰거나 "
             "하나도 쓰지 않으므로(rel1/rel2.xyz 필요) EDA만 쓸 수 없다: 같은 EDA 입력 함수 prep.g1_input(T, rxn, 'g2')를 "
             "D1HF_* 환경의 자식 프로세스(_geominput)에서 직접 불러 inputs/<T>/g2/eda.inp만 쓰고 results/g2_inputs.csv를 "
             "geominputs 형식(job_id, rxn_id, status; X-ok 아님 = not_x_ok, 템플릿 없음 = no_template)으로 쓴다. frag*.inp가 "
             "없으므로 run.py calcs와 build.py outputs가 EDA(eda + ORCA가 쓰는 eda_frag1/2)만 다룬다 = D1-HF g1 파일럿의 "
             "EDA 전용 경로. g1_input·run.py·parse_rxn 경로는 T-3이 검증했다(EDA 채널 파싱은 5개 계산일 때와 같은 코드); "
             "새 시험은 두지 않고 T_3.json 상태를 요약에 적는다.",
    "U-4-3": "입력 내용: eda.inp = 라벨 TS HF 입력(d1_hf/inputs/<T>/hf/eda.inp: HF D3BJ 6-31G* SMD(water) NoSym EDA TightSCF, "
             "%maxcore 3500, %pal nprocs 1)에서 좌표만 Part B TS로 바꾼 것. 조각 표지(FRAG1 = 쌍극자)·FRAG 문자열·전하·다중도는 "
             "라벨 그대로(Part B gate의 partition_match는 B_hf.csv에 기록만). 자식이 parse_inp로 '!' 줄·블록·FRAG·전하·다중도·"
             "원소와 표지가 템플릿과 같고 좌표가 root ts.xyz와 1e-6 Å 안임을 확인한다(아니면 상태 coord_check_failed; "
             "ok가 아닌 입력이 하나라도 있으면 prep STOP, exit 3: 모든 Part B TS가 라벨 원자 순서라 실패는 예상 밖).",
    "U-4-4": "표·요약: B_hf.csv = 반응별 elst, pauli, oi, cpcm(사용자 결정 4의 채널) + disp, cds, bond(설명용 추가 열)의 Part B "
             "TS 값, 라벨 TS 값(analysis/d1_hf/results/labels_d1_hf.parquet geom=dft, status ok), d_<ch> = partB − label "
             "(kcal/mol), |d|; converged_to_label, dE_kcal(unit.json 그대로 = ΔE‡ = E_SP(Part B TS) − E(라벨 TS), 라벨 수준 "
             "SP; converged_to_label 반응은 SP를 하지 않아 빈 값), rmsd_label_A·dform_label_A(rmsd.same_ts), "
             "partition_match, HF 상태와 코어시간. B_hf_summary.json = 요청 채널마다 |d|의 n·중앙값·최대(+ 평균): 전체(HF "
             "ok이고 라벨 HF ok), not-converged, converged 부분집합; ΔE‡ 분포(명령서 E-6의 −0.5 / 0.5 구간); HF gate 실패·"
             "미완 반응은 요약에서 빼고 수를 적는다. 결과는 설명용(명령서 9절 판정에 쓰지 않음).",
    "U-4-5": "currency: prep은 root/<rxn>/ts.xyz를 단위 파일과 바이트 단위로 맞추고(sources.json에 sha256), 기존 "
             "runs/g2/<T>/eda.inp의 좌표 줄이 새 입력과 다르면 그 run 폴더를 $G3/partB_hf/runs_prev/<T>.<시각>으로 옮긴다"
             "(run.py는 .done 반응을 다시 돌지 않으므로). table은 sources.json·root ts.xyz·runs eda.inp 좌표가 지금 단위 "
             "파일과 맞지 않으면 STOP(exit 3: prep 다시).",
    "U-4-6": "결과 위치(명령서 8: 쓰기는 $G3와 results/만): runner 결과(g2_inputs.csv, H_status.csv, "
             "labels_d1_hf_g2.parquet / .json)는 scratch $G3/partB_hf(results/)에 두고(feat.py T-3과 같은 방식), "
             "analysis/d1_g3/results에는 B_hf.csv와 B_hf_summary.json만 쓴다. 명령 기록 = $G3/partB_hf/steps/<명령>.json.",
}


# ================================================================ helpers
def record_decision(key, text):
    O.record_decision(key, text)                     # g3_common.record_decision under the shared fcntl lock


def record_all():
    """spec 11: every decision of this file, before the command acts on it (idempotent per key)."""
    for k, v in DECISIONS.items():
        record_decision(k, v)


def env_b():
    """the D1-HF children: the 'g2' slot on the Part B root (U-4-2)."""
    return dict(os.environ, D1HF_SCRATCH=str(PB), D1HF_RES=str(PB_RES), D1HF_TEMPLATES=str(F.HF_TEMPLATES),
                D1HF_G2_ROOT=str(ROOT), PYTHONDONTWRITEBYTECODE="1")


def step(name, body):
    """body(rec) -> exit code (0 / 2 / 3); $G3/partB_hf/steps/<name>.json (core_h_alloc = wall x cpus)."""
    record_all()
    rec = dict(step=name, code=CODE_VERSION, argv=sys.argv[1:], started=O._now(), host=socket.gethostname(),
               slurm_job=os.environ.get("SLURM_JOB_ID"), slurm_array_task=os.environ.get("SLURM_ARRAY_TASK_ID"),
               cpus=G0F.cpus(), git=G0F.git_head(), python=sys.executable)
    t0 = time.time()
    try:
        rc = int(body(rec) or 0)
    except Stop as e:
        rec.update(fail_stage=e.stage, fail=str(e.msg)[:4000])
        rc = e.rc
    except SystemExit as e:                                # C.stop / H.stop inside a body
        code = e.code
        rc = 0 if code in (None, 0) else (code if isinstance(code, int) else 3)
        rec.update(fail_stage="exit", fail=f"SystemExit({code!r}) inside the step (message on stderr)")
    except Exception as e:                                 # noqa: BLE001 — a crash is a failed step, recorded
        rec.update(fail_stage="error", fail=f"{type(e).__name__}: {e}"[:4000], traceback=traceback.format_exc()[-4000:])
        rc = 3
    rec.update(rc=rc, finished=O._now(), wall_s=round(time.time() - t0, 1))
    rec["core_h_alloc"] = rec["wall_s"] * G0F.cpus() / 3600.0
    C.write_json(STEPS / f"{name}.json", G0F.clean(rec))
    print(json.dumps(G0F.clean({k: v for k, v in rec.items() if k != "traceback"}), default=str)[:6000], flush=True)
    if rec.get("traceback"):
        print(rec["traceback"], file=sys.stderr, flush=True)
    print(f"[{name}] exit {rc} -> {STEPS / f'{name}.json'}", flush=True)
    return rc


def sha_bytes(b):
    return hashlib.sha256(b).hexdigest()


def part_b():
    """(plan, {rxn_id: unit.json}, [uids without unit.json]) of the Part B plan units (U-4-1)."""
    plan = O.load_plan()
    res, missing = {}, []
    for rid in plan["part_b"]:
        r = O._rj(O.UNITS / f"B.{rid}" / "unit.json")
        if r is None:
            missing.append(f"B.{rid}")
        else:
            res[int(rid)] = r
    return plan, res, missing


def ts_source(r):
    return O.UNITS / r["uid"] / (r.get("final_dir") or "optts") / "ts.xyz"


def coord_lines(H, p):
    """the coordinate lines of an ORCA input (feat.hf_stale rule: the SlowConv retry text is not a difference)."""
    return [ln.strip() for ln in Path(p).read_text().splitlines() if H.XYZ_LINE.match(ln)]


def same_xyz(H, inp_text, xyz_path):
    """elements in order and coordinates within COORD_TOL of an ORCA input vs an xyz file."""
    a = H.parse_inp(inp_text)["atoms"]
    b = H.read_xyz(xyz_path)
    return len(a) == len(b) > 0 and all(x[0] == y[0] for x, y in zip(a, b)) and float(
        np.abs(np.array([x[2:] for x in a], float) - np.array([y[1:] for y in b], float)).max()) <= COORD_TOL


def _stats(v):
    a = [float(x) for x in v if x is not None and np.isfinite(float(x))]
    if not a:
        return dict(n=0)
    return dict(n=len(a), median=float(np.median(a)), mean=float(np.mean(a)), max=float(np.max(a)), min=float(np.min(a)))


# ================================================================ prep
def cmd_prep():
    def body(rec):
        import pandas as pd
        F.env_hf(rec)                                      # hf.slurm from analysis/d1_hf, no D1HF_* overrides here
        plan, res, missing = part_b()
        rec["n_part_b"], rec["missing_units"] = len(plan["part_b"]), missing
        if missing:
            raise Stop("partB", f"{len(missing)} Part B units without unit.json {missing} (U-4-1: finish Part B first)", 2)
        ok = {rid: r for rid, r in res.items() if r.get("status") == "ok"}
        rec["units"] = dict(Counter(r.get("status") for r in res.values()))
        rec["fail_units"] = {f"B.{rid}": r.get("reason") for rid, r in res.items() if r.get("status") != "ok"}
        if not ok:
            raise Stop("partB", "no Part B unit with status ok", 3)
        src, n_new = {}, 0
        for rid, r in sorted(ok.items()):                  # root/<rxn>/{ts.xyz, .done} (U-4-5: byte copies)
            R = plan["rxns"][str(rid)]
            if r.get("job_id") != R["job_id"]:
                raise Stop("partB", f"B.{rid}: unit job_id {r.get('job_id')} != plan {R['job_id']}", 3)
            sp = ts_source(r)
            data = sp.read_bytes()
            d = ROOT / str(rid)
            sha = sha_bytes(data)
            if not ((d / ".done").is_file() and (d / "ts.xyz").is_file() and (d / "ts.xyz").read_bytes() == data):
                if (d / ".done").exists():
                    os.remove(C.assert_path(d / ".done"))
                C.write_atomic(d / "ts.xyz", data)
                C.write_atomic(d / ".done", f"{O._now()} {sp} sha256 {sha}\n")
                n_new += 1
            src[str(rid)] = dict(uid=r["uid"], job_id=R["job_id"], n_atoms=int(R["n_atoms"]), source=str(sp), sha256=sha,
                                 final_dir=r.get("final_dir"), unit_finished=r.get("finished"))
        rec["root"] = dict(path=str(ROOT), n=len(src), written_now=n_new)
        C.write_json(SRC_JSON, dict(code=CODE_VERSION, written=O._now(), plan_fingerprint=plan["fingerprint"],
                                    reactions=src))
        jobs = sorted((v["job_id"], int(k)) for k, v in src.items())
        C.write_atomic(LIST_CSV, "job_id,rxn_id\n" + "".join(f"{T},{rid}\n" for T, rid in jobs))
        env = env_b()
        rec["child_env"] = {k: env[k] for k in F.HF_KEYS}
        G0F.child(rec, "geominput", [THIS, "_geominput", LIST_CSV], C.HF_DIR, env)
        gi = pd.read_csv(PB_RES / f"{GEOM}_inputs.csv", dtype=str, keep_default_na=False)
        st = dict(zip(gi.job_id, gi.status))
        rec["inputs"] = dict(Counter(st.values()))
        if sorted(st) != sorted(T for T, _ in jobs):
            raise Stop("inputs", f"{GEOM}_inputs.csv: {len(st)} rows != the {len(jobs)} Part B ok reactions", 3)
        bad = {T: s for T, s in st.items() if s != "ok"}
        if bad:                                            # U-4-3: every Part B TS is in label order -> unexpected
            raise Stop("inputs", f"EDA inputs not built for {bad}", 3)
        H = F.hf_common_mod()                              # parsing only (XYZ_LINE): this process keeps the D1-HF defaults
        moved = {}
        for T, _ in jobs:                                  # U-4-5: a run made from other coordinates goes aside
            run, new = PB / "runs" / GEOM / T, PB / "inputs" / T / GEOM / "eda.inp"
            if (run / "eda.inp").is_file() and coord_lines(H, run / "eda.inp") != coord_lines(H, new):
                q = C.assert_path(RUNS_PREV / f"{T}.{time.strftime('%Y%m%d_%H%M%S')}")
                q.parent.mkdir(parents=True, exist_ok=True)
                os.replace(C.assert_path(run), q)
                moved[T] = str(q)
        rec["stale_runs_moved"] = moved
        size = {v["job_id"]: v["n_atoms"] for v in src.values()}
        order = sorted((T for T, _ in jobs), key=lambda T: (-size[T], T))      # large first (feat.py H)
        C.write_atomic(RUN_CSV, "job_id,n_atoms\n" + "".join(f"{T},{size[T]}\n" for T in order))
        return 0
    return step("prep", body)


def _geominput(lst):
    """child of prep (D1HF_* overrides in its environment): prep.g1_input(T, rxn, 'g2') -> $HF/inputs/<T>/g2/eda.inp
    (only that file; written when its text changed), results/g2_inputs.csv (U-4-2, U-4-3)."""
    import pandas as pd
    sys.path.insert(0, str(C.HF_DIR))                  # first, so 'prep' / 'hf_common' are the D1-HF files (checked)
    import prep as HP
    H = HP.H
    if Path(HP.__file__).resolve().parent != C.HF_DIR.resolve() or Path(H.__file__).resolve().parent != C.HF_DIR.resolve():
        C.stop(f"_geominput: prep / hf_common not from {C.HF_DIR} ({HP.__file__}, {H.__file__})")
    got = dict(D1HF_SCRATCH=H.HF, D1HF_RES=H.RES, D1HF_TEMPLATES=H.TEMPLATES, D1HF_G2_ROOT=H.GEOM_ROOT[GEOM])
    want = dict(D1HF_SCRATCH=PB, D1HF_RES=PB_RES, D1HF_TEMPLATES=F.HF_TEMPLATES, D1HF_G2_ROOT=ROOT)
    bad = {k: (str(got[k]), str(want[k])) for k in want if os.path.abspath(got[k]) != os.path.abspath(want[k])}
    if bad:
        C.stop(f"_geominput: hf_common is not on the D1HF_* overrides: {bad}")
    ids = pd.read_csv(lst, dtype=str)
    okm = set(HP.ok_manifest().job_id)
    rows = []
    for T, rid in zip(ids.job_id, ids.rxn_id.astype(int)):
        tpl = H.TEMPLATES / T / "hf" / "eda.inp"
        if T not in okm:
            s = "not_x_ok"
        elif not tpl.is_file():
            s = "no_template"
        else:
            txt, s = HP.g1_input(T, int(rid), GEOM)
            d = H.INPUTS / T / GEOM
            if txt is not None:
                pt, pn = H.parse_inp(tpl.read_text()), H.parse_inp(txt)
                same_head = all(pt[k] == pn[k] for k in ("bang", "blocks", "frag", "charge", "mult", "pal"))
                same_tags = [a[:2] for a in pt["atoms"]] == [a[:2] for a in pn["atoms"]]
                if not (same_head and same_tags and same_xyz(H, txt, H.GEOM_ROOT[GEOM] / str(rid) / "ts.xyz")):
                    s = "coord_check_failed"
                elif any((d / f"{c}.inp").is_file() for c in H.CALCS if c != "eda"):
                    s = "fragment_inputs_present"          # run.py would run all 5 calculations (U-4-2)
                elif not ((d / "eda.inp").is_file() and (d / "eda.inp").read_text() == txt):
                    H.write_atomic(d / "eda.inp", txt)
        rows.append(dict(job_id=T, rxn_id=int(rid), status=s))
    D = pd.DataFrame(rows, columns=["job_id", "rxn_id", "status"]).sort_values("job_id")
    D["rxn_id"] = D.rxn_id.astype("Int64")
    H.write_atomic(H.RES / f"{GEOM}_inputs.csv", lambda t: D.to_csv(t, index=False))
    print(f"[_geominput] root {H.GEOM_ROOT[GEOM]}, templates {H.TEMPLATES}, inputs {H.INPUTS}/<T>/{GEOM}/eda.inp; "
          f"{len(D)} reactions: {D.status.value_counts().to_dict()}", flush=True)
    return 0


# ================================================================ run
def _slice(i, n):
    if str(n).upper() == "A":
        n = os.environ.get("SLURM_ARRAY_TASK_COUNT")
    if str(i).upper() == "A":
        i = os.environ.get("SLURM_ARRAY_TASK_ID")
    if i is None or n is None or not str(i).isdigit() or not str(n).isdigit() or not 0 <= int(i) < int(n):
        raise SystemExit(f"run: bad slice {i!r} of {n!r}\n{__doc__}")
    return int(i), int(n)


def cmd_run(i=0, n=1):
    def body(rec):
        F.env_hf(rec)
        pj = O._rj(STEPS / "prep.json") or {}
        if pj.get("rc") != 0 or not RUN_CSV.is_file():
            raise Stop("prep", "no successful 'partb_hf.py prep' ($G3/partB_hf/steps/prep.json rc 0, B_run.csv)", 2)
        nw = max(1, G0F.cpus())
        env = env_b()
        rec["child_env"] = {k: env[k] for k in F.HF_KEYS}
        argv = [C.HF_DIR / "run.py", "run", RUN_CSV, GEOM, "hf", nw] + ([i, n] if n > 1 else [])
        return G0F.child(rec, "run", argv, C.HF_DIR, env, ok=(0, 2))
    return step("run" if n == 1 else f"run_{i}_{n}", body)


# ================================================================ table
def cmd_table():
    def body(rec):
        import pandas as pd
        F.env_hf(rec)
        plan, res, missing = part_b()
        if missing:
            raise Stop("partB", f"Part B units without unit.json {missing}", 2)
        ok = {rid: r for rid, r in res.items() if r.get("status") == "ok"}
        src = (O._rj(SRC_JSON) or {}).get("reactions")
        if src is None or not LIST_CSV.is_file():
            raise Stop("prep", "no sources.json / B_list.csv: run 'partb_hf.py prep' first", 2)
        H = F.hf_common_mod()                              # parsing only
        stale = []                                         # U-4-5
        if sorted(int(k) for k in src) != sorted(ok):
            stale.append(f"Part B ok reactions {sorted(set(ok) ^ {int(k) for k in src})} differ from sources.json")
        for k, v in src.items():
            r = ok.get(int(k))
            if r is None:
                continue
            cur = sha_bytes(ts_source(r).read_bytes())
            root = ROOT / k / "ts.xyz"
            if cur != v["sha256"] or not root.is_file() or C.sha256(root) != cur:
                stale.append(f"{k}: root ts.xyz != {ts_source(r)}")
                continue
            run = PB / "runs" / GEOM / v["job_id"] / "eda.inp"
            if run.is_file() and not same_xyz(H, run.read_text(), root):
                stale.append(f"{k}: runs/{GEOM}/{v['job_id']}/eda.inp coordinates != root ts.xyz")
        if stale:
            raise Stop("stale", f"{len(stale)} inconsistencies (rerun 'partb_hf.py prep', then run): {stale[:5]}", 3)
        env = env_b()
        rec["child_env"] = {k: env[k] for k in F.HF_KEYS}
        G0F.child(rec, "allgeom", [C.HF_DIR / "build.py", "allgeom", LIST_CSV, GEOM], C.HF_DIR, env)
        S = pd.read_csv(PB_RES / "H_status.csv", dtype=str, keep_default_na=False)
        hst = {T: (s, rs) for T, s, rs in zip(S.job_id, S.status, S.reason)}
        lj = PB / f"labels_d1_hf_{GEOM}.json"
        hv = {x["job_id"]: x for x in C.read_json(lj)}
        Hl = pd.read_parquet(L.D1HF_LABELS)
        Hl = Hl[Hl.geom == "dft"]
        if Hl.job_id.duplicated().any():
            raise Stop("labels", f"{L.D1HF_LABELS}: duplicate geom=dft job_id rows", 3)
        lab = {r["job_id"]: r for r in Hl.to_dict("records")}
        rows = []
        for rid, r in sorted(ok.items()):
            R, T = plan["rxns"][str(rid)], r["job_id"]
            hs, hr = hst.get(T, ("absent", ""))
            a = hv.get(T) if hs == "ok" else None
            b = lab.get(T)
            b_ok = b is not None and b.get("status") == "ok"
            gt = r.get("gate") or {}
            row = dict(rxn_id=rid, job_id=T, geom="partB", hf_slot=GEOM, dipole=R.get("dipole"), n_atoms=R["n_atoms"],
                       final_dir=r.get("final_dir"), converged_to_label=r.get("converged_to_label"),
                       dE_kcal=O._f(r.get("dE_kcal")), rmsd_label_A=O._f(r.get("rmsd_label_A")),
                       dform_label_A=O._f(r.get("dform_label_A")), same_as_g2_start=r.get("same_as_g2_start"),
                       partition_match=gt.get("partition_match"), partB_reason=r.get("reason"), hf_status=hs,
                       hf_reason=hr, label_hf_status=None if b is None else b.get("status"),
                       hf_core_h=O._f(a.get("coreh_total")) if a else None, used=bool(a is not None and b_ok))
            for ch in CH + CH_EXTRA:
                va = O._f(a.get(ch)) if a else None
                vb = O._f(b.get(ch)) if b_ok else None
                d = va - vb if va is not None and vb is not None else None
                row.update({f"{ch}_partB": va, f"{ch}_label": vb, f"d_{ch}": d, f"abs_d_{ch}": None if d is None else abs(d)})
            rows.append(row)
        D = pd.DataFrame(rows)
        C.write_atomic(B_CSV, lambda t: D.to_csv(t, index=False))
        used = [x for x in rows if x["used"]]
        sub = dict(all=used, not_converged=[x for x in used if x["converged_to_label"] is False],
                   converged=[x for x in used if x["converged_to_label"] is True])
        chan = {ch: {k: dict(_stats(x[f"abs_d_{ch}"] for x in v), signed=_stats(x[f"d_{ch}"] for x in v))
                     for k, v in sub.items()} for ch in CH}
        extra = {ch: {k: _stats(x[f"abs_d_{ch}"] for x in v) for k, v in sub.items()} for ch in CH_EXTRA}
        dE = [x["dE_kcal"] for x in sub["not_converged"] if x["dE_kcal"] is not None]
        open_ = sorted(x["job_id"] for x in rows if x["hf_status"] in H_OPEN)
        fin_bad = {x["job_id"]: f"{x['hf_status']} {x['hf_reason']}" for x in rows
                   if x["hf_status"] not in H_OPEN + ("ok",)}
        t3 = O._rj(F.T3_JSON) or {}
        summ = dict(
            spec="D1G3 U-4 (user decision 2026-10-25 item 4): D1-HF EDA single point at the Part B TS vs the label TS",
            explanatory=True, label="설명용 (명령서 9절 판정에 쓰지 않음)", written=O._now(), code=CODE_VERSION,
            plan_fingerprint=plan["fingerprint"], channels=list(CH), extra_channels=list(CH_EXTRA),
            diff="partB - label (kcal/mol); summaries over |diff| (signed under 'signed')",
            n_part_b_units=len(plan["part_b"]), units_by_status=dict(Counter(x.get("status") for x in res.values())),
            fail_units={f"B.{k}": x.get("reason") for k, x in res.items() if x.get("status") != "ok"},
            n_ok_units=len(ok), hf_by_status=dict(Counter(x["hf_status"] for x in rows)), hf_not_ok=fin_bad,
            hf_open=open_, n_label_hf_ok=sum(x["label_hf_status"] == "ok" for x in rows),
            n_used={k: len(v) for k, v in sub.items()}, abs_diff=chan, abs_diff_extra=extra,
            dE_kcal=dict(note="ΔE‡ = E_SP(Part B TS) - E(label TS), label SP level, from unit.json (not-converged "
                              "subset; converged_to_label had no SP)",
                         stats=_stats(dE), lt_m05=sum(v < -0.5 for v in dE), abs_le_05=sum(abs(v) <= 0.5 for v in dE),
                         gt_05=sum(v > 0.5 for v in dE)),
            hf_core_h=dict(sum=float(sum(x["hf_core_h"] or 0.0 for x in rows)),
                           per_rxn=G0F.qstats(x["hf_core_h"] for x in rows if x["hf_core_h"] is not None)),
            runner=dict(slot=GEOM, env=rec["child_env"], root=str(ROOT), runs=str(PB / "runs" / GEOM),
                        path="prep.g1_input -> run.py run g2 hf -> build.py allgeom g2 (EDA only, U-4-2)"),
            t3=dict(file=str(F.T3_JSON), verdict=t3.get("verdict"), passed=t3.get("pass"),
                    note="T-3 validated the g2 runner path (5 calculations); this EDA-only path uses the same g1_input / "
                         "run.py / parse_rxn code"),
            sources={str(p): C.sha256(p) for p in (L.D1HF_LABELS, PB_RES / "H_status.csv", lj, SRC_JSON, B_CSV)},
            complete=not open_, decisions=DECISIONS)
        C.write_json(B_SUM, G0F.clean(summ))
        rec["summary"] = {k: summ[k] for k in ("n_ok_units", "hf_by_status", "n_used", "complete")}
        rec["summary"]["abs_diff_median_all"] = {ch: chan[ch]["all"].get("median") for ch in CH}
        return 2 if open_ else 0
    return step("table", body)


# ================================================================ CLI
def main(a):
    cmd, n = (a[0] if a else ""), len(a)
    if cmd == "prep" and n == 1:
        return cmd_prep()
    if cmd == "run" and n == 1:
        return cmd_run()
    if cmd == "run" and n == 3:
        return cmd_run(*_slice(a[1], a[2]))
    if cmd == "table" and n == 1:
        return cmd_table()
    if cmd == "_geominput" and n == 2:
        return _geominput(a[1])
    raise SystemExit(__doc__)


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
