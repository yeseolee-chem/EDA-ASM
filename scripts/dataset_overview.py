"""Halo8 데이터셋 전수 스캔 → 한국어 요약 리포트.

10개의 ASE SQLite DB(Halo_1.db..Halo_10.db, ~22M 프레임)를 한 번 순회하며
프레임 단위 통계를 궤적 단위(trajectory id = dand_id.rsplit('_',1)[0])로
집계한다. 각 궤적에서 R = 프레임 0, P = 마지막 프레임, TS = 내부 프레임
중 에너지 최대값으로 잡아 Ea = E(TS) - E(R) 을 계산한다.
"""
from __future__ import annotations

import glob
import os
import re
import sqlite3
import sys
import time
from collections import Counter, defaultdict

import numpy as np
from ase.data import chemical_symbols

DATA_DIR = "data/Halo8"
DAND_RE = re.compile(rb'"dand_id":"([^"]+)"')


def family_of(traj_id: str) -> str:
    low = traj_id.lower()
    if low.startswith("t1x"):
        return "T1x"
    if low.startswith("halo"):
        return "Halogen"
    return "기타"


def fmt_int(n: int) -> str:
    return f"{n:,}"


def hist_lines(values, bins, label):
    if not values:
        return [f"  (데이터 없음)"]
    arr = np.asarray(values)
    counts, edges = np.histogram(arr, bins=bins)
    lines = []
    total = counts.sum()
    for c, lo, hi in zip(counts, edges[:-1], edges[1:]):
        pct = 100 * c / max(total, 1)
        bar = "█" * int(round(pct / 2))
        lines.append(f"  [{lo:6.1f}, {hi:6.1f})  {fmt_int(int(c)):>10}  {pct:5.1f}%  {bar}")
    lines.append(
        f"  ─ {label}: min={arr.min():.2f}  median={np.median(arr):.2f}"
        f"  mean={arr.mean():.2f}  max={arr.max():.2f}  n={len(arr)}"
    )
    return lines


def main() -> int:
    db_files = sorted(glob.glob(os.path.join(DATA_DIR, "Halo_*.db")))
    if not db_files:
        print(f"DB 파일을 찾을 수 없음: {DATA_DIR}", file=sys.stderr)
        return 1

    # 궤적 단위 집계 자료구조
    # traj_id -> {natoms, numbers_bytes, family, formula_counter_key,
    #             n_frames, e_first, e_last, e_max, e_max_idx, e_min,
    #             frame_max_idx}
    traj = {}

    db_rows = {}
    total_rows = 0
    t_global = time.time()

    for db in db_files:
        t0 = time.time()
        conn = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
        conn.row_factory = None
        cur = conn.cursor()
        cur.execute(
            "SELECT natoms, energy, numbers, data FROM systems"
        )
        n = 0
        for natoms, energy, numbers_blob, data_blob in cur:
            n += 1
            # data_blob: 8-byte length header then JSON; dand_id is at start
            m = DAND_RE.search(data_blob, 0, 256)
            if not m:
                continue
            did = m.group(1).decode("ascii", "ignore")
            try:
                stem, idx_str = did.rsplit("_", 1)
                frame_idx = int(idx_str)
            except ValueError:
                continue
            t = traj.get(stem)
            if t is None:
                # numbers: int32 little-endian
                arr = np.frombuffer(numbers_blob, dtype=np.int32)
                t = {
                    "natoms": int(natoms),
                    "numbers": arr.tolist(),
                    "family": family_of(stem),
                    "n_frames": 0,
                    "frame_indices": [],
                    "energies": [],
                }
                traj[stem] = t
            t["n_frames"] += 1
            t["frame_indices"].append(frame_idx)
            t["energies"].append(float(energy) if energy is not None else float("nan"))
        conn.close()
        db_rows[os.path.basename(db)] = n
        total_rows += n
        print(
            f"[scan] {os.path.basename(db):12s}  rows={fmt_int(n):>12}  "
            f"trajs(누적)={fmt_int(len(traj)):>10}  {time.time()-t0:5.1f}s",
            file=sys.stderr,
            flush=True,
        )

    print(f"\n[scan] 총 {fmt_int(total_rows)} 프레임, "
          f"{fmt_int(len(traj))} 궤적, {time.time()-t_global:.1f}s 소요",
          file=sys.stderr)

    # 궤적 단위 파생량
    family_counts = Counter()
    natoms_by_family = defaultdict(list)
    nframes_by_family = defaultdict(list)
    element_atomcount_by_family = defaultdict(Counter)  # 원자 수 가중
    element_molcount_by_family = defaultdict(Counter)   # 분자 수 가중(있/없음)
    formula_by_family = defaultdict(Counter)
    ea_by_family = defaultdict(list)
    bad_ts_count = Counter()  # family -> #궤적 중 TS 검증 실패
    short_traj_count = Counter()  # family -> #궤적 중 프레임<3

    for stem, t in traj.items():
        fam = t["family"]
        family_counts[fam] += 1
        natoms_by_family[fam].append(t["natoms"])
        nframes_by_family[fam].append(t["n_frames"])

        nums = t["numbers"]
        c = Counter(nums)
        for z, k in c.items():
            sym = chemical_symbols[z]
            element_atomcount_by_family[fam][sym] += k
            element_molcount_by_family[fam][sym] += 1

        # 화학식(Hill 표기 비슷하게): symbol -> count 정렬해서 문자열로
        sym_counts = Counter(chemical_symbols[z] for z in nums)
        parts = []
        for s in sorted(sym_counts):
            k = sym_counts[s]
            parts.append(s + (str(k) if k > 1 else ""))
        formula = "".join(parts)
        formula_by_family[fam][formula] += 1

        # Ea = E_TS - E_R (내부 프레임 에너지 최대)
        idxs = np.asarray(t["frame_indices"])
        es = np.asarray(t["energies"])
        order = np.argsort(idxs)
        idxs = idxs[order]
        es = es[order]
        if len(es) < 3:
            short_traj_count[fam] += 1
            continue
        e_r = es[0]
        e_p = es[-1]
        interior = es[1:-1]
        ts_local = int(np.argmax(interior)) + 1
        e_ts = es[ts_local]
        if not (e_ts > e_r and e_ts > e_p):
            bad_ts_count[fam] += 1
            continue
        ea_by_family[fam].append(float(e_ts - e_r))

    # ---------- 리포트 ----------
    out = []
    P = out.append
    P("=" * 72)
    P("Halo8 데이터셋 전수 분석 리포트")
    P("=" * 72)
    P("")
    P(f"스캔 대상 디렉토리 : {DATA_DIR}")
    P(f"DB 파일 수         : {len(db_files)}")
    P(f"총 프레임 수       : {fmt_int(total_rows)}")
    P(f"총 궤적 수         : {fmt_int(len(traj))}")
    P("")
    P("[DB 파일별 프레임 수]")
    for f, n in db_rows.items():
        P(f"  {f:12s}  {fmt_int(n):>14}")
    P("")

    # 가족별 요약
    P("[반응 카테고리 (dand_id 접두사 기준, 대소문자 무시)]")
    P(f"  {'카테고리':<10} {'궤적 수':>12} {'프레임 수':>14} {'평균 프레임/궤적':>18}")
    for fam in ("T1x", "Halogen", "기타"):
        ntr = family_counts.get(fam, 0)
        if ntr == 0:
            continue
        nfr = sum(nframes_by_family[fam])
        avg = nfr / ntr if ntr else 0
        P(f"  {fam:<10} {fmt_int(ntr):>12} {fmt_int(nfr):>14} {avg:>18.1f}")
    P("")

    # 분자 크기
    P("[분자 크기 (궤적당 원자 수) — 카테고리별]")
    for fam in ("T1x", "Halogen", "기타"):
        if family_counts.get(fam, 0) == 0:
            continue
        arr = np.asarray(natoms_by_family[fam])
        P(f"  · {fam} (N={fmt_int(len(arr))} 궤적)")
        P(f"    최소 원자 수 = {arr.min()}")
        P(f"    최대 원자 수 = {arr.max()}")
        P(f"    평균/중앙값 = {arr.mean():.2f} / {int(np.median(arr))}")
        # 분포 (정수 bin)
        lo, hi = int(arr.min()), int(arr.max())
        bin_edges = np.arange(lo, hi + 2)
        counts, _ = np.histogram(arr, bins=bin_edges)
        # 빈도 상위 표시
        top = sorted(zip(bin_edges[:-1], counts), key=lambda x: -x[1])[:8]
        P(f"    원자 수별 상위 분포:")
        for k, c in top:
            pct = 100 * c / len(arr)
            P(f"       {k:>3} 원자 → {fmt_int(int(c)):>8} 궤적 ({pct:5.1f}%)")
    P("")

    # 가장 크고 가장 작은 분자 (예시 궤적 포함)
    P("[극단 사례 — 가장 큰/작은 분자를 가진 궤적]")
    for fam in ("T1x", "Halogen"):
        if family_counts.get(fam, 0) == 0:
            continue
        # 해당 가족 내 궤적 (stem, natoms) 정렬
        fam_trajs = [(s, t["natoms"]) for s, t in traj.items()
                     if t["family"] == fam]
        fam_trajs_sorted = sorted(fam_trajs, key=lambda x: x[1])
        P(f"  · {fam}")
        s, n = fam_trajs_sorted[0]
        P(f"    최소 원자 수 분자: {n} 원자 — 예시 궤적: {s}")
        s, n = fam_trajs_sorted[-1]
        P(f"    최대 원자 수 분자: {n} 원자 — 예시 궤적: {s}")
    P("")

    # 원소
    all_elements = set()
    for c in element_atomcount_by_family.values():
        all_elements.update(c.keys())
    P("[데이터셋에 등장하는 원소]")
    P(f"  전체 합집합 ({len(all_elements)}종): "
      f"{', '.join(sorted(all_elements, key=lambda s: chemical_symbols.index(s)))}")
    for fam in ("T1x", "Halogen"):
        if family_counts.get(fam, 0) == 0:
            continue
        elems = element_atomcount_by_family[fam]
        sorted_elems = sorted(elems.keys(), key=lambda s: chemical_symbols.index(s))
        P(f"  · {fam}: {', '.join(sorted_elems)}")
    P("")

    P("[원소별 통계 — 카테고리별]")
    P("  (원자 수 = 모든 궤적의 numbers 를 단순 합산; "
      "분자 수 = 해당 원소가 1개 이상 포함된 궤적의 개수)")
    for fam in ("T1x", "Halogen", "기타"):
        if family_counts.get(fam, 0) == 0:
            continue
        total_atoms_fam = sum(element_atomcount_by_family[fam].values())
        ntr = family_counts[fam]
        P(f"  · {fam}  (궤적 {fmt_int(ntr)}, 총 원자 {fmt_int(total_atoms_fam)})")
        P(f"    {'원소':<6}{'원자수':>14}{'원자비율':>12}"
          f"{'포함 궤적':>14}{'궤적 비율':>12}")
        for sym in sorted(element_atomcount_by_family[fam].keys(),
                          key=lambda s: chemical_symbols.index(s)):
            ac = element_atomcount_by_family[fam][sym]
            mc = element_molcount_by_family[fam][sym]
            P(f"    {sym:<6}{fmt_int(ac):>14}"
              f"{100*ac/max(total_atoms_fam,1):>11.2f}%"
              f"{fmt_int(mc):>14}{100*mc/max(ntr,1):>11.2f}%")
    P("")

    # 화학식
    P("[고유 화학식 (궤적당 1개) — 카테고리별]")
    for fam in ("T1x", "Halogen"):
        if family_counts.get(fam, 0) == 0:
            continue
        fdict = formula_by_family[fam]
        P(f"  · {fam}: 고유 화학식 {fmt_int(len(fdict))} 종")
        P(f"    가장 흔한 화학식 top 15:")
        for form, c in fdict.most_common(15):
            P(f"       {form:<20} {fmt_int(c):>8} 궤적")
    P("")

    # 프레임 길이 분포
    P("[궤적 길이 (프레임 수) — 카테고리별]")
    for fam in ("T1x", "Halogen"):
        if family_counts.get(fam, 0) == 0:
            continue
        arr = np.asarray(nframes_by_family[fam])
        P(f"  · {fam}: min={arr.min()}  median={int(np.median(arr))}"
          f"  mean={arr.mean():.1f}  max={arr.max()}  std={arr.std():.1f}")
        # 분포
        bins = [0, 50, 100, 150, 200, 250, 300, 400, 500, arr.max() + 1]
        bins = sorted(set(bins))
        counts, edges = np.histogram(arr, bins=bins)
        for c, lo, hi in zip(counts, edges[:-1], edges[1:]):
            pct = 100 * c / len(arr)
            bar = "█" * int(round(pct / 2))
            P(f"       [{lo:>4}, {hi:>4})  {fmt_int(int(c)):>8} 궤적  "
              f"({pct:5.1f}%)  {bar}")
    P("")

    # Ea 분포
    P("[활성화 에너지 Ea = E(TS) - E(R)  (단위: 데이터베이스 원본 단위, eV 가정)]")
    P("  TS = 내부 프레임 에너지 최댓값 (R=프레임0, P=마지막). "
      "E(TS) > E(R) 및 E(TS) > E(P) 인 궤적만 집계.")
    for fam in ("T1x", "Halogen"):
        if family_counts.get(fam, 0) == 0:
            continue
        arr = np.asarray(ea_by_family[fam])
        bad = bad_ts_count.get(fam, 0)
        short = short_traj_count.get(fam, 0)
        total = family_counts[fam]
        P(f"  · {fam}: 유효 {fmt_int(len(arr))} / 총 {fmt_int(total)} "
          f"(TS 검증 실패 {fmt_int(bad)}, 프레임<3 {fmt_int(short)})")
        if len(arr):
            for line in hist_lines(arr.tolist(),
                                   bins=[0, 0.5, 1, 1.5, 2, 3, 4, 6, 8, 12,
                                         float(arr.max()) + 0.01],
                                   label=f"{fam} Ea"):
                P(line)
    P("")

    # 결합 통계: 카테고리별 원자 수 vs Ea 상관
    P("[원자 수와 Ea 상관 — 카테고리별]")
    for fam in ("T1x", "Halogen"):
        if not ea_by_family[fam]:
            continue
        # 동일 궤적에서 natoms / ea
        pairs = []
        for stem, t in traj.items():
            if t["family"] != fam:
                continue
            if t["n_frames"] < 3:
                continue
            idxs = np.argsort(t["frame_indices"])
            es = np.asarray(t["energies"])[idxs]
            if len(es) < 3:
                continue
            e_r, e_p = es[0], es[-1]
            interior = es[1:-1]
            ts_local = int(np.argmax(interior)) + 1
            e_ts = es[ts_local]
            if not (e_ts > e_r and e_ts > e_p):
                continue
            pairs.append((t["natoms"], float(e_ts - e_r)))
        if len(pairs) >= 2:
            arr_n = np.asarray([p[0] for p in pairs])
            arr_e = np.asarray([p[1] for p in pairs])
            r = float(np.corrcoef(arr_n, arr_e)[0, 1])
            P(f"  · {fam}: Pearson r(natoms, Ea) = {r:+.3f}  "
              f"(n={len(pairs)})")
    P("")

    P("=" * 72)
    P("리포트 끝.")
    P("=" * 72)

    print("\n".join(out))
    return 0


if __name__ == "__main__":
    sys.exit(main())
