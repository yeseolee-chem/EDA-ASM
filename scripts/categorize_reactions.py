"""Halo8 데이터셋 반응 유형 카테고리화 → 한국어 리포트.

각 궤적에 대해 R(프레임 0)와 P(마지막 프레임)의 좌표/원자번호를 한 번의
스캔으로 모은 뒤 ASE 의 natural cutoffs (× 1.1) 기반 이웃리스트로 결합
그래프를 만든다. R→P 사이의 결합 변화(끊김/생성), 단편 수 변화, 고리
수 변화로 반응을 분류한다.
"""
from __future__ import annotations

import glob
import math
import os
import re
import sqlite3
import sys
import time
from collections import Counter, defaultdict

import numpy as np
from ase.data import chemical_symbols, covalent_radii

DATA_DIR = "data/Halo8"
DAND_RE = re.compile(rb'"dand_id":"([^"]+)"')
HALOGENS = {"F", "Cl", "Br"}
COVALENT = np.asarray(covalent_radii)  # index = Z


# ─────────────────────────── 유틸 ───────────────────────────


def family_of(traj_id: str) -> str:
    low = traj_id.lower()
    if low.startswith("t1x"):
        return "T1x"
    if low.startswith("halo"):
        return "Halogen"
    return "기타"


def fmt(n) -> str:
    if isinstance(n, float):
        return f"{n:.3f}"
    return f"{n:,}"


def bonds_of(numbers: np.ndarray, positions: np.ndarray,
             mult: float = 1.1) -> set[tuple[int, int]]:
    """직접 거리 계산으로 공유결합 그래프 추출 — N²·작은 N 이라 매우 빠름."""
    r = COVALENT[numbers]  # (N,)
    # pairwise threshold matrix
    thr = (r[:, None] + r[None, :]) * mult
    # pairwise distances
    diff = positions[:, None, :] - positions[None, :, :]
    d2 = (diff * diff).sum(-1)
    # i<j 마스크
    n = len(numbers)
    iu = np.triu_indices(n, k=1)
    d_pair = np.sqrt(d2[iu])
    thr_pair = thr[iu]
    mask = d_pair < thr_pair
    ii = iu[0][mask]
    jj = iu[1][mask]
    return set(zip(ii.tolist(), jj.tolist()))


def n_components(natoms: int, bonds: set[tuple[int, int]]) -> int:
    parent = list(range(natoms))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for a, b in bonds:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb
    return len({find(i) for i in range(natoms)})


def n_rings(natoms: int, bonds: set[tuple[int, int]], ncomp: int) -> int:
    # 순환 인덱스: edges - nodes + components
    return len(bonds) - natoms + ncomp


def pair_label(z1: int, z2: int) -> str:
    s1, s2 = chemical_symbols[z1], chemical_symbols[z2]
    if s1 > s2:
        s1, s2 = s2, s1
    return f"{s1}-{s2}"


def bond_involves(numbers, i, j, predicate):
    return predicate(chemical_symbols[int(numbers[i])]) or predicate(
        chemical_symbols[int(numbers[j])]
    )


def classify(
    numbers, broken, formed, ncomp_r, ncomp_p, rings_r, rings_p
) -> tuple[str, str]:
    """1차/2차 카테고리 라벨 반환."""
    nb, nf = len(broken), len(formed)

    # 1차: 결합 변화 패턴
    if nb == 0 and nf == 0:
        primary = "A. 비반응 (결합 변화 없음)"
    elif nb >= 1 and nf == 0:
        primary = "B. 절단만 (β-scission/단편화)"
    elif nb == 0 and nf >= 1:
        primary = "C. 형성만 (결합/축합)"
    elif nb == 1 and nf == 1:
        b = broken[0]; f = formed[0]
        b_has_h = "H" in (chemical_symbols[numbers[b[0]]], chemical_symbols[numbers[b[1]]])
        f_has_h = "H" in (chemical_symbols[numbers[f[0]]], chemical_symbols[numbers[f[1]]])
        if b_has_h and f_has_h:
            primary = "D1. 1:1 교환 — 수소 이동(H-shift/tautomerization)"
        else:
            b_hal = any(chemical_symbols[numbers[k]] in HALOGENS for k in b)
            f_hal = any(chemical_symbols[numbers[k]] in HALOGENS for k in f)
            if b_hal or f_hal:
                primary = "D2. 1:1 교환 — 할로겐 이동"
            else:
                primary = "D3. 1:1 교환 — 중원자 재배열"
    elif nb == nf and nb >= 2:
        primary = f"E. 다중 1:1 교환 (n={nb})"
    else:
        primary = "F. 비대칭 결합 변화 (nb≠nf, 둘 다 ≥1)"

    # 2차: 단편 수 / 고리 수 변화
    if ncomp_p > ncomp_r:
        secondary = f"단편화 ({ncomp_r}→{ncomp_p})"
    elif ncomp_p < ncomp_r:
        secondary = f"결합 ({ncomp_r}→{ncomp_p})"
    else:
        if rings_p > rings_r:
            secondary = f"고리화 (rings {rings_r}→{rings_p})"
        elif rings_p < rings_r:
            secondary = f"개환 (rings {rings_r}→{rings_p})"
        else:
            secondary = "단편/고리 보존"

    return primary, secondary


# ─────────────────────────── 스캔 ───────────────────────────


def scan():
    db_files = sorted(glob.glob(os.path.join(DATA_DIR, "Halo_*.db")))
    traj = {}
    total_rows = 0
    t_global = time.time()
    for db in db_files:
        t0 = time.time()
        conn = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
        cur = conn.cursor()
        cur.execute(
            "SELECT natoms, energy, numbers, positions, data FROM systems"
        )
        for natoms, energy, numbers_blob, positions_blob, data_blob in cur:
            total_rows += 1
            m = DAND_RE.search(data_blob, 0, 256)
            if not m:
                continue
            did = m.group(1).decode("ascii", "ignore")
            try:
                stem, idx_str = did.rsplit("_", 1)
                fidx = int(idx_str)
            except ValueError:
                continue
            t = traj.get(stem)
            E = float(energy) if energy is not None else math.nan

            if t is None:
                t = {
                    "natoms": int(natoms),
                    "numbers": bytes(numbers_blob),
                    "family": family_of(stem),
                    "n_frames": 1,
                    "min_idx": fidx,
                    "max_idx": fidx,
                    "R_pos": bytes(positions_blob),
                    "P_pos": bytes(positions_blob),
                    "E_R": E,
                    "E_P": E,
                    "E_max": E,
                    "E_max_idx": fidx,
                }
                traj[stem] = t
                continue

            t["n_frames"] += 1
            if fidx < t["min_idx"]:
                t["min_idx"] = fidx
                t["R_pos"] = bytes(positions_blob)
                t["E_R"] = E
            if fidx > t["max_idx"]:
                t["max_idx"] = fidx
                t["P_pos"] = bytes(positions_blob)
                t["E_P"] = E
            if not math.isnan(E) and (math.isnan(t["E_max"]) or E > t["E_max"]):
                t["E_max"] = E
                t["E_max_idx"] = fidx
        conn.close()
        print(
            f"[scan] {os.path.basename(db):12s}  rows+={total_rows:>10,}  "
            f"trajs(누적)={len(traj):>7,}  {time.time()-t0:5.1f}s",
            file=sys.stderr,
            flush=True,
        )
    print(
        f"[scan] 총 {total_rows:,} 행, {len(traj):,} 궤적, "
        f"{time.time()-t_global:.1f}s",
        file=sys.stderr,
    )
    return traj, total_rows


# ─────────────────────────── 분류 ───────────────────────────


def analyze(traj):
    by_primary = defaultdict(lambda: defaultdict(list))   # primary -> family -> [Ea]
    by_secondary = defaultdict(lambda: defaultdict(list)) # secondary -> family -> [Ea]
    by_cross = defaultdict(lambda: defaultdict(int))      # (primary,secondary) -> family -> count
    broken_pairs_counter = defaultdict(Counter)           # family -> Counter
    formed_pairs_counter = defaultdict(Counter)
    halogen_change_count = defaultdict(lambda: defaultdict(list))  # fam -> {True/False} -> [Ea]
    ts_ok = defaultdict(int)
    ts_bad = defaultdict(int)
    bond_eval_bad = 0

    items = list(traj.items())
    n = len(items)
    t0 = time.time()
    for k, (stem, t) in enumerate(items):
        natoms = t["natoms"]
        numbers = np.frombuffer(t["numbers"], dtype=np.int32)
        try:
            R_pos = np.frombuffer(t["R_pos"], dtype=np.float64).reshape(natoms, 3)
            P_pos = np.frombuffer(t["P_pos"], dtype=np.float64).reshape(natoms, 3)
            bonds_R = bonds_of(numbers, R_pos)
            bonds_P = bonds_of(numbers, P_pos)
        except Exception:
            bond_eval_bad += 1
            continue

        broken = sorted(bonds_R - bonds_P)
        formed = sorted(bonds_P - bonds_R)
        ncR = n_components(natoms, bonds_R)
        ncP = n_components(natoms, bonds_P)
        rR = n_rings(natoms, bonds_R, ncR)
        rP = n_rings(natoms, bonds_P, ncP)

        primary, secondary = classify(numbers, broken, formed, ncR, ncP, rR, rP)
        fam = t["family"]

        # Ea
        Ea = math.nan
        if (
            t["E_max_idx"] != t["min_idx"]
            and t["E_max_idx"] != t["max_idx"]
            and t["E_max"] > t["E_R"]
            and t["E_max"] > t["E_P"]
        ):
            Ea = t["E_max"] - t["E_R"]
            ts_ok[fam] += 1
        else:
            ts_bad[fam] += 1

        by_primary[primary][fam].append(Ea)
        by_secondary[secondary][fam].append(Ea)
        by_cross[(primary, secondary)][fam] += 1

        for i, j in broken:
            broken_pairs_counter[fam][pair_label(int(numbers[i]), int(numbers[j]))] += 1
        for i, j in formed:
            formed_pairs_counter[fam][pair_label(int(numbers[i]), int(numbers[j]))] += 1

        hal_changed = any(
            chemical_symbols[int(numbers[i])] in HALOGENS
            or chemical_symbols[int(numbers[j])] in HALOGENS
            for i, j in (broken + formed)
        )
        halogen_change_count[fam][hal_changed].append(Ea)

        if k % 2000 == 0 and k > 0:
            rate = k / (time.time() - t0)
            print(
                f"[classify] {k:>6}/{n}  {rate:.0f}/s",
                file=sys.stderr,
                flush=True,
            )

    print(
        f"[classify] 완료. 결합 평가 실패 = {bond_eval_bad}, "
        f"소요 {time.time()-t0:.1f}s",
        file=sys.stderr,
    )
    return {
        "by_primary": by_primary,
        "by_secondary": by_secondary,
        "by_cross": by_cross,
        "broken_pairs": broken_pairs_counter,
        "formed_pairs": formed_pairs_counter,
        "halogen_change": halogen_change_count,
        "ts_ok": ts_ok,
        "ts_bad": ts_bad,
        "bond_eval_bad": bond_eval_bad,
    }


# ─────────────────────────── 리포트 ───────────────────────────


def stats_line(values):
    arr = np.asarray([v for v in values if not math.isnan(v)])
    if len(arr) == 0:
        return "  (Ea 유효치 없음)"
    return (
        f"n_Ea={len(arr):>6,}  Ea_median={np.median(arr):>5.2f}  "
        f"mean={arr.mean():>5.2f}  min={arr.min():>5.2f}  max={arr.max():>5.2f}"
    )


def report(traj, res):
    out = []
    P = out.append

    fams = ("T1x", "Halogen")
    n_total = {fam: sum(1 for t in traj.values() if t["family"] == fam) for fam in fams}

    P("=" * 78)
    P("Halo8 반응 카테고리화 리포트")
    P("=" * 78)
    P("")
    P(f"분류 대상 궤적 : T1x {n_total['T1x']:,}, Halogen {n_total['Halogen']:,}, "
      f"합계 {n_total['T1x']+n_total['Halogen']:,}")
    P(f"결합 그래프 평가 실패 : {res['bond_eval_bad']:,}")
    P("")
    P("결합 판정 기준 : ASE natural_cutoffs × 1.1 (모든 원소). R = 프레임 0, "
      "P = 마지막 프레임.")
    P("TS = 내부 프레임 중 에너지 최대값. Ea = E(TS) − E(R) [원본 단위, eV 가정].")
    P("")

    # ── 1차 카테고리
    P("─" * 78)
    P("[1차 카테고리 — 결합 변화 패턴]")
    P("─" * 78)
    primary_order = [
        "A. 비반응 (결합 변화 없음)",
        "B. 절단만 (β-scission/단편화)",
        "C. 형성만 (결합/축합)",
        "D1. 1:1 교환 — 수소 이동(H-shift/tautomerization)",
        "D2. 1:1 교환 — 할로겐 이동",
        "D3. 1:1 교환 — 중원자 재배열",
        "F. 비대칭 결합 변화 (nb≠nf, 둘 다 ≥1)",
    ]
    # E. 다중 1:1 교환은 동적이라 별도로 모음
    primary_keys = set(res["by_primary"].keys())
    multi_keys = sorted([k for k in primary_keys if k.startswith("E.")],
                        key=lambda s: int(re.search(r"n=(\d+)", s).group(1)))
    full_order = primary_order[:6] + multi_keys + primary_order[6:]
    full_order = [k for k in full_order if k in primary_keys] + \
                 [k for k in primary_keys if k not in full_order]

    P(f"{'카테고리':<54}{'T1x':>9}{'Halogen':>11}{'합계':>9}")
    grand_total = 0
    for prim in full_order:
        fam_map = res["by_primary"][prim]
        nt = len(fam_map.get("T1x", []))
        nh = len(fam_map.get("Halogen", []))
        tot = nt + nh
        grand_total += tot
        P(f"{prim:<54}{nt:>9,}{nh:>11,}{tot:>9,}")
    P(f"{'합계':<54}{n_total['T1x']:>9,}{n_total['Halogen']:>11,}"
      f"{n_total['T1x']+n_total['Halogen']:>9,}")
    P("")

    # 카테고리별 Ea
    P("[1차 카테고리별 Ea 분포]")
    for prim in full_order:
        fam_map = res["by_primary"][prim]
        P(f"  · {prim}")
        for fam in fams:
            vals = fam_map.get(fam, [])
            if not vals:
                continue
            P(f"      {fam:<9}  {stats_line(vals)}")
    P("")

    # ── 2차 카테고리
    P("─" * 78)
    P("[2차 카테고리 — 단편 / 고리 변화]")
    P("─" * 78)
    sec_keys = sorted(res["by_secondary"].keys())
    # 자연스러운 순서: 보존 → 고리화/개환 → 단편화 → 결합
    def sec_sort(k):
        if "보존" in k:
            return (0, k)
        if "고리화" in k:
            return (1, k)
        if "개환" in k:
            return (2, k)
        if "단편화" in k:
            return (3, k)
        if "결합" in k and "결합" == k.split(" ")[0]:
            return (4, k)
        return (5, k)
    sec_keys.sort(key=sec_sort)

    P(f"{'카테고리':<30}{'T1x':>9}{'Halogen':>11}{'합계':>9}   Ea(T1x med / Halogen med)")
    for sec in sec_keys:
        fam_map = res["by_secondary"][sec]
        nt = len(fam_map.get("T1x", []))
        nh = len(fam_map.get("Halogen", []))
        et = [v for v in fam_map.get("T1x", []) if not math.isnan(v)]
        eh = [v for v in fam_map.get("Halogen", []) if not math.isnan(v)]
        med_t = f"{np.median(et):>5.2f}" if et else "  —  "
        med_h = f"{np.median(eh):>5.2f}" if eh else "  —  "
        P(f"{sec:<30}{nt:>9,}{nh:>11,}{nt+nh:>9,}     {med_t} / {med_h}")
    P("")

    # ── 1차 × 2차 교차
    P("─" * 78)
    P("[1차 × 2차 교차표 (궤적 수 합계)]")
    P("─" * 78)
    P(f"{'1차':<54} × {'2차':<22}  T1x   Halogen")
    rows = sorted(res["by_cross"].items(),
                  key=lambda kv: -(kv[1].get("T1x", 0) + kv[1].get("Halogen", 0)))
    for (prim, sec), fam_map in rows[:30]:
        nt = fam_map.get("T1x", 0)
        nh = fam_map.get("Halogen", 0)
        if nt + nh < 5:
            continue
        P(f"{prim:<54} × {sec:<22}  {nt:>5,}  {nh:>7,}")
    P("")

    # ── 결합 종류
    P("─" * 78)
    P("[끊긴 결합 종류 분포 (R 에는 있고 P 에는 없는 결합 — 가족별)]")
    P("─" * 78)
    for fam in fams:
        c = res["broken_pairs"][fam]
        total = sum(c.values())
        P(f"  · {fam}  (총 {total:,} 결합 끊김 사건)")
        for pair, cnt in c.most_common(15):
            P(f"      {pair:<6}  {cnt:>10,}  ({100*cnt/max(total,1):>5.2f}%)")
    P("")
    P("─" * 78)
    P("[형성된 결합 종류 분포 (P 에는 있고 R 에는 없는 결합 — 가족별)]")
    P("─" * 78)
    for fam in fams:
        c = res["formed_pairs"][fam]
        total = sum(c.values())
        P(f"  · {fam}  (총 {total:,} 결합 형성 사건)")
        for pair, cnt in c.most_common(15):
            P(f"      {pair:<6}  {cnt:>10,}  ({100*cnt/max(total,1):>5.2f}%)")
    P("")

    # ── 할로겐 참여
    P("─" * 78)
    P("[할로겐(F/Cl/Br) 결합 변화 참여 여부 — Halogen 가족 내]")
    P("─" * 78)
    for fam in fams:
        m = res["halogen_change"][fam]
        for tag, label in ((True, "할로겐 결합 변화 있음"), (False, "없음")):
            vals = m.get(tag, [])
            n = len(vals)
            if n == 0:
                continue
            valid = [v for v in vals if not math.isnan(v)]
            med = f"{np.median(valid):>5.2f}" if valid else " — "
            P(f"  {fam:<9} {label:<22}  {n:>6,} 궤적   Ea median = {med}")
    P("")

    P("=" * 78)
    P("리포트 끝.")
    P("=" * 78)
    return "\n".join(out)


def main():
    traj, _ = scan()
    res = analyze(traj)
    print(report(traj, res))
    return 0


if __name__ == "__main__":
    sys.exit(main())
