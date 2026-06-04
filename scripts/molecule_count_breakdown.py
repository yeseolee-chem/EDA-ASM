"""R / P 의 분자 수(연결 성분 수) 매트릭스 분석.

각 궤적의 R(프레임 0)과 P(마지막 프레임)에 대해 ASE 공유반경 × 1.1
cutoff 의 결합 그래프를 만들고 연결 성분 수를 세서 (n_R → n_P) 매트릭스
와 카테고리별 Ea 통계를 보고한다.
"""
from __future__ import annotations

import math
import sys
import time
from collections import Counter, defaultdict

import numpy as np

sys.path.insert(0, "scripts")
from categorize_reactions import bonds_of, n_components, scan  # noqa: E402


def fmt(n) -> str:
    return f"{n:,}"


def main() -> int:
    traj, _ = scan()

    # (n_R, n_P) -> family -> [Ea]
    cell_eas: dict[tuple[int, int], dict[str, list[float]]] = defaultdict(
        lambda: defaultdict(list)
    )
    # (n_R, n_P) -> family -> [natoms]  (분자 크기 영향 확인)
    cell_natoms: dict[tuple[int, int], dict[str, list[int]]] = defaultdict(
        lambda: defaultdict(list)
    )
    # (n_R, n_P) -> family -> [예시 traj id]
    cell_examples: dict[tuple[int, int], dict[str, list[str]]] = defaultdict(
        lambda: defaultdict(list)
    )

    t0 = time.time()
    bad = 0
    for k, (stem, t) in enumerate(traj.items()):
        natoms = t["natoms"]
        numbers = np.frombuffer(t["numbers"], dtype=np.int32)
        try:
            R_pos = np.frombuffer(t["R_pos"], dtype=np.float64).reshape(natoms, 3)
            P_pos = np.frombuffer(t["P_pos"], dtype=np.float64).reshape(natoms, 3)
            ncR = n_components(natoms, bonds_of(numbers, R_pos))
            ncP = n_components(natoms, bonds_of(numbers, P_pos))
        except Exception:
            bad += 1
            continue

        fam = t["family"]
        # Ea (TS 검증 통과한 경우만)
        Ea = math.nan
        if (
            t["E_max_idx"] != t["min_idx"]
            and t["E_max_idx"] != t["max_idx"]
            and t["E_max"] > t["E_R"]
            and t["E_max"] > t["E_P"]
        ):
            Ea = t["E_max"] - t["E_R"]

        cell_eas[(ncR, ncP)][fam].append(Ea)
        cell_natoms[(ncR, ncP)][fam].append(natoms)
        if len(cell_examples[(ncR, ncP)][fam]) < 3:
            cell_examples[(ncR, ncP)][fam].append(stem)

    print(
        f"[분석] 19,176 궤적 처리, 결합 평가 실패 {bad}, {time.time()-t0:.1f}s",
        file=sys.stderr,
    )

    fams = ("T1x", "Halogen")
    n_total = {fam: sum(1 for t in traj.values() if t["family"] == fam) for fam in fams}

    # ── 리포트 ──
    out: list[str] = []
    P = out.append
    P("=" * 78)
    P("Halo8 — R→P 분자 수 매트릭스 (연결 성분 수 기준)")
    P("=" * 78)
    P("")
    P("분자 수 정의 : ASE natural_cutoffs × 1.1 결합 그래프의 연결 성분 수.")
    P(f"전체 궤적   : T1x {fmt(n_total['T1x'])}, Halogen {fmt(n_total['Halogen'])}, "
      f"합계 {fmt(n_total['T1x'] + n_total['Halogen'])}")
    P(f"결합 평가 실패 : {bad}")
    P("")

    # 매트릭스 표시: 가족별
    for fam in fams:
        P("─" * 78)
        P(f"[{fam}] R 분자 수 × P 분자 수 매트릭스 (궤적 수)")
        P("─" * 78)
        # 해당 가족에서 등장하는 n_R, n_P 범위 파악
        keys = [(r, p) for (r, p), m in cell_eas.items()
                if m.get(fam) and len(m[fam]) > 0]
        if not keys:
            P("  (해당 가족 데이터 없음)")
            P("")
            continue
        max_r = max(k[0] for k in keys)
        max_p = max(k[1] for k in keys)

        # 헤더
        header = f"  {'R\\P':>6}"
        for p in range(1, max_p + 1):
            header += f"{p:>9}"
        header += f"{'합계':>10}"
        P(header)
        col_totals = [0] * (max_p + 1)
        grand = 0
        for r in range(1, max_r + 1):
            row = f"  {r:>6}"
            row_total = 0
            for p in range(1, max_p + 1):
                n = len(cell_eas.get((r, p), {}).get(fam, []))
                if n == 0:
                    row += f"{'·':>9}"
                else:
                    row += f"{n:>9,}"
                row_total += n
                col_totals[p] += n
            row += f"{row_total:>10,}"
            grand += row_total
            P(row)
        # 컬럼 합계 행
        col_row = f"  {'합계':>6}"
        for p in range(1, max_p + 1):
            col_row += f"{col_totals[p]:>9,}"
        col_row += f"{grand:>10,}"
        P(col_row)
        P("")

    # 합산 매트릭스 + 비율 + Ea
    P("─" * 78)
    P("[전체] R→P 분자 수 변화 카테고리별 요약")
    P("─" * 78)
    # 셀을 (T1x_n, Halogen_n) 기준으로 정렬해서 모든 등장 셀 표시
    all_cells = sorted(cell_eas.keys())
    P(f"  {'R→P':<8}{'T1x':>9}{'Halogen':>11}{'합계':>9}{'비율':>9}"
      f"   {'T1x Ea med':>12}{'Halogen Ea med':>16}")
    grand_t = sum(n_total.values())
    for cell in all_cells:
        nt = len(cell_eas[cell].get("T1x", []))
        nh = len(cell_eas[cell].get("Halogen", []))
        tot = nt + nh
        pct = 100 * tot / grand_t
        et = [v for v in cell_eas[cell].get("T1x", []) if not math.isnan(v)]
        eh = [v for v in cell_eas[cell].get("Halogen", []) if not math.isnan(v)]
        med_t = f"{np.median(et):.2f}" if et else "—"
        med_h = f"{np.median(eh):.2f}" if eh else "—"
        label = f"{cell[0]}→{cell[1]}"
        P(f"  {label:<8}{nt:>9,}{nh:>11,}{tot:>9,}{pct:>8.2f}%"
          f"   {med_t:>12}{med_h:>16}")
    P("")

    # 변화 방향별 합계
    P("─" * 78)
    P("[변화 방향별 요약]")
    P("─" * 78)
    direction_counts = {
        "분자 수 보존 (n_R == n_P)": 0,
        "단편화 (n_R < n_P)": 0,
        "결합 (n_R > n_P)": 0,
    }
    direction_by_fam = {fam: dict(direction_counts) for fam in fams}
    for cell, fam_map in cell_eas.items():
        r, p = cell
        if r == p:
            key = "분자 수 보존 (n_R == n_P)"
        elif r < p:
            key = "단편화 (n_R < n_P)"
        else:
            key = "결합 (n_R > n_P)"
        for fam in fams:
            n = len(fam_map.get(fam, []))
            direction_by_fam[fam][key] += n

    P(f"  {'방향':<28}{'T1x':>9}{'Halogen':>11}{'합계':>9}{'비율':>9}")
    for key in ["분자 수 보존 (n_R == n_P)", "단편화 (n_R < n_P)", "결합 (n_R > n_P)"]:
        nt = direction_by_fam["T1x"][key]
        nh = direction_by_fam["Halogen"][key]
        tot = nt + nh
        pct = 100 * tot / grand_t
        P(f"  {key:<28}{nt:>9,}{nh:>11,}{tot:>9,}{pct:>8.2f}%")
    P("")

    # 예시 trajectory id
    P("─" * 78)
    P("[셀별 예시 궤적 id (가족당 최대 3개)]")
    P("─" * 78)
    for cell in all_cells:
        r, p = cell
        for fam in fams:
            ex = cell_examples[cell].get(fam, [])
            if not ex:
                continue
            P(f"  {r}→{p}  {fam:<9}  {', '.join(ex)}")
    P("")

    P("=" * 78)
    P("리포트 끝.")
    P("=" * 78)
    print("\n".join(out))
    return 0


if __name__ == "__main__":
    sys.exit(main())
