"""d1_archive_scan.py — D1 작업 아카이브에서 기존 출력만 읽는다 (새 QM 없음). B2 결측 원인 · B5 GEDT · Phase C A_idx 대조의 원자료.

python d1_archive_scan.py STATUS_CSV BUNDLE_DIRS UBAI_JOBS_DIR OUT_DIR [N_PROC]
  STATUS_CSV      확정 d1_status.csv (job_id, class, harvested)
  BUNDLE_DIRS     묶음 tar 폴더들, ':' 로 이음 — 누리온 묶음 (D1_build/nurion) + UBAI relay 의 outbox ($B/d1_scratch/outbox).
                  harvested 열의 묶음 <h>.tar 를 이 폴더들에서 찾는다 (member jobs/<T>/archive.tgz)
  UBAI_JOBS_DIR   아직 거두지 않은 행 (harvested 가 비었거나 작업 폴더가 남아 있음): <dir>/<T>/archive.tgz, 없으면 <dir>/<T>/ 폴더
  거둔 행인데 묶음 tar 를 어느 폴더에서도 못 찾으면 그 행은 읽지 못한 행 (parse_error, no_ts 는 path_verdict 'unread').
  읽지 못한 행이 하나라도 있으면 종료 코드 3 (B2 · B5 를 읽기 전에 고친다); 'no_path' 는 '아카이브는 읽었고 경로 파일이 없음' 만 뜻한다.
출력: OUT_DIR/archive_scan.csv (행 = class ok 또는 no_ts 인 작업), 중간 결과 OUT_DIR/archive_scan_parts/<원천>.<SCAN_VERSION>.d<DFORM_MIN_A>.csv
  (멱등: 같은 판 · 같은 기준 · 같은 작업 목록이면 건너뜀)
ok 행 (sp/):
  a_idx_meta   sp/input_meta.json 의 A_idx (라벨 파이프라인의 조각 분할)
  a_idx_tag    sp/eda.inp 좌표에서 (1) 이 붙은 원자 (EDA 의 조각 1 = role1 = dipole), 0-based
  mull_frag0/1, loew_frag0/1   sp/eda.out 의 "MULLIKEN / LOEWDIN FRAGMENT CHARGES" (복합체 한 블록씩; Fragment 0 = (1))
  mull_tag1_sum, loew_tag1_sum 같은 출력의 원자 전하를 (1) 원자에 대해 더한 값 — Fragment 0 대응의 자체 검사
  GEDT 는 b5_gedt.py 가 만든다: gedt = frag0 − charge1 (양수 = dipole → dipolarophile 전자 흐름 = FEDF, 음수 = REDF)
no_ts 행 (ts/ 또는 ts.tar 안):
  *_hl_ad_<i-j>_<k-l>_path.xyz · *_ll_ad_…_path.xyz 의 각 frame 에너지(주석 줄 "E = …" Eh)와 좌표. 파일 이름의 ad_ 뒤 원자 쌍 =
  형성 결합 (autodE, 0-based: 파일럿 T02894 의 8-13 · 9-10 은 frame 0 에서 1.51 · 1.52 Å, 마지막 frame 에서 3.02 · 3.04 Å). 경로마다
    고개      내부 frame 이 엄격한 극대 (E[i-1] < E[i] > E[i+1]; autodE path.is_saddle 과 같은 검사, 크기와 무관)
    d_react   반응물 쪽 끝 frame (양 끝 중 형성 결합 거리의 최댓값이 큰 쪽) 의 형성 결합 두 개 중 큰 거리 [Å]
    d_start   다른 끝 (생성물 쪽) 의 같은 값; ratio = d_react / d_start
    판정      고개 있음 → peak; 고개 없음 + d_react ≥ DFORM_MIN_A (3.0 Å) → barrierless (반응물까지 에너지가 한쪽으로 계속 내려감);
              고개 없음 + d_react < 3.0 Å → no_peak_short_scan (barrierless 로 세지 않는다)
    v4 메모 (2026-10-14 검토 · 사용자 결정 "v3 유지 + 민감도"): autodE (1.4.5) 는 고개 없는 경로를 늘 BreakingBond 의 끝 거리
              2 × (생성물 결합) 에서 끝낸다 (표본 68 경로 모두 ratio 2.000). 그래서 d_react < 3.0 Å 는 '경로가 잘림' 이 아니라
              '생성물 결합 < 1.5 Å 이라 autodE 의 탐색 범위가 3.0 Å 에 못 미침' (예: isoxazole 1.42 Å → 2.84 Å) 을 뜻한다.
              주 판정은 v3 그대로, missing_cause.py 가 '극대 없음 전체' (barrierless + no_peak_short_scan) 를 민감도로 함께 낸다.
  path_verdict (작업 하나): hl 경로가 있으면 hl 로, 없으면 ll 로. 하나라도 peak → peak; 아니고 읽지 못한 경로가 있으면 unparsed;
    아니고 하나라도 no_peak_short_scan → no_peak_short_scan; 모두 barrierless → barrierless; 경로 파일이 없으면 no_path
  dform_react_min_A · dform_start_min_A · dform_ratio_min: 판정에 쓴 경로들의 최솟값;
  path_detail: 경로마다 "hl|8-13_9-10|peak=0|d_react=3.037|d_start=1.518|ratio=2.0000|end=last|n=17"
  peak_kcal = (내부 frame 최대 − 양 끝 중 큰 값) × 627.509 (참고: 고개가 끝점보다 낮을 수 있다)
  log_no_peak, log_barrierless: ts autode.log 의 "Adaptive path had no peak" · "barrierless bimolecular" 줄 수 (참고; autodE 는 TS 가 없으면 늘 후자를 쓴다)
"""
from __future__ import annotations

import io
import json
import os
import re
import sys
import tarfile
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

EH2KCAL = 627.5094740631
SCAN_VERSION = "v4"           # v4 (user 2026-10-14): v3 rule kept; category no_peak_short_scan (was no_peak_truncated) + d_start, ratio
DFORM_MIN_A = 3.0             # user rule 2026-10-14: barrierless only if the reactant-end longer forming bond >= 3.0 A
COLS = ["job_id", "class", "source", "parse_error", "n_atoms", "a_idx_meta", "a_idx_tag",
        "mull_frag0", "mull_frag1", "loew_frag0", "loew_frag1", "mull_tag1_sum", "loew_tag1_sum",
        "n_hl_paths", "n_ll_paths", "hl_peak_max_kcal", "ll_peak_max_kcal", "hl_frames", "log_no_peak",
        "log_barrierless", "dform_react_min_A", "dform_start_min_A", "dform_ratio_min", "path_detail", "path_verdict"]
PATH_RE = re.compile(r"_(hl|ll)_ad_[^/]*_path\.xyz$")
BOND_RE = re.compile(r"_ad_(.*?)(?:_br_.*)?_path\.xyz$")
E_RE = re.compile(r"E\s*=\s*(-?\d+\.\d+)")
ATOM_RE = re.compile(r"^\s*([A-Z][a-z]?)\((\d+)\)\s+-?\d")


# ---------------------------------------------------------------- parsers
def parse_inp_tags(text: str):
    """(n_atoms, sorted 0-based indices tagged (1)) from the '* xyz' block of eda.inp."""
    lines = text.splitlines()
    k = next(i for i, l in enumerate(lines) if l.strip().lower().startswith("* xyz"))
    tags = []
    for l in lines[k + 1:]:
        if l.strip() == "*":
            break
        m = ATOM_RE.match(l)
        if not m:
            raise ValueError(f"eda.inp atom line not understood: {l!r}")
        tags.append(int(m.group(2)))
    return len(tags), [i for i, t in enumerate(tags) if t == 1]


def _block(lines, title, n):
    k = [i for i, l in enumerate(lines) if l.strip() == title]
    if len(k) != 1:
        raise ValueError(f"{title}: {len(k)} blocks")
    return lines[k[0] + 2:k[0] + 2 + n]


def parse_eda_charges(text: str, n_atoms: int):
    """Mulliken / Löwdin atomic charges (n) and fragment charges (2) of the complex in eda.out."""
    lines = text.splitlines()
    out = {}
    for tag, name in (("mull", "MULLIKEN"), ("loew", "LOEWDIN")):
        at = _block(lines, f"{name} ATOMIC CHARGES", n_atoms)
        q = np.array([float(l.split(":")[1]) for l in at])
        idx = [int(l.split()[0]) for l in at]
        if idx != list(range(n_atoms)):
            raise ValueError(f"{name} atomic charges: atom indices {idx[:3]}…")
        fr = _block(lines, f"{name} FRAGMENT CHARGES", 2)
        f = {}
        for l in fr:
            m = re.match(r"\s*Fragment\s+(\d+)\s*:\s*(-?\d+\.\d+)", l)
            if not m:
                raise ValueError(f"{name} fragment line {l!r}")
            f[int(m.group(1))] = float(m.group(2))
        if sorted(f) != [0, 1]:
            raise ValueError(f"{name} fragments {sorted(f)}")
        out[tag] = (q, f[0], f[1])
    return out


def path_bonds(name: str):
    """Forming bonds (0-based atom pairs) from an autodE path file name '..._ad_8-13_9-10_path.xyz'; [] if not parsed."""
    m = BOND_RE.search(name.split("/")[-1])
    return [(int(a), int(b)) for a, b in re.findall(r"(\d+)-(\d+)", m.group(1))] if m else []


def path_info(name: str, text: str) -> dict:
    """One autodE path xyz: frame energies + the forming-bond distances of the two end frames.
    has_peak = an interior frame is a strict local maximum (E[i-1] < E[i] > E[i+1]; autodE path.is_saddle uses the same
    test); peak_kcal = interior max − the larger endpoint (information: a peak can sit below an endpoint);
    d_react = the longer forming bond at the reactant-side end (the end whose longest forming bond is longer);
    d_start = the longer forming bond at the other (product-side) end; ratio = d_react / d_start (autodE ends a no-peak
    path at its BreakingBond final distance 2 × the product bond, so ratio ≈ 2.000 = autodE scanned its full range);
    verdict = peak | barrierless (no peak, d_react >= DFORM_MIN_A) | no_peak_short_scan (no peak, d_react < DFORM_MIN_A:
    autodE's own end point 2 × r(product) lies below 3.0 Å, i.e. a product bond < 1.5 Å) | unparsed."""
    bonds = path_bonds(name)
    lines = text.splitlines()
    E, X, i = [], [], 0
    while i < len(lines):
        if not lines[i].strip():
            i += 1; continue
        n = int(lines[i].split()[0])
        m = E_RE.search(lines[i + 1])
        E.append(float(m.group(1)) if m else np.nan)
        X.append(np.array([[float(v) for v in l.split()[1:4]] for l in lines[i + 2:i + 2 + n]]))
        i += n + 2
    E = np.array(E)
    out = dict(n=len(E), has_peak=None, peak_kcal=np.nan, d_react=np.nan, d_start=np.nan, ratio=np.nan, end="",
               bonds="_".join(f"{a}-{b}" for a, b in bonds), verdict="unparsed")
    if len(E) < 3 or not np.isfinite(E).all() or len(bonds) < 2 or any(max(p) >= len(X[0]) for p in bonds):
        return out
    dmax = [max(float(np.linalg.norm(x[a] - x[b])) for a, b in bonds) for x in (X[0], X[-1])]
    k = int(np.argmax(dmax))
    has = bool(np.any((E[1:-1] > E[:-2]) & (E[1:-1] > E[2:])))
    out.update(has_peak=has, peak_kcal=float((E[1:-1].max() - max(E[0], E[-1])) * EH2KCAL), d_react=dmax[k],
               d_start=dmax[1 - k], ratio=dmax[k] / dmax[1 - k] if dmax[1 - k] > 0 else np.nan, end=("first", "last")[k],
               verdict="peak" if has else ("barrierless" if dmax[k] >= DFORM_MIN_A else "no_peak_short_scan"))
    return out


# ---------------------------------------------------------------- one job
SP_NEED = {"sp/input_meta.json", "sp/eda.inp", "sp/eda.out"}


def _norm(name: str) -> str:
    return name[2:] if name.startswith("./") else name


def _ts_member(nm: str) -> bool:
    """TS-search path files and the TS autodE log (not the reference / stereo searches, which have their own logs)."""
    return (bool(PATH_RE.search(nm)) or nm.split("/")[-1] == "autode.log") and not nm.startswith(("ref/", "stereo/"))


def read_archive(fobj, cls, got):
    """Stream one archive.tgz / ts.tar (file object); a nested ts.tar is opened in place. ok: stop once SP_NEED is read."""
    with tarfile.open(fileobj=fobj, mode="r|*") as t:
        for m in t:
            if not m.isfile():
                continue
            nm = _norm(m.name)
            if cls == "ok":
                if nm in SP_NEED:
                    got[nm] = t.extractfile(m).read().decode(errors="replace")
                    if SP_NEED <= set(got):
                        return
            elif nm.endswith("ts.tar"):
                read_archive(t.extractfile(m), cls, got)
            elif _ts_member(nm):
                got.setdefault("ts", []).append((nm, t.extractfile(m).read().decode(errors="replace")))


def read_dir(d: Path, cls, got):
    """An unpacked job folder (UBAI job not packed into archive.tgz)."""
    if cls == "ok":
        for nm in SP_NEED:
            if (d / nm).is_file():
                got[nm] = (d / nm).read_text(errors="replace")
        return
    if (d / "ts").is_dir():
        for p in (d / "ts").rglob("*"):
            nm = str(p.relative_to(d))
            if p.is_file() and _ts_member(nm):
                got.setdefault("ts", []).append((nm, p.read_text(errors="replace")))
    elif (d / "ts.tar").is_file():
        with open(d / "ts.tar", "rb") as f:
            read_archive(f, cls, got)


def summarize(jid, cls, src, got) -> dict:
    r = {c: np.nan for c in COLS}
    r.update(job_id=jid, **{"class": cls}, source=src, parse_error="")
    try:
        if cls == "ok":
            miss = sorted({"sp/input_meta.json", "sp/eda.inp", "sp/eda.out"} - set(got))
            if miss:
                raise ValueError(f"missing {miss}")
            meta = json.loads(got["sp/input_meta.json"])
            n, tag = parse_inp_tags(got["sp/eda.inp"])
            ch = parse_eda_charges(got["sp/eda.out"], n)
            r.update(n_atoms=n, a_idx_meta=str(meta.get("A_idx", "")), a_idx_tag=" ".join(map(str, tag)))
            for k in ("mull", "loew"):
                q, f0, f1 = ch[k]
                r[f"{k}_frag0"], r[f"{k}_frag1"], r[f"{k}_tag1_sum"] = f0, f1, float(q[tag].sum())
        else:
            ts = got.get("ts", [])
            info = {"hl": [], "ll": []}
            for nm, txt in sorted(ts):
                m = PATH_RE.search(nm)
                if m:
                    try:
                        info[m.group(1)].append(path_info(nm, txt))
                    except (ValueError, IndexError):
                        info[m.group(1)].append(dict(n=0, peak_kcal=np.nan, d_react=np.nan, d_start=np.nan, ratio=np.nan, end="",
                                                     bonds="", has_peak=None, verdict="unparsed"))
                elif nm.split("/")[-1] == "autode.log":
                    r["log_no_peak"] = (0 if np.isnan(r["log_no_peak"]) else r["log_no_peak"]) + txt.count("Adaptive path had no peak")
                    r["log_barrierless"] = (0 if np.isnan(r["log_barrierless"]) else r["log_barrierless"]) + txt.count("barrierless bimolecular")
            r["n_hl_paths"], r["n_ll_paths"] = len(info["hl"]), len(info["ll"])
            for lv in ("hl", "ll"):
                pk = [p["peak_kcal"] for p in info[lv] if np.isfinite(p["peak_kcal"])]
                r[f"{lv}_peak_max_kcal"] = max(pk) if pk else np.nan
            r["hl_frames"] = " ".join(str(p["n"]) for p in info["hl"])
            r["path_detail"] = "; ".join(f"{lv}|{p['bonds']}|peak={int(bool(p['has_peak'])) if p['has_peak'] is not None else '?'}"
                                         f"|d_react={p['d_react']:.3f}|d_start={p['d_start']:.3f}|ratio={p['ratio']:.4f}"
                                         f"|end={p['end']}|n={p['n']}"
                                         for lv in ("hl", "ll") for p in info[lv])
            use = info["hl"] or info["ll"]                    # the decision uses hl paths when present, else ll
            v = [p["verdict"] for p in use]
            d = [p["d_react"] for p in use if np.isfinite(p["d_react"])]
            s0 = [p["d_start"] for p in use if np.isfinite(p["d_start"])]
            q = [p["ratio"] for p in use if np.isfinite(p["ratio"])]
            r["dform_react_min_A"] = min(d) if d else np.nan
            r["dform_start_min_A"] = min(s0) if s0 else np.nan
            r["dform_ratio_min"] = min(q) if q else np.nan
            if not use:
                r["path_verdict"] = "no_path"
            elif "peak" in v:
                r["path_verdict"] = "peak"
            elif "unparsed" in v:
                r["path_verdict"] = "unparsed"
            elif "no_peak_short_scan" in v:
                r["path_verdict"] = "no_peak_short_scan"
            else:
                r["path_verdict"] = "barrierless"
    except Exception as e:                                               # noqa: BLE001
        r["parse_error"] = f"{type(e).__name__}: {str(e)[:120]}"
    return r


# ---------------------------------------------------------------- sources
PART_TAG = f"{SCAN_VERSION}.d{DFORM_MIN_A:g}"   # part caches are keyed by the rule version and its threshold


def _unread(jid, c, src, why):
    """A job whose archive could not be read: parse_error says why; a no_ts row gets path_verdict 'unread' (never
    'no_path', which means 'archive read, no path file in it')."""
    r = summarize(jid, c, src, {})
    r["parse_error"] = why
    if c == "no_ts":
        r["path_verdict"] = "unread"
    return r


def _read_one(jid, c, src, reader):
    """summarize() of one job; an unreadable archive (truncated gzip / tar) becomes an 'unread' row."""
    got = {}
    try:
        reader(got)
    except Exception as e:                                               # noqa: BLE001
        return _unread(jid, c, src, f"archive {type(e).__name__}: {str(e)[:100]}")
    return summarize(jid, c, src, got)


def scan_source(args):
    src, tar, jobs, out_dir, ubai = args     # jobs: list of (job_id, class); tar None = UBAI job folders
    part = out_dir / "archive_scan_parts" / f"{src}.{PART_TAG}.csv"
    if part.is_file():
        old = pd.read_csv(part, dtype=str, keep_default_na=False)
        if sorted(zip(old.job_id, old["class"])) == sorted(jobs):
            return src, len(old), "cached"
    cls = dict(jobs)
    rows = []
    if tar is not None:
        found = set()
        with tarfile.open(tar, mode="r:") as T:
            for m in T:
                mm = re.match(r"(?:\./)?jobs/([^/]+)/archive\.tgz$", m.name)
                if not mm or mm.group(1) not in cls:
                    continue
                jid = mm.group(1)
                rows.append(_read_one(jid, cls[jid], src, lambda g: read_archive(T.extractfile(m), cls[jid], g)))
                found.add(jid)
        for jid in sorted(set(cls) - found):
            rows.append(_unread(jid, cls[jid], src, "not in bundle tar"))
    else:
        for jid, c in jobs:
            d = Path(ubai) / jid
            if (d / "archive.tgz").is_file():
                def rd(g, d=d, c=c):
                    with open(d / "archive.tgz", "rb") as f:
                        read_archive(f, c, g)
                rows.append(_read_one(jid, c, "ubai:archive", rd))
            elif d.is_dir():
                rows.append(_read_one(jid, c, "ubai:dir", lambda g, d=d, c=c: read_dir(d, c, g)))
            else:
                rows.append(_unread(jid, c, "ubai", "no job folder"))
    df = pd.DataFrame(rows, columns=COLS)
    part.parent.mkdir(parents=True, exist_ok=True)
    tmp = part.with_name(part.name + ".tmp"); df.to_csv(tmp, index=False); os.replace(tmp, part)
    return src, len(df), "scanned"


def main():
    st, ubai, out = sys.argv[1], Path(sys.argv[3]), Path(sys.argv[4])
    bundle_dirs = [Path(x) for x in sys.argv[2].split(":") if x]          # Nurion bundles + the UBAI relay outbox
    nproc = int(sys.argv[5]) if len(sys.argv) > 5 else int(os.environ.get("SLURM_CPUS_PER_TASK", "4"))
    out.mkdir(parents=True, exist_ok=True)
    S = pd.read_csv(st, dtype=str, keep_default_na=False)
    S = S[S["class"].isin(["ok", "no_ts"])]

    def find_tar(h):
        return next((d / f"{h}.tar" for d in bundle_dirs if (d / f"{h}.tar").is_file()), None)

    groups: dict[str, list] = {}
    tar_of: dict[str, Path] = {}
    missing = []
    for jid, c, h in zip(S.job_id, S["class"], S.harvested):
        t = find_tar(h) if h.startswith("d1_") else None
        if t is not None:
            groups.setdefault(h, []).append((jid, c)); tar_of[h] = t
        elif not h or (ubai / jid).is_dir():          # not harvested yet (or folder still there): read the job folder
            groups.setdefault("ubai", []).append((jid, c))
        else:                                         # harvested into a bundle that is in none of the bundle dirs
            missing.append(_unread(jid, c, f"bundle:{h}", f"bundle tar not found: {h}.tar in {':'.join(map(str, bundle_dirs))}"))
    tasks = [(b, tar_of.get(b), js, out, str(ubai)) for b, js in sorted(groups.items(), key=lambda kv: -len(kv[1]))]
    if tasks:
        with Pool(min(nproc, len(tasks))) as p:
            for src, n, how in p.imap_unordered(scan_source, tasks):
                print(f"{src}: {n} rows ({how})", flush=True)
    parts = [pd.read_csv(out / "archive_scan_parts" / f"{b}.{PART_TAG}.csv",
                         dtype={"a_idx_meta": str, "a_idx_tag": str, "hl_frames": str, "path_detail": str}) for b in groups]
    if missing:
        parts.append(pd.DataFrame(missing, columns=COLS))
    df = pd.concat(parts, ignore_index=True).sort_values("job_id")
    tmp = out / "archive_scan.csv.tmp"; df.to_csv(tmp, index=False); os.replace(tmp, out / "archive_scan.csv")
    ok = df[df["class"] == "ok"]
    chk = (ok.mull_frag0 - ok.mull_tag1_sum).abs().max(), (ok.loew_frag0 - ok.loew_tag1_sum).abs().max()
    n_err = int((df.parse_error.fillna("") != "").sum())
    print(json.dumps(dict(n=len(df), by_class=df["class"].value_counts().to_dict(), parse_errors=n_err,
                          bundle_dirs=list(map(str, bundle_dirs)), n_bundle_not_found=len(missing),
                          a_idx_meta_eq_tag=int((ok.a_idx_meta == ok.a_idx_tag).sum()), n_ok=len(ok),
                          max_frag0_minus_tag1_sum=dict(mulliken=float(chk[0]), loewdin=float(chk[1])),
                          no_ts_verdict=df.loc[df["class"] == "no_ts", "path_verdict"].value_counts().to_dict()), indent=1))
    if n_err:
        print(f"STOP: {n_err} rows unread / unparsed (parse_error) — see archive_scan.csv; B2 · B5 must not be read until fixed")
        sys.exit(3)


if __name__ == "__main__":
    main()
