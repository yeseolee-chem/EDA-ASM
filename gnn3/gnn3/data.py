"""Reading the GNN3 inputs: atom tokens, the 182 global features, pair labels, cross-validation folds, standardisation.

Vocabulary
----------
reaction   one D1 cycloaddition (dipole + dipolarophile), identified by an integer ``rxn_id``.
pair       two reactions of the same family: a substituted reaction ``a`` and its reference ``b`` (the H reference, or an
           anchor). Its label is ΔΔ = label(a) - label(b) in kcal/mol, for each of the 4 targets.
token      one atom of the transition state (TS) as a 29-vector: element one-hot (9) | binary flags (6) | numeric
           features (14). A reaction is a padded block of tokens plus a 182-long global feature vector.
fold       the T1 split: 5 outer folds whose test pairs come from held-out cores (core = dipole x skeleton), each with
           one inner train / validation split (the first GroupKFold(4) fold, stored in the split file).

Everything here re-implements ``analysis/deploy/data_lf.py`` (DeployData('lf')) + ``analysis/ch3/data_ch3.py`` (Ch3Data)
with the same order of rows and the same floating-point operations, so that the arrays are bit-identical to the
original ones (equivalence check E-1).
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from . import constants as K
from .channels import labels_to_channels


def file_sha256(path: Path | str) -> str:
    """sha256 hex digest of a file (read in 1 MiB blocks)."""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def list_sha256(ids: list[str]) -> str:
    """sha256 of an ordered id list, one id per line (= d1pair_common.list_sha of one-element rows)."""
    return hashlib.sha256(("\n".join(str(q) for q in ids) + "\n").encode()).hexdigest()


class DataError(RuntimeError):
    """An input file is missing, inconsistent with its report, or fails a consistency check."""


# =========================================================================== atom tokens
@dataclass(frozen=True)
class TokenLayout:
    """Column layout of an atom token and of the global feature vector (stored with every saved model)."""

    elements: tuple[int, ...]
    binary_columns: tuple[str, ...]
    numeric_columns: tuple[str, ...]
    global_columns: tuple[str, ...]
    token_version: str

    @property
    def token_width(self) -> int:
        return len(self.elements) + len(self.binary_columns) + len(self.numeric_columns)

    @classmethod
    def from_saved(cls, layout: dict) -> "TokenLayout":
        """Rebuild from the ``layout`` dict of a saved model file (keys of the original pipeline)."""
        return cls(tuple(int(z) for z in layout["elements"]), tuple(layout["bin_cols"]), tuple(layout["num_cols"]),
                   tuple(layout["glob_cols"]), str(layout["token_version"]))

    def as_saved(self) -> dict:
        return dict(elements=list(self.elements), bin_cols=list(self.binary_columns), num_cols=list(self.numeric_columns),
                    glob_cols=list(self.global_columns), token_version=self.token_version, d_in=self.token_width,
                    d_glob=len(self.global_columns))


def fill_token_arrays(token_paths: list[Path], layout: TokenLayout, max_atoms: int | None = None,
                      expected_sha256: list[str] | None = None) -> dict[str, np.ndarray]:
    """Read token files into zero-padded arrays (one row per reaction, rows in the order of ``token_paths``).

    Parameters
    ----------
    token_paths : list of Path
        ``<rxn_id>.npz`` files with arrays Z [n], bin [n, 6], num [n, 14], pos [n, 3] (angstrom), edges [e, 2],
        etype [e], glob [182].
    layout : TokenLayout
        Element order of the one-hot block and the column counts.
    max_atoms : int, optional
        Padding length; default = the largest reaction among the files.
    expected_sha256 : list of str, optional
        Per-file sha256 that must match (the token manifest), same order as ``token_paths``.

    Returns
    -------
    dict of numpy arrays
        element_onehot [r, N, 9], binary_flags [r, N, 6], numeric_features [r, N, 14], positions_angstrom [r, N, 3]
        (float64), atom_mask [r, N] (bool), edge_types [r, N, N] (int8; 0 none, 1 covalent, 2 forming),
        fragment_ids [r, N] (int8; 0 dipole, 1 dipolarophile), global_features [r, 182] (float64), max_atoms (int).

    Raises
    ------
    DataError
        A file differs from its expected sha256.
    """
    if max_atoms is None:
        sizes = []
        for path in token_paths:
            with np.load(path) as token:
                sizes.append(len(token["Z"]))
        max_atoms = int(max(sizes)) if sizes else 0
    n_reactions = len(token_paths)
    column_of_element = {z: k for k, z in enumerate(layout.elements)}
    element_onehot = np.zeros((n_reactions, max_atoms, len(layout.elements)))
    binary_flags = np.zeros((n_reactions, max_atoms, len(layout.binary_columns)))
    numeric_features = np.zeros((n_reactions, max_atoms, len(layout.numeric_columns)))
    positions = np.zeros((n_reactions, max_atoms, 3))
    atom_mask = np.zeros((n_reactions, max_atoms), dtype=bool)
    edge_types = np.zeros((n_reactions, max_atoms, max_atoms), dtype=np.int8)
    fragment_ids = np.zeros((n_reactions, max_atoms), dtype=np.int8)
    global_features = None
    for k, path in enumerate(token_paths):
        if expected_sha256 is not None and file_sha256(path) != expected_sha256[k]:
            raise DataError(f"{path}: sha256 differs from the token manifest")
        token = np.load(path)
        n_atoms = len(token["Z"])
        element_onehot[k, np.arange(n_atoms), [column_of_element[int(z)] for z in token["Z"]]] = 1.0
        binary_flags[k, :n_atoms] = token["bin"]
        numeric_features[k, :n_atoms] = token["num"]
        positions[k, :n_atoms] = token["pos"]
        atom_mask[k, :n_atoms] = True
        fragment_ids[k, :n_atoms] = token["bin"][:, 0]           # frag_B flag = fragment id
        for (i, j), edge_type in zip(token["edges"], token["etype"]):
            edge_types[k, i, j] = edge_types[k, j, i] = int(edge_type)
        if global_features is None:
            global_features = np.zeros((n_reactions, len(token["glob"])))
        global_features[k] = token["glob"]
    if global_features is None:
        global_features = np.zeros((n_reactions, 0))
    return dict(element_onehot=element_onehot, binary_flags=binary_flags, numeric_features=numeric_features,
                positions_angstrom=positions, atom_mask=atom_mask, edge_types=edge_types, fragment_ids=fragment_ids,
                global_features=global_features, max_atoms=max_atoms)


class ReactionTokens:
    """All token reactions of an input set as padded numpy arrays (row = reaction).

    Attributes
    ----------
    reaction_ids : numpy.ndarray [r] int
    row_of : dict rxn_id -> row
    n_atoms : numpy.ndarray [r] int
    element_onehot, binary_flags, numeric_features, positions_angstrom : numpy.ndarray [r, N, *] float64
    atom_mask : [r, N] bool;  edge_types : [r, N, N] int8;  fragment_ids : [r, N] int8
    global_features : numpy.ndarray [r, 182] float64 (the copy stored in each token file)
    layout : TokenLayout
    """

    ARRAYS = ("element_onehot", "binary_flags", "numeric_features", "positions_angstrom", "atom_mask", "edge_types",
              "fragment_ids", "global_features")

    def __init__(self, reaction_ids, n_atoms, arrays: dict, layout: TokenLayout):
        self.reaction_ids = np.asarray(reaction_ids, dtype=int)
        self.row_of = {int(r): k for k, r in enumerate(self.reaction_ids)}
        self.n_atoms = np.asarray(n_atoms, dtype=int)
        for name in self.ARRAYS:
            setattr(self, name, arrays[name])
        self.max_atoms = int(arrays["max_atoms"])
        self.layout = layout
        self._static: dict = {}

    @property
    def token_width(self) -> int:
        return self.layout.token_width

    @property
    def n_global(self) -> int:
        return self.global_features.shape[1]

    @classmethod
    def from_manifest(cls, token_dir: Path, report_dir: Path) -> tuple["ReactionTokens", dict]:
        """Read every reaction of a token build in manifest order (= the order of the original training data).

        Parameters
        ----------
        token_dir : Path
            Directory with ``<rxn_id>.npz`` token files.
        report_dir : Path
            Directory with ``T_report.json`` (layout + build gates), ``T_manifest.csv`` (rxn_id, n_atoms, sha256,
            n_ok, finite, n_formed) and ``Q_report.json`` (atom-feature build).

        Returns
        -------
        (ReactionTokens, dict)
            The reactions and a dict with ``token_checks_ok`` [r] bool (build gate T-a: all atoms ok, finite values,
            exactly 2 forming bonds) and the report contents / sha256.
        """
        token_report_path, manifest_path, atom_report_path = (report_dir / n for n in
                                                              ("T_report.json", "T_manifest.csv", "Q_report.json"))
        token_report = json.loads(token_report_path.read_text())
        atom_report = json.loads(atom_report_path.read_text())
        problems = [name for name, ok in (
            ("token_version", token_report.get("token_version") == K.TOKEN_VERSION),
            ("atom_version", atom_report.get("atom_version") == K.ATOM_VERSION),
            ("q_report_sha256", token_report.get("q_report_sha256") == file_sha256(atom_report_path)),
            ("manifest_sha256", token_report.get("manifest_sha256") == file_sha256(manifest_path)),
            ("binary columns", tuple(token_report.get("bin_cols") or ()) == K.BINARY_COLUMNS),
            ("numeric columns", tuple(token_report.get("num_cols") or ()) == K.NUMERIC_COLUMNS
             and bool(token_report.get("ext_used"))),
            ("182 global columns", len(token_report.get("glob_cols") or ()) == K.N_GLOBAL_FEATURES),
            ("elements", tuple(int(z) for z in token_report.get("elements") or ()) == K.ELEMENTS),
            ("build gates", all(bool(v) for v in (token_report.get("gate_Ta") or {"missing": False}).values())),
        ) if not ok]
        if problems:
            raise DataError(f"{token_report_path}: token build does not match the expected layout: {problems}")
        layout = TokenLayout(K.ELEMENTS, K.BINARY_COLUMNS, K.NUMERIC_COLUMNS, tuple(token_report["glob_cols"]),
                             token_report["token_version"])
        manifest = pd.read_csv(manifest_path)
        reaction_ids = manifest.rxn_id.astype(int).to_numpy()
        if len(set(reaction_ids.tolist())) != len(reaction_ids):
            raise DataError(f"{manifest_path}: duplicated rxn_id")
        paths = [Path(token_dir) / f"{rid}.npz" for rid in reaction_ids]
        arrays = fill_token_arrays(paths, layout, max_atoms=int(manifest.n_atoms.max()),
                                   expected_sha256=manifest.sha256.astype(str).tolist())
        token_checks_ok = np.ones(len(reaction_ids), dtype=bool)
        for column in ("n_ok", "finite"):
            if column in manifest.columns:
                token_checks_ok &= manifest[column].astype(bool).to_numpy()
        if "n_formed" in manifest.columns:
            token_checks_ok &= manifest.n_formed.to_numpy(int) == 2
        info = dict(token_checks_ok=token_checks_ok, token_report=token_report,
                    sha256=dict(t_report=file_sha256(token_report_path), t_manifest=file_sha256(manifest_path),
                                q_report=file_sha256(atom_report_path)))
        return cls(reaction_ids, manifest.n_atoms.to_numpy(int), arrays, layout), info

    @classmethod
    def from_files(cls, token_dir: Path, reaction_ids, layout: TokenLayout) -> "ReactionTokens":
        """Read the token files of arbitrary reactions (new reactions at prediction time), rows in ascending rxn_id.

        Padding length = the largest of these reactions; ``n_atoms`` is counted from the atom mask.
        """
        ids = np.array(sorted(int(r) for r in reaction_ids), dtype=int)
        arrays = fill_token_arrays([Path(token_dir) / f"{r}.npz" for r in ids], layout)
        return cls(ids, arrays["atom_mask"].sum(1).astype(int), arrays, layout)

    def static_tensors(self, dtype, device) -> dict:
        """Tensors that do not depend on the standardisation (positions, masks, edge types, fragments, sizes); cached.

        Returns
        -------
        dict of torch tensors
            positions_angstrom [r, N, 3] (dtype), atom_mask [r, N] bool, edge_types [r, N, N] int8,
            fragment_ids [r, N] int8, n_atoms [r] int64.
        """
        import torch
        key = (dtype, str(device))
        if key not in self._static:
            self._static[key] = dict(positions_angstrom=torch.tensor(self.positions_angstrom, dtype=dtype, device=device),
                                     atom_mask=torch.tensor(self.atom_mask, device=device),
                                     edge_types=torch.tensor(self.edge_types.astype(np.int8), device=device),
                                     fragment_ids=torch.tensor(self.fragment_ids.astype(np.int8), device=device),
                                     n_atoms=torch.tensor(self.n_atoms, device=device))
        return self._static[key]


def check_global_table(table_path: Path, tokens: ReactionTokens) -> tuple[np.ndarray, dict]:
    """Check the 182-column table against the token reactions.

    Each token file carries its own copy of the 182 global features; the table must hold the same values
    (<= 1e-12 relative) so that the model sees exactly the published inputs.

    Returns
    -------
    (numpy.ndarray [r] bool, dict)
        True where the reaction has a table row whose 182 values are all finite; info (sha256, counts).

    Raises
    ------
    DataError
        Wrong columns, duplicated rxn_id, or a row that differs from the token copy.
    """
    table = pd.read_parquet(table_path)
    if "rxn_id" not in table.columns:
        raise DataError(f"{table_path}: no rxn_id column")
    columns = [c for c in table.columns if c != "rxn_id"]
    if tuple(columns) != tokens.layout.global_columns:
        raise DataError(f"{table_path}: column names / order differ from the token report")
    table = table.assign(rxn_id=table.rxn_id.astype(int))
    if table.rxn_id.duplicated().any():
        raise DataError(f"{table_path}: duplicated rxn_id")
    table = table.set_index("rxn_id")
    common = [int(r) for r in tokens.reaction_ids if int(r) in table.index]
    rows = np.array([tokens.row_of[r] for r in common], dtype=int)
    has_row = np.zeros(len(tokens.reaction_ids), dtype=bool)
    finite = np.zeros(len(tokens.reaction_ids), dtype=bool)
    values = table.loc[common, columns].apply(pd.to_numeric, errors="raise").to_numpy(dtype=float) if common \
        else np.zeros((0, K.N_GLOBAL_FEATURES))
    has_row[rows] = True
    finite[rows] = np.isfinite(values).all(1)
    token_copy = tokens.global_features[rows]
    differs = (np.abs(values - token_copy) > K.GLOBAL_TABLE_RELATIVE_TOLERANCE * np.maximum(1.0, np.abs(token_copy))) \
        & np.isfinite(values)
    if differs.any(1).sum():
        raise DataError(f"{table_path}: {int(differs.any(1).sum())} reactions differ from their token copy")
    return has_row & finite, dict(path=str(table_path), sha256=file_sha256(table_path), n_rows=int(len(table)))


# =========================================================================== pairs, labels, folds
def load_pair_labels(path: Path, label_report: Path | None = K.LABEL_REPORT) -> tuple[pd.DataFrame, dict]:
    """Read the pair table (index pair_id) with the 10 label differences dd_<label> (kcal/mol).

    The label file is accepted only when its sha256 equals the one recorded by the label validation report.
    """
    path = Path(path)
    sha = file_sha256(path)
    if label_report is not None:
        report = json.loads(Path(label_report).read_text())
        recorded = (report.get("outputs") or {}).get(str(path))
        if not report.get("all_pass") or recorded != sha:
            raise DataError(f"{path}: sha256 differs from the validated label file ({label_report})")
    pairs = pd.read_csv(path)
    needed = ["pair_id", "ref_kind", "a_rxn", "b_rxn", "pos"] + [f"dd_{t}" for t in K.LABEL_CHANNELS]
    missing = [c for c in needed if c not in pairs.columns]
    if missing:
        raise DataError(f"{path}: columns missing {missing}")
    if pairs.pair_id.duplicated().any():
        raise DataError(f"{path}: duplicated pair_id")
    pairs = pairs.assign(pair_id=pairs.pair_id.astype(str), ref_kind=pairs.ref_kind.fillna("").astype(str),
                         pos=pairs.pos.fillna("").astype(str), a_rxn=pairs.a_rxn.astype(int),
                         b_rxn=pairs.b_rxn.astype(int))
    return pairs.set_index("pair_id"), dict(path=str(path), sha256=sha, n_pairs=int(len(pairs)))


def load_folds(split_dir: Path = K.SPLIT_DIR, split_report: Path = K.SPLIT_REPORT) -> tuple[dict, dict, dict]:
    """Read the 5 T1 split files ``T1__f<k>.json`` (lists train / test / inner_train / inner_val of pair ids).

    Returns
    -------
    (folds, report, sha256)
        folds[k][list name] = list of pair ids (file order); the split report; sha256 per file.
    """
    report = json.loads(Path(split_report).read_text())
    folds, shas = {}, {}
    for k in range(K.N_FOLDS):
        name = f"T1__f{k}.json"
        path = Path(split_dir) / name
        if not path.is_file() or file_sha256(path) != report["units"][name]["sha256"]:
            raise DataError(f"{path} missing or changed after {split_report}")
        folds[k] = json.loads(path.read_text())
        shas[f"split_T1_f{k}"] = report["units"][name]["sha256"]
    return folds, report, shas


@dataclass
class FoldSplit:
    """Pair rows (indices into the pair table) of one fold; every list holds usable pairs only, in split-file order."""

    train: np.ndarray
    test: np.ndarray
    inner_train: np.ndarray
    inner_val: np.ndarray
    dropped: dict
    pairs_sha256: str
    validation_sha256: str | None
    test_sha256: str | None

    def __getitem__(self, name: str) -> np.ndarray:
        return getattr(self, name)


class PairDataset:
    """The GNN3 training data: token reactions, the pair table with ΔΔ targets, usable pairs and the T1 folds.

    Attributes
    ----------
    reactions : ReactionTokens
    pairs : pandas.DataFrame, index pair_id (columns ref_kind, a_rxn, b_rxn, pos, dd_<label>, ...)
    labels_kcal : numpy.ndarray [n_pairs, 10]   dd_<label> in ``LABEL_CHANNELS`` order
    targets_kcal : numpy.ndarray [n_pairs, 4]   strain, inter, solv, barrier ΔΔ (= labels_kcal @ M)
    rows_a, rows_b : numpy.ndarray [n_pairs] int   reaction row of a and of b (-1 without tokens)
    usable_pair : numpy.ndarray [n_pairs] bool    both reactions have tokens, pass the build gate and the 182 table
    pair_list : list of str                        usable pairs that appear in some fold (sorted ids)
    """

    def __init__(self, token_dir: Path = K.LF_INPUTS / "fu2_lf" / "tokens", report_dir: Path = K.TOKEN_REPORTS,
                 global_table: Path = K.LF_INPUTS / "X182_lf.parquet", pair_labels: Path = K.PAIR_LABELS,
                 split_dir: Path = K.SPLIT_DIR, split_report: Path = K.SPLIT_REPORT,
                 pair_attributes: Path = K.PAIR_ATTRIBUTES, label_report: Path | None = K.LABEL_REPORT):
        self.reactions, token_info = ReactionTokens.from_manifest(Path(token_dir), Path(report_dir))
        table_ok, table_info = check_global_table(Path(global_table), self.reactions)
        recorded = token_info["token_report"].get("x182_sha256")
        if recorded is not None and recorded != table_info["sha256"]:
            raise DataError(f"{global_table}: sha256 differs from the one the token build recorded")
        # usable reaction ("lf" rule): passes the token build gate AND has a finite row in the 182-column table
        self.usable_reaction = token_info["token_checks_ok"] & table_ok
        self.pairs, label_info = load_pair_labels(pair_labels, label_report)
        self.folds, self.split_report, split_sha = load_folds(split_dir, split_report)
        fold_ids = {q for fold in self.folds.values() for name in K.FOLD_LISTS for q in fold[name]}
        unknown = sorted(fold_ids - set(self.pairs.index))
        if unknown:
            raise DataError(f"{len(unknown)} split pair ids are not in the label file: {unknown[:5]}")
        self.pair_attributes = Path(pair_attributes)
        self.labels_kcal = self.pairs[[f"dd_{t}" for t in K.LABEL_CHANNELS]].to_numpy(float)
        self.targets_kcal = labels_to_channels(self.labels_kcal)
        pair_ids = self.pairs.index.to_numpy()
        self.row_of_pair = {p: k for k, p in enumerate(pair_ids)}
        row_of = self.reactions.row_of
        self.rows_a = np.array([row_of.get(int(r), -1) for r in self.pairs.a_rxn])
        self.rows_b = np.array([row_of.get(int(r), -1) for r in self.pairs.b_rxn])
        self.has_tokens = (self.rows_a >= 0) & (self.rows_b >= 0)
        usable = self.usable_reaction
        self.usable_pair = self.has_tokens & usable[np.maximum(self.rows_a, 0)] & usable[np.maximum(self.rows_b, 0)]
        in_folds = np.isin(pair_ids, sorted({q for fold in self.folds.values() for q in fold["train"] + fold["test"]}))
        self.pair_list = sorted(pair_ids[self.usable_pair & in_folds].tolist())
        self.sha256 = dict(token_info["sha256"], x182=table_info["sha256"], labels=label_info["sha256"],
                           s_splits=file_sha256(split_report), **split_sha)

    # ------------------------------------------------------------------ convenience
    @property
    def n_pairs(self) -> int:
        return len(self.pairs)

    def pair_ids(self, rows) -> list[str]:
        return [str(q) for q in self.pairs.index.to_numpy()[np.asarray(rows, dtype=int)]]

    def reactions_of(self, pair_rows) -> np.ndarray:
        """Sorted unique reaction rows used by the given pairs (both a and b)."""
        pair_rows = np.asarray(pair_rows, dtype=int)
        return np.unique(np.r_[self.rows_a[pair_rows], self.rows_b[pair_rows]])

    # ------------------------------------------------------------------ folds
    def fold_split(self, fold: int) -> FoldSplit:
        """Usable pair rows of fold ``fold`` (0..4) with leak checks.

        Raises
        ------
        DataError
            A test pair or a test reaction is also in training, or the inner lists do not partition the training list.
        """
        record = self.folds[int(fold)]
        lists, dropped = {}, {}
        for name in K.FOLD_LISTS:
            rows = np.array([self.row_of_pair[q] for q in record[name]], dtype=int)
            keep = self.usable_pair[rows]
            lists[name] = rows[keep]
            dropped[name] = dict(no_tokens=int((~self.has_tokens[rows]).sum()), not_usable=int((~keep).sum()))
        as_set = {name: set(lists[name].tolist()) for name in K.FOLD_LISTS}
        if as_set["test"] & as_set["train"]:
            raise DataError(f"fold {fold}: test pair in train")
        if as_set["inner_train"] | as_set["inner_val"] != as_set["train"] or as_set["inner_train"] & as_set["inner_val"]:
            raise DataError(f"fold {fold}: inner lists do not partition the training list")
        if set(self.reactions_of(lists["test"]).tolist()) & set(self.reactions_of(lists["train"]).tolist()):
            raise DataError(f"fold {fold}: a test reaction is also a training reaction")
        ids = {name: self.pair_ids(lists[name]) for name in K.FOLD_LISTS}
        return FoldSplit(**lists, dropped=dropped,
                         pairs_sha256=list_sha256(ids["train"] + ["|"] + ids["test"] + ["|"] + ids["inner_train"] + ["|"]
                                                  + ids["inner_val"]),
                         validation_sha256=list_sha256(ids["inner_val"]), test_sha256=list_sha256(ids["test"]))

    def all_training_pairs(self) -> FoldSplit:
        """Every usable pair of the main training pool (union of the 5 training lists) for the final model.

        The union must equal the ``in_T_main`` pairs of the pair attribute file. Test / validation lists are empty.
        """
        ids = sorted({q for fold in self.folds.values() for q in fold["train"]})
        if self.pair_attributes.is_file():
            attributes = pd.read_csv(self.pair_attributes, usecols=["pair_id", "in_T_main"])
            in_main = set(attributes.loc[attributes.in_T_main.astype(bool), "pair_id"].astype(str))
            if in_main != set(ids):
                raise DataError(f"union of the fold training lists ({len(ids)}) != in_T_main ({len(in_main)})")
        rows = np.array(sorted(self.row_of_pair[q] for q in ids), dtype=int)
        keep = self.usable_pair[rows]
        empty = np.zeros(0, dtype=int)
        train_ids = self.pair_ids(rows[keep])
        return FoldSplit(train=rows[keep], test=empty, inner_train=empty, inner_val=empty,
                         dropped=dict(train=dict(no_tokens=int((~self.has_tokens[rows]).sum()),
                                                 not_usable=int((~keep).sum()))),
                         pairs_sha256=list_sha256(train_ids), validation_sha256=None, test_sha256=None)


# =========================================================================== standardisation
class Standardizer:
    """Per-fit feature and target scaling, computed from the fit's TRAINING data only.

    Numeric token columns and the 182 global features are z-scored with the mean / std over the atoms (resp. the
    reactions) of the training reactions; columns with std < 1e-12 are left unscaled (std set to 1). Targets are divided
    by their std over the training pairs (ddof 0) WITHOUT subtracting the mean: a ΔΔ model must give exactly 0 for
    a pair of identical reactions, so the target scale keeps zero at zero.

    Notes
    -----
    Using only training reactions keeps test reactions out of every statistic the model sees (leak check C-5).
    The saved-model keys of the original pipeline are mu, sd, gmu, gsd, sy.
    """

    def __init__(self, numeric_mean, numeric_std, global_mean, global_std, target_scale, reaction_rows=None,
                 pair_rows=None):
        self.numeric_mean = np.asarray(numeric_mean, dtype=float)
        self.numeric_std = np.asarray(numeric_std, dtype=float)
        self.global_mean = np.asarray(global_mean, dtype=float)
        self.global_std = np.asarray(global_std, dtype=float)
        self.target_scale = np.asarray(target_scale, dtype=float)
        self.reaction_rows = reaction_rows
        self.pair_rows = pair_rows

    @classmethod
    def from_training_pairs(cls, dataset: PairDataset, pair_rows) -> "Standardizer":
        """Statistics of the training pairs ``pair_rows`` and their reactions.

        Parameters
        ----------
        dataset : PairDataset
        pair_rows : array [n_train] int

        Returns
        -------
        Standardizer
            numeric_mean / numeric_std [14], global_mean / global_std [182], target_scale [4] (kcal/mol).
        """
        pair_rows = np.asarray(pair_rows)
        reaction_rows = np.unique(np.r_[dataset.rows_a[pair_rows], dataset.rows_b[pair_rows]])
        tokens = dataset.reactions
        mask = tokens.atom_mask[reaction_rows]
        atom_values = tokens.numeric_features[reaction_rows][mask]                 # [atoms, 14]
        numeric_mean, numeric_std = atom_values.mean(0), atom_values.std(0)
        numeric_std[numeric_std < K.SCALE_EPSILON] = 1.0
        global_values = tokens.global_features[reaction_rows]
        global_mean, global_std = global_values.mean(0), global_values.std(0)
        global_std[global_std < K.SCALE_EPSILON] = 1.0
        target_scale = dataset.targets_kcal[pair_rows].std(0)                       # ddof 0, mean NOT removed
        return cls(numeric_mean, numeric_std, global_mean, global_std, target_scale, reaction_rows, pair_rows)

    @classmethod
    def from_saved(cls, saved: dict) -> "Standardizer":
        """From the ``std`` dict of a model file (keys mu, sd, gmu, gsd, sy)."""
        return cls(saved["mu"], saved["sd"], saved["gmu"], saved["gsd"], saved["sy"])

    def as_saved(self) -> dict:
        return dict(mu=self.numeric_mean, sd=self.numeric_std, gmu=self.global_mean, gsd=self.global_std,
                    sy=self.target_scale)

    def sha256(self) -> str:
        """sha256 of the five statistic arrays as float64 bytes (mu, sd, gmu, gsd, sy order)."""
        digest = hashlib.sha256()
        for values in (self.numeric_mean, self.numeric_std, self.global_mean, self.global_std, self.target_scale):
            digest.update(np.ascontiguousarray(np.asarray(values, dtype=np.float64)).tobytes())
        return digest.hexdigest()

    def tensors(self, reactions: ReactionTokens, dtype=None, device=None) -> dict:
        """Model input tensors of every reaction, standardised with these statistics.

        Returns
        -------
        dict of torch tensors
            atom_features [r, N, 29] (one-hot | flags | z-scored numeric; padded atoms 0), global_features [r, 182]
            (z-scored) in ``dtype``, plus the static tensors of ``ReactionTokens.static_tensors``.
        """
        import torch
        dtype = torch.float32 if dtype is None else dtype
        device = default_device() if device is None else device
        numeric = (reactions.numeric_features - self.numeric_mean) / self.numeric_std * reactions.atom_mask[..., None]
        atom_features = np.concatenate([reactions.element_onehot, reactions.binary_flags, numeric], -1)
        global_features = (reactions.global_features - self.global_mean) / self.global_std
        return dict(reactions.static_tensors(dtype, device),
                    atom_features=torch.tensor(atom_features, dtype=dtype, device=device),
                    global_features=torch.tensor(global_features, dtype=dtype, device=device))


def default_device():
    """The CUDA device when one is visible, else the CPU (= fu2/train.py DEV)."""
    import torch
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")
