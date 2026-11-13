"""Run PhaseResSAGE, a residue-level GAT-to-GraphSAGE LLPS experiment.

The script intentionally has no PyTorch Geometric dependency. Each protein is
represented as a residue graph parsed from a PDB file. ESM-2 t6 residue
embeddings are cached to disk, then a GAT-first / GraphSAGE-second encoder is
trained for:
  1. LLPS versus non-LLPS controls
  2. Scaffold versus client proteins

The script supports the formal PDB-first/AlphaFold2-fallback structure workflow
and a lightweight ESM2 contact-proxy mode for machines without enough memory.
It reports the exact data audit and must not be used to claim a benchmark that
was not actually run.
"""

from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import json
import math
import os
import random
import re
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
from Bio.SeqIO import parse as parse_fasta
from scipy.spatial import cKDTree
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split


AA20 = "ACDEFGHIKLMNPQRSTVWY"
AA_TO_IDX = {aa: i for i, aa in enumerate(AA20)}
HYDROPATHY = {
    "A": 1.8, "C": 2.5, "D": -3.5, "E": -3.5, "F": 2.8,
    "G": -0.4, "H": -3.2, "I": 4.5, "K": -3.9, "L": 3.8,
    "M": 1.9, "N": -3.5, "P": -1.6, "Q": -3.5, "R": -4.5,
    "S": -0.8, "T": -0.7, "V": 4.2, "W": -0.9, "Y": -1.3,
}
CHARGE = {"K": 1.0, "R": 1.0, "H": 0.5, "D": -1.0, "E": -1.0}
AROMATIC = set("FWY")
POLAR = set("STNQCY")
FLEXIBLE = set("GSPQ")
THREE_TO_ONE = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C",
    "GLN": "Q", "GLU": "E", "GLY": "G", "HIS": "H", "ILE": "I",
    "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F", "PRO": "P",
    "SER": "S", "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V",
}


@dataclass
class ProteinRecord:
    protein_id: str
    sequence: str
    pdb_path: str
    stage1_label: int
    stage2_label: Optional[int]
    role: str
    source: str


@dataclass
class GraphRecord:
    protein_id: str
    sequence: str
    node_features: torch.Tensor
    edge_index: torch.Tensor
    edge_features: torch.Tensor
    stage1_label: int
    stage2_label: Optional[int]
    role: str
    source: str


@dataclass
class GraphRef:
    protein_id: str
    sequence: str
    cache_path: str
    stage1_label: int
    stage2_label: Optional[int]
    role: str
    source: str


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def read_fasta(path: Path) -> Dict[str, Tuple[str, str]]:
    records: Dict[str, Tuple[str, str]] = {}
    for record in parse_fasta(str(path), "fasta"):
        sequence = str(record.seq).upper().replace(" ", "").replace("\n", "")
        # Biopython's record.id stops at the first whitespace. The PhaseOM
        # headers contain species names with spaces, so retain the full header.
        records[record.description.strip()] = (record.description.strip(), sequence)
    return records


def normalize_name(text: str) -> str:
    text = text.lower().replace(".pdb", "")
    text = text.replace("reviewed", "")
    text = re.sub(r"[^a-z0-9]+", "", text)
    return text


def pdb_stem_from_description(description: str) -> str:
    """Extract the PDB filename stem from a PhaseOM FASTA header."""
    fields = [field.strip() for field in description.split("|") if field.strip()]
    for index, field in enumerate(fields[:-1]):
        if field.lower() in {"sp", "tr"} and re.fullmatch(
            r"[A-Z0-9]{6,10}", fields[index + 1], flags=re.IGNORECASE
        ):
            return fields[index + 1].upper()
    for field in reversed(fields):
        if re.fullmatch(r"(?i)reviewed\d+", field):
            return field[0].upper() + field[1:]
    for field in reversed(fields):
        if re.fullmatch(r"[A-Z][0-9][A-Z0-9]{4,8}", field.upper()):
            return field.upper()
    # Some PhaseOM rows expose only an internal numeric row identifier.  It is
    # not a structural accession, but it is sufficient as a deterministic
    # placeholder in ESM2-proxy mode; the protein_id below also includes a
    # sequence/header hash to prevent collisions.
    if fields:
        fallback = re.sub(r"[^A-Za-z0-9_.-]+", "_", fields[-1])
        if fallback:
            return fallback[:80]
    raise ValueError(f"Cannot infer a stable identifier from FASTA header: {description}")


def stable_record_id(description: str, sequence: str, stem: str) -> str:
    """Keep internal PhaseOM row names unique when Reviewed indices repeat."""
    digest = hashlib.sha1(
        f"{description}\n{sequence}".encode("utf-8")
    ).hexdigest()[:12]
    return f"{stem}_{digest}"


def read_pdb_backbone_and_dssp(
    pdb_path: Path,
) -> Tuple[str, np.ndarray, np.ndarray, np.ndarray]:
    """Read backbone coordinates, pLDDT and DSSP in one PDB pass."""
    import pydssp

    text = pdb_path.read_text(encoding="utf-8", errors="ignore")
    backbone, residue_names = pydssp.read_pdbtext(text, return_sequence=True)
    dssp_labels = np.asarray(pydssp.assign(backbone, out_type="c3"))
    pdb_sequence = "".join(
        THREE_TO_ONE.get(str(name).strip().upper(), "X")
        for name in residue_names
    )
    if not pdb_sequence or len(backbone) != len(pdb_sequence):
        raise ValueError(f"No usable backbone coordinates: {pdb_path}")

    # AlphaFold PDB files store confidence in the B-factor column. pydssp
    # provides the residue order, while this lightweight scan avoids a second
    # full Biopython structure parse.
    plddt = []
    seen_chain = None
    for line in text.splitlines():
        if line.startswith("ENDMDL"):
            break
        if not line.startswith("ATOM") or line[12:16].strip() != "CA":
            continue
        chain = line[21].strip()
        if seen_chain is None:
            seen_chain = chain
        if chain != seen_chain:
            continue
        try:
            plddt.append(float(line[60:66]))
        except ValueError:
            plddt.append(50.0)
    if len(plddt) != len(pdb_sequence):
        plddt = [50.0] * len(pdb_sequence)
    return (
        pdb_sequence,
        np.asarray(backbone[:, 1, :], dtype=np.float32),
        np.asarray(plddt, dtype=np.float32),
        dssp_labels,
    )


def align_structure_to_sequence(
    fasta_sequence: str,
    pdb_sequence: str,
    coords: np.ndarray,
    plddt: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Map PDB residues to the supplied sequence with a simple global alignment."""
    if fasta_sequence == pdb_sequence and len(coords) == len(fasta_sequence):
        return coords, plddt
    from Bio.Align import PairwiseAligner

    aligner = PairwiseAligner()
    aligner.mode = "global"
    aligner.match_score = 2.0
    aligner.mismatch_score = -1.0
    aligner.open_gap_score = -2.0
    aligner.extend_gap_score = -0.2
    alignment = aligner.align(fasta_sequence, pdb_sequence)[0]
    mapped_coords = np.zeros((len(fasta_sequence), 3), dtype=np.float32)
    mapped_plddt = np.zeros(len(fasta_sequence), dtype=np.float32)
    mapped_plddt[:] = 0.0
    for (f_start, f_end), (p_start, p_end) in zip(
        alignment.aligned[0], alignment.aligned[1]
    ):
        width = min(f_end - f_start, p_end - p_start)
        mapped_coords[f_start : f_start + width] = coords[p_start : p_start + width]
        mapped_plddt[f_start : f_start + width] = plddt[p_start : p_start + width]
    return mapped_coords, mapped_plddt


def align_labels_to_sequence(
    fasta_sequence: str,
    pdb_sequence: str,
    labels: np.ndarray,
    fill_value: str = "-",
) -> np.ndarray:
    if fasta_sequence == pdb_sequence and len(labels) == len(fasta_sequence):
        return labels.copy()
    from Bio.Align import PairwiseAligner

    aligner = PairwiseAligner()
    aligner.mode = "global"
    aligner.match_score = 2.0
    aligner.mismatch_score = -1.0
    aligner.open_gap_score = -2.0
    aligner.extend_gap_score = -0.2
    alignment = aligner.align(fasta_sequence, pdb_sequence)[0]
    mapped = np.full(len(fasta_sequence), fill_value, dtype=labels.dtype)
    for (f_start, f_end), (p_start, p_end) in zip(
        alignment.aligned[0], alignment.aligned[1]
    ):
        width = min(f_end - f_start, p_end - p_start)
        mapped[f_start : f_start + width] = labels[p_start : p_start + width]
    return mapped


def calculate_dssp_c3(pdb_path: Path, pdb_sequence: str) -> Optional[np.ndarray]:
    try:
        import pydssp

        coords, residue_names = pydssp.read_pdbtext(
            pdb_path.read_text(encoding="utf-8", errors="ignore"),
            return_sequence=True,
        )
        labels = np.asarray(pydssp.assign(coords, out_type="c3"))
        parsed_sequence = "".join(
            THREE_TO_ONE.get(str(name).strip().upper(), "X")
            for name in residue_names
        )
        if parsed_sequence != pdb_sequence:
            return None
        return labels
    except Exception:
        return None


def residue_hand_features(sequence: str, plddt: np.ndarray) -> np.ndarray:
    length = max(1, len(sequence))
    aa_onehot = np.zeros((length, 20), dtype=np.float32)
    for i, aa in enumerate(sequence):
        if aa in AA_TO_IDX:
            aa_onehot[i, AA_TO_IDX[aa]] = 1.0
    hydro = np.asarray([HYDROPATHY.get(aa, 0.0) for aa in sequence], dtype=np.float32)
    charge = np.asarray([CHARGE.get(aa, 0.0) for aa in sequence], dtype=np.float32)
    aromatic = np.asarray([float(aa in AROMATIC) for aa in sequence], dtype=np.float32)
    polar = np.asarray([float(aa in POLAR) for aa in sequence], dtype=np.float32)
    flexible = np.asarray([float(aa in FLEXIBLE) for aa in sequence], dtype=np.float32)
    rel_pos = np.arange(length, dtype=np.float32) / max(1, length - 1)
    disorder = 1.0 - np.clip(plddt, 0.0, 100.0) / 100.0
    idr_flag = (plddt < 50.0).astype(np.float32)

    # A short sequence-local complexity proxy: 1 - normalized entropy in a
    # 15-residue window. It is intentionally residue-level and interpretable.
    lcr = np.zeros(length, dtype=np.float32)
    local_idr = np.zeros(length, dtype=np.float32)
    radius = 7
    for i in range(length):
        lo, hi = max(0, i - radius), min(length, i + radius + 1)
        window = sequence[lo:hi]
        counts = np.bincount(
            [AA_TO_IDX[a] for a in window if a in AA_TO_IDX], minlength=20
        ).astype(np.float32)
        probs = counts / max(1.0, counts.sum())
        entropy = float(-(probs[probs > 0] * np.log(probs[probs > 0])).sum())
        lcr[i] = 1.0 - entropy / math.log(20.0)
        local_idr[i] = float(idr_flag[lo:hi].mean())
    numeric = np.stack(
        [hydro / 4.5, charge / 4.5, aromatic, polar, flexible, rel_pos,
         disorder, idr_flag, lcr, local_idr],
        axis=1,
    )
    return np.concatenate([aa_onehot, numeric], axis=1)


def make_dssp_like_features(
    sequence: str,
    plddt: np.ndarray,
    dssp_labels: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Return three-state DSSP one-hot plus the residue confidence value."""
    n = len(sequence)
    ss = np.zeros((n, 3), dtype=np.float32)
    if dssp_labels is not None and len(dssp_labels) == n:
        for i, label in enumerate(dssp_labels):
            label = str(label)
            ss[i, 0 if label == "H" else 1 if label == "E" else 2] = 1.0
    else:
        # Explicit fallback for structures with incomplete backbone atoms.
        for i, confidence in enumerate(plddt):
            if confidence >= 75:
                ss[i, 0 if sequence[i] in set("AEIKLMQR") else 1] = 1.0
            else:
                ss[i, 2] = 1.0
    return np.concatenate([ss, (np.clip(plddt, 0, 100) / 100.0)[:, None]], axis=1)


def build_edges(coords: np.ndarray, threshold: float = 8.0) -> Tuple[np.ndarray, np.ndarray]:
    n = coords.shape[0]
    if n <= 1:
        return (
            np.empty((2, 0), dtype=np.int64),
            np.empty((0, 2), dtype=np.float32),
        )
    # cKDTree avoids the quadratic pairwise scan for long human proteins.
    pairs = cKDTree(coords).query_pairs(threshold, output_type="ndarray")
    if pairs.size == 0:
        pairs = np.empty((0, 2), dtype=np.int64)
    else:
        pairs = np.asarray(pairs, dtype=np.int64).reshape(-1, 2)
    sequential = np.column_stack(
        [np.arange(n - 1, dtype=np.int64), np.arange(1, n, dtype=np.int64)]
    )
    pairs = np.vstack([pairs, sequential])
    pairs = np.unique(pairs, axis=0)
    distances = np.linalg.norm(coords[pairs[:, 0]] - coords[pairs[:, 1]], axis=1)
    gaps = np.minimum(pairs[:, 1] - pairs[:, 0], 50) / 50.0
    directed = np.vstack([pairs, pairs[:, ::-1]])
    directed_distances = np.concatenate([distances, distances])
    directed_gaps = np.concatenate([gaps, gaps])
    edge_features = np.column_stack(
        [np.minimum(directed_distances, threshold) / threshold, directed_gaps]
    ).astype(np.float32)
    return directed.T.astype(np.int64), edge_features


def load_esm_model(
    encoder_name: str,
    cache_dir: Path,
    device: torch.device,
):
    import esm

    cache_dir.mkdir(parents=True, exist_ok=True)
    loaders = {
        "esm2_t6_8M_UR50D": esm.pretrained.esm2_t6_8M_UR50D,
        "esm2_t33_650M_UR50D": esm.pretrained.esm2_t33_650M_UR50D,
    }
    if encoder_name not in loaders:
        raise ValueError(
            f"Unsupported encoder {encoder_name}; choose from {sorted(loaders)}"
        )
    model, alphabet = loaders[encoder_name]()
    model = model.to(device).eval()
    return model, alphabet


def encoder_layer(model) -> int:
    """Use the final hidden layer for every supported ESM2 checkpoint."""
    return int(getattr(model, "num_layers", 6))


@torch.no_grad()
def esm_embedding(
    model,
    alphabet,
    sequence: str,
    device: torch.device,
) -> torch.Tensor:
    batch_converter = alphabet.get_batch_converter()
    max_residues = int(getattr(model, "max_positions", 1024)) - 2
    overlap = 128
    if len(sequence) <= max_residues:
        chunks = [(0, sequence)]
    else:
        step = max_residues - overlap
        starts = list(range(0, len(sequence), step))
        if starts[-1] + max_residues > len(sequence):
            starts[-1] = max(0, len(sequence) - max_residues)
        starts = list(dict.fromkeys(starts))
        chunks = [(start, sequence[start : start + max_residues]) for start in starts]

    accumulated = torch.zeros((len(sequence), model.embed_dim), dtype=torch.float32)
    counts = torch.zeros((len(sequence), 1), dtype=torch.float32)
    for offset in range(0, len(chunks), 2):
        batch = chunks[offset : offset + 2]
        _, _, tokens = batch_converter(
            [(f"protein_{start}", chunk) for start, chunk in batch]
        )
        tokens = tokens.to(device)
        layer = encoder_layer(model)
        result = model(tokens, repr_layers=[layer], return_contacts=False)
        representations = result["representations"][layer].detach().cpu()
        for batch_index, (start, chunk) in enumerate(batch):
            length = len(chunk)
            reps = representations[batch_index, 1 : length + 1].float()
            accumulated[start : start + length] += reps
            counts[start : start + length] += 1.0
    return accumulated / counts.clamp_min(1.0)


def sequence_hash(sequence: str) -> str:
    return hashlib.sha1(sequence.upper().encode("utf-8")).hexdigest()[:20]


def cached_esm(
    model,
    alphabet,
    sequence: str,
    protein_id: str,
    cache_dir: Path,
    device: torch.device,
) -> torch.Tensor:
    path = esm_cache_path(protein_id, cache_dir, sequence)
    if path.exists():
        return torch.load(path, map_location="cpu", weights_only=True)
    embedding = esm_embedding(model, alphabet, sequence, device)
    torch.save(embedding, path)
    return embedding


def esm_cache_path(
    protein_id: str,
    cache_dir: Path,
    sequence: Optional[str] = None,
) -> Path:
    cache_key = f"seq_{sequence_hash(sequence)}" if sequence is not None else protein_id
    safe_id = re.sub(r"[^A-Za-z0-9_.-]+", "_", cache_key)
    if len(safe_id) > 100:
        safe_id = hashlib.sha1(cache_key.encode("utf-8")).hexdigest()
    return cache_dir / f"{safe_id}.pt"


@torch.no_grad()
def precompute_esm_cache(
    records: Sequence[ProteinRecord],
    model,
    alphabet,
    cache_dir: Path,
    device: torch.device,
    batch_size: int = 8,
) -> None:
    """Batch short sequences and window long sequences before graph parsing."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    max_residues = int(getattr(model, "max_positions", 1024)) - 2
    missing = []
    seen_cache_paths = set()
    for record in records:
        path = esm_cache_path(record.protein_id, cache_dir, record.sequence)
        if path.exists() or path in seen_cache_paths:
            continue
        seen_cache_paths.add(path)
        missing.append(record)
    short = [record for record in missing if len(record.sequence) <= max_residues]
    short.sort(key=lambda record: len(record.sequence))
    batch_converter = alphabet.get_batch_converter()

    for start in range(0, len(short), batch_size):
        batch = short[start : start + batch_size]
        labels = [
            (record.protein_id, record.sequence)
            for record in batch
        ]
        _, _, tokens = batch_converter(labels)
        tokens = tokens.to(device)
        layer = encoder_layer(model)
        result = model(tokens, repr_layers=[layer], return_contacts=False)
        representations = result["representations"][layer].detach().cpu()
        for batch_index, record in enumerate(batch):
            length = len(record.sequence)
            embedding = representations[batch_index, 1 : length + 1].float()
            torch.save(
                embedding,
                esm_cache_path(record.protein_id, cache_dir, record.sequence),
            )

    for record in missing:
        if len(record.sequence) <= max_residues:
            continue
        embedding = esm_embedding(model, alphabet, record.sequence, device)
        torch.save(
            embedding,
            esm_cache_path(record.protein_id, cache_dir, record.sequence),
        )


def contact_cache_path(
    protein_id: str,
    cache_dir: Path,
    sequence: Optional[str] = None,
) -> Path:
    cache_key = (
        f"seq_{sequence_hash(sequence)}" if sequence is not None else protein_id
    )
    safe_id = re.sub(r"[^A-Za-z0-9_.-]+", "_", cache_key)
    if len(safe_id) > 100:
        safe_id = hashlib.sha1(cache_key.encode("utf-8")).hexdigest()
    return cache_dir / f"{safe_id}.npz"


def sequence_chunks(model, sequence: str) -> List[Tuple[int, str]]:
    max_residues = int(getattr(model, "max_positions", 1024)) - 2
    overlap = 128
    if len(sequence) <= max_residues:
        return [(0, sequence)]
    step = max_residues - overlap
    starts = list(range(0, len(sequence), step))
    if starts[-1] + max_residues > len(sequence):
        starts[-1] = max(0, len(sequence) - max_residues)
    starts = list(dict.fromkeys(starts))
    return [(start, sequence[start : start + max_residues]) for start in starts]


def contact_edges_from_map(
    contacts: np.ndarray,
    top_k: int = 4,
    min_sequence_gap: int = 3,
    minimum_probability: float = 0.20,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Convert an ESM contact map to sparse undirected residue contacts."""
    contacts = np.asarray(contacts, dtype=np.float32)
    contacts = (contacts + contacts.T) / 2.0
    n = contacts.shape[0]
    best: Dict[Tuple[int, int], float] = {}
    for i in range(n):
        row = contacts[i].copy()
        row[max(0, i - min_sequence_gap + 1) : i + min_sequence_gap] = -1.0
        valid = np.flatnonzero(row >= minimum_probability)
        if valid.size == 0:
            continue
        if valid.size > top_k:
            keep = np.argpartition(row[valid], -top_k)[-top_k:]
            valid = valid[keep]
        for j in valid:
            if i == int(j):
                continue
            a, b = sorted((i, int(j)))
            best[(a, b)] = max(best.get((a, b), 0.0), float(contacts[i, j]))
    if not best:
        return (
            np.empty((0, 2), dtype=np.int64),
            np.empty((0,), dtype=np.float32),
            np.zeros(n, dtype=np.float32),
        )
    pairs = np.asarray(list(best.keys()), dtype=np.int64)
    scores = np.asarray([best[tuple(pair)] for pair in pairs], dtype=np.float32)
    strength = np.zeros(n, dtype=np.float32)
    for (i, j), score in zip(pairs, scores):
        strength[i] = max(strength[i], score)
        strength[j] = max(strength[j], score)
    return pairs, scores, strength


def save_contact_proxy(
    path: Path,
    pairs: np.ndarray,
    scores: np.ndarray,
    strength: np.ndarray,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        pairs=pairs.astype(np.int32),
        scores=scores.astype(np.float16),
        strength=strength.astype(np.float16),
    )


@torch.no_grad()
def precompute_esm_contact_cache(
    records: Sequence[ProteinRecord],
    model,
    alphabet,
    cache_dir: Path,
    device: torch.device,
    batch_size: int = 1,
) -> None:
    """Cache sparse ESM2 contact proxies without storing dense L x L maps."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    max_residues = int(getattr(model, "max_positions", 1024)) - 2
    missing = []
    seen_cache_paths = set()
    for record in records:
        path = contact_cache_path(record.protein_id, cache_dir, record.sequence)
        if path.exists() or path in seen_cache_paths:
            continue
        seen_cache_paths.add(path)
        missing.append(record)
    short = [record for record in missing if len(record.sequence) <= max_residues]
    short.sort(key=lambda record: len(record.sequence))
    batch_converter = alphabet.get_batch_converter()
    for start in range(0, len(short), batch_size):
        batch = short[start : start + batch_size]
        _, _, tokens = batch_converter(
            [(record.protein_id, record.sequence) for record in batch]
        )
        result = model(
            tokens.to(device),
            return_contacts=True,
        )
        contacts = result["contacts"].detach().cpu().numpy()
        for batch_index, record in enumerate(batch):
            length = len(record.sequence)
            pairs, scores, strength = contact_edges_from_map(
                contacts[batch_index, :length, :length]
            )
            save_contact_proxy(
                contact_cache_path(record.protein_id, cache_dir, record.sequence),
                pairs,
                scores,
                strength,
            )
        del result, contacts, tokens
        if start % 16 == 0:
            gc.collect()

    for record in missing:
        if len(record.sequence) <= max_residues:
            continue
        all_pairs = []
        all_scores = []
        strength = np.zeros(len(record.sequence), dtype=np.float32)
        chunks = sequence_chunks(model, record.sequence)
        for start, chunk in chunks:
            _, _, tokens = batch_converter([(record.protein_id, chunk)])
            result = model(
                tokens.to(device),
                return_contacts=True,
            )
            contacts = result["contacts"][0].detach().cpu().numpy()
            pairs, scores, local_strength = contact_edges_from_map(
                contacts[: len(chunk), : len(chunk)]
            )
            if len(pairs):
                all_pairs.append(pairs + start)
                all_scores.append(scores)
            strength[start : start + len(chunk)] = np.maximum(
                strength[start : start + len(chunk)],
                local_strength,
            )
            del result, contacts, tokens
            gc.collect()
        if all_pairs:
            pairs = np.vstack(all_pairs)
            scores = np.concatenate(all_scores)
            unique = {}
            for pair, score in zip(pairs, scores):
                key = tuple(sorted((int(pair[0]), int(pair[1]))))
                unique[key] = max(unique.get(key, 0.0), float(score))
            pairs = np.asarray(list(unique.keys()), dtype=np.int64)
            scores = np.asarray(
                [unique[tuple(pair)] for pair in pairs],
                dtype=np.float32,
            )
        else:
            pairs = np.empty((0, 2), dtype=np.int64)
            scores = np.empty((0,), dtype=np.float32)
        save_contact_proxy(
            contact_cache_path(record.protein_id, cache_dir, record.sequence),
            pairs,
            scores,
            strength,
        )


def build_proxy_edges(
    length: int,
    contact_pairs: np.ndarray,
    contact_scores: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Build sequence plus predicted-contact edges for structure-free inputs."""
    sequential = np.column_stack(
        [
            np.arange(max(0, length - 1), dtype=np.int64),
            np.arange(1, length, dtype=np.int64),
        ]
    )
    if len(contact_pairs):
        pairs = np.vstack([sequential, contact_pairs.astype(np.int64)])
        distances = np.concatenate(
            [
                np.full(len(sequential), 3.8 / 8.0, dtype=np.float32),
                1.0 - np.clip(contact_scores, 0.0, 1.0),
            ]
        )
    else:
        pairs = sequential
        distances = np.full(len(sequential), 3.8 / 8.0, dtype=np.float32)
    if len(pairs) == 0:
        return (
            np.empty((2, 0), dtype=np.int64),
            np.empty((0, 2), dtype=np.float32),
        )
    unique_pairs, unique_indices = np.unique(pairs, axis=0, return_index=True)
    pairs = unique_pairs
    distances = distances[unique_indices]
    gaps = np.minimum(pairs[:, 1] - pairs[:, 0], 50) / 50.0
    directed = np.vstack([pairs, pairs[:, ::-1]])
    directed_distances = np.concatenate([distances, distances])
    directed_gaps = np.concatenate([gaps, gaps])
    return (
        directed.T.astype(np.int64),
        np.column_stack([directed_distances, directed_gaps]).astype(np.float32),
    )


def make_graph(
    record: ProteinRecord,
    esm_model,
    esm_alphabet,
    esm_cache: Path,
    device: torch.device,
    structure_mode: str = "pdb",
    contact_cache: Optional[Path] = None,
) -> GraphRecord:
    if structure_mode in {"pdb", "pdb_or_alphafold"}:
        pdb_sequence, coords, plddt, pdb_dssp = read_pdb_backbone_and_dssp(
            Path(record.pdb_path)
        )
        coords, plddt = align_structure_to_sequence(
            record.sequence, pdb_sequence, coords, plddt
        )
        dssp_labels = (
            align_labels_to_sequence(record.sequence, pdb_sequence, pdb_dssp)
            if pdb_dssp is not None
            else None
        )
        valid = (plddt > 0).astype(np.float32)
        if valid.mean() < 0.8:
            raise ValueError(f"Structure coverage below 80% for {record.protein_id}")
        plddt[plddt <= 0] = 50.0
    elif structure_mode == "esm_proxy":
        if contact_cache is None:
            raise ValueError("contact_cache is required for esm_proxy mode")
        proxy_path = contact_cache_path(
            record.protein_id,
            contact_cache,
            record.sequence,
        )
        if not proxy_path.exists():
            raise FileNotFoundError(f"Missing ESM contact proxy: {proxy_path}")
        proxy = np.load(proxy_path)
        contact_pairs = proxy["pairs"].astype(np.int64)
        contact_scores = proxy["scores"].astype(np.float32)
        strength = proxy["strength"].astype(np.float32)
        # Contact strength is used only as a smooth structure-confidence proxy.
        # It is not AlphaFold pLDDT and is reported as such in data_audit.json.
        plddt = 25.0 + 70.0 * np.clip(strength, 0.0, 1.0)
        dssp_labels = None
    else:
        raise ValueError(f"Unknown structure_mode: {structure_mode}")
    esm = cached_esm(
        esm_model, esm_alphabet, record.sequence, record.protein_id, esm_cache, device
    ).numpy()
    hand = residue_hand_features(record.sequence, plddt)
    dssp = make_dssp_like_features(record.sequence, plddt, dssp_labels)
    x = np.concatenate([esm, hand, dssp], axis=1).astype(np.float32)
    if structure_mode in {"pdb", "pdb_or_alphafold"}:
        edge_index, edge_features = build_edges(coords)
    else:
        edge_index, edge_features = build_proxy_edges(
            len(record.sequence),
            contact_pairs,
            contact_scores,
        )
    return GraphRecord(
        protein_id=record.protein_id,
        sequence=record.sequence,
        node_features=torch.from_numpy(x),
        edge_index=torch.from_numpy(edge_index),
        edge_features=torch.from_numpy(edge_features),
        stage1_label=record.stage1_label,
        stage2_label=record.stage2_label,
        role=record.role,
        source=record.source,
    )


def graph_ref_cache_name(record: ProteinRecord, structure_mode: str = "pdb") -> str:
    return f"{structure_mode}_seq_{sequence_hash(record.sequence)}.pt"


def build_graph_ref(
    record: ProteinRecord,
    esm_model,
    esm_alphabet,
    esm_cache: Path,
    graph_cache: Path,
    device: torch.device,
    structure_mode: str = "pdb",
    contact_cache: Optional[Path] = None,
) -> GraphRef:
    graph_cache.mkdir(parents=True, exist_ok=True)
    cache_path = graph_cache / graph_ref_cache_name(record, structure_mode)
    if not cache_path.exists():
        graph = make_graph(
            record,
            esm_model,
            esm_alphabet,
            esm_cache,
            device,
            structure_mode,
            contact_cache,
        )
        # Half precision cuts disk usage substantially; tensors are converted
        # back to float32 when loaded for the CPU model.
        torch.save(
            {
                "node_features": graph.node_features.half(),
                "edge_index": graph.edge_index,
                "edge_features": graph.edge_features.half(),
            },
            cache_path,
        )
    return GraphRef(
        protein_id=record.protein_id,
        sequence=record.sequence,
        cache_path=str(cache_path),
        stage1_label=record.stage1_label,
        stage2_label=record.stage2_label,
        role=record.role,
        source=record.source,
    )


def load_graph(graphs: Sequence[GraphRef], index: int) -> GraphRecord:
    ref = graphs[index]
    payload = torch.load(ref.cache_path, map_location="cpu", weights_only=True)
    return GraphRecord(
        protein_id=ref.protein_id,
        sequence=ref.sequence,
        node_features=payload["node_features"].float(),
        edge_index=payload["edge_index"].long(),
        edge_features=payload["edge_features"].float(),
        stage1_label=ref.stage1_label,
        stage2_label=ref.stage2_label,
        role=ref.role,
        source=ref.source,
    )


def iter_graph_batches(
    graphs: Sequence[GraphRef],
    indices: Sequence[int],
    max_nodes: int = 12000,
    max_graphs: int = 32,
):
    batch: List[int] = []
    node_count = 0
    for index in indices:
        length = len(graphs[index].sequence)
        if batch and (
            len(batch) >= max_graphs or node_count + length > max_nodes
        ):
            yield batch
            batch = []
            node_count = 0
        batch.append(index)
        node_count += length
    if batch:
        yield batch


def load_graph_batch(
    graphs: Sequence[GraphRef],
    indices: Sequence[int],
):
    records = [load_graph(graphs, index) for index in indices]
    node_features = []
    edge_indices = []
    edge_features = []
    graph_ids = []
    node_offset = 0
    for graph_id, graph in enumerate(records):
        node_features.append(graph.node_features)
        edge_indices.append(graph.edge_index + node_offset)
        edge_features.append(graph.edge_features)
        graph_ids.append(
            torch.full(
                (graph.node_features.shape[0],),
                graph_id,
                dtype=torch.long,
            )
        )
        node_offset += graph.node_features.shape[0]
    return (
        torch.cat(node_features, dim=0),
        torch.cat(edge_indices, dim=1),
        torch.cat(edge_features, dim=0),
        torch.cat(graph_ids, dim=0),
        records,
    )


class GraphAttentionLayer(torch.nn.Module):
    def __init__(self, in_dim: int, out_dim: int, heads: int = 4, dropout: float = 0.2):
        super().__init__()
        self.heads = heads
        self.out_dim = out_dim
        self.linear = torch.nn.Linear(in_dim, heads * out_dim, bias=False)
        self.attn_src = torch.nn.Parameter(torch.empty(heads, out_dim))
        self.attn_dst = torch.nn.Parameter(torch.empty(heads, out_dim))
        self.edge_proj = torch.nn.Linear(2, heads, bias=False)
        self.dropout = torch.nn.Dropout(dropout)
        self.norm = torch.nn.LayerNorm(heads * out_dim)
        torch.nn.init.xavier_uniform_(self.attn_src)
        torch.nn.init.xavier_uniform_(self.attn_dst)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, edge_attr: torch.Tensor):
        src, dst = edge_index
        h = self.linear(x).view(x.shape[0], self.heads, self.out_dim)
        scores = (h[src] * self.attn_src).sum(-1) + (h[dst] * self.attn_dst).sum(-1)
        scores = torch.nn.functional.leaky_relu(scores + self.edge_proj(edge_attr), 0.2)
        destination = dst[:, None].expand(-1, self.heads)
        max_scores = torch.full(
            (x.shape[0], self.heads),
            -torch.inf,
            dtype=scores.dtype,
            device=scores.device,
        )
        max_scores.scatter_reduce_(
            0, destination, scores, reduce="amax", include_self=True
        )
        exp_scores = (scores - max_scores[dst]).exp()
        denominators = torch.zeros_like(max_scores)
        denominators.scatter_add_(0, destination, exp_scores)
        alpha = exp_scores / denominators[dst].clamp_min(1e-12)
        messages = h[src] * alpha.unsqueeze(-1)
        out = torch.zeros_like(h)
        out.index_add_(0, dst, messages)
        out = out.reshape(x.shape[0], self.heads * self.out_dim)
        out = self.norm(out)
        return torch.nn.functional.elu(self.dropout(out)), alpha.detach()


class GraphSAGELayer(torch.nn.Module):
    def __init__(self, in_dim: int, out_dim: int, dropout: float = 0.2):
        super().__init__()
        self.self_linear = torch.nn.Linear(in_dim, out_dim)
        self.neighbor_linear = torch.nn.Linear(in_dim, out_dim)
        self.norm = torch.nn.LayerNorm(out_dim)
        self.dropout = torch.nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor):
        src, dst = edge_index
        neighbor = torch.zeros_like(x)
        degree = torch.zeros(x.shape[0], device=x.device)
        neighbor.index_add_(0, dst, x[src])
        degree.index_add_(0, dst, torch.ones_like(dst, dtype=x.dtype))
        neighbor = neighbor / degree.clamp_min(1.0).unsqueeze(1)
        out = self.self_linear(x) + self.neighbor_linear(neighbor)
        return torch.nn.functional.relu(self.norm(self.dropout(out)))


class PhaseResSAGE(torch.nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 64):
        super().__init__()
        self.input_proj = torch.nn.Sequential(
            torch.nn.Linear(input_dim, hidden_dim),
            torch.nn.LayerNorm(hidden_dim),
            torch.nn.GELU(),
        )
        self.gat = GraphAttentionLayer(hidden_dim, 16, heads=4)
        self.sage1 = GraphSAGELayer(64, hidden_dim)
        self.sage2 = GraphSAGELayer(hidden_dim, hidden_dim)
        self.attn_pool = torch.nn.Linear(hidden_dim, 1)
        self.stage1_head = torch.nn.Linear(hidden_dim * 3, 1)
        self.stage2_head = torch.nn.Linear(hidden_dim * 3, 1)

    def encode(self, graph: GraphRecord):
        x = graph.node_features
        edge_index = graph.edge_index
        edge_attr = graph.edge_features
        x = self.input_proj(x)
        x, alpha = self.gat(x, edge_index, edge_attr)
        x = x + self.sage1(x, edge_index)
        x = x + self.sage2(x, edge_index)
        pool_logits = self.attn_pool(x).squeeze(-1)
        pool = torch.softmax(pool_logits, dim=0)
        pooled = torch.cat(
            [x.mean(0), x.max(0).values, (pool[:, None] * x).sum(0)], dim=0
        )
        gat_node_importance = torch.zeros_like(pool).index_add(
            0, graph.edge_index[1], alpha.mean(1)
        )
        residue_importance = pool * gat_node_importance.clamp_min(1e-8)
        return pooled, residue_importance

    def encode_batch(
        self,
        node_features: torch.Tensor,
        edge_index: torch.Tensor,
        edge_features: torch.Tensor,
        graph_ids: torch.Tensor,
        num_graphs: int,
    ):
        x = self.input_proj(node_features)
        x, alpha = self.gat(x, edge_index, edge_features)
        x = x + self.sage1(x, edge_index)
        x = x + self.sage2(x, edge_index)

        counts = torch.bincount(graph_ids, minlength=num_graphs).to(x.dtype)
        summed = torch.zeros(
            (num_graphs, x.shape[1]), dtype=x.dtype, device=x.device
        )
        summed.index_add_(0, graph_ids, x)
        mean = summed / counts.clamp_min(1.0).unsqueeze(1)

        maximum = torch.full_like(summed, -torch.inf)
        expanded_ids = graph_ids[:, None].expand(-1, x.shape[1])
        maximum.scatter_reduce_(
            0, expanded_ids, x, reduce="amax", include_self=True
        )

        pool_logits = self.attn_pool(x).squeeze(-1)
        max_logits = torch.full(
            (num_graphs,),
            -torch.inf,
            dtype=pool_logits.dtype,
            device=pool_logits.device,
        )
        max_logits.scatter_reduce_(
            0, graph_ids, pool_logits, reduce="amax", include_self=True
        )
        weights = (pool_logits - max_logits[graph_ids]).exp()
        denominators = torch.zeros_like(max_logits)
        denominators.scatter_add_(0, graph_ids, weights)
        weights = weights / denominators[graph_ids].clamp_min(1e-12)
        weighted = torch.zeros_like(summed)
        weighted.index_add_(0, graph_ids, weights[:, None] * x)
        pooled = torch.cat([mean, maximum, weighted], dim=1)
        return pooled, alpha

    def forward(self, graph: GraphRecord):
        pooled, importance = self.encode(graph)
        return self.stage1_head(pooled).squeeze(), self.stage2_head(pooled).squeeze(), importance

    def forward_batch(
        self,
        node_features: torch.Tensor,
        edge_index: torch.Tensor,
        edge_features: torch.Tensor,
        graph_ids: torch.Tensor,
        num_graphs: int,
    ):
        pooled, _ = self.encode_batch(
            node_features,
            edge_index,
            edge_features,
            graph_ids,
            num_graphs,
        )
        return self.stage1_head(pooled).squeeze(-1), self.stage2_head(pooled).squeeze(-1)


def effective_class_weights(labels: Sequence[int], beta: float = 0.999) -> torch.Tensor:
    counts = np.bincount(np.asarray(labels, dtype=np.int64), minlength=2).astype(np.float32)
    weights = (1.0 - beta) / np.maximum(1.0 - np.power(beta, counts), 1e-8)
    weights = weights / weights.mean()
    return torch.tensor(weights, dtype=torch.float32)


def class_balanced_focal(logit: torch.Tensor, label: int, weights: torch.Tensor, gamma: float = 2.0):
    target = torch.tensor(float(label), device=logit.device)
    bce = torch.nn.functional.binary_cross_entropy_with_logits(logit, target, reduction="none")
    probability = torch.sigmoid(logit)
    pt = probability if label == 1 else 1.0 - probability
    return weights[label] * (1.0 - pt).pow(gamma) * bce


def class_balanced_focal_batch(
    logits: torch.Tensor,
    labels: torch.Tensor,
    weights: torch.Tensor,
    gamma: float = 2.0,
):
    labels = labels.long()
    targets = labels.float()
    bce = torch.nn.functional.binary_cross_entropy_with_logits(
        logits, targets, reduction="none"
    )
    probability = torch.sigmoid(logits)
    pt = torch.where(labels == 1, probability, 1.0 - probability)
    return (weights[labels] * (1.0 - pt).pow(gamma) * bce).mean()


def sqrt_weighted_bce_batch(
    logits: torch.Tensor,
    labels: torch.Tensor,
    pos_weight: torch.Tensor,
):
    """BCE with a tempered positive-class weight.

    The positive class receives sqrt(N_negative / N_positive) rather than
    the full inverse-frequency weight. This reduces majority-class bias
    without making the classifier overly aggressive on the minority class.
    """
    targets = labels.float()
    return torch.nn.functional.binary_cross_entropy_with_logits(
        logits,
        targets,
        pos_weight=pos_weight,
        reduction="mean",
    )


def metric_dict(y_true: Sequence[int], y_score: Sequence[float], threshold: float = 0.5):
    y_true = np.asarray(y_true, dtype=np.int64)
    y_score = np.asarray(y_score, dtype=np.float64)
    y_pred = (y_score >= threshold).astype(np.int64)
    out = {
        "n": int(len(y_true)),
        "positive": int(y_true.sum()),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "mcc": float(matthews_corrcoef(y_true, y_pred)),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
    }
    out["auroc"] = float(roc_auc_score(y_true, y_score)) if len(np.unique(y_true)) == 2 else None
    out["auprc"] = float(average_precision_score(y_true, y_score)) if y_true.sum() else None
    return out


def train_one(
    model: PhaseResSAGE,
    graphs: Sequence[GraphRef],
    train_idx: List[int],
    val_idx: List[int],
    task: int,
    device: torch.device,
    epochs: int,
    patience: int,
    seed: int,
    max_batch_nodes: int = 12000,
    max_batch_graphs: int = 32,
    loss_name: str = "focal",
) -> Tuple[PhaseResSAGE, float, Dict[str, float]]:
    set_seed(seed)
    model.to(device)
    labels = [
        (graphs[i].stage1_label if task == 1 else int(graphs[i].stage2_label))
        for i in train_idx
    ]
    counts = np.bincount(np.asarray(labels, dtype=np.int64), minlength=2).astype(np.float32)
    if loss_name == "sqrt_bce":
        pos_weight = torch.tensor(
            math.sqrt(float(counts[0]) / max(float(counts[1]), 1.0)),
            dtype=torch.float32,
            device=device,
        )
        weights = None
    elif loss_name == "focal":
        weights = effective_class_weights(labels).to(device)
        pos_weight = None
    else:
        raise ValueError(f"Unknown loss_name: {loss_name}")
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)
    best_state = None
    best_score = -float("inf")
    no_improve = 0
    for epoch in range(1, epochs + 1):
        model.train()
        epoch_indices = train_idx.copy()
        random.shuffle(epoch_indices)
        losses = []
        for batch_indices in iter_graph_batches(
            graphs,
            epoch_indices,
            max_nodes=max_batch_nodes,
            max_graphs=max_batch_graphs,
        ):
            node_features, edge_index, edge_features, graph_ids, records = load_graph_batch(
                graphs, batch_indices
            )
            labels_batch = torch.tensor(
                [
                    record.stage1_label
                    if task == 1
                    else int(record.stage2_label)
                    for record in records
                ],
                dtype=torch.long,
                device=device,
            )
            node_features = node_features.to(device)
            edge_index = edge_index.to(device)
            edge_features = edge_features.to(device)
            graph_ids = graph_ids.to(device)
            optimizer.zero_grad(set_to_none=True)
            out1, out2 = model.forward_batch(
                node_features,
                edge_index,
                edge_features,
                graph_ids,
                len(records),
            )
            logit = out1 if task == 1 else out2
            if loss_name == "sqrt_bce":
                loss = sqrt_weighted_bce_batch(logit, labels_batch, pos_weight)
            else:
                loss = class_balanced_focal_batch(logit, labels_batch, weights)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        model.eval()
        val_true, val_score = [], []
        with torch.no_grad():
            for batch_indices in iter_graph_batches(
                graphs,
                val_idx,
                max_nodes=max_batch_nodes,
                max_graphs=max_batch_graphs,
            ):
                node_features, edge_index, edge_features, graph_ids, records = load_graph_batch(
                    graphs, batch_indices
                )
                out1, out2 = model.forward_batch(
                    node_features.to(device),
                    edge_index.to(device),
                    edge_features.to(device),
                    graph_ids.to(device),
                    len(records),
                )
                val_true.extend(
                    [
                        record.stage1_label
                        if task == 1
                        else int(record.stage2_label)
                        for record in records
                    ]
                )
                val_score.extend(
                    torch.sigmoid(out1 if task == 1 else out2)
                    .detach()
                    .cpu()
                    .tolist()
                )
        if len(set(val_true)) == 2:
            score = average_precision_score(val_true, val_score)
        else:
            score = -float(np.mean(losses))
        if score > best_score:
            best_score = score
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
        if no_improve >= patience:
            break
    if best_state is not None:
        model.load_state_dict(best_state)
    train_info = {
        "epochs": epoch,
        "best_val_auprc": best_score,
        "loss": loss_name,
        "train_negative": int(counts[0]),
        "train_positive": int(counts[1]),
    }
    if pos_weight is not None:
        train_info["positive_weight"] = float(pos_weight.detach().cpu())
    else:
        train_info["effective_class_weights"] = [float(x) for x in weights.detach().cpu()]
    return model, best_score, train_info


def predict(
    model: PhaseResSAGE,
    graphs: Sequence[GraphRef],
    indices: Iterable[int],
    task: int,
    device: torch.device,
    max_batch_nodes: int = 12000,
    max_batch_graphs: int = 32,
):
    model.eval()
    y_true, y_score, ids = [], [], []
    with torch.no_grad():
        selected = list(indices)
        for batch_indices in iter_graph_batches(
            graphs,
            selected,
            max_nodes=max_batch_nodes,
            max_graphs=max_batch_graphs,
        ):
            node_features, edge_index, edge_features, graph_ids, records = load_graph_batch(
                graphs, batch_indices
            )
            out1, out2 = model.forward_batch(
                node_features.to(device),
                edge_index.to(device),
                edge_features.to(device),
                graph_ids.to(device),
                len(records),
            )
            y_true.extend(
                [
                    record.stage1_label if task == 1 else int(record.stage2_label)
                    for record in records
                ]
            )
            y_score.extend(
                torch.sigmoid(out1 if task == 1 else out2).detach().cpu().tolist()
            )
            ids.extend(record.protein_id for record in records)
    return y_true, y_score, ids


def choose_threshold(
    y_true: Sequence[int],
    y_score: Sequence[float],
    objective_name: str = "mcc_f1",
    min_recall: float = 0.2,
) -> float:
    candidates = np.linspace(0.1, 0.9, 81)
    best = (0.5, -float("inf"))
    scored = []
    for threshold in candidates:
        metrics = metric_dict(y_true, y_score, float(threshold))
        if objective_name == "accuracy_recall":
            if metrics["recall"] + 1e-12 < min_recall:
                continue
            # Accuracy is primary, while MCC and recall break near-ties.
            objective = (
                metrics["accuracy"],
                metrics["mcc"],
                metrics["recall"],
                -float(threshold),
            )
            scored.append((float(threshold), objective))
        elif objective_name == "mcc_f1":
            objective = metrics["mcc"] + metrics["f1"]
            scored.append((float(threshold), (objective, metrics["recall"], -float(threshold))))
        else:
            raise ValueError(f"Unknown threshold objective: {objective_name}")
    if scored:
        best_threshold, _ = max(scored, key=lambda item: item[1])
        return best_threshold
    # If a recall floor is infeasible on the validation set, use the
    # threshold that maximizes MCC+F1 rather than silently predicting one class.
    fallback = []
    for threshold in candidates:
        metrics = metric_dict(y_true, y_score, float(threshold))
        fallback.append(
            (
                float(threshold),
                metrics["mcc"] + metrics["f1"],
                metrics["recall"],
            )
        )
    if fallback:
        return max(fallback, key=lambda item: (item[1], item[2], -item[0]))[0]
    return best[0]


def bootstrap_ci(y_true: Sequence[int], y_score: Sequence[float], threshold: float, seed: int, n_boot: int = 500):
    rng = np.random.default_rng(seed)
    values = []
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)
    for _ in range(n_boot):
        idx = rng.integers(0, len(y_true), len(y_true))
        if len(np.unique(y_true[idx])) < 2:
            continue
        values.append(metric_dict(y_true[idx], y_score[idx], threshold)["f1"])
    if not values:
        return [None, None]
    return [float(np.percentile(values, 2.5)), float(np.percentile(values, 97.5))]


def load_structure_manifest(path: Optional[Path]) -> Dict[str, dict]:
    if path is None:
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    records = payload.get("records", payload)
    return records if isinstance(records, dict) else {}


def build_records(
    data_root: Path,
    structure_manifest_path: Optional[Path] = None,
) -> Tuple[List[ProteinRecord], List[ProteinRecord]]:
    structure_manifest = load_structure_manifest(structure_manifest_path)

    def pdb_file(directory: Path, stem: str) -> str:
        target = f"{stem}.pdb".lower()
        for candidate in directory.glob("*.pdb"):
            if candidate.name.lower() == target and candidate.stat().st_size > 0:
                return str(candidate)
        shared = directory.parent.parent / "structures" / f"{stem}.pdb"
        if shared.exists() and shared.stat().st_size > 1000:
            return str(shared)
        # Keep a deterministic path so the audit records the missing file.
        return str(directory / f"{stem}.pdb")

    def structure_file(sequence: str, directory: Path, stem: str) -> str:
        item = structure_manifest.get(sequence_hash(sequence), {})
        path = item.get("path") if isinstance(item, dict) else None
        if path:
            candidate = Path(path)
            if not candidate.is_absolute() and structure_manifest_path is not None:
                candidate = structure_manifest_path.parent / candidate
            if candidate.exists() and candidate.stat().st_size > 1000:
                return str(candidate.resolve())
        return pdb_file(directory, stem)

    stage1_llps_path = data_root / "stage1_llps.fasta"
    stage1_scaffold = read_fasta(
        stage1_llps_path if stage1_llps_path.exists() else data_root / "stage1_scaffold.fasta"
    )
    stage1_nonllps = read_fasta(data_root / "stage1_nonllps.fasta")
    stage2_roles = read_fasta(data_root / "stage2_roles.fasta")
    stage1: List[ProteinRecord] = []
    for key, (description, sequence) in stage1_scaffold.items():
        stem = pdb_stem_from_description(description)
        is_llps = description.lower().startswith("llps|")
        role = "LLPS" if is_llps else "scaffold"
        stage1.append(
                ProteinRecord(
                stable_record_id(description, sequence, stem),
                sequence,
                structure_file(
                    sequence,
                    data_root / "stage1" / "scaffold_pdb",
                    stem,
                ),
                1,
                None,
                role,
                "stage1",
            )
        )
    for key, (description, sequence) in stage1_nonllps.items():
        stem = pdb_stem_from_description(description)
        stage1.append(
            ProteinRecord(
                stable_record_id(description, sequence, stem),
                sequence,
                structure_file(
                    sequence,
                    data_root / "stage1" / "nonllps_pdb",
                    stem,
                ),
                0,
                None,
                "non-LLPS",
                "stage1",
            )
        )
    stage2: List[ProteinRecord] = []
    for key, (description, sequence) in stage2_roles.items():
        role = "scaffold" if description.lower().startswith("scaffold|") else "client"
        label = 1 if role == "scaffold" else 0
        stem = pdb_stem_from_description(description)
        pdb_dir = "scaffold_pdb" if role == "scaffold" else "client_pdb"
        stage2.append(
            ProteinRecord(
                stable_record_id(description, sequence, stem),
                sequence,
                structure_file(sequence, data_root / "stage2" / pdb_dir, stem),
                1,
                label,
                role,
                "stage2",
            )
        )
    return stage1, stage2


def prepare_data_root(data_root: Path, repo_root: Path) -> None:
    """Copy the downloaded public FASTA files into stable experiment names."""
    mapping = {
        "stage1_scaffold.fasta": repo_root / "tmp" / "scaffold_example.fasta",
        "stage1_nonllps.fasta": repo_root / "tmp" / "non-LLPS_example.fasta",
        "stage2_roles.fasta": repo_root / "tmp" / "idr_llps_example.fasta",
    }
    for target, source in mapping.items():
        target_path = data_root / target
        if not target_path.exists():
            target_path.write_text(source.read_text(encoding="utf-8"), encoding="utf-8")


def save_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--data-root", type=Path, default=None)
    parser.add_argument("--out-root", type=Path, default=None)
    parser.add_argument(
        "--cache-root",
        type=Path,
        default=None,
        help="Optional shared directory for ESM and graph caches.",
    )
    parser.add_argument(
        "--structure-mode",
        choices=["pdb", "pdb_or_alphafold", "esm_proxy"],
        default="pdb",
        help="Use PDB/DSSP, PDB-first plus AlphaFold fallback, or ESM2 proxy graphs.",
    )
    parser.add_argument(
        "--structure-manifest",
        type=Path,
        default=None,
        help="JSON manifest produced by prepare_phaseom_structures.py.",
    )
    parser.add_argument(
        "--encoder",
        choices=["esm2_t6_8M_UR50D", "esm2_t33_650M_UR50D"],
        default="esm2_t6_8M_UR50D",
        help="ESM2 checkpoint. The t6 checkpoint is the low-memory default.",
    )
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--patience", type=int, default=12)
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch-nodes", type=int, default=12000)
    parser.add_argument("--batch-graphs", type=int, default=32)
    parser.add_argument(
        "--loss",
        choices=["sqrt_bce", "focal"],
        default="sqrt_bce",
        help="Training loss: tempered sqrt-frequency BCE or the focal ablation.",
    )
    parser.add_argument(
        "--threshold-objective",
        choices=["accuracy_recall", "mcc_f1"],
        default="accuracy_recall",
        help="Validation threshold objective.",
    )
    parser.add_argument(
        "--min-recall",
        type=float,
        default=0.2,
        help="Minimum validation positive recall for accuracy_recall thresholding.",
    )
    args = parser.parse_args()

    repo_root = args.repo_root.resolve()
    data_root = (args.data_root or repo_root / "data" / "phaseom_pilot").resolve()
    out_root = (args.out_root or repo_root / "results" / "phase_res_sage").resolve()
    out_root.mkdir(parents=True, exist_ok=True)
    cache_root = (args.cache_root or out_root).resolve()
    cache_root.mkdir(parents=True, exist_ok=True)
    cache_dir = cache_root / f"{args.encoder}_cache"
    contact_cache = cache_root / "esm2_contact_cache"
    graph_cache = cache_root / "graph_cache"
    prepare_data_root(data_root, repo_root)
    device = torch.device(args.device)
    set_seed(args.seeds[0])
    esm_model, esm_alphabet = load_esm_model(args.encoder, cache_dir, device)

    structure_manifest_path = (
        args.structure_manifest.resolve()
        if args.structure_manifest is not None
        else None
    )
    stage1_records, stage2_records = build_records(
        data_root,
        structure_manifest_path,
    )
    all_records = stage1_records + stage2_records
    # Deduplicate by exact sequence and PDB path while preserving task labels.
    unique: Dict[Tuple[str, str, str], ProteinRecord] = {}
    for record in all_records:
        unique[(record.source, record.protein_id, record.pdb_path)] = record
    unique_records = list(unique.values())
    precompute_esm_cache(
        unique_records,
        esm_model,
        esm_alphabet,
        cache_dir,
        device,
    )
    if args.structure_mode == "esm_proxy":
        precompute_esm_contact_cache(
            unique_records,
            esm_model,
            esm_alphabet,
            contact_cache,
            device,
        )
    graphs: List[GraphRef] = []
    skipped = []
    for record in unique_records:
        try:
            graphs.append(
                build_graph_ref(
                    record,
                    esm_model,
                    esm_alphabet,
                    cache_dir,
                    graph_cache,
                    device,
                    args.structure_mode,
                    contact_cache,
                )
            )
        except Exception as exc:
            skipped.append({"protein_id": record.protein_id, "error": str(exc)})
    structure_source_counts = {}
    if structure_manifest_path is not None:
        for item in load_structure_manifest(structure_manifest_path).values():
            if not isinstance(item, dict):
                continue
            source = item.get("source", "unknown")
            structure_source_counts[source] = structure_source_counts.get(source, 0) + 1
    save_json(out_root / "data_audit.json", {
        "stage1_requested": len(stage1_records),
        "stage2_requested": len(stage2_records),
        "graphs_built": len(graphs),
        "graphs_by_source": {
            "stage1": sum(g.source == "stage1" for g in graphs),
            "stage2": sum(g.source == "stage2" for g in graphs),
        },
        "skipped": skipped,
        "model": "PhaseResSAGE",
        "encoder": args.encoder,
        "encoder_dimension": int(getattr(esm_model, "embed_dim", 0)),
        "structure_mode": args.structure_mode,
        "structure_graph": (
            "RCSB PDB-first; AlphaFold2-derived PDB fallback; sequence edges plus <=8A contacts"
            if args.structure_mode == "pdb_or_alphafold"
            else (
                "PDB C-alpha; sequence edges plus <=8A contacts"
                if args.structure_mode == "pdb"
                else f"{args.encoder} sparse predicted contacts plus sequence edges"
            )
        ),
        "dssp_note": (
            "pydssp is applied to the selected RCSB PDB or AlphaFold2-derived PDB chain; incomplete structures use an explicit pLDDT-informed fallback"
            if args.structure_mode in {"pdb", "pdb_or_alphafold"}
            else "DSSP-like fallback from ESM2 contact-strength confidence; not experimental or AlphaFold pLDDT"
        ),
        "structure_manifest": (
            str(structure_manifest_path)
            if structure_manifest_path is not None
            else None
        ),
        "structure_source_counts": structure_source_counts,
        "contact_cache": str(contact_cache) if args.structure_mode == "esm_proxy" else None,
        "graph_cache": str(graph_cache),
    })
    if not graphs:
        raise RuntimeError("No graphs were built; inspect data_audit.json")
    input_dim = int(load_graph(graphs, 0).node_features.shape[1])
    results = {
        "config": {
            "model": "PhaseResSAGE",
            "encoder": args.encoder,
            "input_dim": int(input_dim),
            "device": str(device),
            "epochs": args.epochs,
            "seeds": args.seeds,
            "loss": (
                "sqrt-frequency weighted BCE; positive weight=sqrt(N_negative/N_positive)"
                if args.loss == "sqrt_bce"
                else "class-balanced focal loss; beta=0.999; gamma=2"
            ),
            "loss_name": args.loss,
            "threshold_objective": args.threshold_objective,
            "minimum_validation_recall": args.min_recall,
            "structure_mode": args.structure_mode,
        },
        "stage1": {},
        "stage2": {},
    }

    def run_task(task_name: str, task: int, selected: List[int]):
        if len(selected) < 4:
            return {"status": "insufficient_graphs", "n": len(selected)}
        labels = [graphs[i].stage1_label if task == 1 else graphs[i].stage2_label for i in selected]
        if len(set(labels)) < 2:
            return {"status": "single_class", "n": len(selected), "labels": labels}
        n_train = max(2, int(len(selected) * 0.6))
        n_val = max(2, int(len(selected) * 0.2))
        n_test = len(selected) - n_train - n_val
        if n_test < 2:
            n_test = 2
            n_train = len(selected) - n_val - n_test
        # Keep train/validation/test disjoint and preserve both classes whenever
        # the class counts allow it. This is especially important for the small
        # public-example pilot, where an accidental overlap is very misleading.
        train_val, test_idx = train_test_split(
            list(selected),
            test_size=n_test,
            stratify=labels,
            random_state=2026 + task,
        )
        train_labels = [
            graphs[i].stage1_label if task == 1 else graphs[i].stage2_label
            for i in train_val
        ]
        train_idx, val_idx = train_test_split(
            train_val,
            test_size=n_val,
            stratify=train_labels,
            random_state=3030 + task,
        )
        seed_runs = []
        all_seed_predictions = {}
        for seed in args.seeds:
            model = PhaseResSAGE(input_dim)
            model, _, train_info = train_one(
                model, graphs, train_idx.copy(), val_idx, task, device,
                args.epochs, args.patience, seed,
                args.batch_nodes,
                args.batch_graphs,
                args.loss,
            )
            val_true, val_score, _ = predict(
                model,
                graphs,
                val_idx,
                task,
                device,
                args.batch_nodes,
                args.batch_graphs,
            )
            threshold = (
                choose_threshold(
                    val_true,
                    val_score,
                    args.threshold_objective,
                    args.min_recall,
                )
                if len(set(val_true)) == 2
                else 0.5
            )
            test_true, test_score, ids = predict(
                model,
                graphs,
                test_idx,
                task,
                device,
                args.batch_nodes,
                args.batch_graphs,
            )
            metrics = metric_dict(test_true, test_score, threshold)
            metrics["threshold"] = threshold
            metrics["threshold_objective"] = args.threshold_objective
            metrics["minimum_validation_recall"] = args.min_recall
            if task == 2:
                metrics["scaffold_recall"] = metrics["recall"]
            metrics["f1_bootstrap_95ci"] = bootstrap_ci(test_true, test_score, threshold, seed)
            seed_runs.append({"seed": seed, "train": train_info, "test": metrics, "ids": ids})
            all_true, all_score, all_ids = predict(
                model,
                graphs,
                selected,
                task,
                device,
                args.batch_nodes,
                args.batch_graphs,
            )
            all_seed_predictions[seed] = {
                "true": all_true,
                "score": all_score,
                "ids": all_ids,
                "threshold": threshold,
            }
            torch.save(model.state_dict(), out_root / f"{task_name}_seed{seed}.pt")
        summary = {}
        summary_keys = [
            "accuracy",
            "balanced_accuracy",
            "precision",
            "recall",
            "f1",
            "mcc",
            "auroc",
            "auprc",
        ]
        if task == 2:
            summary_keys.append("scaffold_recall")
        for key in summary_keys:
            values = [r["test"].get(key) for r in seed_runs if r["test"].get(key) is not None]
            summary[key] = {"mean": float(np.mean(values)), "std": float(np.std(values, ddof=1)) if len(values) > 1 else 0.0}
        prediction_path = out_root / f"{task_name}_predictions.csv"
        split_by_index = {
            **{index: "train" for index in train_idx},
            **{index: "validation" for index in val_idx},
            **{index: "test" for index in test_idx},
        }
        threshold_mean = float(
            np.mean([item["threshold"] for item in all_seed_predictions.values()])
        )
        prediction_fields = [
            "protein_id",
            "source",
            "role",
            "sequence_length",
            "label",
            "split",
            "mean_score",
            "mean_predicted_label",
            "mean_predicted_role",
        ] + [f"score_seed{seed}" for seed in args.seeds]
        with prediction_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=prediction_fields)
            writer.writeheader()
            reference = all_seed_predictions[args.seeds[0]]
            for row_index, graph_index in enumerate(selected):
                seed_scores = {
                    seed: float(all_seed_predictions[seed]["score"][row_index])
                    for seed in args.seeds
                }
                mean_score = float(np.mean(list(seed_scores.values())))
                label = (
                    int(graphs[graph_index].stage1_label)
                    if task == 1
                    else int(graphs[graph_index].stage2_label)
                )
                predicted_label = int(mean_score >= threshold_mean)
                writer.writerow(
                    {
                        "protein_id": reference["ids"][row_index],
                        "source": graphs[graph_index].source,
                        "role": graphs[graph_index].role,
                        "sequence_length": len(graphs[graph_index].sequence),
                        "label": label,
                        "split": split_by_index[graph_index],
                        "mean_score": mean_score,
                        "mean_predicted_label": predicted_label,
                        "mean_predicted_role": (
                            ("LLPS" if predicted_label else "non-LLPS")
                            if task == 1
                            else ("scaffold" if predicted_label else "client")
                        ),
                        **{
                            f"score_seed{seed}": seed_scores[seed]
                            for seed in args.seeds
                        },
                    }
                )
        return {
            "status": "complete",
            "n": len(selected),
            "split": {"train": len(train_idx), "validation": len(val_idx), "test": len(test_idx)},
            "summary": summary,
            "runs": seed_runs,
            "prediction_file": str(prediction_path),
            "prediction_threshold_mean": threshold_mean,
        }

    stage1_indices = [
        i for i, g in enumerate(graphs)
        if g.source == "stage1" and g.role in {"LLPS", "scaffold", "non-LLPS"}
    ]
    stage2_indices = [
        i for i, g in enumerate(graphs)
        if g.source == "stage2"
        and g.role in {"scaffold", "client"}
        and g.stage2_label is not None
    ]
    results["stage1"] = run_task("stage1_llps", 1, stage1_indices)
    results["stage2"] = run_task("stage2_role", 2, stage2_indices)
    save_json(out_root / "results.json", results)
    print(json.dumps(results, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
