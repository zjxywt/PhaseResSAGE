# -*- coding: utf-8 -*-
"""Prepare PDB-first structures for the PhaseResSAGE LLPS experiment.

The structure policy implemented here is:

1. Search the RCSB PDB by protein sequence.
2. Download an experimental PDB entry when a chain passes the identity and
   query-coverage thresholds.
3. If no suitable PDB chain is found, try the AlphaFold Protein Structure
   Database for an accession present in the FASTA header or an optional mapping.
4. Keep unresolved records in the manifest instead of silently substituting an
   ESM contact proxy.

AlphaFold Database files are AlphaFold2-derived predictions. For records with
no PDB hit and no UniProt accession, this script records
``requires_local_alphafold2``; it does not invent an accession or claim that a
local AlphaFold2 prediction was completed.
"""

import argparse
import concurrent.futures
import csv
import hashlib
import json
import re
import time
import urllib.error
import urllib.request
from pathlib import Path

import numpy as np
from Bio import SeqIO
from Bio.Align import PairwiseAligner
from Bio.PDB import PDBIO, PDBParser


RCSB_SEARCH_URL = "https://search.rcsb.org/rcsbsearch/v2/query"
RCSB_ENTITY_URL = "https://data.rcsb.org/rest/v1/core/polymer_entity/{entry}/{entity}"
RCSB_PDB_URL = "https://files.rcsb.org/download/{entry}.pdb"
ALPHAFOLD_URLS = (
    "https://alphafold.ebi.ac.uk/files/AF-{accession}-F1-model_v6.pdb",
    "https://alphafold.ebi.ac.uk/files/AF-{accession}-F1-model_v4.pdb",
)
AA20 = set("ACDEFGHIKLMNPQRSTVWY")
THREE_TO_ONE = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C",
    "GLN": "Q", "GLU": "E", "GLY": "G", "HIS": "H", "ILE": "I",
    "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F", "PRO": "P",
    "SER": "S", "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V",
}


def sequence_hash(sequence):
    return hashlib.sha1(sequence.upper().encode("utf-8")).hexdigest()[:20]


def read_records(paths):
    records = []
    seen = set()
    for path in paths:
        for record in SeqIO.parse(str(path), "fasta"):
            sequence = str(record.seq).upper().replace(" ", "").replace("\n", "")
            if not sequence or not set(sequence).issubset(AA20):
                continue
            key = sequence_hash(sequence)
            if key in seen:
                continue
            seen.add(key)
            records.append(
                {
                    "sequence_hash": key,
                    "description": record.description.strip(),
                    "sequence": sequence,
                }
            )
    return records


def accession_candidates(description):
    candidates = []
    fields = [field.strip() for field in description.split("|") if field.strip()]
    for field in fields:
        if re.fullmatch(r"[OPQ][0-9][A-Z0-9]{3,8}", field.upper()):
            candidates.append(field.upper())
        elif re.fullmatch(r"[A-NR-Z][0-9][A-Z0-9]{4,9}", field.upper()):
            candidates.append(field.upper())
    return list(dict.fromkeys(candidates))


def load_accession_mapping(path):
    if path is None:
        return {}
    mapping = {}
    if path.suffix.lower() == ".json":
        payload = json.loads(path.read_text(encoding="utf-8"))
        for key, value in payload.items():
            if isinstance(value, dict):
                value = value.get("accession")
            mapping[str(key)] = str(value)
        return mapping
    with path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle, delimiter="\t"):
            key = row.get("sequence_hash") or row.get("description")
            accession = row.get("accession")
            if key and accession:
                mapping[key] = accession.strip().upper()
    return mapping


def request_json(url, payload=None, retries=4):
    last_error = ""
    for attempt in range(retries):
        try:
            if payload is None:
                request = urllib.request.Request(
                    url,
                    headers={"User-Agent": "PhaseResSAGE/1.0"},
                )
            else:
                request = urllib.request.Request(
                    url,
                    data=json.dumps(payload).encode("utf-8"),
                    headers={
                        "Content-Type": "application/json",
                        "User-Agent": "PhaseResSAGE/1.0",
                    },
                    method="POST",
                )
            with urllib.request.urlopen(request, timeout=90) as response:
                raw = response.read().decode("utf-8")
                return json.loads(raw) if raw.strip() else {}
        except Exception as exc:
            last_error = repr(exc)
            time.sleep(min(2 ** attempt, 8))
    raise RuntimeError(last_error)


def download_bytes(url, path, retries=4):
    if path.exists() and path.stat().st_size > 1000:
        return "cached"
    temporary = path.with_suffix(path.suffix + ".download")
    last_error = ""
    for attempt in range(retries):
        try:
            request = urllib.request.Request(
                url,
                headers={"User-Agent": "PhaseResSAGE/1.0"},
            )
            with urllib.request.urlopen(request, timeout=180) as response:
                if response.status != 200:
                    raise RuntimeError("HTTP {}".format(response.status))
                with temporary.open("wb") as handle:
                    while True:
                        block = response.read(1024 * 1024)
                        if not block:
                            break
                        handle.write(block)
            if temporary.stat().st_size <= 1000:
                raise RuntimeError("downloaded file is unexpectedly small")
            temporary.replace(path)
            return "downloaded"
        except urllib.error.HTTPError as exc:
            last_error = "HTTP {}".format(exc.code)
            if exc.code == 404:
                break
        except Exception as exc:
            last_error = repr(exc)
        if temporary.exists():
            temporary.unlink()
        time.sleep(min(2 ** attempt, 8))
    if temporary.exists():
        temporary.unlink()
    raise RuntimeError(last_error)


def rcsb_sequence_search(sequence, identity_cutoff, rows):
    query = {
        "query": {
            "type": "terminal",
            "service": "sequence",
            "parameters": {
                "evalue_cutoff": 1e-3,
                "identity_cutoff": identity_cutoff,
                "sequence_type": "protein",
                "value": sequence,
            },
        },
        "return_type": "polymer_entity",
        "request_options": {
            "paginate": {"start": 0, "rows": rows},
            "results_content_type": ["experimental"],
        },
    }
    payload = request_json(RCSB_SEARCH_URL, query)
    return payload.get("result_set", [])


def clean_pdb_sequence(sequence):
    return re.sub(r"[^A-Z]", "", sequence.upper())


def alignment_stats(query, candidate):
    aligner = PairwiseAligner()
    aligner.mode = "local"
    aligner.match_score = 2.0
    aligner.mismatch_score = -1.0
    aligner.open_gap_score = -4.0
    aligner.extend_gap_score = -0.5
    alignment = aligner.align(query, candidate)[0]
    aligned_query = 0
    matches = 0
    for (q_start, q_end), (c_start, c_end) in zip(
        alignment.aligned[0], alignment.aligned[1]
    ):
        width = min(q_end - q_start, c_end - c_start)
        aligned_query += width
        matches += sum(
            query[q_start + i] == candidate[c_start + i]
            for i in range(width)
        )
    query_coverage = aligned_query / max(1, len(query))
    candidate_coverage = aligned_query / max(1, len(candidate))
    identity = matches / max(1, aligned_query)
    return {
        "identity": float(identity),
        "query_coverage": float(query_coverage),
        "candidate_coverage": float(candidate_coverage),
        "score": float(identity * query_coverage),
    }


def entity_metadata(identifier):
    entry, entity = identifier.split("_", 1)
    return request_json(RCSB_ENTITY_URL.format(entry=entry, entity=entity))


def chain_sequence(chain):
    residues = []
    for residue in chain.get_residues():
        if residue.id[0] != " ":
            continue
        if residue.get_resname().strip().upper() not in THREE_TO_ONE:
            continue
        if not residue.has_id("CA"):
            continue
        residues.append(THREE_TO_ONE[residue.get_resname().strip().upper()])
    return "".join(residues)


def select_best_chain(pdb_path, query):
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("candidate", str(pdb_path))
    best = None
    for model in structure:
        for chain in model:
            candidate = chain_sequence(chain)
            if not candidate:
                continue
            stats = alignment_stats(query, candidate)
            item = {
                "model_id": model.id,
                "chain_id": chain.id,
                "stats": stats,
                "chain": chain,
            }
            if best is None or stats["score"] > best["stats"]["score"]:
                best = item
        break
    return best


def save_chain(chain, path):
    io = PDBIO()
    io.set_structure(chain)
    io.save(str(path))


def choose_pdb_structure(record, out_root, identity_cutoff, min_coverage, rows):
    results = rcsb_sequence_search(record["sequence"], identity_cutoff, rows)
    candidate_rows = []
    raw_dir = out_root / "raw_rcsb"
    structure_dir = out_root / "structures"
    raw_dir.mkdir(parents=True, exist_ok=True)
    structure_dir.mkdir(parents=True, exist_ok=True)
    for result in results:
        identifier = result.get("identifier", "")
        if "_" not in identifier:
            continue
        try:
            metadata = entity_metadata(identifier)
        except Exception:
            continue
        entity_sequence = clean_pdb_sequence(
            metadata.get("entity_poly", {}).get("pdbx_seq_one_letter_code_can", "")
        )
        if not entity_sequence:
            continue
        stats = alignment_stats(record["sequence"], entity_sequence)
        candidate_rows.append((stats["score"], identifier, stats))
    candidate_rows.sort(reverse=True, key=lambda item: item[0])
    for _, identifier, canonical_stats in candidate_rows:
        if (
            canonical_stats["identity"] < identity_cutoff
            or canonical_stats["query_coverage"] < min_coverage
        ):
            continue
        entry = identifier.split("_", 1)[0]
        raw_path = raw_dir / "{}.pdb".format(entry)
        try:
            download_bytes(RCSB_PDB_URL.format(entry=entry), raw_path)
            selected = select_best_chain(raw_path, record["sequence"])
        except Exception:
            continue
        if selected is None:
            continue
        stats = selected["stats"]
        if stats["identity"] < identity_cutoff or stats["query_coverage"] < min_coverage:
            continue
        final_path = structure_dir / "{}.pdb".format(record["sequence_hash"])
        save_chain(selected["chain"], final_path)
        return {
            "status": "resolved",
            "source": "rcsb_pdb",
            "path": str(final_path.resolve()),
            "pdb_id": entry,
            "entity_id": identifier,
            "chain_id": selected["chain_id"],
            "identity": stats["identity"],
            "query_coverage": stats["query_coverage"],
            "candidate_coverage": stats["candidate_coverage"],
        }
    return {"status": "no_suitable_pdb", "searched": len(results)}


def choose_alphafold_structure(record, out_root, accessions):
    structure_dir = out_root / "structures"
    structure_dir.mkdir(parents=True, exist_ok=True)
    raw_dir = out_root / "raw_alphafold"
    raw_dir.mkdir(parents=True, exist_ok=True)
    for accession in accessions:
        for template in ALPHAFOLD_URLS:
            raw_path = raw_dir / "{}.pdb".format(accession)
            try:
                download_bytes(template.format(accession=accession), raw_path)
                selected = select_best_chain(raw_path, record["sequence"])
            except Exception:
                continue
            if selected is None:
                continue
            stats = selected["stats"]
            if stats["query_coverage"] < 0.80 or stats["identity"] < 0.90:
                continue
            final_path = structure_dir / "{}.pdb".format(record["sequence_hash"])
            save_chain(selected["chain"], final_path)
            return {
                "status": "resolved",
                "source": "alphafold_db_af2",
                "path": str(final_path.resolve()),
                "accession": accession,
                "model": template.rsplit("model_", 1)[-1].replace(".pdb", ""),
                "identity": stats["identity"],
                "query_coverage": stats["query_coverage"],
                "candidate_coverage": stats["candidate_coverage"],
            }
    return {"status": "alphafold_missing", "accessions_tried": accessions}


def resolve_one(record, out_root, accession_mapping, identity_cutoff, min_coverage, rows):
    try:
        pdb_result = choose_pdb_structure(
            record,
            out_root,
            identity_cutoff,
            min_coverage,
            rows,
        )
    except Exception as exc:
        pdb_result = {"status": "rcsb_error", "error": repr(exc)}
    if pdb_result.get("status") == "resolved":
        return dict(record, **pdb_result)

    accessions = accession_candidates(record["description"])
    accessions += accession_candidates(accession_mapping.get(record["sequence_hash"], ""))
    accessions = list(dict.fromkeys(accessions))
    if accessions:
        try:
            af_result = choose_alphafold_structure(record, out_root, accessions)
        except Exception as exc:
            af_result = {"status": "alphafold_error", "error": repr(exc)}
        if af_result.get("status") == "resolved":
            result = dict(record, **af_result)
            result["pdb_search"] = pdb_result
            return result
    return dict(
        record,
        status="requires_local_alphafold2",
        source="unresolved",
        path=None,
        pdb_search=pdb_result,
        accessions=accessions,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=Path("data") / "phaseom_attachment")
    parser.add_argument("--out-root", type=Path, default=None)
    parser.add_argument("--accession-map", type=Path, default=None)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--rows", type=int, default=20)
    parser.add_argument("--pdb-identity", type=float, default=0.95)
    parser.add_argument("--min-coverage", type=float, default=0.80)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    data_root = args.data_root.resolve()
    out_root = (
        args.out_root.resolve()
        if args.out_root is not None
        else (data_root / "structure_manifest").resolve()
    )
    out_root.mkdir(parents=True, exist_ok=True)
    paths = [
        data_root / "stage1_llps.fasta",
        data_root / "stage1_nonllps.fasta",
    ]
    records = read_records(paths)
    if args.limit is not None:
        records = records[: args.limit]
    mapping = load_accession_mapping(args.accession_map)

    results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = [
            executor.submit(
                resolve_one,
                record,
                out_root,
                mapping,
                args.pdb_identity,
                args.min_coverage,
                args.rows,
            )
            for record in records
        ]
        for index, future in enumerate(futures, 1):
            result = future.result()
            results.append(result)
            print(
                "[{}/{}] {} {}".format(
                    index,
                    len(futures),
                    result["sequence_hash"],
                    result["status"],
                ),
                flush=True,
            )

    manifest = {
        "policy": "RCSB PDB first; AlphaFold Database AlphaFold2-derived fallback; unresolved records require local AlphaFold2",
        "data_root": str(data_root),
        "structure_root": str(out_root),
        "pdb_identity_cutoff": args.pdb_identity,
        "minimum_query_coverage": args.min_coverage,
        "records_requested": len(records),
        "records": {item["sequence_hash"]: item for item in results},
        "status_counts": {},
    }
    for item in results:
        status = item["status"]
        manifest["status_counts"][status] = manifest["status_counts"].get(status, 0) + 1
    (out_root / "structure_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(manifest["status_counts"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
