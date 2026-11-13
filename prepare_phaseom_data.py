"""Prepare the uploaded PhaseOM FASTA files for PhaseResSAGE.

The three main sample files are used as samples:
  - Scaffold_0.4.fasta
  - client_0.4-4244.fasta
  - non-human+else-3891.fasta

The IDR/non-IDR FASTA files and non-LLPS_human_0.4-549.fasta are audited as
subsets and are not appended as additional proteins, because doing so would
duplicate samples.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable, List

from Bio import SeqIO


DEFAULT_SOURCE = Path(r"C:\Users\27873\Downloads\数据\数据")


def read_records(path: Path) -> List[dict]:
    records = []
    for record in SeqIO.parse(str(path), "fasta"):
        sequence = str(record.seq).upper().replace(" ", "").replace("\n", "")
        records.append(
            {
                "description": record.description.strip(),
                "sequence": sequence,
            }
        )
    return records


def write_fasta(path: Path, records: Iterable[dict], prefix: str = "") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for item in records:
            description = item["description"]
            if prefix and not description.lower().startswith(prefix.lower() + "|"):
                description = f"{prefix}|{description}"
            handle.write(f">{description}\n")
            sequence = item["sequence"]
            for start in range(0, len(sequence), 80):
                handle.write(sequence[start : start + 80] + "\n")


def sequence_set(records: Iterable[dict]) -> set[str]:
    return {item["sequence"] for item in records}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-dir", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument(
        "--out-root",
        type=Path,
        default=Path("data") / "phaseom_attachment",
    )
    args = parser.parse_args()

    source = args.source_dir.resolve()
    out_root = args.out_root.resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    source_files = {
        "client": source / "client_0.4-4244.fasta",
        "scaffold": source / "Scaffold_0.4.fasta",
        "non-LLPS_human": source / "non-LLPS_human_0.4-549.fasta",
        "non-LLPS_other": source / "non-human+else-3891.fasta",
        "client_idr": source / "client_idr.fasta",
        "client_non_idr": source / "client_non_idr.fasta",
        "scaffold_idr": source / "scaffold_idr.fasta",
        "scaffold_non_idr": source / "scaffold_non_idr.fasta",
    }
    records = {name: read_records(path) for name, path in source_files.items()}

    scaffold = records["scaffold"]
    client = records["client"]
    nonllps_other = records["non-LLPS_other"]
    nonllps = []
    for item in nonllps_other:
        nonllps.append(
            {
                "description": f"non-LLPS|{item['description']}",
                "sequence": item["sequence"],
            }
        )

    write_fasta(out_root / "stage1_llps.fasta", scaffold + client, "LLPS")
    write_fasta(out_root / "stage1_scaffold.fasta", scaffold, "LLPS")
    write_fasta(out_root / "stage1_nonllps.fasta", nonllps)
    write_fasta(out_root / "stage2_roles.fasta", scaffold, "scaffold")
    with (out_root / "stage2_roles.fasta").open(
        "a", encoding="utf-8", newline="\n"
    ) as handle:
        for item in client:
            description = item["description"]
            if not description.lower().startswith("client|"):
                description = f"client|{description}"
            handle.write(f">{description}\n")
            for start in range(0, len(item["sequence"]), 80):
                handle.write(item["sequence"][start : start + 80] + "\n")

    audit = {
        "source_dir": str(source),
        "source_files": {key: str(value) for key, value in source_files.items()},
        "main_counts": {
            "scaffold": len(scaffold),
            "client": len(client),
            "non-LLPS_main": len(nonllps_other),
            "non-LLPS_human_subset": len(records["non-LLPS_human"]),
            "stage1_llps": len(scaffold) + len(client),
            "stage1_nonllps": len(nonllps),
            "stage2_scaffold": len(scaffold),
            "stage2_client": len(client),
        },
        "idr_subset_counts": {
            "client_idr": len(records["client_idr"]),
            "client_non_idr": len(records["client_non_idr"]),
            "scaffold_idr": len(records["scaffold_idr"]),
            "scaffold_non_idr": len(records["scaffold_non_idr"]),
        },
        "subset_relationships": {
            "nonllps_human_subset_in_nonllps_main": len(
                sequence_set(records["non-LLPS_human"])
                - sequence_set(records["non-LLPS_other"])
            )
            == 0,
            "nonllps_human_subset_overlap_count": len(
                sequence_set(records["non-LLPS_human"])
                & sequence_set(records["non-LLPS_other"])
            ),
            "client_union_in_client": len(
                sequence_set(records["client_idr"])
                | sequence_set(records["client_non_idr"])
            )
            == len(sequence_set(client)),
            "client_idr_non_idr_overlap": len(
                sequence_set(records["client_idr"])
                & sequence_set(records["client_non_idr"])
            ),
            "scaffold_union_in_scaffold": len(
                sequence_set(records["scaffold_idr"])
                | sequence_set(records["scaffold_non_idr"])
            ),
            "scaffold_main_not_in_idr_subsets": len(
                sequence_set(scaffold)
                - (
                    sequence_set(records["scaffold_idr"])
                    | sequence_set(records["scaffold_non_idr"])
                )
            ),
            "scaffold_idr_non_idr_overlap": len(
                sequence_set(records["scaffold_idr"])
                & sequence_set(records["scaffold_non_idr"])
            ),
        },
        "invalid_residue_counts": {},
        "note": (
            "The 549-record non-LLPS_human file and IDR/non-IDR files are "
            "subset annotations and are not added as independent samples."
        ),
    }
    allowed = set("ACDEFGHIKLMNPQRSTVWY")
    for name, items in records.items():
        audit["invalid_residue_counts"][name] = sum(
            sum(aa not in allowed for aa in item["sequence"])
            for item in items
        )
    (out_root / "prepare_manifest.json").write_text(
        json.dumps(audit, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(audit, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
