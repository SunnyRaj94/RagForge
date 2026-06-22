from __future__ import annotations

import json
import shutil
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SAMPLE_ROOT = ROOT / "data" / "ingestion" / "sample"
MANIFEST_DIR = SAMPLE_ROOT / "manifests"


FILES = [
    {
        "file_type": "pdf",
        "source_org": "arXiv",
        "license": "arXiv submission distribution",
        "corpus_id": "sample-docs-pdf",
        "items": [
            {
                "doc_id": "arxiv-2311.01767",
                "source_name": "PPTC Benchmark: Evaluating Large Language Models for PowerPoint Task Completion",
                "source_url": "https://arxiv.org/pdf/2311.01767.pdf",
                "local_name": "2311.01767_pptc_benchmark.pdf",
            },
            {
                "doc_id": "arxiv-2403.03788",
                "source_name": "PPTC-R benchmark: Towards Evaluating the Robustness of Large Language Models for PowerPoint Task Completion",
                "source_url": "https://arxiv.org/pdf/2403.03788.pdf",
                "local_name": "2403.03788_pptc_r_benchmark.pdf",
            },
            {
                "doc_id": "arxiv-2501.03936",
                "source_name": "PPTAgent: Generating and Evaluating Presentations Beyond Text-to-Slides",
                "source_url": "https://arxiv.org/pdf/2501.03936.pdf",
                "local_name": "2501.03936_pptagent.pdf",
            },
            {
                "doc_id": "arxiv-2303.09957",
                "source_name": "A Benchmark of PDF Information Extraction Tools using a Multi-Task and Multi-Domain Evaluation Framework for Academic Documents",
                "source_url": "https://arxiv.org/pdf/2303.09957.pdf",
                "local_name": "2303.09957_pdf_information_extraction_tools.pdf",
            },
            {
                "doc_id": "arxiv-2204.02809",
                "source_name": "Hammer PDF: An Intelligent PDF Reader for Scientific Papers",
                "source_url": "https://arxiv.org/pdf/2204.02809.pdf",
                "local_name": "2204.02809_hammer_pdf.pdf",
            },
        ],
    },
    {
        "file_type": "pptx",
        "source_org": "icip-cas/PPTAgent",
        "license": "MIT",
        "corpus_id": "sample-docs-pptx",
        "items": [
            {
                "doc_id": "pptagent-test",
                "source_name": "PPTAgent test presentation",
                "source_url": "https://raw.githubusercontent.com/icip-cas/PPTAgent/main/pptagent/test/test.pptx",
                "local_name": "pptagent_test.pptx",
            },
            {
                "doc_id": "pptagent-hit-template",
                "source_name": "PPTAgent HIT template",
                "source_url": "https://raw.githubusercontent.com/icip-cas/PPTAgent/main/pptagent/templates/hit/source.pptx",
                "local_name": "hit_source.pptx",
            },
            {
                "doc_id": "pptagent-cip-template",
                "source_name": "PPTAgent CIP template",
                "source_url": "https://raw.githubusercontent.com/icip-cas/PPTAgent/main/pptagent/templates/cip/source.pptx",
                "local_name": "cip_source.pptx",
            },
            {
                "doc_id": "pptagent-default-template",
                "source_name": "PPTAgent default template",
                "source_url": "https://raw.githubusercontent.com/icip-cas/PPTAgent/main/pptagent/templates/default/source.pptx",
                "local_name": "default_source.pptx",
            },
            {
                "doc_id": "pptagent-ucas-template",
                "source_name": "PPTAgent UCAS template",
                "source_url": "https://raw.githubusercontent.com/icip-cas/PPTAgent/main/pptagent/templates/ucas/source.pptx",
                "local_name": "ucas_source.pptx",
            },
        ],
    },
    {
        "file_type": "txt",
        "source_org": "Project Gutenberg",
        "license": "Project Gutenberg License",
        "corpus_id": "sample-docs-txt",
        "items": [
            {
                "doc_id": "gutenberg-11",
                "source_name": "Alice's Adventures in Wonderland",
                "source_url": "https://www.gutenberg.org/cache/epub/11/pg11.txt",
                "local_name": "pg11_alice_in_wonderland.txt",
            },
            {
                "doc_id": "gutenberg-84",
                "source_name": "Frankenstein; Or, The Modern Prometheus",
                "source_url": "https://www.gutenberg.org/cache/epub/84/pg84.txt",
                "local_name": "pg84_frankenstein.txt",
            },
            {
                "doc_id": "gutenberg-1342",
                "source_name": "Pride and Prejudice",
                "source_url": "https://www.gutenberg.org/cache/epub/1342/pg1342.txt",
                "local_name": "pg1342_pride_and_prejudice.txt",
            },
            {
                "doc_id": "gutenberg-2600",
                "source_name": "War and Peace",
                "source_url": "https://www.gutenberg.org/cache/epub/2600/pg2600.txt",
                "local_name": "pg2600_war_and_peace.txt",
            },
            {
                "doc_id": "gutenberg-2701",
                "source_name": "Moby Dick; Or, The Whale",
                "source_url": "https://www.gutenberg.org/cache/epub/2701/pg2701.txt",
                "local_name": "pg2701_moby_dick.txt",
            },
        ],
    },
]


def download(url: str, dest: Path) -> None:
    if dest.exists() and dest.stat().st_size > 0:
        return
    request = urllib.request.Request(
        url,
        headers={
            "User-Agent": "Mozilla/5.0 RagForge sample corpus seeder",
        },
    )
    with urllib.request.urlopen(request) as response, dest.open("wb") as fh:
        shutil.copyfileobj(response, fh)


def main() -> None:
    SAMPLE_ROOT.mkdir(parents=True, exist_ok=True)
    MANIFEST_DIR.mkdir(parents=True, exist_ok=True)

    manifest_entries: list[dict[str, object]] = []

    for group in FILES:
        type_dir = SAMPLE_ROOT / group["file_type"]
        type_dir.mkdir(parents=True, exist_ok=True)

        for item in group["items"]:
            dest = type_dir / item["local_name"]
            download(item["source_url"], dest)
            manifest_entries.append(
                {
                    "doc_id": item["doc_id"],
                    "source_name": item["source_name"],
                    "source_url": item["source_url"],
                    "source_org": group["source_org"],
                    "license": group["license"],
                    "fetched_at": datetime.now(timezone.utc).isoformat(),
                    "local_path": str(dest.relative_to(ROOT)),
                    "file_type": group["file_type"],
                    "corpus_id": group["corpus_id"],
                }
            )

    manifest_path = MANIFEST_DIR / "sample_corpus_manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "generated_at": datetime.now(timezone.utc).isoformat(),
                "files": manifest_entries,
            },
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
