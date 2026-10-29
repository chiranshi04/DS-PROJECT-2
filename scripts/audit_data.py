"""Audit configured dataset sources without fabricating counts or downloading audio."""

from __future__ import annotations

import argparse
import json
import urllib.error
import urllib.request
from datetime import date
from pathlib import Path
from typing import Any

from omegaconf import OmegaConf

from indivoice.config import load_config


def probe_url(url: str) -> dict[str, Any]:
    """Probe an official URL and return only observed HTTP information."""
    request = urllib.request.Request(
        url, method="HEAD", headers={"User-Agent": "InDivoice-data-audit/0.2"}
    )
    try:
        with urllib.request.urlopen(request, timeout=20) as response:
            return {
                "status": "accessible",
                "http_status": response.status,
                "final_url": response.url,
            }
    except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError) as error:
        return {"status": "not_verified", "error": str(error)}


def hf_probe(repo_id: str) -> dict[str, Any]:
    """Inspect a Hugging Face dataset repository's file listing without loading audio."""
    try:
        from huggingface_hub import HfApi

        api = HfApi()
        info = api.dataset_info(repo_id, files_metadata=False)
        siblings = [item.rfilename for item in (info.siblings or [])]
        metadata_files = [
            name
            for name in siblings
            if any(
                token in name.lower()
                for token in ("metadata", "manifest", ".csv", ".json", ".parquet")
            )
        ]
        return {
            "status": "accessible",
            "repo_id": repo_id,
            "file_count": len(siblings),
            "metadata_files": metadata_files[:200],
            "card_data": info.card_data.to_dict() if info.card_data else None,
        }
    except Exception as error:  # API errors are part of the audit result, not a crash.
        return {"status": "not_verified", "repo_id": repo_id, "error": repr(error)}


def audit_source(source: dict[str, Any]) -> dict[str, Any]:
    result = dict(OmegaConf.to_container(source, resolve=True))
    result["checked_date"] = date.today().isoformat()
    result["official_url_probe"] = probe_url(str(source["official_url"]))
    if source.get("hf_repo_id"):
        result["huggingface_probe"] = hf_probe(str(source["hf_repo_id"]))
    result["audio_sampling"] = {
        "attempted": False,
        "reason": "Phase 2 audit records metadata first; no audio was downloaded by default.",
        "maximum_files_per_language": 200,
    }
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("results/data_audit.json"))
    args = parser.parse_args()
    source_config = load_config("data_sources")
    sources = list(source_config.get("audit_sources", []))
    results = [audit_source(source) for source in sources]
    payload = {
        "audit_date": date.today().isoformat(),
        "method": (
            "official URL probe plus Hugging Face repository metadata listing; "
            "no audio download by default"
        ),
        "sources": results,
        "overlap_checks": {
            "status": "not_verified",
            "reason": (
                "Speaker/source-utterance overlap requires comparable utterance metadata "
                "and was not inferred from names."
            ),
        },
        "recommendation": (
            "Keep Hindi, Tamil, and Bengali pending full metadata inspection; "
            "do not substitute languages on incomplete evidence."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(f"Wrote audit result to {args.output}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
