"""Write a post-hoc Stage 1 run manifest from the existing Stage 1 outputs.

Stage 1 (1_preprocess_benchmark.py) writes its outputs to stable paths under
data/processed/ and does not create a run manifest. Stage 15 verifies upstream
stages through eu.assert_current_stage_manifest(1, ...), so this helper records
the SHA-256 of the current Stage 1 outputs in the standard manifest format.

The manifest only attests that these files are the ones in use now. It does not
prove that the current script version produced them, so it is labelled
"post_hoc_from_existing_outputs". Disclose this in the methodology chapter.

Run from the repository root, AFTER experiment_utils.py, 1_preprocess_benchmark.py,
experiment_protocol.md and pipeline.md are final (editing any of them
invalidates the manifest):

    python write_stage1_manifest.py
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import experiment_utils as eu

PROJECT_ROOT = Path.cwd().resolve()
SCRIPT_NAME = "scripts/1_preprocess_benchmark.py"  # as registered in eu.PIPELINE_SCRIPTS
# Edit these if your Stage 1 raw input or output names differ.
RAW_INPUT = Path("data/raw/re_dataset.csv")
OUTPUTS = (
    Path("data/processed/ib_cleaned.csv"),
    Path("data/processed/ib_train.csv"),
    Path("data/processed/ib_val.csv"),
    Path("data/processed/ib_test.csv"),
)
RUN_DIR = Path("results/runs") / eu.RUN_ID / "stage_01"


def _find_stage_key(label: str) -> int | str:
    """Return the registry key for a stage, whether stored as int or str."""
    for key in eu.PIPELINE_SCRIPTS:
        if str(key) == label:
            return key
    raise KeyError(f"Stage {label!r} is not registered in eu.PIPELINE_SCRIPTS")


def _check_files(paths: tuple[Path, ...]) -> None:
    missing = [str(p) for p in paths if not (PROJECT_ROOT / p).is_file()]
    if missing:
        raise FileNotFoundError("Missing Stage 1 files: " + ", ".join(missing))
    # Raw-byte hashing: CRLF copies would not match the LF files on Kaggle/Colab.
    crlf = [str(p) for p in paths if b"\r\n" in (PROJECT_ROOT / p).read_bytes()]
    if crlf:
        print("WARNING: CRLF line endings found (hashes will differ on Linux):",
              ", ".join(crlf))


def main() -> int:
    stage = _find_stage_key("1")
    expected_script = eu.PIPELINE_SCRIPTS[stage]
    if Path(expected_script).as_posix() != SCRIPT_NAME:
        raise ValueError(f"Registry maps Stage 1 to {expected_script!r}, not {SCRIPT_NAME!r}")
    _check_files((RAW_INPUT, *OUTPUTS))

    output_hashes = {
        p.as_posix(): eu.sha256_file(PROJECT_ROOT / p) for p in OUTPUTS
    }
    config: dict[str, Any] = {
        "schema_version": eu.CONFIG_SCHEMA_VERSION,
        "run_id": eu.RUN_ID,
        "stage": stage,
        "script_name": expected_script,
        "outputs": [p.as_posix() for p in OUTPUTS],
    }
    manifest = eu.make_run_manifest(
        stage=stage,
        script_name=expected_script,
        config=config,
        input_paths=[RAW_INPUT],
        metadata={
            "output_sha256": output_hashes,
            "manifest_origin": "post_hoc_from_existing_outputs",
        },
        project_root=PROJECT_ROOT,
    )
    eu.prepare_run_directory(PROJECT_ROOT / RUN_DIR, manifest)
    # Same validator that Stage 15 applies.
    result = eu.assert_current_stage_manifest(stage, PROJECT_ROOT)
    print("Stage 1 manifest written and validated.")
    print("fingerprint:", result["fingerprint"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
