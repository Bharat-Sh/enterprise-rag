"""Download the model weights and vocabulary this platform needs.

Run once per machine and once per container build:

    uv run python scripts/fetch_models.py                 # everything
    uv run python scripts/fetch_models.py --tokenizer-only  # ~17 MB, no GPU needed

Why files on disk rather than a repo id resolved at runtime
-----------------------------------------------------------
Both the model service and the ingestion worker take *paths*. Nothing reaches
HuggingFace while serving traffic. That buys three things: a restart during a
hub outage still works, the bytes being tokenized with are the ones the image was
built with rather than whatever was served today, and there is a single place
where "what did we download, and from where" is answerable.

The tokenizer is separable on purpose. It is 17 MB against 2.3 GB of weights,
and the ingestion worker needs *only* the tokenizer — it never loads a model, so
making it wait on a multi-gigabyte download to size chunks would be absurd.

`huggingface-hub` ships in the `gpu` extra, but `--tokenizer-only` is exactly the
case where you may not have it. The import failure below says so plainly rather
than raising `ModuleNotFoundError` at someone.
"""

from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

#: Repo ids and the local directory each is fetched into. The directory names
#: are what `MODEL_SERVICE_EMBEDDING_MODEL_PATH` and friends default to.
EMBEDDING_REPO = "BAAI/bge-m3"
RERANKER_REPO = "BAAI/bge-reranker-v2-m3"
DEFAULT_ROOT = Path("./var/models")

#: The tokenizer file the worker needs, and the only file `--tokenizer-only`
#: fetches. Sitting inside the model snapshot is what guarantees the vocabulary
#: matches the weights.
TOKENIZER_FILENAME = "tokenizer.json"

#: Everything else is an artefact of a training framework we do not use, and
#: some repos carry several formats of the same weights. Downloading all of them
#: roughly doubles the transfer for files nothing will open.
IGNORED = ["*.msgpack", "*.h5", "*.ot", "*.onnx*", "onnx/*"]


def _hub():  # noqa: ANN202 - the return type lives in an optional dependency
    try:
        from huggingface_hub import hf_hub_download, snapshot_download
    except ImportError:  # pragma: no cover - depends on the installed extras
        sys.exit(
            "huggingface-hub is not installed. It ships with the `gpu` extra:\n"
            "    uv sync --extra gpu\n"
            "or, for the tokenizer alone on a machine with no GPU:\n"
            "    uv pip install huggingface-hub"
        )
    return hf_hub_download, snapshot_download


def fetch_tokenizer(root: Path) -> Path:
    """Download just `tokenizer.json` for the embedding model."""
    hf_hub_download, _ = _hub()
    target = root / "bge-m3"
    target.mkdir(parents=True, exist_ok=True)

    print(f"Fetching {TOKENIZER_FILENAME} from {EMBEDDING_REPO} ...")
    downloaded = hf_hub_download(
        repo_id=EMBEDDING_REPO,
        filename=TOKENIZER_FILENAME,
        local_dir=str(target),
    )
    path = Path(downloaded)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()[:16]
    # Printed because it is what `/v1/info` publishes: if chunking and embedding
    # ever disagree about what a token is, comparing these two strings is how
    # you find out, and it is far from obvious where else to look.
    print(f"  -> {path}  ({path.stat().st_size / 1e6:.1f} MB, fingerprint {digest})")
    return path


def fetch_model(repo: str, root: Path, name: str) -> Path:
    """Download a full model snapshot."""
    _, snapshot_download = _hub()
    target = root / name
    print(f"Fetching {repo} ...")
    snapshot_download(repo_id=repo, local_dir=str(target), ignore_patterns=IGNORED)
    print(f"  -> {target}")
    return target


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        type=Path,
        default=DEFAULT_ROOT,
        help=f"Directory to download into (default: {DEFAULT_ROOT})",
    )
    parser.add_argument(
        "--tokenizer-only",
        action="store_true",
        help="Fetch only tokenizer.json (~17 MB). All the ingestion worker needs.",
    )
    parser.add_argument(
        "--no-reranker",
        action="store_true",
        help="Skip bge-reranker-v2-m3, for a GPU too small to hold both.",
    )
    args = parser.parse_args()

    root: Path = args.root
    root.mkdir(parents=True, exist_ok=True)

    if args.tokenizer_only:
        fetch_tokenizer(root)
        print("\nSet RAG_INGESTION__TOKENIZER=bge-m3 to use it.")
        return

    fetch_model(EMBEDDING_REPO, root, "bge-m3")
    if not args.no_reranker:
        fetch_model(RERANKER_REPO, root, "bge-reranker-v2-m3")

    tokenizer = root / "bge-m3" / TOKENIZER_FILENAME
    if tokenizer.is_file():
        digest = hashlib.sha256(tokenizer.read_bytes()).hexdigest()[:16]
        print(f"\nTokenizer fingerprint: {digest}")
        print("This must match `tokenizer_hash` in the model service's /v1/info.")


if __name__ == "__main__":
    main()
