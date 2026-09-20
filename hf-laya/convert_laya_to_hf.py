"""Upload the ``hf_laya`` remote-code files + a per-repo ``config.json`` to a Hub repo."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from huggingface_hub import HfApi

HUB_REPOS: dict[str, str] = {
    "convaiinnovations/laya": "laya.json",
    "convaiinnovations/laya-multilingual": "laya-multilingual.json",
    "convaiinnovations/laya-typed-decisions": "laya-typed-decisions.json",
}

_PACKAGE_DIR = Path(__file__).parent
_CONFIGS_DIR = _PACKAGE_DIR / "configs"

_SKIP = {"convert_laya_to_hf.py", "__init__.py"}


def _sources() -> list[Path]:
    return sorted(p for p in _PACKAGE_DIR.glob("*.py") if p.name not in _SKIP)


def publish(repo_id: str, token: str | None = None) -> None:
    if repo_id not in HUB_REPOS:
        raise ValueError(f"Unknown Hub repo: {repo_id!r}. Known: {sorted(HUB_REPOS)}")

    config_source = _CONFIGS_DIR / HUB_REPOS[repo_id]
    config_payload = json.dumps(json.loads(config_source.read_text()), indent=2).encode()

    api = HfApi(token=token)
    for path in _sources():
        api.upload_file(path_or_fileobj=str(path), path_in_repo=path.name, repo_id=repo_id)
    api.upload_file(path_or_fileobj=config_payload, path_in_repo="config.json", repo_id=repo_id)


def publish_all(token: str | None = None) -> None:
    for repo_id in HUB_REPOS:
        publish(repo_id, token=token)


def main() -> None:
    parser = argparse.ArgumentParser(description="Publish hf_laya files to Convai's Hub repos.")
    parser.add_argument("repo_id", nargs="?", help="Target Hub repo id; omit to publish to all.")
    parser.add_argument("--token", default=None, help="HF token; defaults to $HF_TOKEN.")
    args = parser.parse_args()
    if args.repo_id:
        publish(args.repo_id, token=args.token)
    else:
        publish_all(token=args.token)


if __name__ == "__main__":
    main()
