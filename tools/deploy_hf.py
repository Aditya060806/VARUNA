"""
Deploy VARUNA to a Hugging Face Space (Streamlit SDK).

Keeps the GitHub README clean (no YAML front-matter) while giving the HF Space
the front-matter it needs — the front-matter is injected only into the copy
uploaded to the Space.

Usage (Windows):
    set HF_TOKEN=hf_xxxxxxxxxxxxxxxxx          # a WRITE token from
                                               # https://huggingface.co/settings/tokens
    python tools/deploy_hf.py                  # creates <your-username>/VARUNA

Optional: set a custom space id ->  set HF_SPACE=my-user/VARUNA
"""
from __future__ import annotations

import os
import sys
import tempfile

FRONTMATTER = """---
title: VARUNA - AI Digital Twin of India's Climate
emoji: 🛰️
colorFrom: indigo
colorTo: red
sdk: streamlit
sdk_version: 1.56.0
app_file: app.py
pinned: true
license: mit
short_description: AI Digital Twin of India's Climate - ISRO BAH 2026 (PS#5)
---

"""

# files/folders that must NOT be uploaded (large, rebuildable, or local-only)
IGNORE = [
    "data/raw/*", "data/processed/anom_cube.npy",
    ".git/*", ".git*/**", ".venv/*", "venv/*", "**/__pycache__/*", "*.pyc",
    "README.md",  # uploaded separately with front-matter
]

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def main():
    from huggingface_hub import HfApi, create_repo, upload_folder, whoami

    token = os.environ.get("HF_TOKEN")
    if not token:
        print("ERROR: set HF_TOKEN (a WRITE token from "
              "https://huggingface.co/settings/tokens).")
        sys.exit(1)

    space = os.environ.get("HF_SPACE")
    if not space:
        user = whoami(token=token)["name"]
        space = f"{user}/VARUNA"
    print(f"[deploy] target Space: {space}")

    create_repo(space, repo_type="space", space_sdk="streamlit",
                exist_ok=True, token=token)
    print("[deploy] Space ready. Uploading project (LFS handled automatically)…")

    upload_folder(repo_id=space, repo_type="space", folder_path=ROOT,
                  ignore_patterns=IGNORE, token=token,
                  commit_message="Deploy VARUNA — AI Digital Twin of India's Climate")

    # README with HF front-matter (GitHub copy stays clean)
    readme = FRONTMATTER + open(os.path.join(ROOT, "README.md"), encoding="utf-8").read()
    api = HfApi(token=token)
    api.upload_file(path_or_fileobj=readme.encode("utf-8"), path_in_repo="README.md",
                    repo_id=space, repo_type="space",
                    commit_message="Add HF Space front-matter")

    print(f"\n[deploy] DONE → https://huggingface.co/spaces/{space}")
    print("The Space will build (install requirements, then launch). First build "
          "takes a few minutes.")


if __name__ == "__main__":
    main()
