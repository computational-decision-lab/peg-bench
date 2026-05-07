#!/usr/bin/env python3
import os
from huggingface_hub import create_repo, upload_large_folder
repo_id = os.environ.get('HF_REPO_ID')
folder_path = os.environ.get('HF_FOLDER', '/root/autodl-tmp/peg_bench_public_artifact_candidate_20260507')
token = os.environ.get('HF_TOKEN')
private = os.environ.get('HF_PRIVATE', 'true').lower() == 'true'
if not repo_id:
    raise SystemExit('Set HF_REPO_ID, e.g. colin-wang-ai/peg-bench-artifact')
if not token:
    raise SystemExit('Set HF_TOKEN to a write-scoped Hugging Face user access token')
create_repo(repo_id=repo_id, token=token, repo_type='dataset', private=private, exist_ok=True)
upload_large_folder(repo_id=repo_id, folder_path=folder_path, repo_type='dataset', token=token)
print('Uploaded to', repo_id)
