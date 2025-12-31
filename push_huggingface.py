#!/usr/bin/env python
# encoding: utf-8
'''
@license: (C) Copyright 2025, Hey.
@author: Hey
@email: sanyuan.hy@alibaba-inc.com
@tel: 137****6540
@datetime: 2025/12/31 14:02
@project: lucaone
@file: push_huggingface.py
@desc: push models to huggingface
'''

from huggingface_hub import HfApi

api = HfApi()

local_paths = [
    "./checkpoints/LucaGroup/LucaOne-default-step36M",
    "./checkpoints/LucaGroup/LucaOne-gene-step36.8M",
    "./checkpoints/LucaGroup/LucaOne-prot-step38.2M",
    "./checkpoints/LucaGroup/LucaOne-mask-step36M",
    "./checkpoints/LucaGroup/LucaOne-default-step17.6M",
    "./checkpoints/LucaGroup/LucaOne-default-step5.6M",
]

for local_path in local_paths:
    repo_id = local_path.replace("./checkpoints/", "")
    api.create_repo(repo_id=repo_id, repo_type="model", exist_ok=True)
    api.upload_folder(
        folder_path=local_path,
        repo_id=repo_id,
        repo_type="model",
        commit_message="Update weights and modeling code to latest version ",
    )
    print("Success %s" % repo_id)
print("done all!")