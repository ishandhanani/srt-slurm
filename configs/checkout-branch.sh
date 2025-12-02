#!/bin/bash
BRANCH="dev/mtp_support_for_eagle_worker_v1"

cd /sgl-workspace/sglang
git remote add trevor https://github.com/trevor-m/sglang.git
git fetch trevor $BRANCH
git checkout "trevor/${BRANCH}"

