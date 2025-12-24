#!/bin/bash
BRANCH="spec-tp-1"

cd /sgl-workspace/sglang
git remote add trevor https://github.com/trevor-m/sglang.git
git fetch trevor
git checkout trevor/${BRANCH}
