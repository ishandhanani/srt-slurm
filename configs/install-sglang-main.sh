#!/bin/bash
# Install sglang from main branch

set -e

cd /sgl-workspace

# Apply fix from PR #14934 - remove incorrect BlockRemoved event emission during node splits
# https://github.com/sgl-project/sglang/pull/14934
sed -i '/_record_remove_event(child)/d' /sgl-workspace/sglang/python/sglang/srt/mem_cache/radix_cache.py

cd /sgl-workspace/sglang

# install sglang router version 0.3.0
pip install sglang-router==0.3.0