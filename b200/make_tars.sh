#!/bin/bash
# make_tars.sh — build upload tarballs on KISTI for transfer to the GPU platform.
set -e
R=/scratch/migrate_k266_to_gpu
mkdir -p "$R"; chmod 777 "$R"
cd /home01/k266a01/kbds_project
echo "START $(date +%T)" > "$R/tar.log"

# 1) code (small, gzip)
tar czf "$R/code.tar.gz" --exclude=work --exclude='core.*' --exclude=__pycache__ \
    -C /home01/k266a01 kbds_project
echo "code done $(date +%T) $(du -h "$R/code.tar.gz" | cut -f1)" >> "$R/tar.log"

# 2) init model 18G (plain tar; safetensors don't compress)
tar cf "$R/model.tar" -C work/checkpoints sft_mixed_merged
echo "model done $(date +%T) $(du -h "$R/model.tar" | cut -f1)" >> "$R/tar.log"

# 3) data 46G (plain tar)
tar cf "$R/data.tar" -C work data
echo "data done $(date +%T) $(du -h "$R/data.tar" | cut -f1)" >> "$R/tar.log"

echo "ALL_DONE $(date +%T)" >> "$R/tar.log"
