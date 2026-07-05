#!/bin/bash
for id in "$@"; do
  echo "=== creduce $id starting $(date +%T) ==="
  ./run_creduce.sh "$id"
  echo "=== creduce $id done $(date +%T) ==="
done
echo "ALL CREDUCE DONE"
