#!/usr/bin/env bash


for i in $(seq 10) ; do

echo "lol i'm running" $i >&2

echo "Step" $i 

sleep 1

done

exit 0
