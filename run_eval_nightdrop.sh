#!/bin/bash

sids=(
    '211' '212' '213' '214' '215' '216' '217' '218'
    '219' '220'
)

for sid in "${sids[@]}"
do
    CUDA_VISIBLE_DEVICES=1 python test.py --sid "$sid"
done
