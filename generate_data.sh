#!/bin/bash

mkdir "${1}/dataset"
mkdir "${1}/dataset/training_data"

echo 'gpu-device?'
read -p 'GPU-device-num: ' gpu_id

for town in town01 town02 town03 town04 town05 town06 town07
do
  echo "start routes_${town}_${2}";
  echo '================================';
  mkdir "${1}/dataset/training_data/${town}_${2}";
  python test_run.py --localization filter --routes ./routes/training_routes/routes_${town}_${2}.xml \
  --scenarios ./routes/scenarios/${town}_all_scenarios.json --log-dir ${1}/dataset/training_data/${town}_${2}/ --timeout 200.0 --record-video \
  --checkpoint "${1}/dataset/training_data/${town}_${2}/simulation_results.json" --silent \
  --record-frame-skip 10 --gpu-device "cuda:${gpu_id}" |& tee -a "${1}/dataset/training_data/${town}_${2}/log_${town}_${2}.txt";
done

echo 'done'