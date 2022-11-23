#!/bin/bash
set -e
cd "${0%/*}"
trap "trap - SIGTERM && kill -- -$$" SIGINT SIGTERM EXIT

if [[ ! -z "$CONDA_ACTIVATE" ]]; then
  conda activate $CONDA_ACTIVATE
fi

NUM_WORKERS=${NUM_WORKERS-1}
if [[ "$NUM_WORKERS" == "auto" ]]; then
  NUM_WORKERS=`nvidia-smi --query-gpu=name --format=csv,noheader | wc -l`
  echo NUM_WORKERS=auto resolved to NUM_WORKERS+${NUM_WORKERS}
fi

for ((i = 0; i < $NUM_WORKERS; i++)); do
  env DEVICE=${DEVICE-cuda:$i} GROUP=${GROUP-group$i} /bin/bash -c 'python app/serving_inference.py' &
done

wait -n
exit $?
