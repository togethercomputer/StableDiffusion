#!/bin/bash

if [[ ! -z "$CONDA_ACTIVATE" ]]; then
  conda activate $CONDA_ACTIVATE
fi

NUM_WORKERS=${NUM_WORKERS-1}
for ((i = 0; i < $NUM_WORKERS; i++)); do
  env DEVICE=${DEVICE-cuda:$i} GROUP=${GROUP-group$i} /bin/bash -c 'python app/serving_inference.py' &
done

wait -n
exit $?
