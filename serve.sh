#!/bin/bash
set -e
cd "${0%/*}"
trap "trap - SIGTERM && kill -- -$$" SIGINT SIGTERM EXIT

if [[ ! -z "$CONDA_ACTIVATE" ]]; then
  conda activate $CONDA_ACTIVATE
fi

NUM_WORKERS=${NUM_WORKERS-1}
if [[ -z "$CUDA_VISIBLE_DEVICES" ]]; then
  unset CUDA_VISIBLE_DEVICES
  if [[ "$NUM_WORKERS" == "auto" ]]; then
    NUM_WORKERS=`nvidia-smi --query-gpu=name --format=csv,noheader | wc -l`
    echo NUM_WORKERS=auto resolved to NUM_WORKERS=${NUM_WORKERS}
  fi
  DEVICES=
  for ((i = 0; i < $NUM_WORKERS; i++)); do
    DEVICES=$DEVICES,$i
  done
else
  DEVICES=$CUDA_VISIBLE_DEVICES
  if [[ "$NUM_WORKERS" == "auto" ]]; then
    NUM_WORKERS=`echo $DEVICES | sed 's/[^,]//g' | wc -c`
    echo NUM_WORKERS=auto resolved to NUM_WORKERS=${NUM_WORKERS}
  fi
fi

count=0
for i in ${DEVICES//,/$IFS}; do
  env DEVICE=${DEVICE-cuda:$i} GROUP=${GROUP-group$i} /bin/bash -c 'python app/serving_inference.py' &
  count=`expr $count + 1`
  if [ $count -eq $NUM_WORKERS ]; then
    break
  fi
done

wait -n
exit $?
