#!/bin/bash

CONFIG_PATH="./configs/test-original"
MODEL=$1
GPU=$2

echo 'set5' &&
echo 'x2' &&
python test.py --config ${CONFIG_PATH}/test-set5-2.yaml --model ${MODEL} --gpu ${GPU} &&
echo 'x3' &&
python test.py --config ${CONFIG_PATH}/test-set5-3.yaml --model ${MODEL} --gpu ${GPU} &&
echo 'x4' &&
python test.py --config ${CONFIG_PATH}/test-set5-4.yaml --model ${MODEL} --gpu ${GPU} &&
echo 'x6*' &&
python test.py --config ${CONFIG_PATH}/test-set5-6.yaml --model ${MODEL} --gpu ${GPU} &&
echo 'x8*' &&
python test.py --config ${CONFIG_PATH}/test-set5-8.yaml --model ${MODEL} --gpu ${GPU} &&

echo 'set14' &&
echo 'x2' &&
python test.py --config ${CONFIG_PATH}/test-set14-2.yaml --model ${MODEL} --gpu ${GPU} &&
echo 'x3' &&
python test.py --config ${CONFIG_PATH}/test-set14-3.yaml --model ${MODEL} --gpu ${GPU} &&
echo 'x4' &&
python test.py --config ${CONFIG_PATH}/test-set14-4.yaml --model ${MODEL} --gpu ${GPU} &&
echo 'x6*' &&
python test.py --config ${CONFIG_PATH}/test-set14-6.yaml --model ${MODEL} --gpu ${GPU} &&
echo 'x8*' &&
python test.py --config ${CONFIG_PATH}/test-set14-8.yaml --model ${MODEL} --gpu ${GPU} &&

echo 'b100' &&
echo 'x2' &&
python test.py --config ${CONFIG_PATH}/test-b100-2.yaml --model ${MODEL} --gpu ${GPU} &&
echo 'x3' &&
python test.py --config ${CONFIG_PATH}/test-b100-3.yaml --model ${MODEL} --gpu ${GPU} &&
echo 'x4' &&
python test.py --config ${CONFIG_PATH}/test-b100-4.yaml --model ${MODEL} --gpu ${GPU} &&
echo 'x6*' &&
python test.py --config ${CONFIG_PATH}/test-b100-6.yaml --model ${MODEL} --gpu ${GPU} &&
echo 'x8*' &&
python test.py --config ${CONFIG_PATH}/test-b100-8.yaml --model ${MODEL} --gpu ${GPU} &&

echo 'urban100' &&
echo 'x2' &&
python test.py --config ${CONFIG_PATH}/test-urban100-2.yaml --model ${MODEL} --gpu ${GPU} &&
echo 'x3' &&
python test.py --config ${CONFIG_PATH}/test-urban100-3.yaml --model ${MODEL} --gpu ${GPU} &&
echo 'x4' &&
python test.py --config ${CONFIG_PATH}/test-urban100-4.yaml --model ${MODEL} --gpu ${GPU} &&
echo 'x6*' &&
python test.py --config ${CONFIG_PATH}/test-urban100-6.yaml --model ${MODEL} --gpu ${GPU} &&
echo 'x8*' &&
python test.py --config ${CONFIG_PATH}/test-urban100-8.yaml --model ${MODEL} --gpu ${GPU} &&

true
