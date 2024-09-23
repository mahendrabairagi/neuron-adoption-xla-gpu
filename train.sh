export YOLOX_DATADIR=/home/ubuntu/YOLOX/datasets/
export XLA_IR_DEBUG=1
export XLA_HLO_DEBUG=1

NEURON_RT_ASYNC_EXEC_MAX_INFLIGHT_REQUESTS=3  NEURON_CC_FLAGS='--cache_dir=./compiler_cache --model-type=cnn-training'  XLA_IR_DEBUG=1 XLA_HLO_DEBUG=1 torchrun --nproc_per_node=1 -m yolox.tools.train -n yolox-s -d 1 -b 1 
