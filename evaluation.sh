export YOLOX_DATADIR=/efs/data/coco_data/
export XLA_IR_DEBUG=1
export XLA_HLO_DEBUG=1

NEURON_RT_ASYNC_EXEC_MAX_INFLIGHT_REQUESTS=3  NEURON_CC_FLAGS='--cache_dir=./compiler_cache --model-type=cnn-training'  XLA_IR_DEBUG=1 XLA_HLO_DEBUG=1 torchrun --nproc_per_node=2 -m yolox.tools.eval -n  yolox-s -c YOLOX_outputs/yolox_s/last_epoch_ckpt.pth -b 1 -d 1 --conf 0.001 
