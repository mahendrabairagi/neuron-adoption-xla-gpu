
SyncTensorsGraph.14SyncTensorsGraph.14²
AddComputation.5$
x.6	parameter*: ’  &
y.7	parameter*: H’  %
add.8add*: ’’  "%
*
**x.6y.7(0Ύ	
SyncTensorsGraph.14ά
p0.2	parameterx*
 2   :
xla__device_dataxla__device_data`/home/ubuntu/aws-neuron-samples/torch-neuronx/training/neuron-adoption/yolox/models/yolo_head.py x’

neff_input_namesinput0 Έ

constant.3constant*:}
	aten__sum	aten__sum`/home/ubuntu/aws-neuron-samples/torch-neuronx/training/neuron-adoption/yolox/models/yolo_head.py xB
*B    ’  »
reduce.9reducex*
 2  :}
	aten__sum	aten__sum`/home/ubuntu/aws-neuron-samples/torch-neuronx/training/neuron-adoption/yolox/models/yolo_head.py xr	’²’  9
constantconstant*: B
*B    ’  Έ
broadcast.11	broadcastx*
 2  :{
aten__gtaten__gt`/home/ubuntu/aws-neuron-samples/torch-neuronx/training/neuron-adoption/yolox/models/yolo_head.py x’’  Β

compare.12comparex*
 2  :{
aten__gtaten__gt`/home/ubuntu/aws-neuron-samples/torch-neuronx/training/neuron-adoption/yolox/models/yolo_head.py x’	ϊGT’ ΒFLOAT W
tuple.13tuple"x*
 2  : ’’

neff_output_namesoutput0 "7
x*
 2   "x*
 2  p0.2(0"5
x*
 2   "x*
 2  p0(0B z	
 	
/usr/lib/python3.10/runpy.py
U/home/ubuntu/aws-neuron-samples/torch-neuronx/training/neuron-adoption/tools/train.py
[/home/ubuntu/aws-neuron-samples/torch-neuronx/training/neuron-adoption/yolox/core/launch.py
~/home/ubuntu/aws-neuron-samples/aws_neuron_venv_pytorch/lib/python3.10/site-packages/loguru-0.7.2-py3.10.egg/loguru/_logger.py
\/home/ubuntu/aws-neuron-samples/torch-neuronx/training/neuron-adoption/yolox/core/trainer.py
o/home/ubuntu/aws-neuron-samples/aws_neuron_venv_pytorch/lib/python3.10/site-packages/torch/nn/modules/module.py
\/home/ubuntu/aws-neuron-samples/torch-neuronx/training/neuron-adoption/yolox/models/yolox.py
`/home/ubuntu/aws-neuron-samples/torch-neuronx/training/neuron-adoption/yolox/models/yolo_head.py_run_module_as_main	_run_code<module>launchcatch_wrappermaintraintrain_in_epochtrain_in_itertrain_one_iter_wrapped_call_impl
_call_implforward
get_lossesΔVbύ	vNX	b
{ξχ"Β"""""""""	"
	"
"""""""  