
SyncTensorsGraph.5569SyncTensorsGraph.5569�
ScatterCombiner.255(
p0.256	parameter*�: ��� � *
p1.257	parameter*�: H��� � *
add.258add*�: ������ � "+
*�
*�*�p0.256p1.257(�0��
ScatterCombiner.365(
p0.366	parameter*�: ��� � *
p1.367	parameter*�: H��� � *
add.368add*�: ������ � "+
*�
*�*�p0.366p1.367(�0��
AddComputation.490'
x.491	parameter*�: ��� � )
y.492	parameter*�: H��� � *
add.493add*�: ������ � ")
*�
*�*�x.491y.492(�0��
AddComputation.517'
x.518	parameter*�: ��� � )
y.519	parameter*�: H��� � *
add.520add*�: ������ � ")
*�
*�*�x.518y.519(�0��
AddComputation.670'
x.671	parameter*�: ��� � )
y.672	parameter*�: H��� � *
add.673add*�: ������ � ")
*�
*�*�x.671y.672(�0��
AddComputation.980'
x.981	parameter*�: ��� � )
y.982	parameter*�: H��� � *
add.983add*�: ������ � ")
*�
*�*�x.981y.982(�0��
AddComputation.1007(
x.1008	parameter*�: ��� � *
y.1009	parameter*�: H��� � +
add.1010add*�: ������ � "+
*�
*�*�x.1008y.1009(�0��
AddComputation.1160(
x.1161	parameter*�: ��	� � *
y.1162	parameter*�: H��	� � +
add.1163add*�: ��	��	�	� � "+
*�
*�*�x.1161y.1162(�	0�	�
AddComputation.1470(
x.1471	parameter*�: ��� � *
y.1472	parameter*�: H��� � +
add.1473add*�: ������ � "+
*�
*�*�x.1471y.1472(�0��
AddComputation.1497(
x.1498	parameter*�: ��� � *
y.1499	parameter*�: H��� � +
add.1500add*�: ������ � "+
*�
*�*�x.1498y.1499(�0��
AddComputation.1650(
x.1651	parameter*�: ��� � *
y.1652	parameter*�: H��� � +
add.1653add*�: ������ � "+
*�
*�*�x.1651y.1652(�0��
xla_ge_computation.3725(
x.3726	parameter*�: ��� � *
y.3727	parameter*�: H��� � @
compare.3728compare*�: ������GE� �FLOAT� "+
*�
*�*�x.3726y.3727(�0��
AddComputation.3729(
x.3730	parameter*�: ��� � *
y.3731	parameter*�: H��� � +
add.3732add*�: ������ � "+
*�
*�*�x.3730y.3731(�0��
xla_ge_computation.3742(
x.3743	parameter*�: ��� � *
y.3744	parameter*�: H��� � @
compare.3745compare*�: ������GE� �FLOAT� "+
*�
*�*�x.3743y.3744(�0��
AddComputation.3746(
x.3747	parameter*�: ��� � *
y.3748	parameter*�: H��� � +
add.3749add*�: ������ � "+
*�
*�*�x.3747y.3748(�0��
xla_ge_computation.3759(
x.3760	parameter*�: ��� � *
y.3761	parameter*�: H��� � @
compare.3762compare*�: ������GE� �FLOAT� "+
*�
*�*�x.3760y.3761(�0��
AddComputation.3763(
x.3764	parameter*�: ��� � *
y.3765	parameter*�: H��� � +
add.3766add*�: ������ � "+
*�
*�*�x.3764y.3765(�0��
!multiply.1.reduce_sub_computation%
lhs	parameter*�: ��0� � '
rhs	parameter*�: H��0� � &
addadd*�: ��0��0�0� � "%
*�
*�*�lhsrhs(�00�0�
!multiply.2.reduce_sub_computation'
lhs.1	parameter*�: ��1� � )
rhs.1	parameter*�: H��1� � (
add.6add*�: ��1��1�1� � ")
*�
*�*�lhs.1rhs.1(�10�1�
!multiply.3.reduce_sub_computation'
lhs.2	parameter*�: ��1� � )
rhs.2	parameter*�: H��1� � )
add.12add*�: ��1��1�1� � ")
*�
*�*�lhs.2rhs.2(�10�1��
SyncTensorsGraph.5569�
	p547.5548	parameter *	
 �2    :$
xla__device_dataxla__device_dataH���+�

neff_input_namesinput547� �
p0.1	parameter*�:�
xla__device_dataxla__device_datar/home/ubuntu/aws-neuron-samples/aws_neuron_venv_pytorch/lib/python3.10/site-packages/torch/cuda/amp/grad_scaler.py �x��

neff_input_namesinput0� �

constant.3constant*�:�
prim__Constantprim__Constantr/home/ubuntu/aws-neuron-samples/aws_neuron_venv_pytorch/lib/python3.10/site-packages/torch/cuda/amp/grad_scaler.py �xB
*�" �� � �

constant.4constantU*
 �2 : Bi
U*
 �2 U                                                                                   ��5� � a
	broadcast	broadcast�U*
 �2   :
xla__unselectxla__unselectr��/��5� � >
constant.124constant*�: B
*�B    ��6� � `
broadcast.3	broadcast�U*
 �2   :
xla__unselectxla__unselect��6��6� � �

constant.6constantU*
 �2 : Bi
U*
 �2 U                                                                                   ��5� � c
broadcast.5	broadcast�U*
 �2   :
xla__unselectxla__unselectr��/��5� � >
constant.125constant*�: B
*�B    ��6� � `
broadcast.7	broadcast�U*
 �2   :
xla__unselectxla__unselect��6��6� � �

constant.9constantU*
 �2 : Bi
U*
 �2 U     ��5� � c
broadcast.8	broadcast�AU*
 �2   :
xla__unselectxla__unselectr��/��5� � ^
constant.359constant*�: 
prim__Constantprim__ConstantB
*�B    ��� � ]
broadcast.363	broadcast�AP*
 �2  :
aten__expandaten__expand��4��� � x
p25.346	parameter*
 �2  :$
xla__device_dataxla__device_dataH���

neff_input_namesinput25� Q
reshape.347reshape*
 �2 :

aten__view
aten__view����� � [
constant.353constant*�: 
prim__Constantprim__ConstantB
*�* ��� � Q
broadcast.354	broadcast*
 �2 :
aten__ltaten__lt����� � ]
compare.355compare*
 �2 :
aten__ltaten__lt������LT� �SIGNED� l
p20.237	parameter*�:$
xla__device_dataxla__device_dataH���

neff_input_namesinput20� Y
broadcast.351	broadcast*
 �2 :
aten__expandaten__expand��4��� � I
add.352add*
 �2 :
	aten__add	aten__add�����4� � U

select.356select*
 �2 :
aten__whereaten__where������� � V
reshape.357reshape*
 �2  :
aten__stackaten__stack����� � x
p24.342	parameterP*
 �2  :$
xla__device_dataxla__device_dataH���

neff_input_namesinput24� \
logistic.343logisticP*
 �2  :
aten__sigmoidaten__sigmoid����� � x
p23.340	parameterP*
 �2  :$
xla__device_dataxla__device_dataH���

neff_input_namesinput23� V
subtract.344subtractP*
 �2  :
	aten__sub	aten__sub������ � j
p12.49	parameter*�:$
xla__device_dataxla__device_dataH�1�

neff_input_namesinput12� D

divide.330divide*�:
	aten__div	aten__div���1� � \
broadcast.334	broadcastP*
 �2  :
aten__expandaten__expand��4��� � V
multiply.345multiplyP*
 �2  :
	aten__mul	aten__mul�����4� � v
scatter.369scatter�AP*
 �2  :"
aten__index_putaten__index_put����4�����
   � � X
reshape.371reshape�AP*
 �2   :

aten__view
aten__view����� � \
constant.380constant*�:
xla__unselectxla__unselectB
*�B    ��� � c
pad.381pad�AU*
 �2   :
xla__unselectxla__unselect�
 
 
������ � ^
constant.372constant*�: 
prim__Constantprim__ConstantB
*�B    ��� � `
broadcast.376	broadcast�AU*
 �2   :
aten__expandaten__expand��4��� � `

select.383select�AU*
 �2   :
xla__unselectxla__unselect����/��4� � �
constant.11constantU*
 �2 : Bi
U*
 �2 U                                                                                    ��5� � d
broadcast.12	broadcast�AU*
 �2   :
xla__unselectxla__unselectr��/��5� � y
p22.302	parameter�A*
 �2  :$
xla__device_dataxla__device_dataH���

neff_input_namesinput22� ]
logistic.303logistic�A*
 �2  :
aten__sigmoidaten__sigmoid����� � y
p21.300	parameter�A*
 �2  :$
xla__device_dataxla__device_dataH���

neff_input_namesinput21� W
subtract.304subtract�A*
 �2  :
	aten__sub	aten__sub������ � D

divide.290divide*�:
	aten__div	aten__div���1� � ]
broadcast.294	broadcast�A*
 �2  :
aten__expandaten__expand��4��� � W
multiply.305multiply�A*
 �2  :
	aten__mul	aten__mul�����4� � X
reshape.306reshape�A*
 �2   :

aten__view
aten__view����� � \
constant.315constant*�:
xla__unselectxla__unselectB
*�B    ��� � e
pad.316pad�AU*
 �2   :
xla__unselectxla__unselect�

 
 
P������ � ^
constant.307constant*�: 
prim__Constantprim__ConstantB
*�B    ��� � `
broadcast.311	broadcast�AU*
 �2   :
aten__expandaten__expand��4��� � `

select.318select�AU*
 �2   :
xla__unselectxla__unselect����/��4� � P
add.394add�AU*
 �2   :
	aten__add	aten__add������ � �
constant.12constantU*
 �2 : Bi
U*
 �2 U                                                                                 ��5� � d
broadcast.19	broadcast�AU*
 �2   :
xla__unselectxla__unselectr��/��5� � ^
constant.249constant*�: 
prim__Constantprim__ConstantB
*�B    ��� � ]
broadcast.253	broadcast�A*
 �2  :
aten__expandaten__expand��4��� � x
p19.235	parameter*
 �2  :$
xla__device_dataxla__device_dataH���

neff_input_namesinput19� Q
reshape.236reshape*
 �2 :

aten__view
aten__view����� � [
constant.243constant*�: 
prim__Constantprim__ConstantB
*�* ��� � Q
broadcast.244	broadcast*
 �2 :
aten__ltaten__lt����� � ]
compare.245compare*
 �2 :
aten__ltaten__lt������LT� �SIGNED� Y
broadcast.241	broadcast*
 �2 :
aten__expandaten__expand��4��� � I
add.242add*
 �2 :
	aten__add	aten__add�����4� � U

select.246select*
 �2 :
aten__whereaten__where������� � V
reshape.247reshape*
 �2  :
aten__stackaten__stack����� � O
constant.14constant*
 �2 : B
*
 �2   ��5� � `
broadcast.20	broadcast*
 �2  :
xla__unselectxla__unselectr��/��5� � j
p14.62	parameter*�:$
xla__device_dataxla__device_dataH�>�

neff_input_namesinput14� F
multiply.66multiply*�:
	aten__mul	aten__mul�B�>� � B
	divide.67divide*�:
	aten__div	aten__div�C�B1� � A
negate.0negate*�:
	aten__neg	aten__neg��/�C� � X
broadcast.24	broadcast*
 �2 :
aten__expandaten__expand��/��/� � s
p13.59	parameter*
 �2 :$
xla__device_dataxla__device_dataH�;�

neff_input_namesinput13� W
broadcast.53	broadcast*
 �2 :
aten__expandaten__expand��4�1� � P
multiply.61multiply*
 �2 :
	aten__mul	aten__mul�=�;�4� � P
multiply.73multiply*
 �2 :
	aten__mul	aten__mul�I��/=� � J
	negate.74negate*
 �2 :
	aten__neg	aten__neg�J�I� � s
p11.46	parameter*
 �2 :$
xla__device_dataxla__device_dataH�.�

neff_input_namesinput11� s
p10.45	parameter*
 �2 :$
xla__device_dataxla__device_dataH
�-�

neff_input_namesinput10� 7
multiplymultiply*
 �2 : ��/�--� � M
	divide.48divide*
 �2 :
	aten__div	aten__div��/�.�/� � P
multiply.75multiply*
 �2 :
	aten__mul	aten__mul�K�J�/� � X
broadcast.212	broadcast*
 �2  :
	aten__mul	aten__mulr ��4�K� � u
p18.203	parameter*
 �2 :$
xla__device_dataxla__device_dataH���

neff_input_namesinput18� Y
broadcast.207	broadcast*
 �2  :
	aten__div	aten__divr ��4��� � x
p17.202	parameter*
 �2  :$
xla__device_dataxla__device_dataH���

neff_input_namesinput17� R

divide.208divide*
 �2  :
	aten__div	aten__div����4�� � V
multiply.213multiply*
 �2  :
	aten__mul	aten__mul����4�� � \
constant.222constant*�:
xla__unselectxla__unselectB
*�B    ��� � ]
pad.223pad*
 �2  :
xla__unselectxla__unselect�
 
������ � ^
constant.214constant*�: 
prim__Constantprim__ConstantB
*�B    ��� � \
broadcast.218	broadcast*
 �2  :
aten__expandaten__expand��4��� � \

select.225select*
 �2  :
xla__unselectxla__unselect����/��4� � O
constant.15constant*
 �2 : B
*
 �2   ��5� � `
broadcast.25	broadcast*
 �2  :
xla__unselectxla__unselectr��/��5� � x
p16.147	parameter*
 �2  :$
xla__device_dataxla__device_dataH���

neff_input_namesinput16� x
p15.146	parameter*
 �2  :$
xla__device_dataxla__device_dataH���

neff_input_namesinput15� _
compare.155compare*
 �2  :
aten__gtaten__gt������GT� �FLOAT� x
constant.156constant*�::
aten__masked_fill%aten__masked_fill.1/aten__masked_fillB
*�B    ��� � z
broadcast.157	broadcast*
 �2  ::
aten__masked_fill%aten__masked_fill.1/aten__masked_fill����� � _
compare.151compare*
 �2  :
aten__eqaten__eq������EQ� �FLOAT� K
	divide.78divide*
 �2 :
	aten__div	aten__div�N�I-� � J
	negate.76negate*
 �2 :
	aten__neg	aten__neg�L�K� � E
add.79add*
 �2 :
	aten__add	aten__add�O�NL� � q
p9.39	parameter*
 �2 :$
xla__device_dataxla__device_dataH	�'�

neff_input_namesinput9� O
multiply.80multiply*
 �2 :
	aten__mul	aten__mul�P�O'� � W
broadcast.84	broadcast*
 �2  :
	aten__mul	aten__mulr ��4�P� � q
p8.33	parameter*
 �2 :$
xla__device_dataxla__device_dataH�!�

neff_input_namesinput8� W
broadcast.37	broadcast*
 �2  :
	aten__div	aten__divr ��4�!� � t
p7.32	parameter*
 �2  :$
xla__device_dataxla__device_dataH� �

neff_input_namesinput7� O
	divide.38divide*
 �2  :
	aten__div	aten__div�&��4 � � S
multiply.85multiply*
 �2  :
	aten__mul	aten__mul�U��4&� � U
broadcast.149	broadcast*
 �2  :
	aten__div	aten__div���1� � Q

divide.150divide*
 �2  :
	aten__div	aten__div���U�� � W

select.152select*
 �2  :
aten__whereaten__where�����U� � x

select.158select*
 �2  ::
aten__masked_fill%aten__masked_fill.1/aten__masked_fill������� � U
broadcast.182	broadcast*
 �2  :
	aten__div	aten__div���1� � R

divide.183divide*
 �2  :
	aten__div	aten__div������ � \
constant.192constant*�:
xla__unselectxla__unselectB
*�B    ��� � ]
pad.193pad*
 �2  :
xla__unselectxla__unselect�
 
������ � ^
constant.184constant*�: 
prim__Constantprim__ConstantB
*�B    ��� � \
broadcast.188	broadcast*
 �2  :
aten__expandaten__expand��4��� � \

select.195select*
 �2  :
xla__unselectxla__unselect����/��4� � L
add.231add*
 �2  :
	aten__add	aten__add������ � O
constant.16constant*
 �2 : B
*
 �2   ��5� � `
broadcast.27	broadcast*
 �2  :
xla__unselectxla__unselectr��0��5� � \
constant.167constant*�:
xla__unselectxla__unselectB
*�B    ��� � ]
pad.168pad*
 �2  :
xla__unselectxla__unselect�
 
������ � ^
constant.159constant*�: 
prim__Constantprim__ConstantB
*�B    ��� � \
broadcast.163	broadcast*
 �2  :
aten__expandaten__expand��4��� � \

select.170select*
 �2  :
xla__unselectxla__unselect����0��4� � L
add.232add*
 �2  :
	aten__add	aten__add������ � O
constant.17constant*
 �2 : B
*
 �2   ��5� � `
broadcast.29	broadcast*
 �2  :
xla__unselectxla__unselectr��0��5� � t
p6.30	parameter*
 �2  :$
xla__device_dataxla__device_dataH��

neff_input_namesinput6� t
p5.29	parameter*
 �2  :$
xla__device_dataxla__device_dataH��

neff_input_namesinput5� [

compare.93compare*
 �2  :
aten__ltaten__lt���LT� �FLOAT� v
constant.94constant*�::
aten__masked_fill%aten__masked_fill.2/aten__masked_fillB
*�B    �^� � w
broadcast.95	broadcast*
 �2  ::
aten__masked_fill%aten__masked_fill.2/aten__masked_fill�_�^� � [

compare.89compare*
 �2  :
aten__eqaten__eq�Y��EQ� �FLOAT� M
	negate.86negate*
 �2  :
	aten__neg	aten__neg�V�U� � S
broadcast.87	broadcast*
 �2  :
	aten__div	aten__div�W�1� � N
	divide.88divide*
 �2  :
	aten__div	aten__div�X�VW� � S
	select.90select*
 �2  :
aten__whereaten__where�Z�YXV� � s
	select.96select*
 �2  ::
aten__masked_fill%aten__masked_fill.2/aten__masked_fill�`�_Z� � N

negate.120negate*
 �2  :
	aten__neg	aten__neg�x�`� � T
broadcast.121	broadcast*
 �2  :
	aten__div	aten__div�y�1� � O

divide.122divide*
 �2  :
	aten__div	aten__div�z�xy� � \
constant.131constant*�:
xla__unselectxla__unselectB
*�B    ��� � \
pad.132pad*
 �2  :
xla__unselectxla__unselect�
 
���z�� � ]
constant.123constant*�: 
prim__Constantprim__ConstantB
*�B    �{� � [
broadcast.127	broadcast*
 �2  :
aten__expandaten__expand��4�{� � \

select.134select*
 �2  :
xla__unselectxla__unselect����0��4� � L
add.233add*
 �2  :
	aten__add	aten__add������ � O
constant.18constant*
 �2 : B
*
 �2   ��5� � `
broadcast.30	broadcast*
 �2  :
xla__unselectxla__unselectr��0��5� � [
constant.105constant*�:
xla__unselectxla__unselectB
*�B    �i� � Z
pad.106pad*
 �2  :
xla__unselectxla__unselect�
 
�j�`i� � \
constant.97constant*�: 
prim__Constantprim__ConstantB
*�B    �a� � [
broadcast.101	broadcast*
 �2  :
aten__expandaten__expand��4�a� � Z

select.108select*
 �2  :
xla__unselectxla__unselect�l��0j�4� � K
add.234add*
 �2  :
	aten__add	aten__add����l� � v
scatter.259scatter�A*
 �2  :"
aten__index_putaten__index_put����4�����
   � � X
reshape.261reshape�A*
 �2   :

aten__view
aten__view����� � \
constant.270constant*�:
xla__unselectxla__unselectB
*�B    ��� � c
pad.271pad�AU*
 �2   :
xla__unselectxla__unselect�
 
 
Q������ � ^
constant.262constant*�: 
prim__Constantprim__ConstantB
*�B    ��� � `
broadcast.266	broadcast�AU*
 �2   :
aten__expandaten__expand��4��� � `

select.273select�AU*
 �2   :
xla__unselectxla__unselect����/��4� � P
add.395add�AU*
 �2   :
	aten__add	aten__add������ � o
	slice.396slice�U*
 �2   :
xla__selectxla__select���>�A�U����� � `

select.437select�U*
 �2   :
xla__unselectxla__unselect����/�6�� � �
constant.20constantU*
 �2 : Bi
U*
 �2 U                                                                                   ��5� � d
broadcast.33	broadcast�U*
 �2   :
xla__unselectxla__unselectr��0��5� � q
	slice.403slice�*
 �2   :
xla__selectxla__select���>�A���7��� � h
p3.12	parameter*�:$
xla__device_dataxla__device_dataH��

neff_input_namesinput3� Y
broadcast.404	broadcast�*
 �2   :
	aten__mul	aten__mul���� � Z
multiply.405multiply�*
 �2   :
	aten__mul	aten__mul����7�� � x
p4.18	parameter�*
 �2   :$
xla__device_dataxla__device_dataH��

neff_input_namesinput4� Y
multiply.406multiply�*
 �2   :
	aten__mul	aten__mul����� � \
constant.415constant*�:
xla__unselectxla__unselectB
*�B    ��� � e
pad.416pad�U*
 �2   :
xla__unselectxla__unselect�

 
 
Q������ � ^
constant.407constant*�: 
prim__Constantprim__ConstantB
*�B    ��� � `
broadcast.411	broadcast�U*
 �2   :
aten__expandaten__expand��4��� � `

select.418select�U*
 �2   :
xla__unselectxla__unselect����0��4� � O
add.90add�U*
 �2   :
	aten__add	aten__add��7���� � `

select.478select�U*
 �2   :
xla__unselectxla__unselect����/�6�7� � �
constant.21constantU*
 �2 : Bi
U*
 �2 U                                                                                   ��5� � d
broadcast.38	broadcast�U*
 �2   :
xla__unselectxla__unselectr��0��5� � l
	slice.445slice�*
 �2   :
xla__selectxla__select��������7� � Y
broadcast.446	broadcast�*
 �2   :
	aten__mul	aten__mul���� � Z
multiply.447multiply�*
 �2   :
	aten__mul	aten__mul������ � \
constant.456constant*�:
xla__unselectxla__unselectB
*�B    ��� � c
pad.457pad�U*
 �2   :
xla__unselectxla__unselect�
 
 
S������ � ^
constant.448constant*�: 
prim__Constantprim__ConstantB
*�B    ��� � `
broadcast.452	broadcast�U*
 �2   :
aten__expandaten__expand��4��� � `

select.459select�U*
 �2   :
xla__unselectxla__unselect����0��4� � O
add.91add�U*
 �2   :
	aten__add	aten__add��7���� � [
reshape.558reshapeU*

 �2     :
	aten__add	aten__add��7��7� � e
transpose.32	transposeU*

 �2     :
	aten__add	aten__addr ��7��7� � X
reshape.505reshapeU*	
 �2    :
	aten__add	aten__add��6��7� � v
	slice.483slice*	
 �2    :
xla__selectxla__select��������6� � �
constant.489constant*�:R
'aten__convolution_backward_overrideable'aten__convolution_backward_overrideableB
*�B    ��� � �

reduce.494reduce*
 �2 :R
'aten__convolution_backward_overrideable'aten__convolution_backward_overrideabler �������� � �
custom-call.244custom-call*
 �2 :�
'xla___op_TransferWithStaticRingTransfer'xla___op_TransferWithStaticRingTransferv/home/ubuntu/aws-neuron-samples/aws_neuron_venv_pytorch/lib/python3.10/site-packages/torch_xla/core/xla_op_registry.py ,x�AwsNeuronTransferWithStaticRing��-����
 �� � z
p2.6	parameter�*	
 �2    :$
xla__device_dataxla__device_dataH��

neff_input_namesinput2� <
reshape.376reshape*
 �2  : ��7��� � e
broadcast.341	broadcast�*	
 �2    :
xla__selectxla__selectr��7��7� � D

multiply.4multiply�*	
 �2    : ��6��7� � =
constant.22constant*�: B
*�B    ��5� � D
reduce.3reduce�*
 �2  : r��7��6�5��0� � �
reshape.120reshape�*	
 �2    :R
'aten__convolution_backward_overrideable'aten__convolution_backward_overrideable��7��7� � �
custom-call.245custom-call�*	
 �2    :�
'xla___op_TransferWithStaticRingTransfer'xla___op_TransferWithStaticRingTransferv/home/ubuntu/aws-neuron-samples/aws_neuron_venv_pytorch/lib/python3.10/site-packages/torch_xla/core/xla_op_registry.py ,x�AwsNeuronTransferWithStaticRing��-��7��
 �� � t
	slice.510slice*	
 �2    :
xla__selectxla__select��������6� � �
constant.516constant*�:R
'aten__convolution_backward_overrideable'aten__convolution_backward_overrideableB
*�B    ��� � �

reduce.521reduce*
 �2 :R
'aten__convolution_backward_overrideable'aten__convolution_backward_overrideabler �������� � �
custom-call.246custom-call*
 �2 :�
'xla___op_TransferWithStaticRingTransfer'xla___op_TransferWithStaticRingTransferv/home/ubuntu/aws-neuron-samples/aws_neuron_venv_pytorch/lib/python3.10/site-packages/torch_xla/core/xla_op_registry.py ,x�AwsNeuronTransferWithStaticRing��-����
 �� � �
convolution.514convolution�*	
 �2    :R
'aten__convolution_backward_overrideable'aten__convolution_backward_overrideablez
(0
(0� 28HPZb ������
  �� � �
transpose.515	transpose�*	
 �2    :R
'aten__convolution_backward_overrideable'aten__convolution_backward_overrideabler ����� � �
custom-call.247custom-call�*	
 �2    :�
'xla___op_TransferWithStaticRingTransfer'xla___op_TransferWithStaticRingTransferv/home/ubuntu/aws-neuron-samples/aws_neuron_venv_pytorch/lib/python3.10/site-packages/torch_xla/core/xla_op_registry.py ,x�AwsNeuronTransferWithStaticRing��-����
 �� � 
p30.539	parameter�*	
 �2    :$
xla__device_dataxla__device_dataH���

neff_input_namesinput30� v
p29.538	parameter�*
 �2 :$
xla__device_dataxla__device_dataH���

neff_input_namesinput29� v
p28.537	parameter�*
 �2 :$
xla__device_dataxla__device_dataH���

neff_input_namesinput28� �
constant.558constant*�:D
 aten__native_batch_norm_backward aten__native_batch_norm_backwardB
*�B  �?��� � �
broadcast.559	broadcast�*
 �2 :D
 aten__native_batch_norm_backward aten__native_batch_norm_backward����� � v
p27.536	parameter�*
 �2 :$
xla__device_dataxla__device_dataH���

neff_input_namesinput27� ~

divide.560divide�*
 �2 :D
 aten__native_batch_norm_backward aten__native_batch_norm_backward������ � �
multiply.561multiply�*
 �2 :D
 aten__native_batch_norm_backward aten__native_batch_norm_backward������ � =
constant.23constant*�: B
*�Bo����5� � �
broadcast.43	broadcast�*
 �2 :D
 aten__native_batch_norm_backward aten__native_batch_norm_backward��0��5� � v
add.1add�*
 �2 :D
 aten__native_batch_norm_backward aten__native_batch_norm_backward��0���0� � z
p1.5	parameter�*	
 �2    :$
xla__device_dataxla__device_dataH��

neff_input_namesinput1� �
reshape.121reshape�*	
 �2    :R
'aten__convolution_backward_overrideable'aten__convolution_backward_overrideable��0�� � �
convolution.486convolution�*	
 �2    :R
'aten__convolution_backward_overrideable'aten__convolution_backward_overrideablez
(0
(0� 2 @PZb�����0��
  �� � 
p26.509	parameter�*	
 �2    :$
xla__device_dataxla__device_dataH���

neff_input_namesinput26� �
transpose.511	transpose�*	
 �2    :R
'aten__convolution_backward_overrideable'aten__convolution_backward_overrideabler ����� � �
convolution.513convolution�*	
 �2    :R
'aten__convolution_backward_overrideable'aten__convolution_backward_overrideablez
(0
(0� 2 @PZb�������
  �� � S
add.547add�*	
 �2    :
	aten__add	aten__add������ � 
p31.540	parameter�*	
 �2    :$
xla__device_dataxla__device_dataH���

neff_input_namesinput31� o
logistic.549logistic�*	
 �2    :*
aten__silu_backwardaten__silu_backward����� � h
constant.548constant*�:*
aten__silu_backwardaten__silu_backwardB
*�B  �?��� � q
broadcast.553	broadcast�*	
 �2    :*
aten__silu_backwardaten__silu_backward����� � q
broadcast.550	broadcast�*	
 �2    :*
aten__silu_backwardaten__silu_backward����� � q
subtract.551subtract�*	
 �2    :*
aten__silu_backwardaten__silu_backward������ � q
multiply.552multiply�*	
 �2    :*
aten__silu_backwardaten__silu_backward������ � g
add.554add�*	
 �2    :*
aten__silu_backwardaten__silu_backward������ � q
multiply.555multiply�*	
 �2    :*
aten__silu_backwardaten__silu_backward������ � q
multiply.556multiply�*	
 �2    :*
aten__silu_backwardaten__silu_backward������ � �
batch-norm-grad.564batch-norm-gradD"�*	
 �2    "�*
 �2 "�*
 �2 :D
 aten__native_batch_norm_backward aten__native_batch_norm_backward�o�:����
����0�� � �
get-tuple-element.567get-tuple-element�*
 �2 :D
 aten__native_batch_norm_backward aten__native_batch_norm_backwardh����� � �
custom-call.248custom-call�*
 �2 :�
'xla___op_TransferWithStaticRingTransfer'xla___op_TransferWithStaticRingTransferv/home/ubuntu/aws-neuron-samples/aws_neuron_venv_pytorch/lib/python3.10/site-packages/torch_xla/core/xla_op_registry.py ,x�AwsNeuronTransferWithStaticRing��-����
 �� � �
get-tuple-element.566get-tuple-element�*
 �2 :D
 aten__native_batch_norm_backward aten__native_batch_norm_backwardh����� � �
custom-call.249custom-call�*
 �2 :�
'xla___op_TransferWithStaticRingTransfer'xla___op_TransferWithStaticRingTransferv/home/ubuntu/aws-neuron-samples/aws_neuron_venv_pytorch/lib/python3.10/site-packages/torch_xla/core/xla_op_registry.py ,x�AwsNeuronTransferWithStaticRing��-����
 �� � 
p33.583	parameter�*	
 �2    :$
xla__device_dataxla__device_dataH!���

neff_input_namesinput33� �
get-tuple-element.565get-tuple-element�*	
 �2    :D
 aten__native_batch_norm_backward aten__native_batch_norm_backward����� � �
convolution.587convolution��*	
 �2    :R
'aten__convolution_backward_overrideable'aten__convolution_backward_overrideablez
 (0
 (0� 28HPZb �������
  �� � �
transpose.588	transpose��*	
 �2    :R
'aten__convolution_backward_overrideable'aten__convolution_backward_overrideabler ����� � �
custom-call.250custom-call��*	
 �2    :�
'xla___op_TransferWithStaticRingTransfer'xla___op_TransferWithStaticRingTransferv/home/ubuntu/aws-neuron-samples/aws_neuron_venv_pytorch/lib/python3.10/site-packages/torch_xla/core/xla_op_registry.py ,x�AwsNeuronTransferWithStaticRing��-����
 �� � 
p37.605	parameter�*	
 �2    :$
xla__device_dataxla__device_dataH%���

neff_input_namesinput37� v
p36.604	parameter�*
 �2 :$
xla__device_dataxla__device_dataH$���

neff_input_namesinput36� v
p35.603	parameter�*
 �2 :$
xla__device_dataxla__device_dataH#���

neff_input_namesinput35� �
constant.617constant*�:D
 aten__native_batch_norm_backward aten__native_batch_norm_backwardB
*�B  �?��� � �
broadcast.618	broadcast�*
 �2 :D
 aten__native_batch_norm_backward aten__native_batch_norm_backward����� � v
p34.602	parameter�*
 �2 :$
xla__device_dataxla__device_dataH"���

neff_input_namesinput34� ~

divide.619divide�*
 �2 :D
 aten__native_batch_norm_backward aten__native_batch_norm_backward������ � �
multiply.620multiply�*
 �2 :D
 aten__native_batch_norm_backward aten__native_batch_norm_backward������ � =
constant.25constant*�: B
*�Bo����5� � �
broadcast.46	broadcast�*
 �2 :D
 aten__native_batch_norm_backward aten__native_batch_norm_backward��0��5� � v
add.2add�*
 �2 :D
 aten__native_batch_norm_backward aten__native_batch_norm_backward��0���0� � �
p32.582	parameter��*	
 �2    :$
xla__device_dataxla__device_dataH ���

neff_input_namesinput32� �
transpose.584	transpose��*	
 �2    :R
'aten__convolution_backward_overrideable'aten__convolution_backward_overrideabler ����� � �
reverse.585reverse��*	
 �2    :R
'aten__convolution_backward_overrideable'aten__convolution_backward_overrideabler ����� � �
convolution.586convolution�*	
 �2    :R
'aten__convolution_backward_overrideable'aten__convolution_backward_overrideablez
 (0
 (0� 2 @PZb�������
  �� � 
p38.606	parameter�*	
 �2    :$
xla__device_dataxla__device_dataH&���

neff_input_namesinput38� o
logistic.608logistic�*	
 �2    :*
aten__silu_backwardaten__silu_backward����� � h
constant.607constant*�:*
aten__silu_backwardaten__silu_backwardB
*�B  �?��� � q
broadcast.612	broadcast�*	
 �2    :*
aten__silu_backwardaten__silu_backward����� � q
broadcast.609	broadcast�*	
 �2    :*
aten__silu_backwardaten__silu_backward����� � q
subtract.610subtract�*	
 �2    :*
aten__silu_backwardaten__silu_backward������ � q
multiply.611multiply�*	
 �2    :*
aten__silu_backwardaten__silu_backward������ � g
add.613add�*	
 �2    :*
aten__silu_backwardaten__silu_backward������ � q
multiply.614multiply�*	
 �2    :*
aten__silu_backwardaten__silu_backward������ � q
multiply.615multiply�*	
 �2    :*
aten__silu_backwardaten__silu_backward������ � �
batch-norm-grad.623batch-norm-gradD"�*	
 �2    "�*
 �2 "�*
 �2 :D
 aten__native_batch_norm_backward aten__native_batch_norm_backward�o�:����
����0�� � �
get-tuple-element.626get-tuple-element�*
 �2 :D
 aten__native_batch_norm_backward aten__native_batch_norm_backwardh����� � �
custom-call.251custom-call�*
 �2 :�
'xla___op_TransferWithStaticRingTransfer'xla___op_TransferWithStaticRingTransferv/home/ubuntu/aws-neuron-samples/aws_neuron_venv_pytorch/lib/python3.10/site-packages/torch_xla/core/xla_op_registry.py ,x�AwsNeuronTransferWithStaticRing��-����
 �� � �
get-tuple-element.625get-tuple-element�*
 �2 :D
 aten__native_batch_norm_backward aten__native_batch_norm_backwardh����� � �
custom-call.252custom-call�*
 �2 :�
'xla___op_TransferWithStaticRingTransfer'xla___op_TransferWithStaticRingTransferv/home/ubuntu/aws-neuron-samples/aws_neuron_venv_pytorch/lib/python3.10/site-packages/torch_xla/core/xla_op_registry.py ,x�AwsNeuronTransferWithStaticRing��-����
 �� � 
p40.642	parameter�*	
 �2    :$
xla__device_dataxla__device_dataH(���

neff_input_namesinput40� �
get-tuple-element.624get-tuple-element�*	
 �2    :D
 aten__native_batch_norm_backward aten__native_batch_norm_backward����� � �
convolution.646convolution��*	
 �2    :R
'aten__convolution_backward_overrideable'aten__convolution_backward_overrideablez
 (0
 (0� 28HPZb �������
  �� � �
transpose.647	transpose��*	
 �2    :R
'aten__convolution_backward_overrideable'aten__convolution_backward_overrideabler ����� � �
custom-call.253custom-call��*	
 �2    :�
'xla___op_TransferWithStaticRingTransfer'xla___op_TransferWithStaticRingTransferv/home/ubuntu/aws-neuron-samples/aws_neuron_venv_pytorch/lib/python3.10/site-packages/torch_xla/core/xla_op_registry.py ,x�AwsNeuronTransferWithStaticRing��-����
 �� � v
	slice.663sliceP*	
 �2    :
xla__selectxla__select��U������6� � �
constant.669constant*�:R
'aten__convolution_backward_overrideable'aten__convolution_backward_overrideableB
*�B    ��� � �

reduce.674reduceP*
 �2 :R
'aten__convolution_backward_overrideable'aten__convolution_backward_overrideabler �������� � �
custom-call.254custom-callP*
 �2 :�
'xla___op_TransferWithStaticRingTransfer'xla___op_TransferWithStaticRingTransferv/home/ubuntu/aws-neuron-samples/aws_neuron_venv_pytorch/lib/python3.10/site-packages/torch_xla/core/xla_op_registry.py ,x�AwsNeuronTransferWithStaticRing��-����
 �� � 
p42.662	parameter�*	
 �2    :$
xla__device_dataxla__device_dataH*���

neff_input_namesinput42� �
convolution.667convolution�P*	
 �2    :R
'aten__convolution_backward_overrideable'aten__convolution_backward_overrideablez
(0
(0� 28HPZb �������
  �� � �
transpose.668	transposeP�*	
 �2    :R
'aten__convolution_backward_overrideable'aten__convolution_backward_overrideabler ����� � �
custom-call.255custom-callP�*	
 �2    :�
'xla___op_TransferWithStaticRingTransfer'xla___op_TransferWithStaticRingTransferv/home/ubuntu/aws-neuron-samples/aws_neuron_venv_pytorch/lib/python3.10/site-packages/torch_xla/core/xla_op_registry.py ,x�AwsNeuronTransferWithStaticRing��-����
 �� � 
p46.692	parameter�*	
 �2    :$
xla__device_dataxla__device_dataH.���

neff_input_namesinput46� v
p45.691	parameter�*
 �2 :$
xla__device_dataxla__device_dataH-���

neff_input_namesinput45� v
p44.690	parameter�*
 �2 :$
xla__device_dataxla__device_dataH,���

neff_input_namesinput44� �
constant.704constant*�:D
 aten__native_batch_norm_backward aten__native_batch_norm_backwardB
*�B  �?��� � �
broadcast.705	broadcast�*
 �2 :D
 aten__native_batch_norm_backward aten__native_batch_norm_backward����� � v
p43.689	parameter�*
 �2 :$
xla__device_dataxla__device_dataH+���

neff_input_namesinput43� ~

divide.706divide�*
 �2 :D
 aten__native_batch_norm_backward aten__native_batch_norm_backward������ � �
multiply.707multiply�*
 �2 :D
 aten__native_batch_norm_backward aten__native_batch_norm_backward������ � =
constant.26constant*�: B
*�Bo����5� � �
broadcast.47	broadcast�*
 �2 :D
 aten__native_batch_norm_backward aten__native_batch_norm_backward��0��5� � v
add.3add�*
 �2 :D
 aten__native_batch_norm_backward aten__native_batch_norm_backward��0���0� � 
p41.661	parameterP�*	
 �2    :$
xla__device_dataxla__device_dataH)���

neff_input_namesinput41� �
transpose.664	transpose�P*	
 �2    :R
'aten__convolution_backward_overrideable'aten__convolution_backward_overrideabler ����� � �
convolution.666convolution�*	
 �2    :R
'aten__convolution_backward_overrideable'aten__convolution_backward_overrideablez
(0
(0� 2 @PZb�������
  �� � 
p47.693	parameter�*	
 �2    :$
xla__device_dataxla__device_dataH/���

neff_input_namesinput47� o
logistic.695logistic�*	
 �2    :*
aten__silu_backwardaten__silu_backward����� � h
constant.694constant*�:*
aten__silu_backwardaten__silu_backwardB
*�B  �?��� � q
broadcast.699	broadcast�*	
 �2    :*
aten__silu_backwardaten__silu_backward����� � q
broadcast.696	broadcast�*	
 �2    :*
aten__silu_backwardaten__silu_backward����� � q
subtract.697subtract�*	
 �2    :*
aten__silu_backwardaten__silu_backward������ � q
multiply.698multiply�*	
 �2    :*
aten__silu_backwardaten__silu_backward������ � g
add.700add�*	
 �2    :*
aten__silu_backwardaten__silu_backward������ � q
multiply.701multiply�*	
 �2    :*
aten__silu_backwardaten__silu_backward������ � q
multiply.702multiply�*	
 �2    :*
aten__silu_backwardaten__silu_backward������ � �
batch-norm-grad.710batch-norm-gradD"�*	
 �2    "�*
 �2 "�*
 �2 :D
 aten__native_batch_norm_backward aten__native_batch_norm_backward�o�:����
����0�� � �
get-tuple-element.713get-tuple-element�*
 �2 :D
 aten__native_batch_norm_backward aten__native_batch_norm_backwardh����� � �
custom-call.256custom-call�*
 �2 :�
'xla___op_TransferWithStaticRingTransfer'xla___op_TransferWithStaticRingTransferv/home/ubuntu/aws-neuron-samples/aws_neuron_venv_pytorch/lib/python3.10/site-packages/torch_xla/core/xla_op_registry.py ,x�AwsNeuronTransferWithStaticRing��-����
 �� � �
get-tuple-element.712get-tuple-element�*
 �2 :D
 aten__native_batch_norm_backward aten__native_batch_norm_backwardh����� � �
custom-call.257custom-call�*
 �2 :�
'xla___op_TransferWithStaticRingTransfer'xla___op_TransferWithStaticRingTransferv/home/ubuntu/aws-neuron-samples/aws_neuron_venv_pytorch/lib/python3.10/site-packages/torch_xla/core/xla_op_registry.py ,x�AwsNeuronTransferWithStaticRing��-����
 �� � 
p49.729	parameter�*	
 �2    :$
xla__device_dataxla__device_dataH1���

neff_input_namesinput49� �
get-tuple-element.711get-tuple-element�*	
 �2    :D
 aten__native_batch_norm_backward aten__native_batch_norm_backward����� � �
convolution.733convolution��*	
 �2    :R
'aten__convolution_backward_overrideable'aten__convolution_backward_overrideablez
 (0
 (0� 28HPZb �������
  �� � �
transpose.734	transpose��*	
 �2    :R
'aten__convolution_backward_overrideable'aten__convolution_backward_overrideabler ����� � �
custom-call.258custom-call��*	
 �2    :�
'xla___op_TransferWithStaticRingTransfer'xla___op_TransferWithStaticRingTransferv/home/ubuntu/aws-neuron-samples/aws_neuron_venv_pytorch/lib/python3.10/site-packages/torch_xla/core/xla_op_registry.py ,x�AwsNeuronTransferWithStaticRing��-����
 �� � 
p53.751	parameter�*	
 �2    :$
xla__device_dataxla__device_dataH5���

neff_input_namesinput53� v
p52.750	parameter�*
 �2 :$
xla__device_dataxla__device_dataH4���

neff_input_namesinput52� v
p51.749	parameter�*
 �2 :$
xla__device_dataxla__device_dataH3���

neff_input_namesinput51� �
constant.763constant*�:D
 aten__native_batch_norm_backward aten__native_batch_norm_backwardB
*�B  �?��� � �
broadcast.764	broadcast�*
 �2 :D
 aten__native_batch_norm_backward aten__native_batch_norm_backward����� � v
p50.748	parameter�*
 �2 :$
xla__device_dataxla__device_dataH2���

neff_input_namesinput50� ~

divide.765divide�*
 �2 :D
 aten__native_batch_norm_backward aten__native_batch_norm_backward������ � �
multiply.766multiply�*
 �2 :D
 aten__native_batch_norm_backward aten__native_batch_norm_backward������ � =
constant.27constant*�: B
*�Bo����5� � �
broadcast.48	broadcast�*
 �2 :D
 aten__native_batch_norm_backward aten__native_batch_norm_backward��0��5� � v
add.4add�*
 �2 :D
 aten__native_batch_norm_backward aten__native_batch_norm_backward��0���0� � �
p48.728	parameter��*	
 �2    :$
xla__device_dataxla__device_dataH0���

neff_input_namesinput48� �
transpose.730	transpose��*	
 �2    :R
'aten__convolution_backward_overrideable'aten__convolution_backward_overrideabler ����� � �
reverse.731reverse��*	
 �2    :R
'aten__convolution_backward_overrideable'aten__convolution_backward_overrideabler ����� � �
convolution.732convolution�*	
 �2    :R
'aten__convolution_backward_overrideable'aten__convolution_backward_overrideablez
 (0
 (0� 2 @PZb�������
  �� � 
p54.752	parameter�*	
 �2    :$
xla__device_dataxla__device_dataH6���

neff_input_namesinput54� o
logistic.754logistic�*	
 �2    :*
aten__silu_backwardaten__silu_backward����� � h
constant.753constant*�:*
aten__silu_backwardaten__silu_backwardB
*�B  �?��� � q
broadcast.758	broadcast�*	
 �2    :*
aten__silu_backwardaten__silu_backward����� � q
broadcast.755	broadcast�*	
 �2    :*
aten__silu_backwardaten__silu_backward����� � q
subtract.756subtract�*	
 �2    :*
aten__silu_backwardaten__silu_backward������ � q
multiply.757multiply�*	
 �2    :*
aten__silu_backwardaten__silu_backward������ � g
add.759add�*	
 �2    :*
aten__silu_backwardaten__silu_backward������ � q
multiply.760multiply�*	
 �2    :*
aten__silu_backwardaten__silu_backward������ � q
multiply.761multiply�*	
 �2    :*
aten__silu_backwardaten__silu_backward������ � �
batch-norm-grad.769batch-norm-gradD"�*	
 �2    "�*
 �2 "�*
 �2 :D
 aten__native_batch_norm_backward aten__native_batch_norm_backward�o�:����
����0�� � �
get-tuple-element.772get-tuple-element�*
 �2 :D
 aten__native_batch_norm_backward aten__native_batch_norm_backwardh����� � �
custom-call.259custom-call�*
 �2 :�
'xla___op_TransferWithStaticRingTransfer'xla___op_TransferWithStaticRingTransferv/home/ubuntu/aws-neuron-samples/aws_neuron_venv_pytorch/lib/python3.10/site-packages/torch_xla/core/xla_op_registry.py ,x�AwsNeuronTransferWithStaticRing��-����
 �� � �
get-tuple-element.771get-tuple-element�*
 �2 :D
 aten__native_batch_norm_backward aten__native_batch_norm_backwardh����� � �
custom-call.260custom-call�*
 �2 :�
'xla___op_TransferWithStaticRingTransfer'xla___op_TransferWithStaticRingTransferv/home/ubuntu/aws-neuron-samples/aws_neuron_venv_pytorch/lib/python3.10/site-packages/torch_xla/core/xla_op_registry.py ,x�AwsNeuronTransferWithStaticRing��-����
 �� � �
get-tuple-element.770get-tuple-element�*	
 �2    :D
 aten__native_batch_norm_backward aten__native_batch_norm_backward����� � �
convolution.791convolution��*	
 �2    :R
'aten__convolution_backward_overrideable'aten__convolution_backward_overrideablez
 (0
 (0� 28HPZb �������
  �� � �
transpose.792	transpose��*	
 �2    :R
'aten__convolution_backward_overrideable'aten__convolution_backward_overrideabler ����� � �
custom-call.261custom-call��*	
 �2    :�
'xla___op_TransferWithStaticRingTransfer'xla___op_TransferWithStaticRingTransferv/home/ubuntu/aws-neuron-samples/aws_neuron_venv_pytorch/lib/python3.10/site-packages/torch_xla/core/xla_op_registry.py ,x�AwsNeuronTransferWithStaticRing��-����
 �� � 
p59.809	parameter�*	
 �2    :$
xla__device_dataxla__device_dataH;���

neff_input_namesinput59� v
p58.808	parameter�*
 �2 :$
xla__device_dataxla__device_dataH:���

neff_input_namesinput58� v
p57.807	parameter�*
 �2 :$
xla__device_dataxla__device_dataH9���

neff_input_namesinput57� �
constant.828constant*�:D
 aten__native_batch_norm_backward aten__native_batch_norm_backwardB
*�B  �?��� � �
broadcast.829	broadcast�*
 �2 :D
 aten__native_batch_norm_backward aten__native_batch_norm_backward����� � v
p56.806	parameter�*
 �2 :$
xla__device_dataxla__device_dataH8���

neff_input_namesinput56� ~

divide.830divide�*
 �2 :D
 aten__native_batch_norm_backward aten__native_batch_norm_backward������ � �
multiply.831multiply�*
 �2 :D
 aten__native_batch_norm_backward aten__native_batch_norm_backward������ � =
constant.28constant*�: B
*�Bo����5� � �
broadcast.49	broadcast�*
 �2 :D
 aten__native_batch_norm_backward aten__native_batch_norm_backward��0��5� � v
add.5add�*
 �2 :D
 aten__native_batch_norm_backward aten__native_batch_norm_backward��0���0� � �
p39.641	parameter��*	
 �2    :$
xla__device_dataxla__device_dataH'���

neff_input_namesinput39� �
transpose.643	transpose��*	
 �2    :R
'aten__convolution_backward_overrideable'aten__convolution_backward_overrideabler ����� � �
reverse.644reverse��*	
 �2    :R
'aten__convolution_backward_overrideable'aten__convolution_backward_overrideabler ����� � �
convolution.645convolution�*	
 �2    :R
'aten__convolution_backward_overrideable'aten__convolution_backward_overrideablez
 (0
 (0� 2 @PZb�������
  �� � �
p55.787	parameter��*	
 �2    :$
xla__device_dataxla__device_dataH7���

neff_input_namesinput55� �
transpose.788	transpose��*	
 �2    :R
'aten__convolution_backward_overrideable'aten__convolution_backward_overrideabler ����� � �
reverse.789reverse��*	
 �2    :R
'aten__convolution_backward_overrideable'aten__convolution_backward_overrideabler ����� � �
convolution.790convolution�*	
 �2    :R
'aten__convolution_backward_overrideable'aten__convolution_backward_overrideablez
 (0
 (0� 2 @PZb�������
  �� � S
add.817add�*	
 �2    :
	aten__add	aten__add������ � 
p60.810	parameter�*	
 �2    :$
xla__device_dataxla__device_dataH<���

neff_input_namesinput60� o
logistic.819logistic�*	
 �2    :*
aten__silu_backwardaten__silu_backward����� � h
constant.818constant*�:*
aten__silu_backwardaten__silu_backwardB
*�B  �?��� � q
broadcast.823	broadcast�*	
 �2    :*
aten__silu_backwardaten__silu_backward����� � q
broadcast.820	broadcast�*	
 �2    :*
aten__silu_backwardaten__silu_backward����� � q
subtract.821subtract�*	
 �2    :*
aten__silu_backwardaten__silu_backward������ � q
multiply.822multiply�*	
 �2    :*
aten__silu_backwardaten__silu_backward������ � g
add.824add�*	
 �2    :*
aten__silu_backwardaten__silu_backward������ � q
multiply.825multiply�*	
 �2    :*
aten__silu_backwardaten__silu_backward������ � q
multiply.826multiply�*	
 �2    :*
aten__silu_backwardaten__silu_backward������ � �
batch-norm-grad.834batch-norm-gradD"�*	
 �2    "�*
 �2 "�*
 �2 :D
 aten__native_batch_norm_backward aten__native_batch_norm_backward�o�:����
����0�� � �
get-tuple-element.837get-tuple-element�*
 �2 :D
 aten__native_batch_norm_backward aten__native_batch_norm_backwardh����� � �
custom-call.262custom-call�*
 �2 :�
'xla___op_TransferWithStaticRingTransfer'xla___op_TransferWithStaticRingTransferv/home/ubuntu/aws-neuron-samples/aws_neuron_venv_pytorch/lib/python3.10/site-packages/torch_xla/core/xla_op_registry.py ,x�AwsNeuronTransferWithStaticRing��-����
 �� � �
get-tuple-element.836get-tuple-element�*
 �2 :D
 aten__native_batch_norm_backward aten__native_batch_norm_backwardh����� � �
custom-call.263custom-call�*
 �2 :�
'xla___op_TransferWithStaticRingTransfer'xla___op_TransferWithStaticRingTransferv/home/ubuntu/aws-neuron-samples/aws_neuron_venv_pytorch/lib/python3.10/site-packages/torch_xla/core/xla_op_registry.py ,x�AwsNeuronTransferWithStaticRing��-����
 �� � 
p62.853	parameter�*	
 �2    :$
xla__device_dataxla__device_dataH>���

neff_input_namesinput62� �
get-tuple-element.835get-tuple-element�*	
 �2    :D
 aten__native_batch_norm_backward aten__native_batch_norm_backward����� � �
convolution.857convolution��*	
 �2    :R
'aten__convolution_backward_overrideable'aten__convolution_backward_overrideablez
(0
(0� 28HPZb �������
  �� � �
transpose.858	transpose��*	
 �2    :R
'aten__convolution_backward_overrideable'aten__convolution_backward_overrideabler ����� � �
custom-call.264custom-call��*	
 �2    :�
'xla___op_TransferWithStaticRingTransfer'xla___op_TransferWithStaticRingTransferv/home/ubuntu/aws-neuron-samples/aws_neuron_venv_pytorch/lib/python3.10/site-packages/torch_xla/core/xla_op_registry.py ,x�AwsNeuronTransferWithStaticRing��-����
 �� � �
constant.29constantU*
 �2 : Bi
U*
 �2 U                                                                                   ��5� � d
broadcast.52	broadcast�U*
 �2   :
xla__unselectxla__unselectr��0��5� � >
constant.126constant*�: B
*�B    ��7� � a
broadcast.55	broadcast�U*
 �2   :
xla__unselectxla__unselect��7��7� � �
constant.31constantU*
 �2 : Bi
U*
 �2 U                                                                                   ��5� � d
broadcast.57	broadcast�U*
 �2   :
xla__unselectxla__unselectr��0��5� � >
constant.127constant*�: B
*�B    ��7� � a
broadcast.60	broadcast�U*
 �2   :
xla__unselectxla__unselect��7��7� � o
	slice.886slice�U*
 �2   :
xla__selectxla__select���2�>�U����� � `

select.927select�U*
 �2   :
xla__unselectxla__unselect����0�7�� � �
constant.33constantU*
 �2 : Bi
U*
 �2 U                                                                                   ��5� � d
broadcast.61	broadcast�U*
 �2   :
xla__unselectxla__unselectr��0��5� � q
	slice.893slice�*
 �2   :
xla__selectxla__select���2�>���7��� � l
p65.879	parameter*�:$
xla__device_dataxla__device_dataHA���

neff_input_namesinput65� Z
broadcast.894	broadcast�*
 �2   :
	aten__mul	aten__mul����� � Z
multiply.895multiply�*
 �2   :
	aten__mul	aten__mul����7�� � |
p66.885	parameter�*
 �2   :$
xla__device_dataxla__device_dataHB���

neff_input_namesinput66� Z
multiply.896multiply�*
 �2   :
	aten__mul	aten__mul������ � \
constant.905constant*�:
xla__unselectxla__unselectB
*�B    ��� � e
pad.906pad�U*
 �2   :
xla__unselectxla__unselect�

 
 
Q������ � ^
constant.897constant*�: 
prim__Constantprim__ConstantB
*�B    ��� � `
broadcast.901	broadcast�U*
 �2   :
aten__expandaten__expand��4��� � `

select.908select�U*
 �2   :
xla__unselectxla__unselect����0��4� � O
add.92add�U*
 �2   :
	aten__add	aten__add��7���� � `

select.968select�U*
 �2   :
xla__unselectxla__unselect����0�7�7� � �
constant.34constantU*
 �2 : Bi
U*
 �2 U                                                                                   ��6� � d
broadcast.65	broadcast�U*
 �2   :
xla__unselectxla__unselectr��0��6� � l
	slice.935slice�*
 �2   :
xla__selectxla__select��������7� � Z
broadcast.936	broadcast�*
 �2   :
	aten__mul	aten__mul����� � Z
multiply.937multiply�*
 �2   :
	aten__mul	aten__mul������ � \
constant.946constant*�:
xla__unselectxla__unselectB
*�B    ��� � c
pad.947pad�U*
 �2   :
xla__unselectxla__unselect�
 
 
S������ � ^
constant.938constant*�: 
prim__Constantprim__ConstantB
*�B    ��� � `
broadcast.942	broadcast�U*
 �2   :
aten__expandaten__expand��4��� � `

select.949select�U*
 �2   :
xla__unselectxla__unselect����0��4� � O
add.93add�U*
 �2   :
	aten__add	aten__add��7���� � [
reshape.564reshape((U*

 �2     :
	aten__add	aten__add��7��7� � e
transpose.38	transposeU((*

 �2     :
	aten__add	aten__addr ��7��7� � X
reshape.511reshapeU((*	
 �2    :
	aten__add	aten__add��6��7� � v
	slice.973slice((*	
 �2    :
xla__selectxla__select���(�(����6� � �
constant.979constant*�:R
'aten__convolution_backward_overrideable'aten__convolution_backward_overrideableB
*�B    ��� � �

reduce.984reduce*
 �2 :R
'aten__convolution_backward_overrideable'aten__convolution_backward_overrideabler �������� � �
custom-call.265custom-call*
 �2 :�
'xla___op_TransferWithStaticRingTransfer'xla___op_TransferWithStaticRingTransferv/home/ubuntu/aws-neuron-samples/aws_neuron_venv_pytorch/lib/python3.10/site-packages/torch_xla/core/xla_op_registry.py ,x�AwsNeuronTransferWithStaticRing��-����
 �� � 
p64.873	parameter�((*	
 �2    :$
xla__device_dataxla__device_dataH@���

neff_input_namesinput64� <
reshape.404reshape((*
 �2  : ��7��� � e
broadcast.346	broadcast�((*	
 �2    :
xla__selectxla__selectr��7��7� � E

multiply.5multiply�((*	
 �2    : ��6���7� � =
constant.35constant*�: B
*�B    ��6� � D
reduce.4reduce�*
 �2  : r��7��6�6��1� � �
reshape.156reshape�*	
 �2    :R
'aten__convolution_backward_overrideable'aten__convolution_backward_overrideable��7��7� � �
custom-call.266custom-call�*	
 �2    :�
'xla___op_TransferWithStaticRingTransfer'xla___op_TransferWithStaticRingTransferv/home/ubuntu/aws-neuron-samples/aws_neuron_venv_pytorch/lib/python3.10/site-packages/torch_xla/core/xla_op_registry.py ,x�AwsNeuronTransferWithStaticRing��-��7��
 �� � u

slice.1000slice((*	
 �2    :
xla__selectxla__select���(�(����6� � �
constant.1006constant*�:R
'aten__convolution_backward_overrideable'aten__convolution_backward_overrideableB
*�B    ��� � �
reduce.1011reduce*
 �2 :R
'aten__convolution_backward_overrideable'aten__convolution_backward_overrideabler �������� � �
custom-call.267custom-call*
 �2 :�
'xla___op_TransferWithStaticRingTransfer'xla___op_TransferWithStaticRingTransferv/home/ubuntu/aws-neuron-samples/aws_neuron_venv_pytorch/lib/python3.10/site-packages/torch_xla/core/xla_op_registry.py ,x�AwsNeuronTransferWithStaticRing��-����
 �� � �
convolution.1004convolution�*	
 �2    :R
'aten__convolution_backward_overrideable'aten__convolution_backward_overrideablez
((0
((0� 28HPZb �������
  �� � �
transpose.1005	transpose�*	
 �2    :R
'aten__convolution_backward_overrideable'aten__convolution_backward_overrideabler ����� � �
custom-call.268custom-call�*	
 �2    :�
'xla___op_TransferWithStaticRingTransfer'xla___op_TransferWithStaticRingTransferv/home/ubuntu/aws-neuron-samples/aws_neuron_venv_pytorch/lib/python3.10/site-packages/torch_xla/core/xla_op_registry.py ,x�AwsNeuronTransferWithStaticRing��-����
 �� � �
p71.1029	parameter�((*	
 �2    :$
xla__device_dataxla__device_dataHG���

neff_input_namesinput71� w
p70.1028	parameter�*
 �2 :$
xla__device_dataxla__device_dataHF���

neff_input_namesinput70� w
p69.1027	parameter�*
 �2 :$
xla__device_dataxla__device_dataHE���

neff_input_namesinput69� �
constant.1048constant*�:D
 aten__native_batch_norm_backward aten__native_batch_norm_backwardB
*�B  �?��� � �
broadcast.1049	broadcast�*
 �2 :D
 aten__native_batch_norm_backward aten__native_batch_norm_backward����� � w
p68.1026	parameter�*
 �2 :$
xla__device_dataxla__device_dataHD���

neff_input_namesinput68� 
divide.1050divide�*
 �2 :D
 aten__native_batch_norm_backward aten__native_batch_norm_backward������ � �
multiply.1051multiply�*
 �2 :D
 aten__native_batch_norm_backward aten__native_batch_norm_backward������ � =
constant.36constant*�: B
*�Bo����6� � �
broadcast.70	broadcast�*
 �2 :D
 aten__native_batch_norm_backward aten__native_batch_norm_backward��1��6� � v
add.7add�*
 �2 :D
 aten__native_batch_norm_backward aten__native_batch_norm_backward��1���1� � 
p63.872	parameter�*	
 �2    :$
xla__device_dataxla__device_dataH?���

neff_input_namesinput63� �
reshape.157reshape�*	
 �2    :R
'aten__convolution_backward_overrideable'aten__convolution_backward_overrideable��1��� � �
convolution.976convolution�((*	
 �2    :R
'aten__convolution_backward_overrideable'aten__convolution_backward_overrideablez
(0
(0� 2 @PZb�����1��
  �� � 
p67.999	parameter�*	
 �2    :$
xla__device_dataxla__device_dataHC���

neff_input_namesinput67� �
transpose.1001	transpose�*	
 �2    :R
'aten__convolution_backward_overrideable'aten__convolution_backward_overrideabler ����� � �
convolution.1003convolution�((*	
 �2    :R
'aten__convolution_backward_overrideable'aten__convolution_backward_overrideablez
(0
(0� 2 @PZb�������
  �� � T
add.1037add�((*	
 �2    :
	aten__add	aten__add������ � �
p72.1030	parameter�((*	
 �2    :$
xla__device_dataxla__device_dataHH���

neff_input_namesinput72� p
logistic.1039logistic�((*	
 �2    :*
aten__silu_backwardaten__silu_backward����� � i
constant.1038constant*�:*
aten__silu_backwardaten__silu_backwardB
*�B  �?��� � r
broadcast.1043	broadcast�((*	
 �2    :*
aten__silu_backwardaten__silu_backward����� � r
broadcast.1040	broadcast�((*	
 �2    :*
aten__silu_backwardaten__silu_backward����� � r
subtract.1041subtract�((*	
 �2    :*
aten__silu_backwardaten__silu_backward������ � r
multiply.1042multiply�((*	
 �2    :*
aten__silu_backwardaten__silu_backward������ � h
add.1044add�((*	
 �2    :*
aten__silu_backwardaten__silu_backward������ � r
multiply.1045multiply�((*	
 �2    :*
aten__silu_backwardaten__silu_backward������ � r
multiply.1046multiply�((*	
 �2    :*
aten__silu_backwardaten__silu_backward������ � �
batch-norm-grad.1054batch-norm-gradD"�((*	
 �2    "�*
 �2 "�*
 �2 :D
 aten__native_batch_norm_backward aten__native_batch_norm_backward�o�:����
����1�� � �
get-tuple-element.1057get-tuple-element�*
 �2 :D
 aten__native_batch_norm_backward aten__native_batch_norm_backwardh����� � �
custom-call.269custom-call�*
 �2 :�
'xla___op_TransferWithStaticRingTransfer'xla___op_TransferWithStaticRingTransferv/home/ubuntu/aws-neuron-samples/aws_neuron_venv_pytorch/lib/python3.10/site-packages/torch_xla/core/xla_op_registry.py ,x�AwsNeuronTransferWithStaticRing��-����
 �� � �
get-tuple-element.1056get-tuple-element�*
 �2 :D
 aten__native_batch_norm_backward aten__native_batch_norm_backwardh����� � �
custom-call.270custom-call�*
 �2 :�
'xla___op_TransferWithStaticRingTransfer'xla___op_TransferWithStaticRingTransferv/home/ubuntu/aws-neuron-samples/aws_neuron_venv_pytorch/lib/python3.10/site-packages/torch_xla/core/xla_op_registry.py ,x�AwsNeuronTransferWithStaticRing��-����
 �� � �
p74.1073	parameter�((*	
 �2    :$
xla__device_dataxla__device_dataHJ���

neff_input_namesinput74� �
get-tuple-element.1055get-tuple-element�((*	
 �2    :D
 aten__native_batch_norm_backward aten__native_batch_norm_backward����� � �
convolution.1077convolution��*	
 �2    :R
'aten__convolution_backward_overrideable'aten__convolution_backward_overrideablez
( (0
( (0� 28HPZb �������
  �� � �
transpose.1078	transpose��*	
 �2    :R
'aten__convolution_backward_overrideable'aten__convolution_backward_overrideabler ����� � �
custom-call.271custom-call��*	
 �2    :�
'xla___op_TransferWithStaticRingTransfer'xla___op_TransferWithStaticRingTransferv/home/ubuntu/aws-neuron-samples/aws_neuron_venv_pytorch/lib/python3.10/site-packages/torch_xla/core/xla_op_registry.py ,x�AwsNeuronTransferWithStaticRing��-����
 �� � �
p78.1095	parameter�((*	
 �2    :$
xla__device_dataxla__device_dataHN���

neff_input_namesinput78� w
p77.1094	parameter�*
 �2 :$
xla__device_dataxla__device_dataHM���

neff_input_namesinput77� w
p76.1093	parameter�*
 �2 :$
xla__device_dataxla__device_dataHL���

neff_input_namesinput76� �
constant.1107constant*�:D
 aten__native_batch_norm_backward aten__native_batch_norm_backwardB
*�B  �?��� � �
broadcast.1108	broadcast�*
 �2 :D
 aten__native_batch_norm_backward aten__native_batch_norm_backward����� � w
p75.1092	parameter�*
 �2 :$
xla__device_dataxla__device_dataHK���

neff_input_namesinput75� 
divide.1109divide�*
 �2 :D
 aten__native_batch_norm_backward aten__native_batch_norm_backward������ � �
multiply.1110multiply�*
 �2 :D
 aten__native_batch_norm_backward aten__native_batch_norm_backward������ � =
constant.37constant*�: B
*�Bo����6� � �
broadcast.73	broadcast�*
 �2 :D
 aten__native_batch_norm_backward aten__native_batch_norm_backward��1��6� � v
add.8add�*
 �2 :D
 aten__native_batch_norm_backward aten__native_batch_norm_backward��1���1� � �
p73.1072	parameter��*	
 �2    :$
xla__device_dataxla__device_dataHI���

neff_input_namesinput73� �
transpose.1074	transpose��*	
 �2    :R
'aten__convolution_backward_overrideable'aten__convolution_backward_overrideabler ����� � �
reverse.1075reverse��*	
 �2    :R
'aten__convolution_backward_overrideable'aten__convolution_backward_overrideabler ����� � �
convolution.1076convolution�((*	
 �2    :R
'aten__convolution_backward_overrideable'aten__convolution_backward_overrideablez
 (0
 (0� 2 @PZb�������
  �� � �
p79.1096	parameter�((*	
 �2    :$
xla__device_dataxla__device_dataHO���

neff_input_namesinput79� p
logistic.1098logistic�((*	
 �2    :*
aten__silu_backwardaten__silu_backward����� � i
constant.1097constant*�:*
aten__silu_backwardaten__silu_backwardB
*�B  �?��� � r
broadcast.1102	broadcast�((*	
 �2    :*
aten__silu_backwardaten__silu_backward����� � r
broadcast.1099	broadcast�((*	
 �2    :*
aten__silu_backwardaten__silu_backward����� � r
subtract.1100subtract�((*	
 �2    :*
aten__silu_backwardaten__silu_backward������ � r
multiply.1101multiply�((*	
 �2    :*
aten__silu_backwardaten__silu_backward������ � h
add.1103add�((*	
 �2    :*
aten__silu_backwardaten__silu_backward������ � r
multiply.1104multiply�((*	
 �2    :*
aten__silu_backwardaten__silu_backward������ � r
multiply.1105multiply�((*	
 �2    :*
aten__silu_backwardaten__silu_backward������ � �
batch-norm-grad.1113batch-norm-gradD"�((*	
 �2    "�*
 �2 "�*
 �2 :D
 aten__native_batch_norm_backward aten__native_batch_norm_backward�o�:����
����1�� � �
get-tuple-element.1116get-tuple-element�*
 �2 :D
 aten__native_batch_norm_backward aten__native_batch_norm_backwardh����� � �
custom-call.272custom-call�*
 �2 :�
'xla___op_TransferWithStaticRingTransfer'xla___op_TransferWithStaticRingTransferv/home/ubuntu/aws-neuron-samples/aws_neuron_venv_pytorch/lib/python3.10/site-packages/torch_xla/core/xla_op_registry.py ,x�AwsNeuronTransferWithStaticRing��-����
 �� � �
get-tuple-element.1115get-tuple-element�*
 �2 :D
 aten__native_batch_norm_backward aten__native_batch_norm_backwardh����� � �
custom-call.273custom-call�*
 �2 :�
'xla___op_TransferWithStaticRingTransfer'xla___op_TransferWithStaticRingTransferv/home/ubuntu/aws-neuron-samples/aws_neuron_venv_pytorch/lib/python3.10/site-packages/torch_xla/core/xla_op_registry.py ,x�AwsNeuronTransferWithStaticRing��-����
 �� � �
p81.1132	parameter�((*	
 �2    :$
xla__device_dataxla__device_dataHQ���

neff_input_namesinput81� �
get-tuple-element.1114get-tuple-element�((*	
 �2    :D
 aten__native_batch_norm_backward aten__native_batch_norm_backward����� � �
convolution.1136convolution��*	
 �2    :R
'aten__convolution_backward_overrideable'aten__convolution_backward_overrideablez
( (0
( (0� 28HPZb �������
  �� � �
transpose.1137	transpose��*	
 �2    :R
'aten__convolution_backward_overrideable'aten__convolution_backward_overrideabler ����� � �
custom-call.274custom-call��*	
 �2    :�
'xla___op_TransferWithStaticRingTransfer'xla___op_TransferWithStaticRingTransferv/home/ubuntu/aws-neuron-samples/aws_neuron_venv_pytorch/lib/python3.10/site-packages/torch_xla/core/xla_op_registry.py ,x�AwsNeuronTransferWithStaticRing��-����
 �� � w

slice.1153sliceP((*	
 �2    :
xla__selectxla__select��U�(�(��	��6� � �
constant.1159constant*�:R
'aten__convolution_backward_overrideable'aten__convolution_backward_overrideableB
*�B    ��	� � �
reduce.1164reduceP*
 �2 :R
'aten__convolution_backward_overrideable'aten__convolution_backward_overrideabler ��	��	�	��	� � �
custom-call.275custom-callP*
 �2 :�
'xla___op_TransferWithStaticRingTransfer'xla___op_TransferWithStaticRingTransferv/home/ubuntu/aws-neuron-samples/aws_neuron_venv_pytorch/lib/python3.10/site-packages/torch_xla/core/xla_op_registry.py ,x�AwsNeuronTransferWithStaticRing��-��	��
 �� � �
p83.1152	parameter�((*	
 �2    :$
xla__device_dataxla__device_dataHS��	�

neff_input_namesinput83� �
convolution.1157convolution�P*	
 �2    :R
'aten__convolution_backward_overrideable'aten__convolution_backward_overrideablez
((0
((0� 28HPZb ��	��	�	��
  �� � �
transpose.1158	transposeP�*	
 �2    :R
'aten__convolution_backward_overrideable'aten__convolution_backward_overrideabler ��	��	� � �
custom-call.276custom-callP�*	
 �2    :�
'xla___op_TransferWithStaticRingTransfer'xla___op_TransferWithStaticRingTransferv/home/ubuntu/aws-neuron-samples/aws_neuron_venv_pytorch/lib/python3.10/site-packages/torch_xla/core/xla_op_registry.py ,x�AwsNeuronTransferWithStaticRing��-��	��
 �� � �
p87.1182	parameter�((*	
 �2    :$
xla__device_dataxla__device_dataHW��	�

neff_input_namesinput87� w
p86.1181	parameter�*
 �2 :$
xla__device_dataxla__device_dataHV��	�

neff_input_namesinput86� w
p85.1180	parameter�*
 �2 :$
xla__device_dataxla__device_dataHU��	�

neff_input_namesinput85� �
constant.1194constant*�:D
 aten__native_batch_norm_backward aten__native_batch_norm_backwardB
*�B  �?��	� � �
broadcast.1195	broadcast�*
 �2 :D
 aten__native_batch_norm_backward aten__native_batch_norm_backward��	��	� � w
p84.1179	parameter�*
 �2 :$
xla__device_dataxla__device_dataHT��	�

neff_input_namesinput84� 
divide.1196divide�*
 �2 :D
 aten__native_batch_norm_backward aten__native_batch_norm_backward��	��	�	� � �
multiply.1197multiply�*
 �2 :D
 aten__native_batch_norm_backward aten__native_batch_norm_backward��	��	�	� � =
constant.38constant*�: B
*�Bo����6� � �
broadcast.74	broadcast�*
 �2 :D
 aten__native_batch_norm_backward aten__native_batch_norm_backward��1��6� � v
add.9add�*
 �2 :D
 aten__native_batch_norm_backward aten__native_batch_norm_backward��1��	�1� � �
p82.1151	parameterP�*	
 �2    :$
xla__device_dataxla__device_dataHR���

neff_input_namesinput82� �
transpose.1154	transpose�P*	
 �2    :R
'aten__convolution_backward_overrideable'aten__convolution_backward_overrideabler ��	��� � �
convolution.1156convolution�((*	
 �2    :R
'aten__convolution_backward_overrideable'aten__convolution_backward_overrideablez
(0
(0� 2 @PZb��	��	�	��
  �� � �
p88.1183	parameter�((*	
 �2    :$
xla__device_dataxla__device_dataHX��	�

neff_input_namesinput88� p
logistic.1185logistic�((*	
 �2    :*
aten__silu_backwardaten__silu_backward��	��	� � i
constant.1184constant*�:*
aten__silu_backwardaten__silu_backwardB
*�B  �?��	� � r
broadcast.1189	broadcast�((*	
 �2    :*
aten__silu_backwardaten__silu_backward��	��	� � r
broadcast.1186	broadcast�((*	
 �2    :*
aten__silu_backwardaten__silu_backward��	��	� � r
subtract.1187subtract�((*	
 �2    :*
aten__silu_backwardaten__silu_backward��	��	�	� � r
multiply.1188multiply�((*	
 �2    :*
aten__silu_backwardaten__silu_backward��	��	�	� � h
add.1190add�((*	
 �2    :*
aten__silu_backwardaten__silu_backward��	��	�	� � r
multiply.1191multiply�((*	
 �2    :*
aten__silu_backwardaten__silu_backward��	��	�	� � r
multiply.1192multiply�((*	
 �2    :*
aten__silu_backwardaten__silu_backward��	��	�	� � �
batch-norm-grad.1200batch-norm-gradD"�((*	
 �2    "�*
 �2 "�*
 �2 :D
 aten__native_batch_norm_backward aten__native_batch_norm_backward�o�:���	�
�	�	�	�1�	� � �
get-tuple-element.1203get-tuple-element�*
 �2 :D
 aten__native_batch_norm_backward aten__native_batch_norm_backwardh��	��	� � �
custom-call.277custom-call�*
 �2 :�
'xla___op_TransferWithStaticRingTransfer'xla___op_TransferWithStaticRingTransferv/home/ubuntu/aws-neuron-samples/aws_neuron_venv_pytorch/lib/python3.10/site-packages/torch_xla/core/xla_op_registry.py ,x�AwsNeuronTransferWithStaticRing��-��	��
 �� � �
get-tuple-element.1202get-tuple-element�*
 �2 :D
 aten__native_batch_norm_backward aten__native_batch_norm_backwardh��	��	� � �
custom-call.278custom-call�*
 �2 :�
'xla___op_TransferWithStaticRingTransfer'xla___op_TransferWithStaticRingTransferv/home/ubuntu/aws-neuron-samples/aws_neuron_venv_pytorch/lib/python3.10/site-packages/torch_xla/core/xla_op_registry.py ,x�AwsNeuronTransferWithStaticRing��-��	��
 �� � �
p90.1219	parameter�((*	
 �2    :$
xla__device_dataxla__device_dataHZ��	�

neff_input_namesinput90� �
get-tuple-element.1201get-tuple-element�((*	
 �2    :D
 aten__native_batch_norm_backward aten__native_batch_norm_backward��	��	� � �
convolution.1223convolution��*	
 �2    :R
'aten__convolution_backward_overrideable'aten__convolution_backward_overrideablez
( (0
( (0� 28HPZb ��	��	�	��
  �� � �
transpose.1224	transpose��*	
 �2    :R
'aten__convolution_backward_overrideable'aten__convolution_backward_overrideabler ��	��	� � �
custom-call.279custom-call��*	
 �2    :�
'xla___op_TransferWithStaticRingTransfer'xla___op_TransferWithStaticRingTransferv/home/ubuntu/aws-neuron-samples/aws_neuron_venv_pytorch/lib/python3.10/site-packages/torch_xla/core/xla_op_registry.py ,x�AwsNeuronTransferWithStaticRing��-��	��
 �� � �
p94.1241	parameter�((*	
 �2    :$
xla__device_dataxla__device_dataH^��	�

neff_input_namesinput94� w
p93.1240	parameter�*
 �2 :$
xla__device_dataxla__device_dataH]��	�

neff_input_namesinput93� w
p92.1239	parameter�*
 �2 :$
xla__device_dataxla__device_dataH\��	�

neff_input_namesinput92� �
constant.1253constant*�:D
 aten__native_batch_norm_backward aten__native_batch_norm_backwardB
*�B  �?��	� � �
broadcast.1254	broadcast�*
 �2 :D
 aten__native_batch_norm_backward aten__native_batch_norm_backward��	��	� � w
p91.1238	parameter�*
 �2 :$
xla__device_dataxla__device_dataH[��	�

neff_input_namesinput91� 
divide.1255divide�*
 �2 :D
 aten__native_batch_norm_backward aten__native_batch_norm_backward��	��	�	� � �
multiply.1256multiply�*
 �2 :D
 aten__native_batch_norm_backward aten__native_batch_norm_backward��	��	�	� � =
constant.39constant*�: B
*�Bo����6� � �
broadcast.75	broadcast�*
 �2 :D
 aten__native_batch_norm_backward aten__native_batch_norm_backward��1��6� � w
add.10add�*
 �2 :D
 aten__native_batch_norm_backward aten__native_batch_norm_backward��1��	�1� � �
p89.1218	parameter��*	
 �2    :$
xla__device_dataxla__device_dataHY��	�

neff_input_namesinput89� �
transpose.1220	transpose��*	
 �2    :R
'aten__convolution_backward_overrideable'aten__convolution_backward_overrideabler ��	��	� � �
reverse.1221reverse��*	
 �2    :R
'aten__convolution_backward_overrideable'aten__convolution_backward_overrideabler ��	��	� � �
convolution.1222convolution�((*	
 �2    :R
'aten__convolution_backward_overrideable'aten__convolution_backward_overrideablez
 (0
 (0� 2 @PZb��	��	�	��
  �� � �
p95.1242	parameter�((*	
 �2    :$
xla__device_dataxla__device_dataH_��	�

neff_input_namesinput95� p
logistic.1244logistic�((*	
 �2    :*
aten__silu_backwardaten__silu_backward��	��	� � i
constant.1243constant*�:*
aten__silu_backwardaten__silu_backwardB
*�B  �?��	� � r
broadcast.1248	broadcast�((*	
 �2    :*
aten__silu_backwardaten__silu_backward��	��	� � r
broadcast.1245	broadcast�((*	
 �2    :*
aten__silu_backwardaten__silu_backward��	��	� � r
subtract.1246subtract�((*	
 �2    :*
aten__silu_backwardaten__silu_backward��	��	�	� � r
multiply.1247multiply�((*	
 �2    :*
aten__silu_backwardaten__silu_backward��	��	�	� � h
add.1249add�((*	
 �2    :*
aten__silu_backwardaten__silu_backward��	��	�	� � r
multiply.1250multiply�((*	
 �2    :*
aten__silu_backwardaten__silu_backward��	��	�	� � r
multiply.1251multiply�((*	
 �2    :*
aten__silu_backwardaten__silu_backward��	��	�	� � �
batch-norm-grad.1259batch-norm-gradD"�((*	
 �2    "�*
 �2 "�*
 �2 :D
 aten__native_batch_norm_backward aten__native_batch_norm_backward�o�:���	�
�	�	�	�1�	� � �
get-tuple-element.1262get-tuple-element�*
 �2 :D
 aten__native_batch_norm_backward aten__native_batch_norm_backwardh��	��	� � �
custom-call.280custom-call�*
 �2 :�
'xla___op_TransferWithStaticRingTransfer'xla___op_TransferWithStaticRingTransferv/home/ubuntu/aws-neuron-samples/aws_neuron_venv_pytorch/lib/python3.10/site-packages/torch_xla/core/xla_op_registry.py ,x�AwsNeuronTransferWithStaticRing��-��	��
 �� � �
get-tuple-element.1261get-tuple-element�*
 �2 :D
 aten__native_batch_norm_backward aten__native_batch_norm_backwardh��	��	� � �
custom-call.281custom-call�*
 �2 :�
'xla___op_TransferWithStaticRingTransfer'xla___op_TransferWithStaticRingTransferv/home/ubuntu/aws-neuron-samples/aws_neuron_venv_pytorch/lib/python3.10/site-packages/torch_xla/core/xla_op_registry.py ,x�AwsNeuronTransferWithStaticRing��-��	��
 �� � �
get-tuple-element.1260get-tuple-element�((*	
 �2    :D
 aten__native_batch_norm_backward aten__native_batch_norm_backward��	��	� � �
convolution.1281convolution��*	
 �2    :R
'aten__convolution_backward_overrideable'aten__convolution_backward_overrideablez
( (0
( (0� 28HPZb ��
���	��
  �� � �
transpose.1282	transpose��*	
 �2    :R
'aten__convolution_backward_overrideable'aten__convolution_backward_overrideabler ��
��
� � �
custom-call.282custom-call��*	
 �2    :�
'xla___op_TransferWithStaticRingTransfer'xla___op_TransferWithStaticRingTransferv/home/ubuntu/aws-neuron-samples/aws_neuron_venv_pytorch/lib/python3.10/site-packages/torch_xla/core/xla_op_registry.py ,x�AwsNeuronTransferWithStaticRing��-��
��
 �� � �
	p100.1299	parameter�((*	
 �2    :$
xla__device_dataxla__device_dataHd��
�

neff_input_namesinput100� w
p99.1298	parameter�*
 �2 :$
xla__device_dataxla__device_dataHc��
�

neff_input_namesinput99� w
p98.1297	parameter�*
 �2 :$
xla__device_dataxla__device_dataHb��
�

neff_input_namesinput98� �
constant.1318constant*�:D
 aten__native_batch_norm_backward aten__native_batch_norm_backwardB
*�B  �?��
� � �
broadcast.1319	broadcast�*
 �2 :D
 aten__native_batch_norm_backward aten__native_batch_norm_backward��
��
� � w
p97.1296	parameter�*
 �2 :$
xla__device_dataxla__device_dataHa��
�

neff_input_namesinput97� 
divide.1320divide�*
 �2 :D
 aten__native_batch_norm_backward aten__native_batch_norm_backward��
��
�
� � �
multiply.1321multiply�*
 �2 :D
 aten__native_batch_norm_backward aten__native_batch_norm_backward��
��
�
� � =
constant.41constant*�: B
*�Bo����6� � �
broadcast.76	broadcast�*
 �2 :D
 aten__native_batch_norm_backward aten__native_batch_norm_backward��1��6� � w
add.11add�*
 �2 :D
 aten__native_batch_norm_backward aten__native_batch_norm_backward��1��
�1� � �
p80.1131	parameter��*	
 �2    :$
xla__device_dataxla__device_dataHP���

neff_input_namesinput80� �
transpose.1133	transpose��*	
 �2    :R
'aten__convolution_backward_overrideable'aten__convolution_backward_overrideabler ����� � �
reverse.1134reverse��*	
 �2    :R
'aten__convolution_backward_overrideable'aten__convolution_backward_overrideabler ����� � �
convolution.1135convolution�((*	
 �2    :R
'aten__convolution_backward_overrideable'aten__convolution_backward_overrideablez
 (0
 (0� 2 @PZb�������
  �� � �
p96.1277	parameter��*	
 �2    :$
xla__device_dataxla__device_dataH`��	�

neff_input_namesinput96� �
transpose.1278	transpose��*	
 �2    :R
'aten__convolution_backward_overrideable'aten__convolution_backward_overrideabler ��	��	� � �
reverse.1279reverse��*	
 �2    :R
'aten__convolution_backward_overrideable'aten__convolution_backward_overrideabler ��	��	� � �
convolution.1280convolution�((*	
 �2    :R
'aten__convolution_backward_overrideable'aten__convolution_backward_overrideablez
 (0
 (0� 2 @PZb��
��	�	��
  �� � T
add.1307add�((*	
 �2    :
	aten__add	aten__add��
���
� � �
	p101.1300	parameter�((*	
 �2    :$
xla__device_dataxla__device_dataHe��
�

neff_input_namesinput101� p
logistic.1309logistic�((*	
 �2    :*
aten__silu_backwardaten__silu_backward��
��
� � i
constant.1308constant*�:*
aten__silu_backwardaten__silu_backwardB
*�B  �?��
� � r
broadcast.1313	broadcast�((*	
 �2    :*
aten__silu_backwardaten__silu_backward��
��
� � r
broadcast.1310	broadcast�((*	
 �2    :*
aten__silu_backwardaten__silu_backward��
��
� � r
subtract.1311subtract�((*	
 �2    :*
aten__silu_backwardaten__silu_backward��
��
�
� � r
multiply.1312multiply�((*	
 �2    :*
aten__silu_backwardaten__silu_backward��
��
�
� � h
add.1314add�((*	
 �2    :*
aten__silu_backwardaten__silu_backward��
��
�
� � r
multiply.1315multiply�((*	
 �2    :*
aten__silu_backwardaten__silu_backward��
��
�
� � r
multiply.1316multiply�((*	
 �2    :*
aten__silu_backwardaten__silu_backward��
��
�
� � �
batch-norm-grad.1324batch-norm-gradD"�((*	
 �2    "�*
 �2 "�*
 �2 :D
 aten__native_batch_norm_backward aten__native_batch_norm_backward�o�:���
�
�
�
�
�1�
� � �
get-tuple-element.1327get-tuple-element�*
 �2 :D
 aten__native_batch_norm_backward aten__native_batch_norm_backwardh��
��
� � �
custom-call.283custom-call�*
 �2 :�
'xla___op_TransferWithStaticRingTransfer'xla___op_TransferWithStaticRingTransferv/home/ubuntu/aws-neuron-samples/aws_neuron_venv_pytorch/lib/python3.10/site-packages/torch_xla/core/xla_op_registry.py ,x�AwsNeuronTransferWithStaticRing��-��
��
 �� � �
get-tuple-element.1326get-tuple-element�*
 �2 :D
 aten__native_batch_norm_backward aten__native_batch_norm_backwardh��
��
� � �
custom-call.284custom-call�*
 �2 :�
'xla___op_TransferWithStaticRingTransfer'xla___op_TransferWithStaticRingTransferv/home/ubuntu/aws-neuron-samples/aws_neuron_venv_pytorch/lib/python3.10/site-packages/torch_xla/core/xla_op_registry.py ,x�AwsNeuronTransferWithStaticRing��-��
��
 �� � �
	p103.1343	parameter�((*	
 �2    :$
xla__device_dataxla__device_dataHg��
�

neff_input_namesinput103� �
get-tuple-element.1325get-tuple-element�((*	
 �2    :D
 aten__native_batch_norm_backward aten__native_batch_norm_backward��
��
� � �
convolution.1347convolution��*	
 �2    :R
'aten__convolution_backward_overrideable'aten__convolution_backward_overrideablez
((0
((0� 28HPZb ��
��
�
��
  �� � �
transpose.1348	transpose��*	
 �2    :R
'aten__convolution_backward_overrideable'aten__convolution_backward_overrideabler ��
��
� � �
custom-call.285custom-call��*	
 �2    :�
'xla___op_TransferWithStaticRingTransfer'xla___op_TransferWithStaticRingTransferv/home/ubuntu/aws-neuron-samples/aws_neuron_venv_pytorch/lib/python3.10/site-packages/torch_xla/core/xla_op_registry.py ,x�AwsNeuronTransferWithStaticRing��-��
��
 �� � �
constant.42constantU*
 �2 : Bi
U*
 �2 U                                                                                   ��6� � d
broadcast.78	broadcast�2U*
 �2   :
xla__unselectxla__unselectr��1��6� � >
constant.129constant*�: B
*�B    ��7� � a
broadcast.80	broadcast�2U*
 �2   :
xla__unselectxla__unselect��7��7� � �
constant.44constantU*
 �2 : Bi
U*
 �2 U                                                                                   ��6� � d
broadcast.81	broadcast�2U*
 �2   :
xla__unselectxla__unselectr��1��6� � >
constant.132constant*�: B
*�B    ��7� � a
broadcast.85	broadcast�2U*
 �2   :
xla__unselectxla__unselect��7��7� � m

slice.1376slice�2U*
 �2   :
xla__selectxla__select���2�U��
��� � a
select.1417select�2U*
 �2   :
xla__unselectxla__unselect����1�7�
� � �
constant.46constantU*
 �2 : Bi
U*
 �2 U                                                                                   ��6� � d
broadcast.86	broadcast�2U*
 �2   :
xla__unselectxla__unselectr��1��6� � o

slice.1383slice�2*
 �2   :
xla__selectxla__select���2���7��� � o
	p106.1369	parameter*�:$
xla__device_dataxla__device_dataHj��
�

neff_input_namesinput106� [
broadcast.1384	broadcast�2*
 �2   :
	aten__mul	aten__mul��
��
� � [
multiply.1385multiply�2*
 �2   :
	aten__mul	aten__mul��
��7�
� � 
	p107.1375	parameter�2*
 �2   :$
xla__device_dataxla__device_dataHk��
�

neff_input_namesinput107� [
multiply.1386multiply�2*
 �2   :
	aten__mul	aten__mul��
��
�
� � ]
constant.1395constant*�:
xla__unselectxla__unselectB
*�B    ��
� � f
pad.1396pad�2U*
 �2   :
xla__unselectxla__unselect�

 
 
Q��
��
�
� � _
constant.1387constant*�: 
prim__Constantprim__ConstantB
*�B    ��
� � a
broadcast.1391	broadcast�2U*
 �2   :
aten__expandaten__expand��5��
� � a
select.1398select�2U*
 �2   :
xla__unselectxla__unselect��
��1�
�5� � O
add.94add�2U*
 �2   :
	aten__add	aten__add��7���
� � a
select.1458select�2U*
 �2   :
xla__unselectxla__unselect����1�7�7� � �
constant.47constantU*
 �2 : Bi
U*
 �2 U                                                                                   ��6� � d
broadcast.90	broadcast�2U*
 �2   :
xla__unselectxla__unselectr��1��6� � m

slice.1425slice�2*
 �2   :
xla__selectxla__select���2�����7� � [
broadcast.1426	broadcast�2*
 �2   :
	aten__mul	aten__mul����
� � [
multiply.1427multiply�2*
 �2   :
	aten__mul	aten__mul������ � ]
constant.1436constant*�:
xla__unselectxla__unselectB
*�B    ��� � d
pad.1437pad�2U*
 �2   :
xla__unselectxla__unselect�
 
 
S������ � _
constant.1428constant*�: 
prim__Constantprim__ConstantB
*�B    ��� � a
broadcast.1432	broadcast�2U*
 �2   :
aten__expandaten__expand��5��� � a
select.1439select�2U*
 �2   :
xla__unselectxla__unselect����1��5� � O
add.95add�2U*
 �2   :
	aten__add	aten__add��7���� � [
reshape.570reshapePPU*

 �2     :
	aten__add	aten__add��7��7� � e
transpose.44	transposeUPP*

 �2     :
	aten__add	aten__addr ��7��7� � X
reshape.517reshapeUPP*	
 �2    :
	aten__add	aten__add��6��7� � w

slice.1463slicePP*	
 �2    :
xla__selectxla__select���P�P����6� � �
constant.1469constant*�:R
'aten__convolution_backward_overrideable'aten__convolution_backward_overrideableB
*�B    ��� � �
reduce.1474reduce*
 �2 :R
'aten__convolution_backward_overrideable'aten__convolution_backward_overrideabler �������� � �
custom-call.286custom-call*
 �2 :�
'xla___op_TransferWithStaticRingTransfer'xla___op_TransferWithStaticRingTransferv/home/ubuntu/aws-neuron-samples/aws_neuron_venv_pytorch/lib/python3.10/site-packages/torch_xla/core/xla_op_registry.py ,x�AwsNeuronTransferWithStaticRing��-����
 �� � �
	p105.1363	parameter�PP*	
 �2    :$
xla__device_dataxla__device_dataHi��
�

neff_input_namesinput105� <
reshape.432reshapePP*
 �2  : ��7��� � e
broadcast.353	broadcast�PP*	
 �2    :
xla__selectxla__selectr��7��7� � E

multiply.6multiply�PP*	
 �2    : ��6��
�7� � =
constant.48constant*�: B
*�B    ��6� � D
reduce.5reduce�*
 �2  : r��7��6�6��1� � �
reshape.194reshape�*	
 �2    :R
'aten__convolution_backward_overrideable'aten__convolution_backward_overrideable��7��7� � �
custom-call.287custom-call�*	
 �2    :�
'xla___op_TransferWithStaticRingTransfer'xla___op_TransferWithStaticRingTransferv/home/ubuntu/aws-neuron-samples/aws_neuron_venv_pytorch/lib/python3.10/site-packages/torch_xla/core/xla_op_registry.py ,x�AwsNeuronTransferWithStaticRing��-��7��
 �� � u

slice.1490slicePP*	
 �2    :
xla__selectxla__select���P�P����6� � �
constant.1496constant*�:R
'aten__convolution_backward_overrideable'aten__convolution_backward_overrideableB
*�B    ��� � �
reduce.1501reduce*
 �2 :R
'aten__convolution_backward_overrideable'aten__convolution_backward_overrideabler �������� � �
custom-call.288custom-call*
 �2 :�
'xla___op_TransferWithStaticRingTransfer'xla___op_TransferWithStaticRingTransferv/home/ubuntu/aws-neuron-samples/aws_neuron_venv_pytorch/lib/python3.10/site-packages/torch_xla/core/xla_op_registry.py ,x�AwsNeuronTransferWithStaticRing��-����
 �� � �
convolution.1494convolution�*	
 �2    :R
'aten__convolution_backward_overrideable'aten__convolution_backward_overrideablez
P(0
P(0� 28HPZb ����
���
  �� � �
transpose.1495	transpose�*	
 �2    :R
'aten__convolution_backward_overrideable'aten__convolution_backward_overrideabler ����� � �
custom-call.289custom-call�*	
 �2    :�
'xla___op_TransferWithStaticRingTransfer'xla___op_TransferWithStaticRingTransferv/home/ubuntu/aws-neuron-samples/aws_neuron_venv_pytorch/lib/python3.10/site-packages/torch_xla/core/xla_op_registry.py ,x�AwsNeuronTransferWithStaticRing��-����
 �� � �
	p112.1519	parameter�PP*	
 �2    :$
xla__device_dataxla__device_dataHp���

neff_input_namesinput112� y
	p111.1518	parameter�*
 �2 :$
xla__device_dataxla__device_dataHo���

neff_input_namesinput111� y
	p110.1517	parameter�*
 �2 :$
xla__device_dataxla__device_dataHn���

neff_input_namesinput110� �
constant.1538constant*�:D
 aten__native_batch_norm_backward aten__native_batch_norm_backwardB
*�B  �?��� � �
broadcast.1539	broadcast�*
 �2 :D
 aten__native_batch_norm_backward aten__native_batch_norm_backward����� � y
	p109.1516	parameter�*
 �2 :$
xla__device_dataxla__device_dataHm���

neff_input_namesinput109� 
divide.1540divide�*
 �2 :D
 aten__native_batch_norm_backward aten__native_batch_norm_backward������ � �
multiply.1541multiply�*
 �2 :D
 aten__native_batch_norm_backward aten__native_batch_norm_backward������ � =
constant.49constant*�: B
*�Bo����6� � �
broadcast.96	broadcast�*
 �2 :D
 aten__native_batch_norm_backward aten__native_batch_norm_backward��1��6� � w
add.13add�*
 �2 :D
 aten__native_batch_norm_backward aten__native_batch_norm_backward��1���1� � �
	p104.1362	parameter�*	
 �2    :$
xla__device_dataxla__device_dataHh��
�

neff_input_namesinput104� �
reshape.195reshape�*	
 �2    :R
'aten__convolution_backward_overrideable'aten__convolution_backward_overrideable��1��
� � �
convolution.1466convolution�PP*	
 �2    :R
'aten__convolution_backward_overrideable'aten__convolution_backward_overrideablez
(0
(0� 2 @PZb�����1��
  �� � �
	p108.1489	parameter�*	
 �2    :$
xla__device_dataxla__device_dataHl���

neff_input_namesinput108� �
transpose.1491	transpose�*	
 �2    :R
'aten__convolution_backward_overrideable'aten__convolution_backward_overrideabler ����� � �
convolution.1493convolution�PP*	
 �2    :R
'aten__convolution_backward_overrideable'aten__convolution_backward_overrideablez
(0
(0� 2 @PZb�������
  �� � T
add.1527add�PP*	
 �2    :
	aten__add	aten__add������ � �
	p113.1520	parameter�PP*	
 �2    :$
xla__device_dataxla__device_dataHq���

neff_input_namesinput113� p
logistic.1529logistic�PP*	
 �2    :*
aten__silu_backwardaten__silu_backward����� � i
constant.1528constant*�:*
aten__silu_backwardaten__silu_backwardB
*�B  �?��� � r
broadcast.1533	broadcast�PP*	
 �2    :*
aten__silu_backwardaten__silu_backward����� � r
broadcast.1530	broadcast�PP*	
 �2    :*
aten__silu_backwardaten__silu_backward����� � r
subtract.1531subtract�PP*	
 �2    :*
aten__silu_backwardaten__silu_backward������ � r
multiply.1532multiply�PP*	
 �2    :*
aten__silu_backwardaten__silu_backward������ � h
add.1534add�PP*	
 �2    :*
aten__silu_backwardaten__silu_backward������ � r
multiply.1535multiply�PP*	
 �2    :*
aten__silu_backwardaten__silu_backward������ � r
multiply.1536multiply�PP*	
 �2    :*
aten__silu_backwardaten__silu_backward������ � �
batch-norm-grad.1544batch-norm-gradD"�PP*	
 �2    "�*
 �2 "�*
 �2 :D
 aten__native_batch_norm_backward aten__native_batch_norm_backward�o�:����
����1�� � �
get-tuple-element.1547get-tuple-element�*
 �2 :D
 aten__native_batch_norm_backward aten__native_batch_norm_backwardh����� � �
custom-call.290custom-call�*
 �2 :�
'xla___op_TransferWithStaticRingTransfer'xla___op_TransferWithStaticRingTransferv/home/ubuntu/aws-neuron-samples/aws_neuron_venv_pytorch/lib/python3.10/site-packages/torch_xla/core/xla_op_registry.py ,x�AwsNeuronTransferWithStaticRing��-����
 �� � �
get-tuple-element.1546get-tuple-element�*
 �2 :D
 aten__native_batch_norm_backward aten__native_batch_norm_backwardh����� � �
custom-call.291custom-call�*
 �2 :�
'xla___op_TransferWithStaticRingTransfer'xla___op_TransferWithStaticRingTransferv/home/ubuntu/aws-neuron-samples/aws_neuron_venv_pytorch/lib/python3.10/site-packages/torch_xla/core/xla_op_registry.py ,x�AwsNeuronTransferWithStaticRing��-����
 �� � �
	p115.1563	parameter�PP*	
 �2    :$
xla__device_dataxla__device_dataHs���

neff_input_namesinput115� �
get-tuple-element.1545get-tuple-element�PP*	
 �2    :D
 aten__native_batch_norm_backward aten__native_batch_norm_backward����� � �
convolution.1567convolution��*	
 �2    :R
'aten__convolution_backward_overrideable'aten__convolution_backward_overrideablez
P (0
P (0� 28HPZb �������
  �� � �
transpose.1568	transpose��*	
 �2    :R
'aten__convolution_backward_overrideable'aten__convolution_backward_overrideabler ����� � �
custom-call.292custom-call��*	
 �2    :�
'xla___op_TransferWithStaticRingTransfer'xla___op_TransferWithStaticRingTransferv/home/ubuntu/aws-neuron-samples/aws_neuron_venv_pytorch/lib/python3.10/site-packages/torch_xla/core/xla_op_registry.py ,x�AwsNeuronTransferWithStaticRing��-����
 �� � �
	p119.1585	parameter�PP*	
 �2    :$
xla__device_dataxla__device_dataHw���

neff_input_namesinput119� y
	p118.1584	parameter�*
 �2 :$
xla__device_dataxla__device_dataHv���

neff_input_namesinput118� y
	p117.1583	parameter�*
 �2 :$
xla__device_dataxla__device_dataHu���

neff_input_namesinput117� �
constant.1597constant*�:D
 aten__native_batch_norm_backward aten__native_batch_norm_backwardB
*�B  �?��� � �
broadcast.1598	broadcast�*
 �2 :D
 aten__native_batch_norm_backward aten__native_batch_norm_backward����� � y
	p116.1582	parameter�*
 �2 :$
xla__device_dataxla__device_dataHt���

neff_input_namesinput116� 
divide.1599divide�*
 �2 :D
 aten__native_batch_norm_backward aten__native_batch_norm_backward������ � �
multiply.1600multiply�*
 �2 :D
 aten__native_batch_norm_backward aten__native_batch_norm_backward������ � =
constant.50constant*�: B
*�Bo����6� � �
broadcast.98	broadcast�*
 �2 :D
 aten__native_batch_norm_backward aten__native_batch_norm_backward��1��6� � w
add.14add�*
 �2 :D
 aten__native_batch_norm_backward aten__native_batch_norm_backward��1���1� � �
	p114.1562	parameter��*	
 �2    :$
xla__device_dataxla__device_dataHr���

neff_input_namesinput114� �
transpose.1564	transpose��*	
 �2    :R
'aten__convolution_backward_overrideable'aten__convolution_backward_overrideabler ����� � �
reverse.1565reverse��*	
 �2    :R
'aten__convolution_backward_overrideable'aten__convolution_backward_overrideabler ����� � �
convolution.1566convolution�PP*	
 �2    :R
'aten__convolution_backward_overrideable'aten__convolution_backward_overrideablez
 (0
 (0� 2 @PZb�������
  �� � �
	p120.1586	parameter�PP*	
 �2    :$
xla__device_dataxla__device_dataHx���

neff_input_namesinput120� p
logistic.1588logistic�PP*	
 �2    :*
aten__silu_backwardaten__silu_backward����� � i
constant.1587constant*�:*
aten__silu_backwardaten__silu_backwardB
*�B  �?��� � r
broadcast.1592	broadcast�PP*	
 �2    :*
aten__silu_backwardaten__silu_backward����� � r
broadcast.1589	broadcast�PP*	
 �2    :*
aten__silu_backwardaten__silu_backward����� � r
subtract.1590subtract�PP*	
 �2    :*
aten__silu_backwardaten__silu_backward������ � r
multiply.1591multiply�PP*	
 �2    :*
aten__silu_backwardaten__silu_backward������ � h
add.1593add�PP*	
 �2    :*
aten__silu_backwardaten__silu_backward������ � r
multiply.1594multiply�PP*	
 �2    :*
aten__silu_backwardaten__silu_backward������ � r
multiply.1595multiply�PP*	
 �2    :*
aten__silu_backwardaten__silu_backward������ � �
batch-norm-grad.1603batch-norm-gradD"�PP*	
 �2    "�*
 �2 "�*
 �2 :D
 aten__native_batch_norm_backward aten__native_batch_norm_backward�o�:����
����1�� � �
get-tuple-element.1606get-tuple-element�*
 �2 :D
 aten__native_batch_norm_backward aten__native_batch_norm_backwardh����� � �
custom-call.293custom-call�*
 �2 :�
'xla___op_TransferWithStaticRingTransfer'xla___op_TransferWithStaticRingTransferv/home/ubuntu/aws-neuron-samples/aws_neuron_venv_pytorch/lib/python3.10/site-packages/torch_xla/core/xla_op_registry.py ,x�AwsNeuronTransferWithStaticRing��-����
 �� � �
get-tuple-element.1605get-tuple-element�*
 �2 :D
 aten__native_batch_norm_backward aten__native_batch_norm_backwardh����� � �
custom-call.294custom-call�*
 �2 :�
'xla___op_TransferWithStaticRingTransfer'xla___op_TransferWithStaticRingTransferv/home/ubuntu/aws-neuron-samples/aws_neuron_venv_pytorch/lib/python3.10/site-packages/torch_xla/core/xla_op_registry.py ,x�AwsNeuronTransferWithStaticRing��-����
 �� � �
	p122.1622	parameter�PP*	
 �2    :$
xla__device_dataxla__device_dataHz���

neff_input_namesinput122� �
get-tuple-element.1604get-tuple-element�PP*	
 �2    :D
 aten__native_batch_norm_backward aten__native_batch_norm_backward����� � �
convolution.1626convolution��*	
 �2    :R
'aten__convolution_backward_overrideable'aten__convolution_backward_overrideablez
P (0
P (0� 28HPZb �������
  �� � �
transpose.1627	transpose��*	
 �2    :R
'aten__convolution_backward_overrideable'aten__convolution_backward_overrideabler ����� � �
custom-call.295custom-call��*	
 �2    :�
'xla___op_TransferWithStaticRingTransfer'xla___op_TransferWithStaticRingTransferv/home/ubuntu/aws-neuron-samples/aws_neuron_venv_pytorch/lib/python3.10/site-packages/torch_xla/core/xla_op_registry.py ,x�AwsNeuronTransferWithStaticRing��-����
 �� � w

slice.1643slicePPP*	
 �2    :
xla__selectxla__select��U�P�P����6� � �
constant.1649constant*�:R
'aten__convolution_backward_overrideable'aten__convolution_backward_overrideableB
*�B    ��� � �
reduce.1654reduceP*
 �2 :R
'aten__convolution_backward_overrideable'aten__convolution_backward_overrideabler �������� � �
custom-call.296custom-callP*
 �2 :�
'xla___op_TransferWithStaticRingTransfer'xla___op_TransferWithStaticRingTransferv/home/ubuntu/aws-neuron-samples/aws_neuron_venv_pytorch/lib/python3.10/site-packages/torch_xla/core/xla_op_registry.py ,x�AwsNeuronTransferWithStaticRing��-����
 �� � �
	p124.1642	parameter�PP*	
 �2    :$
xla__device_dataxla__device_dataH|���

neff_input_namesinput124� �
convolution.1647convolution�P*	
 �2    :R
'aten__convolution_backward_overrideable'aten__convolution_backward_overrideablez
P(0
P(0� 28HPZb �������
  �� � �
transpose.1648	transposeP�*	
 �2    :R
'aten__convolution_backward_overrideable'aten__convolution_backward_overrideabler ����� � �
custom-call.297custom-callP�*	
 �2    :�
'xla___op_TransferWithStaticRingTransfer'xla___op_TransferWithStaticRingTransferv/home/ubuntu/aws-neuron-samples/aws_neuron_venv_pytorch/lib/python3.10/site-packages/torch_xla/core/xla_op_registry.py ,x�AwsNeuronTransferWithStaticRing��-����
 �� � �
	p128.1672	parameter�PP*	
 �2    :$
xla__device_dataxla__device_dataH����

neff_input_namesinput128� y
	p127.1671	parameter�*
 �2 :$
xla__device_dataxla__device_dataH���

neff_input_namesinput127� y
	p126.1670	parameter�*
 �2 :$
xla__device_dataxla__device_dataH~���

neff_input_namesinput126� �
constant.1684constant*�:D
 aten__native_batch_norm_backward aten__native_batch_norm_backwardB
*�B  �?��� � �
broadcast.1685	broadcast�*
 �2 :D
 aten__native_batch_norm_backward aten__native_batch_norm_backward����� � y
	p125.1669	parameter�*
 �2 :$
xla__device_dataxla__device_dataH}���

neff_input_namesinput125� 
divide.1686divide�*
 �2 :D
 aten__native_batch_norm_backward aten__native_batch_norm_backward������ � �
multiply.1687multiply�*
 �2 :D
 aten__native_batch_norm_backward aten__native_batch_norm_backward������ � =
constant.51constant*�: B
*�Bo����6� � �
broadcast.100	broadcast�*
 �2 :D
 aten__native_batch_norm_backward aten__native_batch_norm_backward��1��6� � w
add.15add�*
 �2 :D
 aten__native_batch_norm_backward aten__native_batch_norm_backward��1���1� � �
	p123.1641	parameterP�*	
 �2    :$
xla__device_dataxla__device_dataH{���

neff_input_namesinput123� �
transpose.1644	transpose�P*	
 �2    :R
'aten__convolution_backward_overrideable'aten__convolution_backward_overrideabler ����� � �
convolution.1646convolution�PP*	
 �2    :R
'aten__convolution_backward_overrideable'aten__convolution_backward_overrideablez
(0
(0� 2 @PZb�������
  �� � �
	p129.1673	parameter�PP*	
 �2    :$
xla__device_dataxla__device_dataH����

neff_input_namesinput129� p
logistic.1675logistic�PP*	
 �2    :*
aten__silu_backwardaten__silu_backward����� � i
constant.1674constant*�:*
aten__silu_backwardaten__silu_backwardB
*�B  �?��� � r
broadcast.1679	broadcast�PP*	
 �2    :*
aten__silu_backwardaten__silu_backward����� � r
broadcast.1676	broadcast�PP*	
 �2    :*
aten__silu_backwardaten__silu_backward����� � r
subtract.1677subtract�PP*	
 �2    :*
aten__silu_backwardaten__silu_backward������ � r
multiply.1678multiply�PP*	
 �2    :*
aten__silu_backwardaten__silu_backward������ � h
add.1680add�PP*	
 �2    :*
aten__silu_backwardaten__silu_backward������ � r
multiply.1681multiply�PP*	
 �2    :*
aten__silu_backwardaten__silu_backward������ � r
multiply.1682multiply�PP*	
 �2    :*
aten__silu_backwardaten__silu_backward������ � �
batch-norm-grad.1690batch-norm-gradD"�PP*	
 �2    "�*
 �2 "�*
 �2 :D
 aten__native_batch_norm_backward aten__native_batch_norm_backward�o�:����
����1�� � �
get-tuple-element.1693get-tuple-element�*
 �2 :D
 aten__native_batch_norm_backward aten__native_batch_norm_backwardh����� � �
custom-call.298custom-call�*
 �2 :�
'xla___op_TransferWithStaticRingTransfer'xla___op_TransferWithStaticRingTransferv/home/ubuntu/aws-neuron-samples/aws_neuron_venv_pytorch/lib/python3.10/site-packages/torch_xla/core/xla_op_registry.py ,x�AwsNeuronTransferWithStaticRing��-����
 �� � �
get-tuple-element.1692get-tuple-element�*
 �2 :D
 aten__native_batch_norm_backward aten__native_batch_norm_backwardh����� � �
custom-call.299custom-call�*
 �2 :�
'xla___op_TransferWithStaticRingTransfer'xla___op_TransferWithStaticRingTransferv/home/ubuntu/aws-neuron-samples/aws_neuron_venv_pytorch/lib/python3.10/site-packages/torch_xla/core/xla_op_registry.py ,x�AwsNeuronTransferWithStaticRing��-����
 �� � �
	p131.1709	parameter�PP*	
 �2    :$
xla__device_dataxla__device_dataH����

neff_input_namesinput131� �
get-tuple-element.1691get-tuple-element�PP*	
 �2    :D
 aten__native_batch_norm_backward aten__native_batch_norm_backward����� � �
convolution.1713convolution��*	
 �2    :R
'aten__convolution_backward_overrideable'aten__convolution_backward_overrideablez
P (0
P (0� 28HPZb �������
  �� � �
transpose.1714	transpose��*	
 �2    :R
'aten__convolution_backward_overrideable'aten__convolution_backward_overrideabler ����� � �
custom-call.300custom-call��*	
 �2    :�
'xla___op_TransferWithStaticRingTransfer'xla___op_TransferWithStaticRingTransferv/home/ubuntu/aws-neuron-samples/aws_neuron_venv_pytorch/lib/python3.10/site-packages/torch_xla/core/xla_op_registry.py ,x�AwsNeuronTransferWithStaticRing��-����
 �� � �
	p135.1731	parameter�PP*	
 �2    :$
xla__device_dataxla__device_dataH����

neff_input_namesinput135� z
	p134.1730	parameter�*
 �2 :$
xla__device_dataxla__device_dataH����

neff_input_namesinput134� z
	p133.1729	parameter�*
 �2 :$
xla__device_dataxla__device_dataH����

neff_input_namesinput133� �
constant.1743constant*�:D
 aten__native_batch_norm_backward aten__native_batch_norm_backwardB
*�B  �?��� � �
broadcast.1744	broadcast�*
 �2 :D
 aten__native_batch_norm_backward aten__native_batch_norm_backward����� � z
	p132.1728	parameter�*
 �2 :$
xla__device_dataxla__device_dataH����

neff_input_namesinput132� 
divide.1745divide�*
 �2 :D
 aten__native_batch_norm_backward aten__native_batch_norm_backward������ � �
multiply.1746multiply�*
 �2 :D
 aten__native_batch_norm_backward aten__native_batch_norm_backward������ � =
constant.52constant*�: B
*�Bo����6� � �
broadcast.102	broadcast�*
 �2 :D
 aten__native_batch_norm_backward aten__native_batch_norm_backward��1��6� � w
add.16add�*
 �2 :D
 aten__native_batch_norm_backward aten__native_batch_norm_backward��1���1� � �
	p130.1708	parameter��*	
 �2    :$
xla__device_dataxla__device_dataH����

neff_input_namesinput130� �
transpose.1710	transpose��*	
 �2    :R
'aten__convolution_backward_overrideable'aten__convolution_backward_overrideabler ����� � �
reverse.1711reverse��*	
 �2    :R
'aten__convolution_backward_overrideable'aten__convolution_backward_overrideabler ����� � �
convolution.1712convolution�PP*	
 �2    :R
'aten__convolution_backward_overrideable'aten__convolution_backward_overrideablez
 (0
 (0� 2 @PZb�������
  �� � �
	p136.1732	parameter�PP*	
 �2    :$
xla__device_dataxla__device_dataH����

neff_input_namesinput136� p
logistic.1734logistic�PP*	
 �2    :*
aten__silu_backwardaten__silu_backward����� � i
constant.1733constant*�:*
aten__silu_backwardaten__silu_backwardB
*�B  �?��� � r
broadcast.1738	broadcast�PP*	
 �2    :*
aten__silu_backwardaten__silu_backward����� � r
broadcast.1735	broadcast�PP*	
 �2    :*
aten__silu_backwardaten__silu_backward����� � r
subtract.1736subtract�PP*	
 �2    :*
aten__silu_backwardaten__silu_backward������ � r
multiply.1737multiply�PP*	
 �2    :*
aten__silu_backwardaten__silu_backward������ � h
add.1739add�PP*	
 �2    :*
aten__silu_backwardaten__silu_backward������ � r
multiply.1740multiply�PP*	
 �2    :*
aten__silu_backwardaten__silu_backward������ � r
multiply.1741multiply�PP*	
 �2    :*
aten__silu_backwardaten__silu_backward������ � �
batch-norm-grad.1749batch-norm-gradD"�PP*	
 �2    "�*
 �2 "�*
 �2 :D
 aten__native_batch_norm_backward aten__native_batch_norm_backward�o�:����
����1�� � �
get-tuple-element.1752get-tuple-element�*
 �2 :D
 aten__native_batch_norm_backward aten__native_batch_norm_backwardh����� � �
custom-call.301custom-call�*
 �2 :�
'xla___op_TransferWithStaticRingTransfer'xla___op_TransferWithStaticRingTransferv/home/ubuntu/aws-neuron-samples/aws_neuron_venv_pytorch/lib/python3.10/site-packages/torch_xla/core/xla_op_registry.py ,x�AwsNeuronTransferWithStaticRing��-����
 �� � �
get-tuple-element.1751get-tuple-element�*
 �2 :D
 aten__native_batch_norm_backward aten__native_batch_norm_backwardh����� � �
custom-call.302custom-call�*
 �2 :�
'xla___op_TransferWithStaticRingTransfer'xla___op_TransferWithStaticRingTransferv/home/ubuntu/aws-neuron-samples/aws_neuron_venv_pytorch/lib/python3.10/site-packages/torch_xla/core/xla_op_registry.py ,x�AwsNeuronTransferWithStaticRing��-����
 �� � �
get-tuple-element.1750get-tuple-element�PP*	
 �2    :D
 aten__native_batch_norm_backward aten__native_batch_norm_backward����� � �
convolution.1771convolution��*	
 �2    :R
'aten__convolution_backward_overrideable'aten__convolution_backward_overrideablez
P (0
P (0� 28HPZb �������
  �� � �
transpose.1772	transpose��*	
 �2    :R
'aten__convolution_backward_overrideable'aten__convolution_backward_overrideabler ����� � �
custom-call.303custom-call��*	
 �2    :�
'xla___op_TransferWithStaticRingTransfer'xla___op_TransferWithStaticRingTransferv/home/ubuntu/aws-neuron-samples/aws_neuron_venv_pytorch/lib/python3.10/site-packages/torch_xla/core/xla_op_registry.py ,x�AwsNeuronTransferWithStaticRing��-����
 �� � �
	p141.1789	parameter�PP*	
 �2    :$
xla__device_dataxla__device_dataH����

neff_input_namesinput141� z
	p140.1788	parameter�*
 �2 :$
xla__device_dataxla__device_dataH����

neff_input_namesinput140� z
	p139.1787	parameter�*
 �2 :$
xla__device_dataxla__device_dataH����

neff_input_namesinput139� �
constant.1808constant*�:D
 aten__native_batch_norm_backward aten__native_batch_norm_backwardB
*�B  �?��� � �
broadcast.1809	broadcast�*
 �2 :D
 aten__native_batch_norm_backward aten__native_batch_norm_backward����� � z
	p138.1786	parameter�*
 �2 :$
xla__device_dataxla__device_dataH����

neff_input_namesinput138� 
divide.1810divide�*
 �2 :D
 aten__native_batch_norm_backward aten__native_batch_norm_backward������ � �
multiply.1811multiply�*
 �2 :D
 aten__native_batch_norm_backward aten__native_batch_norm_backward������ � =
constant.53constant*�: B
*�Bo����6� � �
broadcast.104	broadcast�*
 �2 :D
 aten__native_batch_norm_backward aten__native_batch_norm_backward��1��6� � w
add.17add�*
 �2 :D
 aten__native_batch_norm_backward aten__native_batch_norm_backward��1���1� � �
	p121.1621	parameter��*	
 �2    :$
xla__device_dataxla__device_dataHy���

neff_input_namesinput121� �
transpose.1623	transpose��*	
 �2    :R
'aten__convolution_backward_overrideable'aten__convolution_backward_overrideabler ����� � �
reverse.1624reverse��*	
 �2    :R
'aten__convolution_backward_overrideable'aten__convolution_backward_overrideabler ����� � �
convolution.1625convolution�PP*	
 �2    :R
'aten__convolution_backward_overrideable'aten__convolution_backward_overrideablez
 (0
 (0� 2 @PZb�������
  �� � �
	p137.1767	parameter��*	
 �2    :$
xla__device_dataxla__device_dataH����

neff_input_namesinput137� �
transpose.1768	transpose��*	
 �2    :R
'aten__convolution_backward_overrideable'aten__convolution_backward_overrideabler ����� � �
reverse.1769reverse��*	
 �2    :R
'aten__convolution_backward_overrideable'aten__convolution_backward_overrideabler ����� � �
convolution.1770convolution�PP*	
 �2    :R
'aten__convolution_backward_overrideable'aten__convolution_backward_overrideablez
 (0
 (0� 2 @PZb�������
  �� � T
add.1797add�PP*	
 �2    :
	aten__add	aten__add������ � �
	p142.1790	parameter�PP*	
 �2    :$
xla__device_dataxla__device_dataH����

neff_input_namesinput142� p
logistic.1799logistic�PP*	
 �2    :*
aten__silu_backwardaten__silu_backward����� � i
constant.1798constant*�:*
aten__silu_backwardaten__silu_backwardB
*�B  �?��� � r
broadcast.1803	broadcast�PP*	
 �2    :*
aten__silu_backwardaten__silu_backward����� � r
broadcast.1800	broadcast�PP*	
 �2    :*
aten__silu_backwardaten__silu_backward����� � r
subtract.1801subtract�PP*	
 �2    :*
aten__silu_backwardaten__silu_backward������ � r
multiply.1802multiply�PP*	
 �2    :*
aten__silu_backwardaten__silu_backward������ � h
add.1804add�PP*	
 �2    :*
aten__silu_backwardaten__silu_backward������ � r
multiply.1805multiply�PP*	
 �2    :*
aten__silu_backwardaten__silu_backward������ � r
multiply.1806multiply�PP*	
 �2    :*
aten__silu_backwardaten__silu_backward������ � �
batch-norm-grad.1814batch-norm-gradD"�PP*	
 �2    "�*
 �2 "�*
 �2 :D
 aten__native_batch_norm_backward aten__native_batch_norm_backward�o�:����
����1�� � �
get-tuple-element.1817get-tuple-element�*
 �2 :D
 aten__native_batch_norm_backward aten__native_batch_norm_backwardh����� � �
custom-call.304custom-call�*
 �2 :�
'xla___op_TransferWithStaticRingTransfer'xla___op_TransferWithStaticRingTransferv/home/ubuntu/aws-neuron-samples/aws_neuron_venv_pytorch/lib/python3.10/site-packages/torch_xla/core/xla_op_registry.py ,x�AwsNeuronTransferWithStaticRing��-����
 �� � �
get-tuple-element.1816get-tuple-element�*
 �2 :D
 aten__native_batch_norm_backward aten__native_batch_norm_backwardh����� � �
custom-call.305custom-call�*
 �2 :�
'xla___op_TransferWithStaticRingTransfer'xla___op_TransferWithStaticRingTransferv/home/ubuntu/aws-neuron-samples/aws_neuron_venv_pytorch/lib/python3.10/site-packages/torch_xla/core/xla_op_registry.py ,x�AwsNeuronTransferWithStaticRing��-����
 �� � �
	p144.1833	parameter�PP*	
 �2    :$
xla__device_dataxla__device_dataH����

neff_input_namesinput144� �
get-tuple-element.1815get-tuple-element�PP*	
 �2    :D
 aten__native_batch_norm_backward aten__native_batch_norm_backward����� � �
convolution.1837convolution��*	
 �2    :R
'aten__convolution_backward_overrideable'aten__convolution_backward_overrideablez
P(0
P(0� 28HPZb �������
  �� � �
transpose.1838	transpose��*	
 �2    :R
'aten__convolution_backward_overrideable'aten__convolution_backward_overrideabler ����� � �
custom-call.306custom-call��*	
 �2    :�
'xla___op_TransferWithStaticRingTransfer'xla___op_TransferWithStaticRingTransferv/home/ubuntu/aws-neuron-samples/aws_neuron_venv_pytorch/lib/python3.10/site-packages/torch_xla/core/xla_op_registry.py ,x�AwsNeuronTransferWithStaticRing��-����
 �� � �
	p148.1855	parameter�*	
 �2    :$
xla__device_dataxla__device_dataH����

neff_input_namesinput148� z
	p147.1854	parameter�*
 �2 :$
xla__device_dataxla__device_dataH����

neff_input_namesinput147� z
	p146.1853	parameter�*
 �2 :$
xla__device_dataxla__device_dataH����

neff_input_namesinput146� �
constant.1867constant*�:D
 aten__native_batch_norm_backward aten__native_batch_norm_backwardB
*�B  �?��� � �
broadcast.1868	broadcast�*
 �2 :D
 aten__native_batch_norm_backward aten__native_batch_norm_backward����� � z
	p145.1852	parameter�*
 �2 :$
xla__device_dataxla__device_dataH����

neff_input_namesinput145� 
divide.1869divide�*
 �2 :D
 aten__native_batch_norm_backward aten__native_batch_norm_backward������ � �
multiply.1870multiply�*
 �2 :D
 aten__native_batch_norm_backward aten__native_batch_norm_backward������ � =
constant.55constant*�: B
*�Bo����6� � �
broadcast.106	broadcast�*
 �2 :D
 aten__native_batch_norm_backward aten__native_batch_norm_backward��1��6� � w
add.18add�*
 �2 :D
 aten__native_batch_norm_backward aten__native_batch_norm_backward��1���1� � �
p61.852	parameter��*	
 �2    :$
xla__device_dataxla__device_dataH=���

neff_input_namesinput61� �
transpose.854	transpose��*	
 �2    :R
'aten__convolution_backward_overrideable'aten__convolution_backward_overrideabler ����� � �
convolution.856convolution�*	
 �2    :R
'aten__convolution_backward_overrideable'aten__convolution_backward_overrideablez
(0
(0� 2 @PZb�������
  �� � �
	p149.1856	parameter�*	
 �2    :$
xla__device_dataxla__device_dataH����

neff_input_namesinput149� p
logistic.1858logistic�*	
 �2    :*
aten__silu_backwardaten__silu_backward����� � i
constant.1857constant*�:*
aten__silu_backwardaten__silu_backwardB
*�B  �?��� � r
broadcast.1862	broadcast�*	
 �2    :*
aten__silu_backwardaten__silu_backward����� � r
broadcast.1859	broadcast�*	
 �2    :*
aten__silu_backwardaten__silu_backward����� � r
subtract.1860subtract�*	
 �2    :*
aten__silu_backwardaten__silu_backward������ � r
multiply.1861multiply�*	
 �2    :*
aten__silu_backwardaten__silu_backward������ � h
add.1863add�*	
 �2    :*
aten__silu_backwardaten__silu_backward������ � r
multiply.1864multiply�*	
 �2    :*
aten__silu_backwardaten__silu_backward������ � r
multiply.1865multiply�*	
 �2    :*
aten__silu_backwardaten__silu_backward������ � �
batch-norm-grad.1873batch-norm-gradD"�*	
 �2    "�*
 �2 "�*
 �2 :D
 aten__native_batch_norm_backward aten__native_batch_norm_backward�o�:����
����1�� � �
get-tuple-element.1876get-tuple-element�*
 �2 :D
 aten__native_batch_norm_backward aten__native_batch_norm_backwardh����� � �
custom-call.307custom-call�*
 �2 :�
'xla___op_TransferWithStaticRingTransfer'xla___op_TransferWithStaticRingTransferv/home/ubuntu/aws-neuron-samples/aws_neuron_venv_pytorch/lib/python3.10/site-packages/torch_xla/core/xla_op_registry.py ,x�AwsNeuronTransferWithStaticRing��-����
 �� � �
get-tuple-element.1875get-tuple-element�*
 �2 :D
 aten__native_batch_norm_backward aten__native_batch_norm_backwardh����� � �
custom-call.308custom-call�*
 �2 :�
'xla___op_TransferWithStaticRingTransfer'xla___op_TransferWithStaticRingTransferv/home/ubuntu/aws-neuron-samples/aws_neuron_venv_pytorch/lib/python3.10/site-packages/torch_xla/core/xla_op_registry.py ,x�AwsNeuronTransferWithStaticRing��-����
 �� � �
	p151.1892	parameter�*	
 �2    :$
xla__device_dataxla__device_dataH����

neff_input_namesinput151� �
get-tuple-element.1874get-tuple-element�*	
 �2    :D
 aten__native_batch_norm_backward aten__native_batch_norm_backward����� � �
convolution.1896convolution��*	
 �2    :R
'aten__convolution_backward_overrideable'aten__convolution_backward_overrideablez
(0
(0� 28HPZb �������
  �� � �
transpose.1897	transpose��*	
 �2    :R
'aten__convolution_backward_overrideable'aten__convolution_backward_overrideabler ����� � �
custom-call.309custom-call��*	
 �2    :�
'xla___op_TransferWithStaticRingTransfer'xla___op_TransferWithStaticRingTransferv/home/ubuntu/aws-neuron-samples/aws_neuron_venv_pytorch/lib/python3.10/site-packages/torch_xla/core/xla_op_registry.py ,x�AwsNeuronTransferWithStaticRing��-����
 �� � �
	p155.1914	parameter�*	
 �2    :$
xla__device_dataxla__device_dataH����

neff_input_namesinput155� z
	p154.1913	parameter�*
 �2 :$
xla__device_dataxla__device_dataH����

neff_input_namesinput154� z
	p153.1912	parameter�*
 �2 :$
xla__device_dataxla__device_dataH����

neff_input_namesinput153� �
constant.1927constant*�:D
 aten__native_batch_norm_backward aten__native_batch_norm_backwardB
*�B  �?��� � �
broadcast.1928	broadcast�*
 �2 :D
 aten__native_batch_norm_backward aten__native_batch_norm_backward����� � z
	p152.1911	parameter�*
 �2 :$
xla__device_dataxla__device_dataH����

neff_input_namesinput152� 
divide.1929divide�*
 �2 :D
 aten__native_batch_norm_backward aten__native_batch_norm_backward������ � �
multiply.1930multiply�*
 �2 :D
 aten__native_batch_norm_backward aten__native_batch_norm_backward������ � =
constant.56constant*�: B
*�Bo����6� � �
broadcast.107	broadcast�*
 �2 :D
 aten__native_batch_norm_backward aten__native_batch_norm_backward��1��6� � w
add.19add�*
 �2 :D
 aten__native_batch_norm_backward aten__native_batch_norm_backward��1���1� � �
	p150.1891	parameter��*	
 �2    :$
xla__device_dataxla__device_dataH����

neff_input_namesinput150� �
transpose.1893	transpose��*	
 �2    :R
'aten__convolution_backward_overrideable'aten__convolution_backward_overrideabler ����� � �
convolution.1895convolution�*	
 �2    :R
'aten__convolution_backward_overrideable'aten__convolution_backward_overrideablez
(0
(0� 2 @PZb�������
  �� � w

slice.1916slice�*	
 �2    :
xla__selectxla__select���������� � �
	p156.1915	parameter�*	
 �2    :$
xla__device_dataxla__device_dataH����

neff_input_namesinput156� p
logistic.1918logistic�*	
 �2    :*
aten__silu_backwardaten__silu_backward����� � i
constant.1917constant*�:*
aten__silu_backwardaten__silu_backwardB
*�B  �?��� � r
broadcast.1922	broadcast�*	
 �2    :*
aten__silu_backwardaten__silu_backward����� � r
broadcast.1919	broadcast�*	
 �2    :*
aten__silu_backwardaten__silu_backward����� � r
subtract.1920subtract�*	
 �2    :*
aten__silu_backwardaten__silu_backward������ � r
multiply.1921multiply�*	
 �2    :*
aten__silu_backwardaten__silu_backward������ � h
add.1923add�*	
 �2    :*
aten__silu_backwardaten__silu_backward������ � r
multiply.1924multiply�*	
 �2    :*
aten__silu_backwardaten__silu_backward������ � r
multiply.1925multiply�*	
 �2    :*
aten__silu_backwardaten__silu_backward������ � �
batch-norm-grad.1933batch-norm-gradD"�*	
 �2    "�*
 �2 "�*
 �2 :D
 aten__native_batch_norm_backward aten__native_batch_norm_backward�o�:����
����1�� � �
get-tuple-element.1936get-tuple-element�*
 �2 :D
 aten__native_batch_norm_backward aten__native_batch_norm_backwardh����� � �
custom-call.310custom-call�*
 �2 :�
'xla___op_TransferWithStaticRingTransfer'xla___op_TransferWithStaticRingTransferv/home/ubuntu/aws-neuron-samples/aws_neuron_venv_pytorch/lib/python3.10/site-packages/torch_xla/core/xla_op_registry.py ,x�AwsNeuronTransferWithStaticRing��-����
 �� � �
get-tuple-element.1935get-tuple-element�*
 �2 :D
 aten__native_batch_norm_backward aten__native_batch_norm_backwardh����� � �
custom-call.311custom-call�*
 �2 :�
'xla___op_TransferWithStaticRingTransfer'xla___op_TransferWithStaticRingTransferv/home/ubuntu/aws-neuron-samples/aws_neuron_venv_pytorch/lib/python3.10/site-packages/torch_xla/core/xla_op_registry.py ,x�AwsNeuronTransferWithStaticRing��-����
 �� � �
	p158.1952	parameter�*	
 �2    :$
xla__device_dataxla__device_dataH����

neff_input_namesinput158� �
get-tuple-element.1934get-tuple-element�*	
 �2    :D
 aten__native_batch_norm_backward aten__native_batch_norm_backward����� � �
convolution.1956convolution��*	
 �2    :R
'aten__convolution_backward_overrideable'aten__convolution_backward_overrideablez
 (0
 (0� 28HPZb �������
  �� � �
transpose.1957	transpose��*	
 �2    :R
'aten__convolution_backward_overrideable'aten__convolution_backward_overrideabler ����� � �
custom-call.312custom-call��*	
 �2    :�
'xla___op_TransferWithStaticRingTransfer'xla___op_TransferWithStaticRingTransferv/home/ubuntu/aws-neuron-samples/aws_neuron_venv_pytorch/lib/python3.10/site-packages/torch_xla/core/xla_op_registry.py ,x�AwsNeuronTransferWithStaticRing��-����
 �� � �
	p162.1974	parameter�*	
 �2    :$
xla__device_dataxla__device_dataH����

neff_input_namesinput162� z
	p161.1973	parameter�*
 �2 :$
xla__device_dataxla__device_dataH����

neff_input_namesinput161� z
	p160.1972	parameter�*
 �2 :$
xla__device_dataxla__device_dataH����

neff_input_namesinput160� �
constant.1986constant*�:D
 aten__native_batch_norm_backward aten__native_batch_norm_backwardB
*�B  �?��� � �
broadcast.1987	broadcast�*
 �2 :D
 aten__native_batch_norm_backward aten__native_batch_norm_backward����� � z
	p159.1971	parameter�*
 �2 :$
xla__device_dataxla__device_dataH����

neff_input_namesinput159� 
divide.1988divide�*
 �2 :D
 aten__native_batch_norm_backward aten__native_batch_norm_backward������ � �
multiply.1989multiply�*
 �2 :D
 aten__native_batch_norm_backward aten__native_batch_norm_backward������ � =
constant.57constant*�: B
*�Bo����6� � �
broadcast.108	broadcast�*
 �2 :D
 aten__native_batch_norm_backward aten__native_batch_norm_backward��1��6� � w
add.20add�*
 �2 :D
 aten__native_batch_norm_backward aten__native_batch_norm_backward��1���1� � �
	p157.1951	parameter��*	
 �2    :$
xla__device_dataxla__device_dataH����

neff_input_namesinput157� �
transpose.1953	transpose��*	
 �2    :R
'aten__convolution_backward_overrideable'aten__convolution_backward_overrideabler ����� � �
reverse.1954reverse��*	
 �2    :R
'aten__convolution_backward_overrideable'aten__convolution_backward_overrideabler ����� � �
convolution.1955convolution�*	
 �2    :R
'aten__convolution_backward_overrideable'aten__convolution_backward_overrideablez
 (0
 (0� 2 @PZb�������
  �� � �
	p163.1975	parameter�*	
 �2    :$
xla__device_dataxla__device_dataH����

neff_input_namesinput163� p
logistic.1977logistic�*	
 �2    :*
aten__silu_backwardaten__silu_backward����� � i
constant.1976constant*�:*
aten__silu_backwardaten__silu_backwardB
*�B  �?��� � r
broadcast.1981	broadcast�*	
 �2    :*
aten__silu_backwardaten__silu_backward����� � r
broadcast.1978	broadcast�*	
 �2    :*
aten__silu_backwardaten__silu_backward����� � r
subtract.1979subtract�*	
 �2    :*
aten__silu_backwardaten__silu_backward������ � r
multiply.1980multiply�*	
 �2    :*
aten__silu_backwardaten__silu_backward������ � h
add.1982add�*	
 �2    :*
aten__silu_backwardaten__silu_backward������ � r
multiply.1983multiply�*	
 �2    :*
aten__silu_backwardaten__silu_backward������ � r
multiply.1984multiply�*	
 �2    :*
aten__silu_backwardaten__silu_backward������ � �
batch-norm-grad.1992batch-norm-gradD"�*	
 �2    "�*
 �2 "�*
 �2 :D
 aten__native_batch_norm_backward aten__native_batch_norm_backward�o�:����
����1�� � �
get-tuple-element.1995get-tuple-element�*
 �2 :D
 aten__native_batch_norm_backward aten__native_batch_norm_backwardh����� � �
custom-call.313custom-call�*
 �2 :�
'xla___op_TransferWithStaticRingTransfer'xla___op_TransferWithStaticRingTransferv/home/ubuntu/aws-neuron-samples/aws_neuron_venv_pytorch/lib/python3.10/site-packages/torch_xla/core/xla_op_registry.py ,x�AwsNeuronTransferWithStaticRing��-����
 �� � �
get-tuple-element.1994get-tuple-element�*
 �2 :D
 aten__native_batch_norm_backward aten__native_batch_norm_backwardh����� � �
custom-call.314custom-call�*
 �2 :�
'xla___op_TransferWithStaticRingTransfer'xla___op_TransferWithStaticRingTransferv/home/ubuntu/aws-neuron-samples/aws_neuron_venv_pytorch/lib/python3.10/site-packages/torch_xla/core/xla_op_registry.py ,x�AwsNeuronTransferWithStaticRing��-����
 �� � �
	p165.2011	parameter�*	
 �2    :$
xla__device_dataxla__device_dataH����

neff_input_namesinput165� �
get-tuple-element.1993get-tuple-element�*	
 �2    :D
 aten__native_batch_norm_backward aten__native_batch_norm_backward����� � �
convolution.2015convolution��*	
 �2    :R
'aten__convolution_backward_overrideable'aten__convolution_backward_overrideablez
(0
(0� 28HPZb �������
  �� � �
transpose.2016	transpose��*	
 �2    :R
'aten__convolution_backward_overrideable'aten__convolution_backward_overrideabler ����� � �
custom-call.315custom-call��*	
 �2    :�
'xla___op_TransferWithStaticRingTransfer'xla___op_TransferWithStaticRingTransferv/home/ubuntu/aws-neuron-samples/aws_neuron_venv_pytorch/lib/python3.10/site-packages/torch_xla/core/xla_op_registry.py ,x�AwsNeuronTransferWithStaticRing��-����
 �� � �
	p169.2033	parameter�*	
 �2    :$
xla__device_dataxla__device_dataH����

neff_input_namesinput169� z
	p168.2032	parameter�*
 �2 :$
xla__device_dataxla__device_dataH����

neff_input_namesinput168� z
	p167.2031	parameter�*
 �2 :$
xla__device_dataxla__device_dataH����

neff_input_namesinput167� �
constant.2046constant*�:D
 aten__native_batch_norm_backward aten__native_batch_norm_backwardB
*�B  �?��� � �
broadcast.2047	broadcast�*
 �2 :D
 aten__native_batch_norm_backward aten__native_batch_norm_backward����� � z
	p166.2030	parameter�*
 �2 :$
xla__device_dataxla__device_dataH����

neff_input_namesinput166� 
divide.2048divide�*
 �2 :D
 aten__native_batch_norm_backward aten__native_batch_norm_backward������ � �
multiply.2049multiply�*
 �2 :D
 aten__native_batch_norm_backward aten__native_batch_norm_backward������ � =
constant.58constant*�: B
*�Bo����6� � �
broadcast.109	broadcast�*
 �2 :D
 aten__native_batch_norm_backward aten__native_batch_norm_backward��1��6� � w
add.21add�*
 �2 :D
 aten__native_batch_norm_backward aten__native_batch_norm_backward��1���1� � z

slice.2035slice�*	
 �2    :
xla__selectxla__select����������� � �
	p170.2034	parameter�*	
 �2    :$
xla__device_dataxla__device_dataH����

neff_input_namesinput170� p
logistic.2037logistic�*	
 �2    :*
aten__silu_backwardaten__silu_backward����� � i
constant.2036constant*�:*
aten__silu_backwardaten__silu_backwardB
*�B  �?��� � r
broadcast.2041	broadcast�*	
 �2    :*
aten__silu_backwardaten__silu_backward����� � r
broadcast.2038	broadcast�*	
 �2    :*
aten__silu_backwardaten__silu_backward����� � r
subtract.2039subtract�*	
 �2    :*
aten__silu_backwardaten__silu_backward������ � r
multiply.2040multiply�*	
 �2    :*
aten__silu_backwardaten__silu_backward������ � h
add.2042add�*	
 �2    :*
aten__silu_backwardaten__silu_backward������ � r
multiply.2043multiply�*	
 �2    :*
aten__silu_backwardaten__silu_backward������ � r
multiply.2044multiply�*	
 �2    :*
aten__silu_backwardaten__silu_backward������ � �
batch-norm-grad.2052batch-norm-gradD"�*	
 �2    "�*
 �2 "�*
 �2 :D
 aten__native_batch_norm_backward aten__native_batch_norm_backward�o�:����
����1�� � �
get-tuple-element.2055get-tuple-element�*
 �2 :D
 aten__native_batch_norm_backward aten__native_batch_norm_backwardh����� � �
custom-call.316custom-call�*
 �2 :�
'xla___op_TransferWithStaticRingTransfer'xla___op_TransferWithStaticRingTransferv/home/ubuntu/aws-neuron-samples/aws_neuron_venv_pytorch/lib/python3.10/site-packages/torch_xla/core/xla_op_registry.py ,x�AwsNeuronTransferWithStaticRing��-����
 �� � �
get-tuple-element.2054get-tuple-element�*
 �2 :D
 aten__native_batch_norm_backward aten__native_batch_norm_backwardh����� � �
custom-call.317custom-call�*
 �2 :�
'xla___op_TransferWithStaticRingTransfer'xla___op_TransferWithStaticRingTransferv/home/ubuntu/aws-neuron-samples/aws_neuron_venv_pytorch/lib/python3.10/site-packages/torch_xla/core/xla_op_registry.py ,x�AwsNeuronTransferWithStaticRing��-����
 �� � �
	p172.2071	parameter�*	
 �2    :$
xla__device_dataxla__device_dataH����

neff_input_namesinput172� �
get-tuple-element.2053get-tuple-element�*	
 �2    :D
 aten__native_batch_norm_backward aten__native_batch_norm_backward����� � �
convolution.2075convolution��*	
 �2    :R
'aten__convolution_backward_overrideable'aten__convolution_backward_overrideablez
(0
(0� 28HPZb �������
  �� � �
transpose.2076	transpose��*	
 �2    :R
'aten__convolution_backward_overrideable'aten__convolution_backward_overrideabler ����� � �
custom-call.318custom-call��*	
 �2    :�
'xla___op_TransferWithStaticRingTransfer'xla___op_TransferWithStaticRingTransferv/home/ubuntu/aws-neuron-samples/aws_neuron_venv_pytorch/lib/python3.10/site-packages/torch_xla/core/xla_op_registry.py ,x�AwsNeuronTransferWithStaticRing��.����
 �� � �
	p176.2093	parameter�*	
 �2    :$
xla__device_dataxla__device_dataH����

neff_input_namesinput176� z
	p175.2092	parameter�*
 �2 :$
xla__device_dataxla__device_dataH����

neff_input_namesinput175� z
	p174.2091	parameter�*
 �2 :$
xla__device_dataxla__device_dataH����

neff_input_namesinput174� �
constant.2105constant*�:D
 aten__native_batch_norm_backward aten__native_batch_norm_backwardB
*�B  �?��� � �
broadcast.2106	broadcast�*
 �2 :D
 aten__native_batch_norm_backward aten__native_batch_norm_backward����� � z
	p173.2090	parameter�*
 �2 :$
xla__device_dataxla__device_dataH����

neff_input_namesinput173� 
divide.2107divide�*
 �2 :D
 aten__native_batch_norm_backward aten__native_batch_norm_backward������ � �
multiply.2108multiply�*
 �2 :D
 aten__native_batch_norm_backward aten__native_batch_norm_backward������ � =
constant.59constant*�: B
*�Bo����6� � �
broadcast.110	broadcast�*
 �2 :D
 aten__native_batch_norm_backward aten__native_batch_norm_backward��1��6� � w
add.22add�*
 �2 :D
 aten__native_batch_norm_backward aten__native_batch_norm_backward��1���1� � �
	p164.2010	parameter��*	
 �2    :$
xla__device_dataxla__device_dataH����

neff_input_namesinput164� �
transpose.2012	transpose��*	
 �2    :R
'aten__convolution_backward_overrideable'aten__convolution_backward_overrideabler ����� � �
convolution.2014convolution�*	
 �2    :R
'aten__convolution_backward_overrideable'aten__convolution_backward_overrideablez
(0
(0� 2 @PZb�������
  �� � �
	p177.2094	parameter�*	
 �2    :$
xla__device_dataxla__device_dataH����

neff_input_namesinput177� p
logistic.2096logistic�*	
 �2    :*
aten__silu_backwardaten__silu_backward����� � i
constant.2095constant*�:*
aten__silu_backwardaten__silu_backwardB
*�B  �?��� � r
broadcast.2100	broadcast�*	
 �2    :*
aten__silu_backwardaten__silu_backward����� � r
broadcast.2097	broadcast�*	
 �2    :*
aten__silu_backwardaten__silu_backward����� � r
subtract.2098subtract�*	
 �2    :*
aten__silu_backwardaten__silu_backward������ � r
multiply.2099multiply�*	
 �2    :*
aten__silu_backwardaten__silu_backward������ � h
add.2101add�*	
 �2    :*
aten__silu_backwardaten__silu_backward������ � r
multiply.2102multiply�*	
 �2    :*
aten__silu_backwardaten__silu_backward������ � r
multiply.2103multiply�*	
 �2    :*
aten__silu_backwardaten__silu_backward������ � �
batch-norm-grad.2111batch-norm-gradD"�*	
 �2    "�*
 �2 "�*
 �2 :D
 aten__native_batch_norm_backward aten__native_batch_norm_backward�o�:����
����1�� � �
get-tuple-element.2114get-tuple-element�*
 �2 :D
 aten__native_batch_norm_backward aten__native_batch_norm_backwardh����� � �
custom-call.319custom-call�*
 �2 :�
'xla___op_TransferWithStaticRingTransfer'xla___op_TransferWithStaticRingTransferv/home/ubuntu/aws-neuron-samples/aws_neuron_venv_pytorch/lib/python3.10/site-packages/torch_xla/core/xla_op_registry.py ,x�AwsNeuronTransferWithStaticRing��.����
 �� � �
get-tuple-element.2113get-tuple-element�*
 �2 :D
 aten__native_batch_norm_backward aten__native_batch_norm_backwardh����� � �
custom-call.320custom-call�*
 �2 :�
'xla___op_TransferWithStaticRingTransfer'xla___op_TransferWithStaticRingTransferv/home/ubuntu/aws-neuron-samples/aws_neuron_venv_pytorch/lib/python3.10/site-packages/torch_xla/core/xla_op_registry.py ,x�AwsNeuronTransferWithStaticRing��.����
 �� � �
get-tuple-element.2112get-tuple-element�*	
 �2    :D
 aten__native_batch_norm_backward aten__native_batch_norm_backward����� � �
convolution.2133convolution��*	
 �2    :R
'aten__convolution_backward_overrideable'aten__convolution_backward_overrideablez
(0
(0� 28HPZb �������
  �� � �
transpose.2134	transpose��*	
 �2    :R
'aten__convolution_backward_overrideable'aten__convolution_backward_overrideabler ����� � �
custom-call.321custom-call��*	
 �2    :�
'xla___op_TransferWithStaticRingTransfer'xla___op_TransferWithStaticRingTransferv/home/ubuntu/aws-neuron-samples/aws_neuron_venv_pytorch/lib/python3.10/site-packages/torch_xla/core/xla_op_registry.py ,x�AwsNeuronTransferWithStaticRing��.����
 �� � �
	p182.2151	parameter�*	
 �2    :$
xla__device_dataxla__device_dataH����

neff_input_namesinput182� z
	p181.2150	parameter�*
 �2 :$
xla__device_dataxla__device_dataH����

neff_input_namesinput181� z
	p180.2149	parameter�*
 �2 :$
xla__device_dataxla__device_dataH����

neff_input_namesinput180� �
constant.2171constant*�:D
 aten__native_batch_norm_backward aten__native_batch_norm_backwardB
*�B  �?��� � �
broadcast.2172	broadcast�*
 �2 :D
 aten__native_batch_norm_backward aten__native_batch_norm_backward����� � z
	p179.2148	parameter�*
 �2 :$
xla__device_dataxla__device_dataH����

neff_input_namesinput179� 
divide.2173divide�*
 �2 :D
 aten__native_batch_norm_backward aten__native_batch_norm_backward������ � �
multiply.2174multiply�*
 �2 :D
 aten__native_batch_norm_backward aten__native_batch_norm_backward������ � =
constant.60constant*�: B
*�Bo����6� � �
broadcast.112	broadcast�*
 �2 :D
 aten__native_batch_norm_backward aten__native_batch_norm_backward��1��6� � w
add.23add�*
 �2 :D
 aten__native_batch_norm_backward aten__native_batch_norm_backward��1���1� � �
	p171.2070	parameter��*	
 �2    :$
xla__device_dataxla__device_dataH����

neff_input_namesinput171� �
transpose.2072	transpose��*	
 �2    :R
'aten__convolution_backward_overrideable'aten__convolution_backward_overrideabler ����� � �
convolution.2074convolution�*	
 �2    :R
'aten__convolution_backward_overrideable'aten__convolution_backward_overrideablez
(0
(0� 2 @PZb�������
  �� � �
	p178.2129	parameter��*	
 �2    :$
xla__device_dataxla__device_dataH����

neff_input_namesinput178� �
transpose.2130	transpose��*	
 �2    :R
'aten__convolution_backward_overrideable'aten__convolution_backward_overrideabler ����� � �
convolution.2132convolution�*	
 �2    :R
'aten__convolution_backward_overrideable'aten__convolution_backward_overrideablez
(0
(0� 2 @PZb�������
  �� � T
add.2159add�*	
 �2    :
	aten__add	aten__add������ � w

slice.2160slice�*	
 �2    :
xla__selectxla__select���������� � �
	p183.2152	parameter�*	
 �2    :$
xla__device_dataxla__device_dataH����

neff_input_namesinput183� p
logistic.2162logistic�*	
 �2    :*
aten__silu_backwardaten__silu_backward����� � i
constant.2161constant*�:*
aten__silu_backwardaten__silu_backwardB
*�B  �?��� � r
broadcast.2166	broadcast�*	
 �2    :*
aten__silu_backwardaten__silu_backward����� � r
broadcast.2163	broadcast�*	
 �2    :*
aten__silu_backwardaten__silu_backward����� � r
subtract.2164subtract�*	
 �2    :*
aten__silu_backwardaten__silu_backward������ � r
multiply.2165multiply�*	
 �2    :*
aten__silu_backwardaten__silu_backward������ � h
add.2167add�*	
 �2    :*
aten__silu_backwardaten__silu_backward������ � r
multiply.2168multiply�*	
 �2    :*
aten__silu_backwardaten__silu_backward������ � r
multiply.2169multiply�*	
 �2    :*
aten__silu_backwardaten__silu_backward������ � �
batch-norm-grad.2177batch-norm-gradD"�*	
 �2    "�*
 �2 "�*
 �2 :D
 aten__native_batch_norm_backward aten__native_batch_norm_backward�o�:����
����1�� � �
get-tuple-element.2180get-tuple-element�*
 �2 :D
 aten__native_batch_norm_backward aten__native_batch_norm_backwardh����� � �
custom-call.322custom-call�*
 �2 :�
'xla___op_TransferWithStaticRingTransfer'xla___op_TransferWithStaticRingTransferv/home/ubuntu/aws-neuron-samples/aws_neuron_venv_pytorch/lib/python3.10/site-packages/torch_xla/core/xla_op_registry.py ,x�AwsNeuronTransferWithStaticRing��.����
 �� � �
get-tuple-element.2179get-tuple-element�*
 �2 :D
 aten__native_batch_norm_backward aten__native_batch_norm_backwardh����� � �
custom-call.323custom-call�*
 �2 :�
'xla___op_TransferWithStaticRingTransfer'xla___op_TransferWithStaticRingTransferv/home/ubuntu/aws-neuron-samples/aws_neuron_venv_pytorch/lib/python3.10/site-packages/torch_xla/core/xla_op_registry.py ,x�AwsNeuronTransferWithStaticRing��.����
 �� � �
get-tuple-element.2178get-tuple-element�*	
 �2    :D
 aten__native_batch_norm_backward aten__native_batch_norm_backward����� � �
convolution.2199convolution��*	
 �2    :R
'aten__convolution_backward_overrideable'aten__convolution_backward_overrideablez

(0

(0� 28HPZb ����
���
  �� � �
transpose.2200	transpose��*	
 �2    :R
'aten__convolution_backward_overrideable'aten__convolution_backward_overrideabler ����� � �
custom-call.324custom-call��*	
 �2    :�
'xla___op_TransferWithStaticRingTransfer'xla___op_TransferWithStaticRingTransferv/home/ubuntu/aws-neuron-samples/aws_neuron_venv_pytorch/lib/python3.10/site-packages/torch_xla/core/xla_op_registry.py ,x�AwsNeuronTransferWithStaticRing��.����
 �� � �
	p188.2217	parameter�((*	
 �2    :$
xla__device_dataxla__device_dataH����

neff_input_namesinput188� z
	p187.2216	parameter�*
 �2 :$
xla__device_dataxla__device_dataH����

neff_input_namesinput187� z
	p186.2215	parameter�*
 �2 :$
xla__device_dataxla__device_dataH����

neff_input_namesinput186� �
constant.2236constant*�:D
 aten__native_batch_norm_backward aten__native_batch_norm_backwardB
*�B  �?��� � �
broadcast.2237	broadcast�*
 �2 :D
 aten__native_batch_norm_backward aten__native_batch_norm_backward����� � z
	p185.2214	parameter�*
 �2 :$
xla__device_dataxla__device_dataH����

neff_input_namesinput185� 
divide.2238divide�*
 �2 :D
 aten__native_batch_norm_backward aten__native_batch_norm_backward������ � �
multiply.2239multiply�*
 �2 :D
 aten__native_batch_norm_backward aten__native_batch_norm_backward������ � =
constant.61constant*�: B
*�Bo����6� � �
broadcast.115	broadcast�*
 �2 :D
 aten__native_batch_norm_backward aten__native_batch_norm_backward��2��6� � w
add.24add�*
 �2 :D
 aten__native_batch_norm_backward aten__native_batch_norm_backward��2���2� � �
	p102.1342	parameter��*	
 �2    :$
xla__device_dataxla__device_dataHf��
�

neff_input_namesinput102� �
transpose.1344	transpose��*	
 �2    :R
'aten__convolution_backward_overrideable'aten__convolution_backward_overrideabler ��
��
� � �
convolution.1346convolution�((*	
 �2    :R
'aten__convolution_backward_overrideable'aten__convolution_backward_overrideablez
(0
(0� 2 @PZb��
��
�
��
  �� � �
	p184.2195	parameter��*	
 �2    :$
xla__device_dataxla__device_dataH����

neff_input_namesinput184� �
transpose.2196	transpose��*	
 �2    :R
'aten__convolution_backward_overrideable'aten__convolution_backward_overrideabler ����� � �
reverse.2197reverse��*	
 �2    :R
'aten__convolution_backward_overrideable'aten__convolution_backward_overrideabler ����� � �
convolution.2198convolution�((*	
 �2    :R
'aten__convolution_backward_overrideable'aten__convolution_backward_overrideablez
 (0
 (0� 2 @PZb�������
  �� � T
add.2225add�((*	
 �2    :
	aten__add	aten__add����
�� � �
	p189.2218	parameter�((*	
 �2    :$
xla__device_dataxla__device_dataH����

neff_input_namesinput189� p
logistic.2227logistic�((*	
 �2    :*
aten__silu_backwardaten__silu_backward����� � i
constant.2226constant*�:*
aten__silu_backwardaten__silu_backwardB
*�B  �?��� � r
broadcast.2231	broadcast�((*	
 �2    :*
aten__silu_backwardaten__silu_backward����� � r
broadcast.2228	broadcast�((*	
 �2    :*
aten__silu_backwardaten__silu_backward����� � r
subtract.2229subtract�((*	
 �2    :*
aten__silu_backwardaten__silu_backward������ � r
multiply.2230multiply�((*	
 �2    :*
aten__silu_backwardaten__silu_backward������ � h
add.2232add�((*	
 �2    :*
aten__silu_backwardaten__silu_backward������ � r
multiply.2233multiply�((*	
 �2    :*
aten__silu_backwardaten__silu_backward������ � r
multiply.2234multiply�((*	
 �2    :*
aten__silu_backwardaten__silu_backward������ � �
batch-norm-grad.2242batch-norm-gradD"�((*	
 �2    "�*
 �2 "�*
 �2 :D
 aten__native_batch_norm_backward aten__native_batch_norm_backward�o�:����
����2�� � �
get-tuple-element.2245get-tuple-element�*
 �2 :D
 aten__native_batch_norm_backward aten__native_batch_norm_backwardh����� � �
custom-call.325custom-call�*
 �2 :�
'xla___op_TransferWithStaticRingTransfer'xla___op_TransferWithStaticRingTransferv/home/ubuntu/aws-neuron-samples/aws_neuron_venv_pytorch/lib/python3.10/site-packages/torch_xla/core/xla_op_registry.py ,x�AwsNeuronTransferWithStaticRing��.����
 �� � �
get-tuple-element.2244get-tuple-element�*
 �2 :D
 aten__native_batch_norm_backward aten__native_batch_norm_backwardh����� � �
custom-call.326custom-call�*
 �2 :�
'xla___op_TransferWithStaticRingTransfer'xla___op_TransferWithStaticRingTransferv/home/ubuntu/aws-neuron-samples/aws_neuron_venv_pytorch/lib/python3.10/site-packages/torch_xla/core/xla_op_registry.py ,x�AwsNeuronTransferWithStaticRing��.����
 �� � �
	p191.2261	parameter�((*	
 �2    :$
xla__device_dataxla__device_dataH����

neff_input_namesinput191� �
get-tuple-element.2243get-tuple-element�((*	
 �2    :D
 aten__native_batch_norm_backward aten__native_batch_norm_backward����� � �
convolution.2265convolution��*	
 �2    :R
'aten__convolution_backward_overrideable'aten__convolution_backward_overrideablez
((0
((0� 28HPZb �������
  �� � �
transpose.2266	transpose��*	
 �2    :R
'aten__convolution_backward_overrideable'aten__convolution_backward_overrideabler ����� � �
custom-call.327custom-call��*	
 �2    :�
'xla___op_TransferWithStaticRingTransfer'xla___op_TransferWithStaticRingTransferv/home/ubuntu/aws-neuron-samples/aws_neuron_venv_pytorch/lib/python3.10/site-packages/torch_xla/core/xla_op_registry.py ,x�AwsNeuronTransferWithStaticRing��.����
 �� � �
	p195.2283	parameter�((*	
 �2    :$
xla__device_dataxla__device_dataH����

neff_input_namesinput195� z
	p194.2282	parameter�*
 �2 :$
xla__device_dataxla__device_dataH����

neff_input_namesinput194� z
	p193.2281	parameter�*
 �2 :$
xla__device_dataxla__device_dataH����

neff_input_namesinput193� �
constant.2296constant*�:D
 aten__native_batch_norm_backward aten__native_batch_norm_backwardB
*�B  �?��� � �
broadcast.2297	broadcast�*
 �2 :D
 aten__native_batch_norm_backward aten__native_batch_norm_backward����� � z
	p192.2280	parameter�*
 �2 :$
xla__device_dataxla__device_dataH����

neff_input_namesinput192� 
divide.2298divide�*
 �2 :D
 aten__native_batch_norm_backward aten__native_batch_norm_backward������ � �
multiply.2299multiply�*
 �2 :D
 aten__native_batch_norm_backward aten__native_batch_norm_backward������ � =
constant.62constant*�: B
*�Bo����6� � �
broadcast.118	broadcast�*
 �2 :D
 aten__native_batch_norm_backward aten__native_batch_norm_backward��2��6� � w
add.25add�*
 �2 :D
 aten__native_batch_norm_backward aten__native_batch_norm_backward��2���2� � �
	p190.2260	parameter��*	
 �2    :$
xla__device_dataxla__device_dataH����

neff_input_namesinput190� �
transpose.2262	transpose��*	
 �2    :R
'aten__convolution_backward_overrideable'aten__convolution_backward_overrideabler ����� � �
convolution.2264convolution�((*	
 �2    :R
'aten__convolution_backward_overrideable'aten__convolution_backward_overrideablez
(0
(0� 2 @PZb�������
  �� � w

slice.2285slice�((*	
 �2    :
xla__selectxla__select����(�(����� � �
	p196.2284	parameter�((*	
 �2    :$
xla__device_dataxla__device_dataH����

neff_input_namesinput196� p
logistic.2287logistic�((*	
 �2    :*
aten__silu_backwardaten__silu_backward����� � i
constant.2286constant*�:*
aten__silu_backwardaten__silu_backwardB
*�B  �?��� � r
broadcast.2291	broadcast�((*	
 �2    :*
aten__silu_backwardaten__silu_backward����� � r
broadcast.2288	broadcast�((*	
 �2    :*
aten__silu_backwardaten__silu_backward����� � r
subtract.2289subtract�((*	
 �2    :*
aten__silu_backwardaten__silu_backward������ � r
multiply.2290multiply�((*	
 �2    :*
aten__silu_backwardaten__silu_backward������ � h
add.2292add�((*	
 �2    :*
aten__silu_backwardaten__silu_backward������ � r
multiply.2293multiply�((*	
 �2    :*
aten__silu_backwardaten__silu_backward������ � r
multiply.2294multiply�((*	
 �2    :*
aten__silu_backwardaten__silu_backward������ � �
batch-norm-grad.2302batch-norm-gradD"�((*	
 �2    "�*
 �2 "�*
 �2 :D
 aten__native_batch_norm_backward aten__native_batch_norm_backward�o�:����
����2�� � �
get-tuple-element.2305get-tuple-element�*
 �2 :D
 aten__native_batch_norm_backward aten__native_batch_norm_backwardh����� � �
custom-call.328custom-call�*
 �2 :�
'xla___op_TransferWithStaticRingTransfer'xla___op_TransferWithStaticRingTransferv/home/ubuntu/aws-neuron-samples/aws_neuron_venv_pytorch/lib/python3.10/site-packages/torch_xla/core/xla_op_registry.py ,x�AwsNeuronTransferWithStaticRing��.����
 �� � �
get-tuple-element.2304get-tuple-element�*
 �2 :D
 aten__native_batch_norm_backward aten__native_batch_norm_backwardh����� � �
custom-call.329custom-call�*
 �2 :�
'xla___op_TransferWithStaticRingTransfer'xla___op_TransferWithStaticRingTransferv/home/ubuntu/aws-neuron-samples/aws_neuron_venv_pytorch/lib/python3.10/site-packages/torch_xla/core/xla_op_registry.py ,x�AwsNeuronTransferWithStaticRing��.����
 �� � �
	p198.2321	parameter�((*	
 �2    :$
xla__device_dataxla__device_dataH����

neff_input_namesinput198� �
get-tuple-element.2303get-tuple-element�((*	
 �2    :D
 aten__native_batch_norm_backward aten__native_batch_norm_backward����� � �
convolution.2325convolution��*	
 �2    :R
'aten__convolution_backward_overrideable'aten__convolution_backward_overrideablez
( (0
( (0� 28HPZb �������
  �� � �
transpose.2326	transpose��*	
 �2    :R
'aten__convolution_backward_overrideable'aten__convolution_backward_overrideabler ����� � �
custom-call.330custom-call��*	
 �2    :�
'xla___op_TransferWithStaticRingTransfer'xla___op_TransferWithStaticRingTransferv/home/ubuntu/aws-neuron-samples/aws_neuron_venv_pytorch/lib/python3.10/site-packages/torch_xla/core/xla_op_registry.py ,x�AwsNeuronTransferWithStaticRing��.����
 �� � �
	p202.2343	parameter�((*	
 �2    :$
xla__device_dataxla__device_dataH����

neff_input_namesinput202� z
	p201.2342	parameter�*
 �2 :$
xla__device_dataxla__device_dataH����

neff_input_namesinput201� z
	p200.2341	parameter�*
 �2 :$
xla__device_dataxla__device_dataH����

neff_input_namesinput200� �
constant.2355constant*�:D
 aten__native_batch_norm_backward aten__native_batch_norm_backwardB
*�B  �?��� � �
broadcast.2356	broadcast�*
 �2 :D
 aten__native_batch_norm_backward aten__native_batch_norm_backward����� � z
	p199.2340	parameter�*
 �2 :$
xla__device_dataxla__device_dataH����

neff_input_namesinput199� 
divide.2357divide�*
 �2 :D
 aten__native_batch_norm_backward aten__native_batch_norm_backward������ � �
multiply.2358multiply�*
 �2 :D
 aten__native_batch_norm_backward aten__native_batch_norm_backward������ � =
constant.64constant*�: B
*�Bo����6� � �
broadcast.120	broadcast�*
 �2 :D
 aten__native_batch_norm_backward aten__native_batch_norm_backward��2��6� � w
add.26add�*
 �2 :D
 aten__native_batch_norm_backward aten__native_batch_norm_backward��2���2� � �
	p197.2320	parameter��*	
 �2    :$
xla__device_dataxla__device_dataH����

neff_input_namesinput197� �
transpose.2322	transpose��*	
 �2    :R
'aten__convolution_backward_overrideable'aten__convolution_backward_overrideabler ����� � �
reverse.2323reverse��*	
 �2    :R
'aten__convolution_backward_overrideable'aten__convolution_backward_overrideabler ����� � �
convolution.2324convolution�((*	
 �2    :R
'aten__convolution_backward_overrideable'aten__convolution_backward_overrideablez
 (0
 (0� 2 @PZb�������
  �� � �
	p203.2344	parameter�((*	
 �2    :$
xla__device_dataxla__device_dataH����

neff_input_namesinput203� p
logistic.2346logistic�((*	
 �2    :*
aten__silu_backwardaten__silu_backward����� � i
constant.2345constant*�:*
aten__silu_backwardaten__silu_backwardB
*�B  �?��� � r
broadcast.2350	broadcast�((*	
 �2    :*
aten__silu_backwardaten__silu_backward����� � r
broadcast.2347	broadcast�((*	
 �2    :*
aten__silu_backwardaten__silu_backward����� � r
subtract.2348subtract�((*	
 �2    :*
aten__silu_backwardaten__silu_backward������ � r
multiply.2349multiply�((*	
 �2    :*
aten__silu_backwardaten__silu_backward������ � h
add.2351add�((*	
 �2    :*
aten__silu_backwardaten__silu_backward������ � r
multiply.2352multiply�((*	
 �2    :*
aten__silu_backwardaten__silu_backward������ � r
multiply.2353multiply�((*	
 �2    :*
aten__silu_backwardaten__silu_backward������ � �
batch-norm-grad.2361batch-norm-gradD"�((*	
 �2    "�*
 �2 "�*
 �2 :D
 aten__native_batch_norm_backward aten__native_batch_norm_backward�o�:����
����2�� � �
get-tuple-element.2364get-tuple-element�*
 �2 :D
 aten__native_batch_norm_backward aten__native_batch_norm_backwardh����� � �
custom-call.331custom-call�*
 �2 :�
'xla___op_TransferWithStaticRingTransfer'xla___op_TransferWithStaticRingTransferv/home/ubuntu/aws-neuron-samples/aws_neuron_venv_pytorch/lib/python3.10/site-packages/torch_xla/core/xla_op_registry.py ,x�AwsNeuronTransferWithStaticRing��.����
 �� � �
get-tuple-element.2363get-tuple-element�*
 �2 :D
 aten__native_batch_norm_backward aten__native_batch_norm_backwardh����� � �
custom-call.332custom-call�*
 �2 :�
'xla___op_TransferWithStaticRingTransfer'xla___op_TransferWithStaticRingTransferv/home/ubuntu/aws-neuron-samples/aws_neuron_venv_pytorch/lib/python3.10/site-packages/torch_xla/core/xla_op_registry.py ,x�AwsNeuronTransferWithStaticRing��.����
 �� � �
	p205.2380	parameter�((*	
 �2    :$
xla__device_dataxla__device_dataH����

neff_input_namesinput205� �
get-tuple-element.2362get-tuple-element�((*	
 �2    :D
 aten__native_batch_norm_backward aten__native_batch_norm_backward����� � �
convolution.2384convolution��*	
 �2    :R
'aten__convolution_backward_overrideable'aten__convolution_backward_overrideablez
((0
((0� 28HPZb �������
  �� � �
transpose.2385	transpose��*	
 �2    :R
'aten__convolution_backward_overrideable'aten__convolution_backward_overrideabler ����� � �
custom-call.333custom-call��*	
 �2    :�
'xla___op_TransferWithStaticRingTransfer'xla___op_TransferWithStaticRingTransferv/home/ubuntu/aws-neuron-samples/aws_neuron_venv_pytorch/lib/python3.10/site-packages/torch_xla/core/xla_op_registry.py ,x�AwsNeuronTransferWithStaticRing��.����
 �� � �
	p209.2402	parameter�((*	
 �2    :$
xla__device_dataxla__device_dataH����

neff_input_namesinput209� z
	p208.2401	parameter�*
 �2 :$
xla__device_dataxla__device_dataH����

neff_input_namesinput208� z
	p207.2400	parameter�*
 �2 :$
xla__device_dataxla__device_dataH����

neff_input_namesinput207� �
constant.2415constant*�:D
 aten__native_batch_norm_backward aten__native_batch_norm_backwardB
*�B  �?��� � �
broadcast.2416	broadcast�*
 �2 :D
 aten__native_batch_norm_backward aten__native_batch_norm_backward����� � z
	p206.2399	parameter�*
 �2 :$
xla__device_dataxla__device_dataH����

neff_input_namesinput206� 
divide.2417divide�*
 �2 :D
 aten__native_batch_norm_backward aten__native_batch_norm_backward������ � �
multiply.2418multiply�*
 �2 :D
 aten__native_batch_norm_backward aten__native_batch_norm_backward������ � =
constant.65constant*�: B
*�Bo����6� � �
broadcast.122	broadcast�*
 �2 :D
 aten__native_batch_norm_backward aten__native_batch_norm_backward��2��6� � w
add.27add�*
 �2 :D
 aten__native_batch_norm_backward aten__native_batch_norm_backward��2���2� � z

slice.2404slice�((*	
 �2    :
xla__selectxla__select�����(�(����� � �
	p210.2403	parameter�((*	
 �2    :$
xla__device_dataxla__device_dataH����

neff_input_namesinput210� p
logistic.2406logistic�((*	
 �2    :*
aten__silu_backwardaten__silu_backward����� � i
constant.2405constant*�:*
aten__silu_backwardaten__silu_backwardB
*�B  �?��� � r
broadcast.2410	broadcast�((*	
 �2    :*
aten__silu_backwardaten__silu_backward����� � r
broadcast.2407	broadcast�((*	
 �2    :*
aten__silu_backwardaten__silu_backward����� � r
subtract.2408subtract�((*	
 �2    :*
aten__silu_backwardaten__silu_backward������ � r
multiply.2409multiply�((*	
 �2    :*
aten__silu_backwardaten__silu_backward������ � h
add.2411add�((*	
 �2    :*
aten__silu_backwardaten__silu_backward������ � r
multiply.2412multiply�((*	
 �2    :*
aten__silu_backwardaten__silu_backward������ � r
multiply.2413multiply�((*	
 �2    :*
aten__silu_backwardaten__silu_backward������ � �
batch-norm-grad.2421batch-norm-gradD"�((*	
 �2    "�*
 �2 "�*
 �2 :D
 aten__native_batch_norm_backward aten__native_batch_norm_backward�o�:����
����2�� � �
get-tuple-element.2424get-tuple-element�*
 �2 :D
 aten__native_batch_norm_backward aten__native_batch_norm_backwardh����� � �
custom-call.334custom-call�*
 �2 :�
'xla___op_TransferWithStaticRingTransfer'xla___op_TransferWithStaticRingTransferv/home/ubuntu/aws-neuron-samples/aws_neuron_venv_pytorch/lib/python3.10/site-packages/torch_xla/core/xla_op_registry.py ,x�AwsNeuronTransferWithStaticRing��.����
 �� � �
get-tuple-element.2423get-tuple-element�*
 �2 :D
 aten__native_batch_norm_backward aten__native_batch_norm_backwardh����� � �
custom-call.335custom-call�*
 �2 :�
'xla___op_TransferWithStaticRingTransfer'xla___op_TransferWithStaticRingTransferv/home/ubuntu/aws-neuron-samples/aws_neuron_venv_pytorch/lib/python3.10/site-packages/torch_xla/core/xla_op_registry.py ,x�AwsNeuronTransferWithStaticRing��.����
 �� � �
	p212.2440	parameter�((*	
 �2    :$
xla__device_dataxla__device_dataH����

neff_input_namesinput212� �
get-tuple-element.2422get-tuple-element�((*	
 �2    :D
 aten__native_batch_norm_backward aten__native_batch_norm_backward����� � �
convolution.2444convolution��*	
 �2    :R
'aten__convolution_backward_overrideable'aten__convolution_backward_overrideablez
((0
((0� 28HPZb �������
  �� � �
transpose.2445	transpose��*	
 �2    :R
'aten__convolution_backward_overrideable'aten__convolution_backward_overrideabler ����� � �
custom-call.336custom-call��*	
 �2    :�
'xla___op_TransferWithStaticRingTransfer'xla___op_TransferWithStaticRingTransferv/home/ubuntu/aws-neuron-samples/aws_neuron_venv_pytorch/lib/python3.10/site-packages/torch_xla/core/xla_op_registry.py ,x�AwsNeuronTransferWithStaticRing��.����
 �� � �
	p216.2462	parameter�((*	
 �2    :$
xla__device_dataxla__device_dataH����

neff_input_namesinput216� z
	p215.2461	parameter�*
 �2 :$
xla__device_dataxla__device_dataH����

neff_input_namesinput215� z
	p214.2460	parameter�*
 �2 :$
xla__device_dataxla__device_dataH����

neff_input_namesinput214� �
constant.2474constant*�:D
 aten__native_batch_norm_backward aten__native_batch_norm_backwardB
*�B  �?��� � �
broadcast.2475	broadcast�*
 �2 :D
 aten__native_batch_norm_backward aten__native_batch_norm_backward����� � z
	p213.2459	parameter�*
 �2 :$
xla__device_dataxla__device_dataH����

neff_input_namesinput213� 
divide.2476divide�*
 �2 :D
 aten__native_batch_norm_backward aten__native_batch_norm_backward������ � �
multiply.2477multiply�*
 �2 :D
 aten__native_batch_norm_backward aten__native_batch_norm_backward������ � =
constant.66constant*�: B
*�Bo����6� � �
broadcast.123	broadcast�*
 �2 :D
 aten__native_batch_norm_backward aten__native_batch_norm_backward��2��6� � w
add.28add�*
 �2 :D
 aten__native_batch_norm_backward aten__native_batch_norm_backward��2���2� � �
	p204.2379	parameter��*	
 �2    :$
xla__device_dataxla__device_dataH����

neff_input_namesinput204� �
transpose.2381	transpose��*	
 �2    :R
'aten__convolution_backward_overrideable'aten__convolution_backward_overrideabler ����� � �
convolution.2383convolution�((*	
 �2    :R
'aten__convolution_backward_overrideable'aten__convolution_backward_overrideablez
(0
(0� 2 @PZb�������
  �� � �
	p217.2463	parameter�((*	
 �2    :$
xla__device_dataxla__device_dataH����

neff_input_namesinput217� p
logistic.2465logistic�((*	
 �2    :*
aten__silu_backwardaten__silu_backward����� � i
constant.2464constant*�:*
aten__silu_backwardaten__silu_backwardB
*�B  �?��� � r
broadcast.2469	broadcast�((*	
 �2    :*
aten__silu_backwardaten__silu_backward����� � r
broadcast.2466	broadcast�((*	
 �2    :*
aten__silu_backwardaten__silu_backward����� � r
subtract.2467subtract�((*	
 �2    :*
aten__silu_backwardaten__silu_backward������ � r
multiply.2468multiply�((*	
 �2    :*
aten__silu_backwardaten__silu_backward������ � h
add.2470add�((*	
 �2    :*
aten__silu_backwardaten__silu_backward������ � r
multiply.2471multiply�((*	
 �2    :*
aten__silu_backwardaten__silu_backward������ � r
multiply.2472multiply�((*	
 �2    :*
aten__silu_backwardaten__silu_backward������ � �
batch-norm-grad.2480batch-norm-gradD"�((*	
 �2    "�*
 �2 "�*
 �2 :D
 aten__native_batch_norm_backward aten__native_batch_norm_backward�o�:����
����2�� � �
get-tuple-element.2483get-tuple-element�*
 �2 :D
 aten__native_batch_norm_backward aten__native_batch_norm_backwardh����� � �
custom-call.337custom-call�*
 �2 :�
'xla___op_TransferWithStaticRingTransfer'xla___op_TransferWithStaticRingTransferv/home/ubuntu/aws-neuron-samples/aws_neuron_venv_pytorch/lib/python3.10/site-packages/torch_xla/core/xla_op_registry.py ,x�AwsNeuronTransferWithStaticRing��.����
 �� � �
get-tuple-element.2482get-tuple-element�*
 �2 :D
 aten__native_batch_norm_backward aten__native_batch_norm_backwardh����� � �
custom-call.338custom-call�*
 �2 :�
'xla___op_TransferWithStaticRingTransfer'xla___op_TransferWithStaticRingTransferv/home/ubuntu/aws-neuron-samples/aws_neuron_venv_pytorch/lib/python3.10/site-packages/torch_xla/core/xla_op_registry.py ,x�AwsNeuronTransferWithStaticRing��.����
 �� � �
get-tuple-element.2481get-tuple-element�((*	
 �2    :D
 aten__native_batch_norm_backward aten__native_batch_norm_backward����� � �
convolution.2502convolution��*	
 �2    :R
'aten__convolution_backward_overrideable'aten__convolution_backward_overrideablez
((0
((0� 28HPZb �������
  �� � �
transpose.2503	transpose��*	
 �2    :R
'aten__convolution_backward_overrideable'aten__convolution_backward_overrideabler ����� � �
custom-call.339custom-call��*	
 �2    :�
'xla___op_TransferWithStaticRingTransfer'xla___op_TransferWithStaticRingTransferv/home/ubuntu/aws-neuron-samples/aws_neuron_venv_pytorch/lib/python3.10/site-packages/torch_xla/core/xla_op_registry.py ,x�AwsNeuronTransferWithStaticRing��.����
 �� � �
	p222.2520	parameter�((*	
 �2    :$
xla__device_dataxla__device_dataH����

neff_input_namesinput222� z
	p221.2519	parameter�*
 �2 :$
xla__device_dataxla__device_dataH����

neff_input_namesinput221� z
	p220.2518	parameter�*
 �2 :$
xla__device_dataxla__device_dataH����

neff_input_namesinput220� �
constant.2540constant*�:D
 aten__native_batch_norm_backward aten__native_batch_norm_backwardB
*�B  �?��� � �
broadcast.2541	broadcast�*
 �2 :D
 aten__native_batch_norm_backward aten__native_batch_norm_backward����� � z
	p219.2517	parameter�*
 �2 :$
xla__device_dataxla__device_dataH����

neff_input_namesinput219� 
divide.2542divide�*
 �2 :D
 aten__native_batch_norm_backward aten__native_batch_norm_backward������ � �
multiply.2543multiply�*
 �2 :D
 aten__native_batch_norm_backward aten__native_batch_norm_backward������ � =
constant.67constant*�: B
*�Bo����6� � �
broadcast.124	broadcast�*
 �2 :D
 aten__native_batch_norm_backward aten__native_batch_norm_backward��2��6� � w
add.29add�*
 �2 :D
 aten__native_batch_norm_backward aten__native_batch_norm_backward��2���2� � �
	p211.2439	parameter��*	
 �2    :$
xla__device_dataxla__device_dataH����

neff_input_namesinput211� �
transpose.2441	transpose��*	
 �2    :R
'aten__convolution_backward_overrideable'aten__convolution_backward_overrideabler ����� � �
convolution.2443convolution�((*	
 �2    :R
'aten__convolution_backward_overrideable'aten__convolution_backward_overrideablez
(0
(0� 2 @PZb�������
  �� � �
	p218.2498	parameter��*	
 �2    :$
xla__device_dataxla__device_dataH����

neff_input_namesinput218� �
transpose.2499	transpose��*	
 �2    :R
'aten__convolution_backward_overrideable'aten__convolution_backward_overrideabler ����� � �
convolution.2501convolution�((*	
 �2    :R
'aten__convolution_backward_overrideable'aten__convolution_backward_overrideablez
(0
(0� 2 @PZb�������
  �� � T
add.2528add�((*	
 �2    :
	aten__add	aten__add������ � w

slice.2529slice�((*	
 �2    :
xla__selectxla__select����(�(����� � �
	p223.2521	parameter�((*	
 �2    :$
xla__device_dataxla__device_dataH����

neff_input_namesinput223� p
logistic.2531logistic�((*	
 �2    :*
aten__silu_backwardaten__silu_backward����� � i
constant.2530constant*�:*
aten__silu_backwardaten__silu_backwardB
*�B  �?��� � r
broadcast.2535	broadcast�((*	
 �2    :*
aten__silu_backwardaten__silu_backward����� � r
broadcast.2532	broadcast�((*	
 �2    :*
aten__silu_backwardaten__silu_backward����� � r
subtract.2533subtract�((*	
 �2    :*
aten__silu_backwardaten__silu_backward������ � r
multiply.2534multiply�((*	
 �2    :*
aten__silu_backwardaten__silu_backward������ � h
add.2536add�((*	
 �2    :*
aten__silu_backwardaten__silu_backward������ � r
multiply.2537multiply�((*	
 �2    :*
aten__silu_backwardaten__silu_backward������ � r
multiply.2538multiply�((*	
 �2    :*
aten__silu_backwardaten__silu_backward������ � �
batch-norm-grad.2546batch-norm-gradD"�((*	
 �2    "�*
 �2 "�*
 �2 :D
 aten__native_batch_norm_backward aten__native_batch_norm_backward�o�:����
����2�� � �
get-tuple-element.2549get-tuple-element�*
 �2 :D
 aten__native_batch_norm_backward aten__native_batch_norm_backwardh����� � �
custom-call.340custom-call�*
 �2 :�
'xla___op_TransferWithStaticRingTransfer'xla___op_TransferWithStaticRingTransferv/home/ubuntu/aws-neuron-samples/aws_neuron_venv_pytorch/lib/python3.10/site-packages/torch_xla/core/xla_op_registry.py ,x�AwsNeuronTransferWithStaticRing��.����
 �� � �
get-tuple-element.2548get-tuple-element�*
 �2 :D
 aten__native_batch_norm_backward aten__native_batch_norm_backwardh����� � �
custom-call.341custom-call�*
 �2 :�
'xla___op_TransferWithStaticRingTransfer'xla___op_TransferWithStaticRingTransferv/home/ubuntu/aws-neuron-samples/aws_neuron_venv_pytorch/lib/python3.10/site-packages/torch_xla/core/xla_op_registry.py ,x�AwsNeuronTransferWithStaticRing��.����
 �� � �
get-tuple-element.2547get-tuple-element�((*	
 �2    :D
 aten__native_batch_norm_backward aten__native_batch_norm_backward����� � �
convolution.2568convolution��*	
 �2    :R
'aten__convolution_backward_overrideable'aten__convolution_backward_overrideablez

((0

((0� 28HPZb �������
  �� � �
transpose.2569	transpose��*	
 �2    :R
'aten__convolution_backward_overrideable'aten__convolution_backward_overrideabler ����� � �
custom-call.342custom-call��*	
 �2    :�
'xla___op_TransferWithStaticRingTransfer'xla___op_TransferWithStaticRingTransferv/home/ubuntu/aws-neuron-samples/aws_neuron_venv_pytorch/lib/python3.10/site-packages/torch_xla/core/xla_op_registry.py ,x�AwsNeuronTransferWithStaticRing��.����
 �� � �
	p228.2586	parameter�PP*	
 �2    :$
xla__device_dataxla__device_dataH����

neff_input_namesinput228� z
	p227.2585	parameter�*
 �2 :$
xla__device_dataxla__device_dataH����

neff_input_namesinput227� z
	p226.2584	parameter�*
 �2 :$
xla__device_dataxla__device_dataH����

neff_input_namesinput226� �
constant.2605constant*�:D
 aten__native_batch_norm_backward aten__native_batch_norm_backwardB
*�B  �?��� � �
broadcast.2606	broadcast�*
 �2 :D
 aten__native_batch_norm_backward aten__native_batch_norm_backward����� � z
	p225.2583	parameter�*
 �2 :$
xla__device_dataxla__device_dataH����

neff_input_namesinput225� 
divide.2607divide�*
 �2 :D
 aten__native_batch_norm_backward aten__native_batch_norm_backward������ � �
multiply.2608multiply�*
 �2 :D
 aten__native_batch_norm_backward aten__native_batch_norm_backward������ � =
constant.68constant*�: B
*�Bo����6� � �
broadcast.128	broadcast�*
 �2 :D
 aten__native_batch_norm_backward aten__native_batch_norm_backward��2��6� � w
add.30add�*
 �2 :D
 aten__native_batch_norm_backward aten__native_batch_norm_backward��2���2� � �
	p143.1832	parameter��*	
 �2    :$
xla__device_dataxla__device_dataH����

neff_input_namesinput143� �
transpose.1834	transpose��*	
 �2    :R
'aten__convolution_backward_overrideable'aten__convolution_backward_overrideabler ����� � �
convolution.1836convolution�PP*	
 �2    :R
'aten__convolution_backward_overrideable'aten__convolution_backward_overrideablez
(0
(0� 2 @PZb�������
  �� � �
	p224.2564	parameter��*	
 �2    :$
xla__device_dataxla__device_dataH����

neff_input_namesinput224� �
transpose.2565	transpose��*	
 �2    :R
'aten__convolution_backward_overrideable'aten__convolution_backward_overrideabler ����� � �
reverse.2566reverse��*	
 �2    :R
'aten__convolution_backward_overrideable'aten__convolution_backward_overrideabler ����� � �
convolution.2567convolution�PP*	
 �2    :R
'aten__convolution_backward_overrideable'aten__convolution_backward_overrideablez
 (0
 (0� 2 @PZb�������
  �� � T
add.2594add�PP*	
 �2    :
	aten__add	aten__add������ � �
	p229.2587	parameter�PP*	
 �2    :$
xla__device_dataxla__device_dataH����

neff_input_namesinput229� p
logistic.2596logistic�PP*	
 �2    :*
aten__silu_backwardaten__silu_backward����� � i
constant.2595constant*�:*
aten__silu_backwardaten__silu_backwardB
*�B  �?��� � r
broadcast.2600	broadcast�PP*	
 �2    :*
aten__silu_backwardaten__silu_backward����� � r
broadcast.2597	broadcast�PP*	
 �2    :*
aten__silu_backwardaten__silu_backward����� � r
subtract.2598subtract�PP*	
 �2    :*
aten__silu_backwardaten__silu_backward������ � r
multiply.2599multiply�PP*	
 �2    :*
aten__silu_backwardaten__silu_backward������ � h
add.2601add�PP*	
 �2    :*
aten__silu_backwardaten__silu_backward������ � r
multiply.2602multiply�PP*	
 �2    :*
aten__silu_backwardaten__silu_backward������ � r
multiply.2603multiply�PP*	
 �2    :*
aten__silu_backwardaten__silu_backward������ � �
batch-norm-grad.2611batch-norm-gradD"�PP*	
 �2    "�*
 �2 "�*
 �2 :D
 aten__native_batch_norm_backward aten__native_batch_norm_backward�o�:����
����2�� � �
get-tuple-element.2614get-tuple-element�*
 �2 :D
 aten__native_batch_norm_backward aten__native_batch_norm_backwardh����� � �
custom-call.343custom-call�*
 �2 :�
'xla___op_TransferWithStaticRingTransfer'xla___op_TransferWithStaticRingTransferv/home/ubuntu/aws-neuron-samples/aws_neuron_venv_pytorch/lib/python3.10/site-packages/torch_xla/core/xla_op_registry.py ,x�AwsNeuronTransferWithStaticRing��.����
 �� � �
get-tuple-element.2613get-tuple-element�*
 �2 :D
 aten__native_batch_norm_backward aten__native_batch_norm_backwardh����� � �
custom-call.344custom-call�*
 �2 :�
'xla___op_TransferWithStaticRingTransfer'xla___op_TransferWithStaticRingTransferv/home/ubuntu/aws-neuron-samples/aws_neuron_venv_pytorch/lib/python3.10/site-packages/torch_xla/core/xla_op_registry.py ,x�AwsNeuronTransferWithStaticRing��.����
 �� � �
	p231.2630	parameter�PP*	
 �2    :$
xla__device_dataxla__device_dataH����

neff_input_namesinput231� �
get-tuple-element.2612get-tuple-element�PP*	
 �2    :D
 aten__native_batch_norm_backward aten__native_batch_norm_backward����� � �
convolution.2634convolution��*	
 �2    :R
'aten__convolution_backward_overrideable'aten__convolution_backward_overrideablez
P(0
P(0� 28HPZb �������
  �� � �
transpose.2635	transpose��*	
 �2    :R
'aten__convolution_backward_overrideable'aten__convolution_backward_overrideabler ����� � �
custom-call.345custom-call��*	
 �2    :�
'xla___op_TransferWithStaticRingTransfer'xla___op_TransferWithStaticRingTransferv/home/ubuntu/aws-neuron-samples/aws_neuron_venv_pytorch/lib/python3.10/site-packages/torch_xla/core/xla_op_registry.py ,x�AwsNeuronTransferWithStaticRing��.����
 �� � �
	p235.2652	parameter@PP*	
 �2    :$
xla__device_dataxla__device_dataH����

neff_input_namesinput235� y
	p234.2651	parameter@*
 �2 :$
xla__device_dataxla__device_dataH����

neff_input_namesinput234� y
	p233.2650	parameter@*
 �2 :$
xla__device_dataxla__device_dataH����

neff_input_namesinput233� �
constant.2665constant*�:D
 aten__native_batch_norm_backward aten__native_batch_norm_backwardB
*�B  �?��� � �
broadcast.2666	broadcast@*
 �2 :D
 aten__native_batch_norm_backward aten__native_batch_norm_backward����� � y
	p232.2649	parameter@*
 �2 :$
xla__device_dataxla__device_dataH����

neff_input_namesinput232� ~
divide.2667divide@*
 �2 :D
 aten__native_batch_norm_backward aten__native_batch_norm_backward������ � �
multiply.2668multiply@*
 �2 :D
 aten__native_batch_norm_backward aten__native_batch_norm_backward������ � =
constant.69constant*�: B
*�Bo����6� � �
broadcast.131	broadcast@*
 �2 :D
 aten__native_batch_norm_backward aten__native_batch_norm_backward��2��6� � v
add.31add@*
 �2 :D
 aten__native_batch_norm_backward aten__native_batch_norm_backward��2���2� � �
	p230.2629	parameter��*	
 �2    :$
xla__device_dataxla__device_dataH����

neff_input_namesinput230� �
transpose.2631	transpose��*	
 �2    :R
'aten__convolution_backward_overrideable'aten__convolution_backward_overrideabler ����� � �
convolution.2633convolution�PP*	
 �2    :R
'aten__convolution_backward_overrideable'aten__convolution_backward_overrideablez
(0
(0� 2 @PZb�������
  �� � u

slice.2654slice@PP*	
 �2    :
xla__selectxla__select��@�P�P����� � �
	p236.2653	parameter@PP*	
 �2    :$
xla__device_dataxla__device_dataH����

neff_input_namesinput236� o
logistic.2656logistic@PP*	
 �2    :*
aten__silu_backwardaten__silu_backward����� � i
constant.2655constant*�:*
aten__silu_backwardaten__silu_backwardB
*�B  �?��� � q
broadcast.2660	broadcast@PP*	
 �2    :*
aten__silu_backwardaten__silu_backward����� � q
broadcast.2657	broadcast@PP*	
 �2    :*
aten__silu_backwardaten__silu_backward����� � q
subtract.2658subtract@PP*	
 �2    :*
aten__silu_backwardaten__silu_backward������ � q
multiply.2659multiply@PP*	
 �2    :*
aten__silu_backwardaten__silu_backward������ � g
add.2661add@PP*	
 �2    :*
aten__silu_backwardaten__silu_backward������ � q
multiply.2662multiply@PP*	
 �2    :*
aten__silu_backwardaten__silu_backward������ � q
multiply.2663multiply@PP*	
 �2    :*
aten__silu_backwardaten__silu_backward������ � �
batch-norm-grad.2671batch-norm-gradA"@PP*	
 �2    "@*
 �2 "@*
 �2 :D
 aten__native_batch_norm_backward aten__native_batch_norm_backward�o�:����
����2�� � �
get-tuple-element.2674get-tuple-element@*
 �2 :D
 aten__native_batch_norm_backward aten__native_batch_norm_backwardh����� � �
custom-call.346custom-call@*
 �2 :�
'xla___op_TransferWithStaticRingTransfer'xla___op_TransferWithStaticRingTransferv/home/ubuntu/aws-neuron-samples/aws_neuron_venv_pytorch/lib/python3.10/site-packages/torch_xla/core/xla_op_registry.py ,x�AwsNeuronTransferWithStaticRing��.����
 �� � �
get-tuple-element.2673get-tuple-element@*
 �2 :D
 aten__native_batch_norm_backward aten__native_batch_norm_backwardh����� � �
custom-call.347custom-call@*
 �2 :�
'xla___op_TransferWithStaticRingTransfer'xla___op_TransferWithStaticRingTransferv/home/ubuntu/aws-neuron-samples/aws_neuron_venv_pytorch/lib/python3.10/site-packages/torch_xla/core/xla_op_registry.py ,x�AwsNeuronTransferWithStaticRing��.����
 �� � �
	p238.2690	parameter@PP*	
 �2    :$
xla__device_dataxla__device_dataH����

neff_input_namesinput238� �
get-tuple-element.2672get-tuple-element@PP*	
 �2    :D
 aten__native_batch_norm_backward aten__native_batch_norm_backward����� � �
convolution.2694convolution@@*	
 �2    :R
'aten__convolution_backward_overrideable'aten__convolution_backward_overrideablez
P (0
P (0� 28HPZb �������
  �� � �
transpose.2695	transpose@@*	
 �2    :R
'aten__convolution_backward_overrideable'aten__convolution_backward_overrideabler ����� � �
custom-call.348custom-call@@*	
 �2    :�
'xla___op_TransferWithStaticRingTransfer'xla___op_TransferWithStaticRingTransferv/home/ubuntu/aws-neuron-samples/aws_neuron_venv_pytorch/lib/python3.10/site-packages/torch_xla/core/xla_op_registry.py ,x�AwsNeuronTransferWithStaticRing��.����
 �� � �
	p242.2712	parameter@PP*	
 �2    :$
xla__device_dataxla__device_dataH����

neff_input_namesinput242� y
	p241.2711	parameter@*
 �2 :$
xla__device_dataxla__device_dataH����

neff_input_namesinput241� y
	p240.2710	parameter@*
 �2 :$
xla__device_dataxla__device_dataH����

neff_input_namesinput240� �
constant.2724constant*�:D
 aten__native_batch_norm_backward aten__native_batch_norm_backwardB
*�B  �?��� � �
broadcast.2725	broadcast@*
 �2 :D
 aten__native_batch_norm_backward aten__native_batch_norm_backward����� � y
	p239.2709	parameter@*
 �2 :$
xla__device_dataxla__device_dataH����

neff_input_namesinput239� ~
divide.2726divide@*
 �2 :D
 aten__native_batch_norm_backward aten__native_batch_norm_backward������ � �
multiply.2727multiply@*
 �2 :D
 aten__native_batch_norm_backward aten__native_batch_norm_backward������ � =
constant.70constant*�: B
*�Bo����6� � �
broadcast.132	broadcast@*
 �2 :D
 aten__native_batch_norm_backward aten__native_batch_norm_backward��2��6� � v
add.32add@*
 �2 :D
 aten__native_batch_norm_backward aten__native_batch_norm_backward��2���2� � �
	p237.2689	parameter@@*	
 �2    :$
xla__device_dataxla__device_dataH����

neff_input_namesinput237� �
transpose.2691	transpose@@*	
 �2    :R
'aten__convolution_backward_overrideable'aten__convolution_backward_overrideabler ����� � �
reverse.2692reverse@@*	
 �2    :R
'aten__convolution_backward_overrideable'aten__convolution_backward_overrideabler ����� � �
convolution.2693convolution@PP*	
 �2    :R
'aten__convolution_backward_overrideable'aten__convolution_backward_overrideablez
 (0
 (0� 2 @PZb�������
  �� � �
	p243.2713	parameter@PP*	
 �2    :$
xla__device_dataxla__device_dataH����

neff_input_namesinput243� o
logistic.2715logistic@PP*	
 �2    :*
aten__silu_backwardaten__silu_backward����� � i
constant.2714constant*�:*
aten__silu_backwardaten__silu_backwardB
*�B  �?��� � q
broadcast.2719	broadcast@PP*	
 �2    :*
aten__silu_backwardaten__silu_backward����� � q
broadcast.2716	broadcast@PP*	
 �2    :*
aten__silu_backwardaten__silu_backward����� � q
subtract.2717subtract@PP*	
 �2    :*
aten__silu_backwardaten__silu_backward������ � q
multiply.2718multiply@PP*	
 �2    :*
aten__silu_backwardaten__silu_backward������ � g
add.2720add@PP*	
 �2    :*
aten__silu_backwardaten__silu_backward������ � q
multiply.2721multiply@PP*	
 �2    :*
aten__silu_backwardaten__silu_backward������ � q
multiply.2722multiply@PP*	
 �2    :*
aten__silu_backwardaten__silu_backward������ � �
batch-norm-grad.2730batch-norm-gradA"@PP*	
 �2    "@*
 �2 "@*
 �2 :D
 aten__native_batch_norm_backward aten__native_batch_norm_backward�o�:����
����2�� � �
get-tuple-element.2733get-tuple-element@*
 �2 :D
 aten__native_batch_norm_backward aten__native_batch_norm_backwardh����� � �
custom-call.349custom-call@*
 �2 :�
'xla___op_TransferWithStaticRingTransfer'xla___op_TransferWithStaticRingTransferv/home/ubuntu/aws-neuron-samples/aws_neuron_venv_pytorch/lib/python3.10/site-packages/torch_xla/core/xla_op_registry.py ,x�AwsNeuronTransferWithStaticRing��.����
 �� � �
get-tuple-element.2732get-tuple-element@*
 �2 :D
 aten__native_batch_norm_backward aten__native_batch_norm_backwardh����� � �
custom-call.350custom-call@*
 �2 :�
'xla___op_TransferWithStaticRingTransfer'xla___op_TransferWithStaticRingTransferv/home/ubuntu/aws-neuron-samples/aws_neuron_venv_pytorch/lib/python3.10/site-packages/torch_xla/core/xla_op_registry.py ,x�AwsNeuronTransferWithStaticRing��.����
 �� � �
	p245.2749	parameter@PP*	
 �2    :$
xla__device_dataxla__device_dataH����

neff_input_namesinput245� �
get-tuple-element.2731get-tuple-element@PP*	
 �2    :D
 aten__native_batch_norm_backward aten__native_batch_norm_backward����� � �
convolution.2753convolution@@*	
 �2    :R
'aten__convolution_backward_overrideable'aten__convolution_backward_overrideablez
P(0
P(0� 28HPZb �������
  �� � �
transpose.2754	transpose@@*	
 �2    :R
'aten__convolution_backward_overrideable'aten__convolution_backward_overrideabler ����� � �
custom-call.351custom-call@@*	
 �2    :�
'xla___op_TransferWithStaticRingTransfer'xla___op_TransferWithStaticRingTransferv/home/ubuntu/aws-neuron-samples/aws_neuron_venv_pytorch/lib/python3.10/site-packages/torch_xla/core/xla_op_registry.py ,x�AwsNeuronTransferWithStaticRing��.����
 �� � �
	p249.2771	parameter@PP*	
 �2    :$
xla__device_dataxla__device_dataH����

neff_input_namesinput249� y
	p248.2770	parameter@*
 �2 :$
xla__device_dataxla__device_dataH����

neff_input_namesinput248� y
	p247.2769	parameter@*
 �2 :$
xla__device_dataxla__device_dataH����

neff_input_namesinput247� �
constant.2784constant*�:D
 aten__native_batch_norm_backward aten__native_batch_norm_backwardB
*�B  �?��� � �
broadcast.2785	broadcast@*
 �2 :D
 aten__native_batch_norm_backward aten__native_batch_norm_backward����� � y
	p246.2768	parameter@*
 �2 :$
xla__device_dataxla__device_dataH����

neff_input_namesinput246� ~
divide.2786divide@*
 �2 :D
 aten__native_batch_norm_backward aten__native_batch_norm_backward������ � �
multiply.2787multiply@*
 �2 :D
 aten__native_batch_norm_backward aten__native_batch_norm_backward������ � =
constant.71constant*�: B
*�Bo����6� � �
broadcast.133	broadcast@*
 �2 :D
 aten__native_batch_norm_backward aten__native_batch_norm_backward��2��6� � v
add.33add@*
 �2 :D
 aten__native_batch_norm_backward aten__native_batch_norm_backward��2���2� � x

slice.2773slice@PP*	
 �2    :
xla__selectxla__select��@��P�P����� � �
	p250.2772	parameter@PP*	
 �2    :$
xla__device_dataxla__device_dataH����

neff_input_namesinput250� o
logistic.2775logistic@PP*	
 �2    :*
aten__silu_backwardaten__silu_backward����� � i
constant.2774constant*�:*
aten__silu_backwardaten__silu_backwardB
*�B  �?��� � q
broadcast.2779	broadcast@PP*	
 �2    :*
aten__silu_backwardaten__silu_backward����� � q
broadcast.2776	broadcast@PP*	
 �2    :*
aten__silu_backwardaten__silu_backward����� � q
subtract.2777subtract@PP*	
 �2    :*
aten__silu_backwardaten__silu_backward������ � q
multiply.2778multiply@PP*	
 �2    :*
aten__silu_backwardaten__silu_backward������ � g
add.2780add@PP*	
 �2    :*
aten__silu_backwardaten__silu_backward������ � q
multiply.2781multiply@PP*	
 �2    :*
aten__silu_backwardaten__silu_backward������ � q
multiply.2782multiply@PP*	
 �2    :*
aten__silu_backwardaten__silu_backward������ � �
batch-norm-grad.2790batch-norm-gradA"@PP*	
 �2    "@*
 �2 "@*
 �2 :D
 aten__native_batch_norm_backward aten__native_batch_norm_backward�o�:����
����2�� � �
get-tuple-element.2793get-tuple-element@*
 �2 :D
 aten__native_batch_norm_backward aten__native_batch_norm_backwardh����� � �
custom-call.352custom-call@*
 �2 :�
'xla___op_TransferWithStaticRingTransfer'xla___op_TransferWithStaticRingTransferv/home/ubuntu/aws-neuron-samples/aws_neuron_venv_pytorch/lib/python3.10/site-packages/torch_xla/core/xla_op_registry.py ,x�AwsNeuronTransferWithStaticRing��.����
 �� � �
get-tuple-element.2792get-tuple-element@*
 �2 :D
 aten__native_batch_norm_backward aten__native_batch_norm_backwardh����� � �
custom-call.353custom-call@*
 �2 :�
'xla___op_TransferWithStaticRingTransfer'xla___op_TransferWithStaticRingTransferv/home/ubuntu/aws-neuron-samples/aws_neuron_venv_pytorch/lib/python3.10/site-packages/torch_xla/core/xla_op_registry.py ,x�AwsNeuronTransferWithStaticRing��.����
 �� � �
	p252.2809	parameter�PP*	
 �2    :$
xla__device_dataxla__device_dataH����

neff_input_namesinput252� �
get-tuple-element.2791get-tuple-element@PP*	
 �2    :D
 aten__native_batch_norm_backward aten__native_batch_norm_backward����� � �
convolution.2813convolution�@*	
 �2    :R
'aten__convolution_backward_overrideable'aten__convolution_backward_overrideablez
P(0
P(0� 28HPZb �������
  �� � �
transpose.2814	transpose@�*	
 �2    :R
'aten__convolution_backward_overrideable'aten__convolution_backward_overrideabler ����� � �
custom-call.354custom-call@�*	
 �2    :�
'xla___op_TransferWithStaticRingTransfer'xla___op_TransferWithStaticRingTransferv/home/ubuntu/aws-neuron-samples/aws_neuron_venv_pytorch/lib/python3.10/site-packages/torch_xla/core/xla_op_registry.py ,x�AwsNeuronTransferWithStaticRing��.����
 �� � �
	p256.2831	parameter@PP*	
 �2    :$
xla__device_dataxla__device_dataH����

neff_input_namesinput256� y
	p255.2830	parameter@*
 �2 :$
xla__device_dataxla__device_dataH����

neff_input_namesinput255� y
	p254.2829	parameter@*
 �2 :$
xla__device_dataxla__device_dataH����

neff_input_namesinput254� �
constant.2843constant*�:D
 aten__native_batch_norm_backward aten__native_batch_norm_backwardB
*�B  �?��� � �
broadcast.2844	broadcast@*
 �2 :D
 aten__native_batch_norm_backward aten__native_batch_norm_backward����� � y
	p253.2828	parameter@*
 �2 :$
xla__device_dataxla__device_dataH����

neff_input_namesinput253� ~
divide.2845divide@*
 �2 :D
 aten__native_batch_norm_backward aten__native_batch_norm_backward������ � �
multiply.2846multiply@*
 �2 :D
 aten__native_batch_norm_backward aten__native_batch_norm_backward������ � =
constant.72constant*�: B
*�Bo����6� � �
broadcast.134	broadcast@*
 �2 :D
 aten__native_batch_norm_backward aten__native_batch_norm_backward��2��6� � v
add.34add@*
 �2 :D
 aten__native_batch_norm_backward aten__native_batch_norm_backward��2���2� � �
	p244.2748	parameter@@*	
 �2    :$
xla__device_dataxla__device_dataH����

neff_input_namesinput244� �
transpose.2750	transpose@@*	
 �2    :R
'aten__convolution_backward_overrideable'aten__convolution_backward_overrideabler ����� � �
convolution.2752convolution@PP*	
 �2    :R
'aten__convolution_backward_overrideable'aten__convolution_backward_overrideablez
(0
(0� 2 @PZb�������
  �� � �
	p257.2832	parameter@PP*	
 �2    :$
xla__device_dataxla__device_dataH����

neff_input_namesinput257� o
logistic.2834logistic@PP*	
 �2    :*
aten__silu_backwardaten__silu_backward����� � i
constant.2833constant*�:*
aten__silu_backwardaten__silu_backwardB
*�B  �?��� � q
broadcast.2838	broadcast@PP*	
 �2    :*
aten__silu_backwardaten__silu_backward����� � q
broadcast.2835	broadcast@PP*	
 �2    :*
aten__silu_backwardaten__silu_backward����� � q
subtract.2836subtract@PP*	
 �2    :*
aten__silu_backwardaten__silu_backward������ � q
multiply.2837multiply@PP*	
 �2    :*
aten__silu_backwardaten__silu_backward������ � g
add.2839add@PP*	
 �2    :*
aten__silu_backwardaten__silu_backward������ � q
multiply.2840multiply@PP*	
 �2    :*
aten__silu_backwardaten__silu_backward������ � q
multiply.2841multiply@PP*	
 �2    :*
aten__silu_backwardaten__silu_backward������ � �
batch-norm-grad.2849batch-norm-gradA"@PP*	
 �2    "@*
 �2 "@*
 �2 :D
 aten__native_batch_norm_backward aten__native_batch_norm_backward�o�:����
����2�� � �
get-tuple-element.2852get-tuple-element@*
 �2 :D
 aten__native_batch_norm_backward aten__native_batch_norm_backwardh����� � �
custom-call.355custom-call@*
 �2 :�
'xla___op_TransferWithStaticRingTransfer'xla___op_TransferWithStaticRingTransferv/home/ubuntu/aws-neuron-samples/aws_neuron_venv_pytorch/lib/python3.10/site-packages/torch_xla/core/xla_op_registry.py ,x�AwsNeuronTransferWithStaticRing��.����
 �� � �
get-tuple-element.2851get-tuple-element@*
 �2 :D
 aten__native_batch_norm_backward aten__native_batch_norm_backwardh����� � �
custom-call.356custom-call@*
 �2 :�
'xla___op_TransferWithStaticRingTransfer'xla___op_TransferWithStaticRingTransferv/home/ubuntu/aws-neuron-samples/aws_neuron_venv_pytorch/lib/python3.10/site-packages/torch_xla/core/xla_op_registry.py ,x�AwsNeuronTransferWithStaticRing��.����
 �� � �
get-tuple-element.2850get-tuple-element@PP*	
 �2    :D
 aten__native_batch_norm_backward aten__native_batch_norm_backward����� � �
convolution.2871convolution�@*	
 �2    :R
'aten__convolution_backward_overrideable'aten__convolution_backward_overrideablez
P(0
P(0� 28HPZb �������
  �� � �
transpose.2872	transpose@�*	
 �2    :R
'aten__convolution_backward_overrideable'aten__convolution_backward_overrideabler ����� � �
custom-call.357custom-call@�*	
 �2    :�
'xla___op_TransferWithStaticRingTransfer'xla___op_TransferWithStaticRingTransferv/home/ubuntu/aws-neuron-samples/aws_neuron_venv_pytorch/lib/python3.10/site-packages/torch_xla/core/xla_op_registry.py ,x�AwsNeuronTransferWithStaticRing��.����
 �� � �
	p262.2889	parameter�((*	
 �2    :$
xla__device_dataxla__device_dataH����

neff_input_namesinput262� z
	p261.2888	parameter�*
 �2 :$
xla__device_dataxla__device_dataH����

neff_input_namesinput261� z
	p260.2887	parameter�*
 �2 :$
xla__device_dataxla__device_dataH����

neff_input_namesinput260� �
constant.2926constant*�:D
 aten__native_batch_norm_backward aten__native_batch_norm_backwardB
*�B  �?��� � �
broadcast.2927	broadcast�*
 �2 :D
 aten__native_batch_norm_backward aten__native_batch_norm_backward����� � z
	p259.2886	parameter�*
 �2 :$
xla__device_dataxla__device_dataH����

neff_input_namesinput259� 
divide.2928divide�*
 �2 :D
 aten__native_batch_norm_backward aten__native_batch_norm_backward������ � �
multiply.2929multiply�*
 �2 :D
 aten__native_batch_norm_backward aten__native_batch_norm_backward������ � =
constant.73constant*�: B
*�Bo����6� � �
broadcast.135	broadcast�*
 �2 :D
 aten__native_batch_norm_backward aten__native_batch_norm_backward��2��6� � w
add.35add�*
 �2 :D
 aten__native_batch_norm_backward aten__native_batch_norm_backward��2���2� � z

slice.2914slice�((*	
 �2    :
xla__selectxla__select�����(�(����� � �
	p251.2808	parameter@�*	
 �2    :$
xla__device_dataxla__device_dataH����

neff_input_namesinput251� �
transpose.2810	transpose�@*	
 �2    :R
'aten__convolution_backward_overrideable'aten__convolution_backward_overrideabler ����� � �
convolution.2812convolution�PP*	
 �2    :R
'aten__convolution_backward_overrideable'aten__convolution_backward_overrideablez
(0
(0� 2 @PZb�������
  �� � �
	p258.2867	parameter@�*	
 �2    :$
xla__device_dataxla__device_dataH����

neff_input_namesinput258� �
transpose.2868	transpose�@*	
 �2    :R
'aten__convolution_backward_overrideable'aten__convolution_backward_overrideabler ����� � �
convolution.2870convolution�PP*	
 �2    :R
'aten__convolution_backward_overrideable'aten__convolution_backward_overrideablez
(0
(0� 2 @PZb�������
  �� � T
add.2902add�PP*	
 �2    :
	aten__add	aten__add������ � �

slice.2903slice�PP*	
 �2    :�
xla__selectxla__selectv/home/ubuntu/aws-neuron-samples/aws_neuron_venv_pytorch/lib/python3.10/site-packages/torch_xla/core/xla_op_registry.py ,x����P�P����� � �
transpose.2904	transposePP�*	
 �2    :�
aten__permuteaten__permutev/home/ubuntu/aws-neuron-samples/aws_neuron_venv_pytorch/lib/python3.10/site-packages/torch_xla/core/xla_op_registry.py ,xr ����� � �
custom-call.242custom-call((�*	
 �2    :�
.xla___op_UpSampleNearestNeighbor2dBackwardImpl.xla___op_UpSampleNearestNeighbor2dBackwardImplv/home/ubuntu/aws-neuron-samples/aws_neuron_venv_pytorch/lib/python3.10/site-packages/torch_xla/core/xla_op_registry.py ,x�ResizeNearestGrad��-���"00"��
 �� � l
transpose.2912	transpose�((*	
 �2    :
aten__permuteaten__permuter ����-� � T
add.2915add�((*	
 �2    :
	aten__add	aten__add������ � �
	p263.2890	parameter�((*	
 �2    :$
xla__device_dataxla__device_dataH����

neff_input_namesinput263� p
logistic.2917logistic�((*	
 �2    :*
aten__silu_backwardaten__silu_backward����� � i
constant.2916constant*�:*
aten__silu_backwardaten__silu_backwardB
*�B  �?��� � r
broadcast.2921	broadcast�((*	
 �2    :*
aten__silu_backwardaten__silu_backward����� � r
broadcast.2918	broadcast�((*	
 �2    :*
aten__silu_backwardaten__silu_backward����� � r
subtract.2919subtract�((*	
 �2    :*
aten__silu_backwardaten__silu_backward������ � r
multiply.2920multiply�((*	
 �2    :*
aten__silu_backwardaten__silu_backward������ � h
add.2922add�((*	
 �2    :*
aten__silu_backwardaten__silu_backward������ � r
multiply.2923multiply�((*	
 �2    :*
aten__silu_backwardaten__silu_backward������ � r
multiply.2924multiply�((*	
 �2    :*
aten__silu_backwardaten__silu_backward������ � �
batch-norm-grad.2932batch-norm-gradD"�((*	
 �2    "�*
 �2 "�*
 �2 :D
 aten__native_batch_norm_backward aten__native_batch_norm_backward�o�:����
����2�� � �
get-tuple-element.2935get-tuple-element�*
 �2 :D
 aten__native_batch_norm_backward aten__native_batch_norm_backwardh����� � �
custom-call.358custom-call�*
 �2 :�
'xla___op_TransferWithStaticRingTransfer'xla___op_TransferWithStaticRingTransferv/home/ubuntu/aws-neuron-samples/aws_neuron_venv_pytorch/lib/python3.10/site-packages/torch_xla/core/xla_op_registry.py ,x�AwsNeuronTransferWithStaticRing��.����
 �� � �
get-tuple-element.2934get-tuple-element�*
 �2 :D
 aten__native_batch_norm_backward aten__native_batch_norm_backwardh����� � �
custom-call.359custom-call�*
 �2 :�
'xla___op_TransferWithStaticRingTransfer'xla___op_TransferWithStaticRingTransferv/home/ubuntu/aws-neuron-samples/aws_neuron_venv_pytorch/lib/python3.10/site-packages/torch_xla/core/xla_op_registry.py ,x�AwsNeuronTransferWithStaticRing��.����
 �� � �
	p265.2951	parameter�((*	
 �2    :$
xla__device_dataxla__device_dataH����

neff_input_namesinput265� �
get-tuple-element.2933get-tuple-element�((*	
 �2    :D
 aten__native_batch_norm_backward aten__native_batch_norm_backward����� � �
convolution.2955convolution��*	
 �2    :R
'aten__convolution_backward_overrideable'aten__convolution_backward_overrideablez
((0
((0� 28HPZb �������
  �� � �
transpose.2956	transpose��*	
 �2    :R
'aten__convolution_backward_overrideable'aten__convolution_backward_overrideabler ����� � �
custom-call.360custom-call��*	
 �2    :�
'xla___op_TransferWithStaticRingTransfer'xla___op_TransferWithStaticRingTransferv/home/ubuntu/aws-neuron-samples/aws_neuron_venv_pytorch/lib/python3.10/site-packages/torch_xla/core/xla_op_registry.py ,x�AwsNeuronTransferWithStaticRing��.����
 �� � �
	p269.2973	parameter�((*	
 �2    :$
xla__device_dataxla__device_dataH����

neff_input_namesinput269� z
	p268.2972	parameter�*
 �2 :$
xla__device_dataxla__device_dataH����

neff_input_namesinput268� z
	p267.2971	parameter�*
 �2 :$
xla__device_dataxla__device_dataH����

neff_input_namesinput267� �
constant.2985constant*�:D
 aten__native_batch_norm_backward aten__native_batch_norm_backwardB
*�B  �?��� � �
broadcast.2986	broadcast�*
 �2 :D
 aten__native_batch_norm_backward aten__native_batch_norm_backward����� � z
	p266.2970	parameter�*
 �2 :$
xla__device_dataxla__device_dataH����

neff_input_namesinput266� 
divide.2987divide�*
 �2 :D
 aten__native_batch_norm_backward aten__native_batch_norm_backward������ � �
multiply.2988multiply�*
 �2 :D
 aten__native_batch_norm_backward aten__native_batch_norm_backward������ � =
constant.74constant*�: B
*�Bo����6� � �
broadcast.140	broadcast�*
 �2 :D
 aten__native_batch_norm_backward aten__native_batch_norm_backward��2��6� � w
add.36add�*
 �2 :D
 aten__native_batch_norm_backward aten__native_batch_norm_backward��2���2� � �
	p264.2950	parameter��*	
 �2    :$
xla__device_dataxla__device_dataH����

neff_input_namesinput264� �
transpose.2952	transpose��*	
 �2    :R
'aten__convolution_backward_overrideable'aten__convolution_backward_overrideabler ����� � �
convolution.2954convolution�((*	
 �2    :R
'aten__convolution_backward_overrideable'aten__convolution_backward_overrideablez
(0
(0� 2 @PZb�������
  �� � �
	p270.2974	parameter�((*	
 �2    :$
xla__device_dataxla__device_dataH����

neff_input_namesinput270� p
logistic.2976logistic�((*	
 �2    :*
aten__silu_backwardaten__silu_backward����� � i
constant.2975constant*�:*
aten__silu_backwardaten__silu_backwardB
*�B  �?��� � r
broadcast.2980	broadcast�((*	
 �2    :*
aten__silu_backwardaten__silu_backward����� � r
broadcast.2977	broadcast�((*	
 �2    :*
aten__silu_backwardaten__silu_backward����� � r
subtract.2978subtract�((*	
 �2    :*
aten__silu_backwardaten__silu_backward������ � r
multiply.2979multiply�((*	
 �2    :*
aten__silu_backwardaten__silu_backward������ � h
add.2981add�((*	
 �2    :*
aten__silu_backwardaten__silu_backward������ � r
multiply.2982multiply�((*	
 �2    :*
aten__silu_backwardaten__silu_backward������ � r
multiply.2983multiply�((*	
 �2    :*
aten__silu_backwardaten__silu_backward������ � �
batch-norm-grad.2991batch-norm-gradD"�((*	
 �2    "�*
 �2 "�*
 �2 :D
 aten__native_batch_norm_backward aten__native_batch_norm_backward�o�:����
����2�� � �
get-tuple-element.2994get-tuple-element�*
 �2 :D
 aten__native_batch_norm_backward aten__native_batch_norm_backwardh����� � �
custom-call.361custom-call�*
 �2 :�
'xla___op_TransferWithStaticRingTransfer'xla___op_TransferWithStaticRingTransferv/home/ubuntu/aws-neuron-samples/aws_neuron_venv_pytorch/lib/python3.10/site-packages/torch_xla/core/xla_op_registry.py ,x�AwsNeuronTransferWithStaticRing��.����
 �� � �
get-tuple-element.2993get-tuple-element�*
 �2 :D
 aten__native_batch_norm_backward aten__native_batch_norm_backwardh����� � �
custom-call.362custom-call�*
 �2 :�
'xla___op_TransferWithStaticRingTransfer'xla___op_TransferWithStaticRingTransferv/home/ubuntu/aws-neuron-samples/aws_neuron_venv_pytorch/lib/python3.10/site-packages/torch_xla/core/xla_op_registry.py ,x�AwsNeuronTransferWithStaticRing��.����
 �� � �
	p272.3010	parameter�((*	
 �2    :$
xla__device_dataxla__device_dataH����

neff_input_namesinput272� �
get-tuple-element.2992get-tuple-element�((*	
 �2    :D
 aten__native_batch_norm_backward aten__native_batch_norm_backward����� � �
convolution.3014convolution��*	
 �2    :R
'aten__convolution_backward_overrideable'aten__convolution_backward_overrideablez
((0
((0� 28HPZb �������
  �� � �
transpose.3015	transpose��*	
 �2    :R
'aten__convolution_backward_overrideable'aten__convolution_backward_overrideabler ����� � �
custom-call.363custom-call��*	
 �2    :�
'xla___op_TransferWithStaticRingTransfer'xla___op_TransferWithStaticRingTransferv/home/ubuntu/aws-neuron-samples/aws_neuron_venv_pytorch/lib/python3.10/site-packages/torch_xla/core/xla_op_registry.py ,x�AwsNeuronTransferWithStaticRing��.����
 �� � �
	p276.3032	parameter�((*	
 �2    :$
xla__device_dataxla__device_dataH����

neff_input_namesinput276� z
	p275.3031	parameter�*
 �2 :$
xla__device_dataxla__device_dataH����

neff_input_namesinput275� z
	p274.3030	parameter�*
 �2 :$
xla__device_dataxla__device_dataH����

neff_input_namesinput274� �
constant.3045constant*�:D
 aten__native_batch_norm_backward aten__native_batch_norm_backwardB
*�B  �?��� � �
broadcast.3046	broadcast�*
 �2 :D
 aten__native_batch_norm_backward aten__native_batch_norm_backward����� � z
	p273.3029	parameter�*
 �2 :$
xla__device_dataxla__device_dataH����

neff_input_namesinput273� 
divide.3047divide�*
 �2 :D
 aten__native_batch_norm_backward aten__native_batch_norm_backward������ � �
multiply.3048multiply�*
 �2 :D
 aten__native_batch_norm_backward aten__native_batch_norm_backward������ � =
constant.75constant*�: B
*�Bo����6� � �
broadcast.141	broadcast�*
 �2 :D
 aten__native_batch_norm_backward aten__native_batch_norm_backward��2��6� � w
add.37add�*
 �2 :D
 aten__native_batch_norm_backward aten__native_batch_norm_backward��2���2� � �
	p271.3009	parameter��*	
 �2    :$
xla__device_dataxla__device_dataH����

neff_input_namesinput271� �
transpose.3011	transpose��*	
 �2    :R
'aten__convolution_backward_overrideable'aten__convolution_backward_overrideabler ����� � �
convolution.3013convolution�((*	
 �2    :R
'aten__convolution_backward_overrideable'aten__convolution_backward_overrideablez
(0
(0� 2 @PZb�������
  �� � w

slice.3034slice�((*	
 �2    :
xla__selectxla__select����(�(����� � �
	p277.3033	parameter�((*	
 �2    :$
xla__device_dataxla__device_dataH����

neff_input_namesinput277� p
logistic.3036logistic�((*	
 �2    :*
aten__silu_backwardaten__silu_backward����� � i
constant.3035constant*�:*
aten__silu_backwardaten__silu_backwardB
*�B  �?��� � r
broadcast.3040	broadcast�((*	
 �2    :*
aten__silu_backwardaten__silu_backward����� � r
broadcast.3037	broadcast�((*	
 �2    :*
aten__silu_backwardaten__silu_backward����� � r
subtract.3038subtract�((*	
 �2    :*
aten__silu_backwardaten__silu_backward������ � r
multiply.3039multiply�((*	
 �2    :*
aten__silu_backwardaten__silu_backward������ � h
add.3041add�((*	
 �2    :*
aten__silu_backwardaten__silu_backward������ � r
multiply.3042multiply�((*	
 �2    :*
aten__silu_backwardaten__silu_backward������ � r
multiply.3043multiply�((*	
 �2    :*
aten__silu_backwardaten__silu_backward������ � �
batch-norm-grad.3051batch-norm-gradD"�((*	
 �2    "�*
 �2 "�*
 �2 :D
 aten__native_batch_norm_backward aten__native_batch_norm_backward�o�:����
����2�� � �
get-tuple-element.3054get-tuple-element�*
 �2 :D
 aten__native_batch_norm_backward aten__native_batch_norm_backwardh����� � �
custom-call.364custom-call�*
 �2 :�
'xla___op_TransferWithStaticRingTransfer'xla___op_TransferWithStaticRingTransferv/home/ubuntu/aws-neuron-samples/aws_neuron_venv_pytorch/lib/python3.10/site-packages/torch_xla/core/xla_op_registry.py ,x�AwsNeuronTransferWithStaticRing��.����
 �� � �
get-tuple-element.3053get-tuple-element�*
 �2 :D
 aten__native_batch_norm_backward aten__native_batch_norm_backwardh����� � �
custom-call.365custom-call�*
 �2 :�
'xla___op_TransferWithStaticRingTransfer'xla___op_TransferWithStaticRingTransferv/home/ubuntu/aws-neuron-samples/aws_neuron_venv_pytorch/lib/python3.10/site-packages/torch_xla/core/xla_op_registry.py ,x�AwsNeuronTransferWithStaticRing��.����
 �� � �
	p279.3070	parameter�((*	
 �2    :$
xla__device_dataxla__device_dataH����

neff_input_namesinput279� �
get-tuple-element.3052get-tuple-element�((*	
 �2    :D
 aten__native_batch_norm_backward aten__native_batch_norm_backward����� � �
convolution.3074convolution��*	
 �2    :R
'aten__convolution_backward_overrideable'aten__convolution_backward_overrideablez
( (0
( (0� 28HPZb �������
  �� � �
transpose.3075	transpose��*	
 �2    :R
'aten__convolution_backward_overrideable'aten__convolution_backward_overrideabler ����� � �
custom-call.366custom-call��*	
 �2    :�
'xla___op_TransferWithStaticRingTransfer'xla___op_TransferWithStaticRingTransferv/home/ubuntu/aws-neuron-samples/aws_neuron_venv_pytorch/lib/python3.10/site-packages/torch_xla/core/xla_op_registry.py ,x�AwsNeuronTransferWithStaticRing��.����
 �� � �
	p283.3092	parameter�((*	
 �2    :$
xla__device_dataxla__device_dataH����

neff_input_namesinput283� z
	p282.3091	parameter�*
 �2 :$
xla__device_dataxla__device_dataH����

neff_input_namesinput282� z
	p281.3090	parameter�*
 �2 :$
xla__device_dataxla__device_dataH����

neff_input_namesinput281� �
constant.3104constant*�:D
 aten__native_batch_norm_backward aten__native_batch_norm_backwardB
*�B  �?��� � �
broadcast.3105	broadcast�*
 �2 :D
 aten__native_batch_norm_backward aten__native_batch_norm_backward����� � z
	p280.3089	parameter�*
 �2 :$
xla__device_dataxla__device_dataH����

neff_input_namesinput280� 
divide.3106divide�*
 �2 :D
 aten__native_batch_norm_backward aten__native_batch_norm_backward������ � �
multiply.3107multiply�*
 �2 :D
 aten__native_batch_norm_backward aten__native_batch_norm_backward������ � =
constant.76constant*�: B
*�Bo����6� � �
broadcast.142	broadcast�*
 �2 :D
 aten__native_batch_norm_backward aten__native_batch_norm_backward��2��6� � w
add.38add�*
 �2 :D
 aten__native_batch_norm_backward aten__native_batch_norm_backward��2���2� � �
	p278.3069	parameter��*	
 �2    :$
xla__device_dataxla__device_dataH����

neff_input_namesinput278� �
transpose.3071	transpose��*	
 �2    :R
'aten__convolution_backward_overrideable'aten__convolution_backward_overrideabler ����� � �
reverse.3072reverse��*	
 �2    :R
'aten__convolution_backward_overrideable'aten__convolution_backward_overrideabler ����� � �
convolution.3073convolution�((*	
 �2    :R
'aten__convolution_backward_overrideable'aten__convolution_backward_overrideablez
 (0
 (0� 2 @PZb�������
  �� � �
	p284.3093	parameter�((*	
 �2    :$
xla__device_dataxla__device_dataH����

neff_input_namesinput284� p
logistic.3095logistic�((*	
 �2    :*
aten__silu_backwardaten__silu_backward����� � i
constant.3094constant*�:*
aten__silu_backwardaten__silu_backwardB
*�B  �?��� � r
broadcast.3099	broadcast�((*	
 �2    :*
aten__silu_backwardaten__silu_backward����� � r
broadcast.3096	broadcast�((*	
 �2    :*
aten__silu_backwardaten__silu_backward����� � r
subtract.3097subtract�((*	
 �2    :*
aten__silu_backwardaten__silu_backward������ � r
multiply.3098multiply�((*	
 �2    :*
aten__silu_backwardaten__silu_backward������ � h
add.3100add�((*	
 �2    :*
aten__silu_backwardaten__silu_backward������ � r
multiply.3101multiply�((*	
 �2    :*
aten__silu_backwardaten__silu_backward������ � r
multiply.3102multiply�((*	
 �2    :*
aten__silu_backwardaten__silu_backward������ � �
batch-norm-grad.3110batch-norm-gradD"�((*	
 �2    "�*
 �2 "�*
 �2 :D
 aten__native_batch_norm_backward aten__native_batch_norm_backward�o�:����
����2�� � �
get-tuple-element.3113get-tuple-element�*
 �2 :D
 aten__native_batch_norm_backward aten__native_batch_norm_backwardh����� � �
custom-call.367custom-call�*
 �2 :�
'xla___op_TransferWithStaticRingTransfer'xla___op_TransferWithStaticRingTransferv/home/ubuntu/aws-neuron-samples/aws_neuron_venv_pytorch/lib/python3.10/site-packages/torch_xla/core/xla_op_registry.py ,x�AwsNeuronTransferWithStaticRing��.����
 �� � �
get-tuple-element.3112get-tuple-element�*
 �2 :D
 aten__native_batch_norm_backward aten__native_batch_norm_backwardh����� � �
custom-call.368custom-call�*
 �2 :�
'xla___op_TransferWithStaticRingTransfer'xla___op_TransferWithStaticRingTransferv/home/ubuntu/aws-neuron-samples/aws_neuron_venv_pytorch/lib/python3.10/site-packages/torch_xla/core/xla_op_registry.py ,x�AwsNeuronTransferWithStaticRing��.����
 �� � �
	p286.3129	parameter�((*	
 �2    :$
xla__device_dataxla__device_dataH����

neff_input_namesinput286� �
get-tuple-element.3111get-tuple-element�((*	
 �2    :D
 aten__native_batch_norm_backward aten__native_batch_norm_backward����� � �
convolution.3133convolution��*	
 �2    :R
'aten__convolution_backward_overrideable'aten__convolution_backward_overrideablez
((0
((0� 28HPZb �������
  �� � �
transpose.3134	transpose��*	
 �2    :R
'aten__convolution_backward_overrideable'aten__convolution_backward_overrideabler ����� � �
custom-call.369custom-call��*	
 �2    :�
'xla___op_TransferWithStaticRingTransfer'xla___op_TransferWithStaticRingTransferv/home/ubuntu/aws-neuron-samples/aws_neuron_venv_pytorch/lib/python3.10/site-packages/torch_xla/core/xla_op_registry.py ,x�AwsNeuronTransferWithStaticRing��.����
 �� � �
	p290.3151	parameter�((*	
 �2    :$
xla__device_dataxla__device_dataH����

neff_input_namesinput290� z
	p289.3150	parameter�*
 �2 :$
xla__device_dataxla__device_dataH����

neff_input_namesinput289� z
	p288.3149	parameter�*
 �2 :$
xla__device_dataxla__device_dataH����

neff_input_namesinput288� �
constant.3164constant*�:D
 aten__native_batch_norm_backward aten__native_batch_norm_backwardB
*�B  �?��� � �
broadcast.3165	broadcast�*
 �2 :D
 aten__native_batch_norm_backward aten__native_batch_norm_backward����� � z
	p287.3148	parameter�*
 �2 :$
xla__device_dataxla__device_dataH����

neff_input_namesinput287� 
divide.3166divide�*
 �2 :D
 aten__native_batch_norm_backward aten__native_batch_norm_backward������ � �
multiply.3167multiply�*
 �2 :D
 aten__native_batch_norm_backward aten__native_batch_norm_backward������ � =
constant.77constant*�: B
*�Bo����6� � �
broadcast.144	broadcast�*
 �2 :D
 aten__native_batch_norm_backward aten__native_batch_norm_backward��2��6� � w
add.39add�*
 �2 :D
 aten__native_batch_norm_backward aten__native_batch_norm_backward��2���2� � z

slice.3153slice�((*	
 �2    :
xla__selectxla__select�����(�(����� � �
	p291.3152	parameter�((*	
 �2    :$
xla__device_dataxla__device_dataH����

neff_input_namesinput291� p
logistic.3155logistic�((*	
 �2    :*
aten__silu_backwardaten__silu_backward����� � i
constant.3154constant*�:*
aten__silu_backwardaten__silu_backwardB
*�B  �?��� � r
broadcast.3159	broadcast�((*	
 �2    :*
aten__silu_backwardaten__silu_backward����� � r
broadcast.3156	broadcast�((*	
 �2    :*
aten__silu_backwardaten__silu_backward����� � r
subtract.3157subtract�((*	
 �2    :*
aten__silu_backwardaten__silu_backward������ � r
multiply.3158multiply�((*	
 �2    :*
aten__silu_backwardaten__silu_backward������ � h
add.3160add�((*	
 �2    :*
aten__silu_backwardaten__silu_backward������ � r
multiply.3161multiply�((*	
 �2    :*
aten__silu_backwardaten__silu_backward������ � r
multiply.3162multiply�((*	
 �2    :*
aten__silu_backwardaten__silu_backward������ � �
batch-norm-grad.3170batch-norm-gradD"�((*	
 �2    "�*
 �2 "�*
 �2 :D
 aten__native_batch_norm_backward aten__native_batch_norm_backward�o�:����
����2�� � �
get-tuple-element.3173get-tuple-element�*
 �2 :D
 aten__native_batch_norm_backward aten__native_batch_norm_backwardh����� � �
custom-call.370custom-call�*
 �2 :�
'xla___op_TransferWithStaticRingTransfer'xla___op_TransferWithStaticRingTransferv/home/ubuntu/aws-neuron-samples/aws_neuron_venv_pytorch/lib/python3.10/site-packages/torch_xla/core/xla_op_registry.py ,x�AwsNeuronTransferWithStaticRing��.����
 �� � �
get-tuple-element.3172get-tuple-element�*
 �2 :D
 aten__native_batch_norm_backward aten__native_batch_norm_backwardh����� � �
custom-call.371custom-call�*
 �2 :�
'xla___op_TransferWithStaticRingTransfer'xla___op_TransferWithStaticRingTransferv/home/ubuntu/aws-neuron-samples/aws_neuron_venv_pytorch/lib/python3.10/site-packages/torch_xla/core/xla_op_registry.py ,x�AwsNeuronTransferWithStaticRing��.����
 �� � �
	p293.3189	parameter�((*	
 �2    :$
xla__device_dataxla__device_dataH����

neff_input_namesinput293� �
get-tuple-element.3171get-tuple-element�((*	
 �2    :D
 aten__native_batch_norm_backward aten__native_batch_norm_backward����� � �
convolution.3193convolution��*	
 �2    :R
'aten__convolution_backward_overrideable'aten__convolution_backward_overrideablez
((0
((0� 28HPZb �������
  �� � �
transpose.3194	transpose��*	
 �2    :R
'aten__convolution_backward_overrideable'aten__convolution_backward_overrideabler ����� � �
custom-call.372custom-call��*	
 �2    :�
'xla___op_TransferWithStaticRingTransfer'xla___op_TransferWithStaticRingTransferv/home/ubuntu/aws-neuron-samples/aws_neuron_venv_pytorch/lib/python3.10/site-packages/torch_xla/core/xla_op_registry.py ,x�AwsNeuronTransferWithStaticRing��.����
 �� � �
	p297.3211	parameter�((*	
 �2    :$
xla__device_dataxla__device_dataH����

neff_input_namesinput297� z
	p296.3210	parameter�*
 �2 :$
xla__device_dataxla__device_dataH����

neff_input_namesinput296� z
	p295.3209	parameter�*
 �2 :$
xla__device_dataxla__device_dataH����

neff_input_namesinput295� �
constant.3223constant*�:D
 aten__native_batch_norm_backward aten__native_batch_norm_backwardB
*�B  �?��� � �
broadcast.3224	broadcast�*
 �2 :D
 aten__native_batch_norm_backward aten__native_batch_norm_backward����� � z
	p294.3208	parameter�*
 �2 :$
xla__device_dataxla__device_dataH����

neff_input_namesinput294� 
divide.3225divide�*
 �2 :D
 aten__native_batch_norm_backward aten__native_batch_norm_backward������ � �
multiply.3226multiply�*
 �2 :D
 aten__native_batch_norm_backward aten__native_batch_norm_backward������ � =
constant.78constant*�: B
*�Bo����6� � �
broadcast.146	broadcast�*
 �2 :D
 aten__native_batch_norm_backward aten__native_batch_norm_backward��2��6� � w
add.40add�*
 �2 :D
 aten__native_batch_norm_backward aten__native_batch_norm_backward��2���2� � �
	p285.3128	parameter��*	
 �2    :$
xla__device_dataxla__device_dataH����

neff_input_namesinput285� �
transpose.3130	transpose��*	
 �2    :R
'aten__convolution_backward_overrideable'aten__convolution_backward_overrideabler ����� � �
convolution.3132convolution�((*	
 �2    :R
'aten__convolution_backward_overrideable'aten__convolution_backward_overrideablez
(0
(0� 2 @PZb�������
  �� � �
	p298.3212	parameter�((*	
 �2    :$
xla__device_dataxla__device_dataH����

neff_input_namesinput298� p
logistic.3214logistic�((*	
 �2    :*
aten__silu_backwardaten__silu_backward����� � i
constant.3213constant*�:*
aten__silu_backwardaten__silu_backwardB
*�B  �?��� � r
broadcast.3218	broadcast�((*	
 �2    :*
aten__silu_backwardaten__silu_backward����� � r
broadcast.3215	broadcast�((*	
 �2    :*
aten__silu_backwardaten__silu_backward����� � r
subtract.3216subtract�((*	
 �2    :*
aten__silu_backwardaten__silu_backward������ � r
multiply.3217multiply�((*	
 �2    :*
aten__silu_backwardaten__silu_backward������ � h
add.3219add�((*	
 �2    :*
aten__silu_backwardaten__silu_backward������ � r
multiply.3220multiply�((*	
 �2    :*
aten__silu_backwardaten__silu_backward������ � r
multiply.3221multiply�((*	
 �2    :*
aten__silu_backwardaten__silu_backward������ � �
batch-norm-grad.3229batch-norm-gradD"�((*	
 �2    "�*
 �2 "�*
 �2 :D
 aten__native_batch_norm_backward aten__native_batch_norm_backward�o�:����
����2�� � �
get-tuple-element.3232get-tuple-element�*
 �2 :D
 aten__native_batch_norm_backward aten__native_batch_norm_backwardh����� � �
custom-call.373custom-call�*
 �2 :�
'xla___op_TransferWithStaticRingTransfer'xla___op_TransferWithStaticRingTransferv/home/ubuntu/aws-neuron-samples/aws_neuron_venv_pytorch/lib/python3.10/site-packages/torch_xla/core/xla_op_registry.py ,x�AwsNeuronTransferWithStaticRing��.����
 �� � �
get-tuple-element.3231get-tuple-element�*
 �2 :D
 aten__native_batch_norm_backward aten__native_batch_norm_backwardh����� � �
custom-call.374custom-call�*
 �2 :�
'xla___op_TransferWithStaticRingTransfer'xla___op_TransferWithStaticRingTransferv/home/ubuntu/aws-neuron-samples/aws_neuron_venv_pytorch/lib/python3.10/site-packages/torch_xla/core/xla_op_registry.py ,x�AwsNeuronTransferWithStaticRing��.����
 �� � �
get-tuple-element.3230get-tuple-element�((*	
 �2    :D
 aten__native_batch_norm_backward aten__native_batch_norm_backward����� � �
convolution.3251convolution��*	
 �2    :R
'aten__convolution_backward_overrideable'aten__convolution_backward_overrideablez
((0
((0� 28HPZb �������
  �� � �
transpose.3252	transpose��*	
 �2    :R
'aten__convolution_backward_overrideable'aten__convolution_backward_overrideabler ����� � �
custom-call.375custom-call��*	
 �2    :�
'xla___op_TransferWithStaticRingTransfer'xla___op_TransferWithStaticRingTransferv/home/ubuntu/aws-neuron-samples/aws_neuron_venv_pytorch/lib/python3.10/site-packages/torch_xla/core/xla_op_registry.py ,x�AwsNeuronTransferWithStaticRing��.����
 �� � �
	p303.3269	parameter�*	
 �2    :$
xla__device_dataxla__device_dataH����

neff_input_namesinput303� z
	p302.3268	parameter�*
 �2 :$
xla__device_dataxla__device_dataH����

neff_input_namesinput302� z
	p301.3267	parameter�*
 �2 :$
xla__device_dataxla__device_dataH����

neff_input_namesinput301� �
constant.3306constant*�:D
 aten__native_batch_norm_backward aten__native_batch_norm_backwardB
*�B  �?��� � �
broadcast.3307	broadcast�*
 �2 :D
 aten__native_batch_norm_backward aten__native_batch_norm_backward����� � z
	p300.3266	parameter�*
 �2 :$
xla__device_dataxla__device_dataH����

neff_input_namesinput300� 
divide.3308divide�*
 �2 :D
 aten__native_batch_norm_backward aten__native_batch_norm_backward������ � �
multiply.3309multiply�*
 �2 :D
 aten__native_batch_norm_backward aten__native_batch_norm_backward������ � =
constant.79constant*�: B
*�Bo����6� � �
broadcast.147	broadcast�*
 �2 :D
 aten__native_batch_norm_backward aten__native_batch_norm_backward��2��6� � w
add.41add�*
 �2 :D
 aten__native_batch_norm_backward aten__native_batch_norm_backward��2���2� � z

slice.3294slice�*	
 �2    :
xla__selectxla__select����������� � �
	p292.3188	parameter��*	
 �2    :$
xla__device_dataxla__device_dataH����

neff_input_namesinput292� �
transpose.3190	transpose��*	
 �2    :R
'aten__convolution_backward_overrideable'aten__convolution_backward_overrideabler ����� � �
convolution.3192convolution�((*	
 �2    :R
'aten__convolution_backward_overrideable'aten__convolution_backward_overrideablez
(0
(0� 2 @PZb�������
  �� � �
	p299.3247	parameter��*	
 �2    :$
xla__device_dataxla__device_dataH����

neff_input_namesinput299� �
transpose.3248	transpose��*	
 �2    :R
'aten__convolution_backward_overrideable'aten__convolution_backward_overrideabler ����� � �
convolution.3250convolution�((*	
 �2    :R
'aten__convolution_backward_overrideable'aten__convolution_backward_overrideablez
(0
(0� 2 @PZb�������
  �� � T
add.3282add�((*	
 �2    :
	aten__add	aten__add������ � �

slice.3283slice�((*	
 �2    :�
xla__selectxla__selectv/home/ubuntu/aws-neuron-samples/aws_neuron_venv_pytorch/lib/python3.10/site-packages/torch_xla/core/xla_op_registry.py ,x����(�(����� � �
transpose.3284	transpose((�*	
 �2    :�
aten__permuteaten__permutev/home/ubuntu/aws-neuron-samples/aws_neuron_venv_pytorch/lib/python3.10/site-packages/torch_xla/core/xla_op_registry.py ,xr ����� � �
custom-call.243custom-call�*	
 �2    :�
.xla___op_UpSampleNearestNeighbor2dBackwardImpl.xla___op_UpSampleNearestNeighbor2dBackwardImplv/home/ubuntu/aws-neuron-samples/aws_neuron_venv_pytorch/lib/python3.10/site-packages/torch_xla/core/xla_op_registry.py ,x�ResizeNearestGrad��-���"00"��
 �� � l
transpose.3292	transpose�*	
 �2    :
aten__permuteaten__permuter ����-� � T
add.3295add�*	
 �2    :
	aten__add	aten__add������ � �
	p304.3270	parameter�*	
 �2    :$
xla__device_dataxla__device_dataH����

neff_input_namesinput304� p
logistic.3297logistic�*	
 �2    :*
aten__silu_backwardaten__silu_backward����� � i
constant.3296constant*�:*
aten__silu_backwardaten__silu_backwardB
*�B  �?��� � r
broadcast.3301	broadcast�*	
 �2    :*
aten__silu_backwardaten__silu_backward����� � r
broadcast.3298	broadcast�*	
 �2    :*
aten__silu_backwardaten__silu_backward����� � r
subtract.3299subtract�*	
 �2    :*
aten__silu_backwardaten__silu_backward������ � r
multiply.3300multiply�*	
 �2    :*
aten__silu_backwardaten__silu_backward������ � h
add.3302add�*	
 �2    :*
aten__silu_backwardaten__silu_backward������ � r
multiply.3303multiply�*	
 �2    :*
aten__silu_backwardaten__silu_backward������ � r
multiply.3304multiply�*	
 �2    :*
aten__silu_backwardaten__silu_backward������ � �
batch-norm-grad.3312batch-norm-gradD"�*	
 �2    "�*
 �2 "�*
 �2 :D
 aten__native_batch_norm_backward aten__native_batch_norm_backward�o�:����
����2�� � �
get-tuple-element.3315get-tuple-element�*
 �2 :D
 aten__native_batch_norm_backward aten__native_batch_norm_backwardh����� � �
custom-call.376custom-call�*
 �2 :�
'xla___op_TransferWithStaticRingTransfer'xla___op_TransferWithStaticRingTransferv/home/ubuntu/aws-neuron-samples/aws_neuron_venv_pytorch/lib/python3.10/site-packages/torch_xla/core/xla_op_registry.py ,x�AwsNeuronTransferWithStaticRing��.����
 �� � �
get-tuple-element.3314get-tuple-element�*
 �2 :D
 aten__native_batch_norm_backward aten__native_batch_norm_backwardh����� � �
custom-call.377custom-call�*
 �2 :�
'xla___op_TransferWithStaticRingTransfer'xla___op_TransferWithStaticRingTransferv/home/ubuntu/aws-neuron-samples/aws_neuron_venv_pytorch/lib/python3.10/site-packages/torch_xla/core/xla_op_registry.py ,x�AwsNeuronTransferWithStaticRing��.����
 �� � �
	p306.3331	parameter�*	
 �2    :$
xla__device_dataxla__device_dataH����

neff_input_namesinput306� �
get-tuple-element.3313get-tuple-element�*	
 �2    :D
 aten__native_batch_norm_backward aten__native_batch_norm_backward����� � �
convolution.3335convolution��*	
 �2    :R
'aten__convolution_backward_overrideable'aten__convolution_backward_overrideablez
(0
(0� 28HPZb �������
  �� � �
transpose.3336	transpose��*	
 �2    :R
'aten__convolution_backward_overrideable'aten__convolution_backward_overrideabler ����� � �
custom-call.378custom-call��*	
 �2    :�
'xla___op_TransferWithStaticRingTransfer'xla___op_TransferWithStaticRingTransferv/home/ubuntu/aws-neuron-samples/aws_neuron_venv_pytorch/lib/python3.10/site-packages/torch_xla/core/xla_op_registry.py ,x�AwsNeuronTransferWithStaticRing��.����
 �� � �
	p310.3353	parameter�*	
 �2    :$
xla__device_dataxla__device_dataH����

neff_input_namesinput310� z
	p309.3352	parameter�*
 �2 :$
xla__device_dataxla__device_dataH����

neff_input_namesinput309� z
	p308.3351	parameter�*
 �2 :$
xla__device_dataxla__device_dataH����

neff_input_namesinput308� �
constant.3365constant*�:D
 aten__native_batch_norm_backward aten__native_batch_norm_backwardB
*�B  �?��� � �
broadcast.3366	broadcast�*
 �2 :D
 aten__native_batch_norm_backward aten__native_batch_norm_backward����� � z
	p307.3350	parameter�*
 �2 :$
xla__device_dataxla__device_dataH����

neff_input_namesinput307� 
divide.3367divide�*
 �2 :D
 aten__native_batch_norm_backward aten__native_batch_norm_backward������ � �
multiply.3368multiply�*
 �2 :D
 aten__native_batch_norm_backward aten__native_batch_norm_backward������ � =
constant.80constant*�: B
*�Bo����6� � �
broadcast.151	broadcast�*
 �2 :D
 aten__native_batch_norm_backward aten__native_batch_norm_backward��2��6� � w
add.42add�*
 �2 :D
 aten__native_batch_norm_backward aten__native_batch_norm_backward��2���2� � �
	p305.3330	parameter��*	
 �2    :$
xla__device_dataxla__device_dataH����

neff_input_namesinput305� �
transpose.3332	transpose��*	
 �2    :R
'aten__convolution_backward_overrideable'aten__convolution_backward_overrideabler ����� � �
convolution.3334convolution�*	
 �2    :R
'aten__convolution_backward_overrideable'aten__convolution_backward_overrideablez
(0
(0� 2 @PZb�������
  �� � �
	p311.3354	parameter�*	
 �2    :$
xla__device_dataxla__device_dataH����

neff_input_namesinput311� p
logistic.3356logistic�*	
 �2    :*
aten__silu_backwardaten__silu_backward����� � i
constant.3355constant*�:*
aten__silu_backwardaten__silu_backwardB
*�B  �?��� � r
broadcast.3360	broadcast�*	
 �2    :*
aten__silu_backwardaten__silu_backward����� � r
broadcast.3357	broadcast�*	
 �2    :*
aten__silu_backwardaten__silu_backward����� � r
subtract.3358subtract�*	
 �2    :*
aten__silu_backwardaten__silu_backward������ � r
multiply.3359multiply�*	
 �2    :*
aten__silu_backwardaten__silu_backward������ � h
add.3361add�*	
 �2    :*
aten__silu_backwardaten__silu_backward������ � r
multiply.3362multiply�*	
 �2    :*
aten__silu_backwardaten__silu_backward������ � r
multiply.3363multiply�*	
 �2    :*
aten__silu_backwardaten__silu_backward������ � �
batch-norm-grad.3371batch-norm-gradD"�*	
 �2    "�*
 �2 "�*
 �2 :D
 aten__native_batch_norm_backward aten__native_batch_norm_backward�o�:����
����2�� � �
get-tuple-element.3374get-tuple-element�*
 �2 :D
 aten__native_batch_norm_backward aten__native_batch_norm_backwardh����� � �
custom-call.379custom-call�*
 �2 :�
'xla___op_TransferWithStaticRingTransfer'xla___op_TransferWithStaticRingTransferv/home/ubuntu/aws-neuron-samples/aws_neuron_venv_pytorch/lib/python3.10/site-packages/torch_xla/core/xla_op_registry.py ,x�AwsNeuronTransferWithStaticRing��.����
 �� � �
get-tuple-element.3373get-tuple-element�*
 �2 :D
 aten__native_batch_norm_backward aten__native_batch_norm_backwardh����� � �
custom-call.380custom-call�*
 �2 :�
'xla___op_TransferWithStaticRingTransfer'xla___op_TransferWithStaticRingTransferv/home/ubuntu/aws-neuron-samples/aws_neuron_venv_pytorch/lib/python3.10/site-packages/torch_xla/core/xla_op_registry.py ,x�AwsNeuronTransferWithStaticRing��.����
 �� � �
	p313.3390	parameter�*	
 �2    :$
xla__device_dataxla__device_dataH����

neff_input_namesinput313� �
get-tuple-element.3372get-tuple-element�*	
 �2    :D
 aten__native_batch_norm_backward aten__native_batch_norm_backward����� � �
convolution.3394convolution��*	
 �2    :R
'aten__convolution_backward_overrideable'aten__convolution_backward_overrideablez
(0
(0� 28HPZb �������
  �� � �
transpose.3395	transpose��*	
 �2    :R
'aten__convolution_backward_overrideable'aten__convolution_backward_overrideabler ����� � �
custom-call.381custom-call��*	
 �2    :�
'xla___op_TransferWithStaticRingTransfer'xla___op_TransferWithStaticRingTransferv/home/ubuntu/aws-neuron-samples/aws_neuron_venv_pytorch/lib/python3.10/site-packages/torch_xla/core/xla_op_registry.py ,x�AwsNeuronTransferWithStaticRing��.����
 �� � �
	p317.3412	parameter�*	
 �2    :$
xla__device_dataxla__device_dataH����

neff_input_namesinput317� z
	p316.3411	parameter�*
 �2 :$
xla__device_dataxla__device_dataH����

neff_input_namesinput316� z
	p315.3410	parameter�*
 �2 :$
xla__device_dataxla__device_dataH����

neff_input_namesinput315� �
constant.3425constant*�:D
 aten__native_batch_norm_backward aten__native_batch_norm_backwardB
*�B  �?��� � �
broadcast.3426	broadcast�*
 �2 :D
 aten__native_batch_norm_backward aten__native_batch_norm_backward����� � z
	p314.3409	parameter�*
 �2 :$
xla__device_dataxla__device_dataH����

neff_input_namesinput314� 
divide.3427divide�*
 �2 :D
 aten__native_batch_norm_backward aten__native_batch_norm_backward������ � �
multiply.3428multiply�*
 �2 :D
 aten__native_batch_norm_backward aten__native_batch_norm_backward������ � =
constant.81constant*�: B
*�Bo����6� � �
broadcast.152	broadcast�*
 �2 :D
 aten__native_batch_norm_backward aten__native_batch_norm_backward��2��6� � w
add.43add�*
 �2 :D
 aten__native_batch_norm_backward aten__native_batch_norm_backward��2���2� � �
	p312.3389	parameter��*	
 �2    :$
xla__device_dataxla__device_dataH����

neff_input_namesinput312� �
transpose.3391	transpose��*	
 �2    :R
'aten__convolution_backward_overrideable'aten__convolution_backward_overrideabler ����� � �
convolution.3393convolution�*	
 �2    :R
'aten__convolution_backward_overrideable'aten__convolution_backward_overrideablez
(0
(0� 2 @PZb�������
  �� � w

slice.3414slice�*	
 �2    :
xla__selectxla__select���������� � �
	p318.3413	parameter�*	
 �2    :$
xla__device_dataxla__device_dataH����

neff_input_namesinput318� p
logistic.3416logistic�*	
 �2    :*
aten__silu_backwardaten__silu_backward����� � i
constant.3415constant*�:*
aten__silu_backwardaten__silu_backwardB
*�B  �?��� � r
broadcast.3420	broadcast�*	
 �2    :*
aten__silu_backwardaten__silu_backward����� � r
broadcast.3417	broadcast�*	
 �2    :*
aten__silu_backwardaten__silu_backward����� � r
subtract.3418subtract�*	
 �2    :*
aten__silu_backwardaten__silu_backward������ � r
multiply.3419multiply�*	
 �2    :*
aten__silu_backwardaten__silu_backward������ � h
add.3421add�*	
 �2    :*
aten__silu_backwardaten__silu_backward������ � r
multiply.3422multiply�*	
 �2    :*
aten__silu_backwardaten__silu_backward������ � r
multiply.3423multiply�*	
 �2    :*
aten__silu_backwardaten__silu_backward������ � �
batch-norm-grad.3431batch-norm-gradD"�*	
 �2    "�*
 �2 "�*
 �2 :D
 aten__native_batch_norm_backward aten__native_batch_norm_backward�o�:����
����2�� � �
get-tuple-element.3434get-tuple-element�*
 �2 :D
 aten__native_batch_norm_backward aten__native_batch_norm_backwardh����� � �
custom-call.382custom-call�*
 �2 :�
'xla___op_TransferWithStaticRingTransfer'xla___op_TransferWithStaticRingTransferv/home/ubuntu/aws-neuron-samples/aws_neuron_venv_pytorch/lib/python3.10/site-packages/torch_xla/core/xla_op_registry.py ,x�AwsNeuronTransferWithStaticRing��.����
 �� � �
get-tuple-element.3433get-tuple-element�*
 �2 :D
 aten__native_batch_norm_backward aten__native_batch_norm_backwardh����� � �
custom-call.383custom-call�*
 �2 :�
'xla___op_TransferWithStaticRingTransfer'xla___op_TransferWithStaticRingTransferv/home/ubuntu/aws-neuron-samples/aws_neuron_venv_pytorch/lib/python3.10/site-packages/torch_xla/core/xla_op_registry.py ,x�AwsNeuronTransferWithStaticRing��.����
 �� � �
	p320.3450	parameter�*	
 �2    :$
xla__device_dataxla__device_dataH����

neff_input_namesinput320� �
get-tuple-element.3432get-tuple-element�*	
 �2    :D
 aten__native_batch_norm_backward aten__native_batch_norm_backward����� � �
convolution.3454convolution��*	
 �2    :R
'aten__convolution_backward_overrideable'aten__convolution_backward_overrideablez
 (0
 (0� 28HPZb �������
  �� � �
transpose.3455	transpose��*	
 �2    :R
'aten__convolution_backward_overrideable'aten__convolution_backward_overrideabler ����� � �
custom-call.384custom-call��*	
 �2    :�
'xla___op_TransferWithStaticRingTransfer'xla___op_TransferWithStaticRingTransferv/home/ubuntu/aws-neuron-samples/aws_neuron_venv_pytorch/lib/python3.10/site-packages/torch_xla/core/xla_op_registry.py ,x�AwsNeuronTransferWithStaticRing��.����
 �� � �
	p324.3472	parameter�*	
 �2    :$
xla__device_dataxla__device_dataH����

neff_input_namesinput324� z
	p323.3471	parameter�*
 �2 :$
xla__device_dataxla__device_dataH����

neff_input_namesinput323� z
	p322.3470	parameter�*
 �2 :$
xla__device_dataxla__device_dataH����

neff_input_namesinput322� �
constant.3484constant*�:D
 aten__native_batch_norm_backward aten__native_batch_norm_backwardB
*�B  �?��� � �
broadcast.3485	broadcast�*
 �2 :D
 aten__native_batch_norm_backward aten__native_batch_norm_backward����� � z
	p321.3469	parameter�*
 �2 :$
xla__device_dataxla__device_dataH����

neff_input_namesinput321� 
divide.3486divide�*
 �2 :D
 aten__native_batch_norm_backward aten__native_batch_norm_backward������ � �
multiply.3487multiply�*
 �2 :D
 aten__native_batch_norm_backward aten__native_batch_norm_backward������ � =
constant.82constant*�: B
*�Bo����6� � �
broadcast.153	broadcast�*
 �2 :D
 aten__native_batch_norm_backward aten__native_batch_norm_backward��2��6� � w
add.44add�*
 �2 :D
 aten__native_batch_norm_backward aten__native_batch_norm_backward��2���2� � �
	p319.3449	parameter��*	
 �2    :$
xla__device_dataxla__device_dataH����

neff_input_namesinput319� �
transpose.3451	transpose��*	
 �2    :R
'aten__convolution_backward_overrideable'aten__convolution_backward_overrideabler ����� � �
reverse.3452reverse��*	
 �2    :R
'aten__convolution_backward_overrideable'aten__convolution_backward_overrideabler ����� � �
convolution.3453convolution�*	
 �2    :R
'aten__convolution_backward_overrideable'aten__convolution_backward_overrideablez
 (0
 (0� 2 @PZb�������
  �� � �
	p325.3473	parameter�*	
 �2    :$
xla__device_dataxla__device_dataH����

neff_input_namesinput325� p
logistic.3475logistic�*	
 �2    :*
aten__silu_backwardaten__silu_backward����� � i
constant.3474constant*�:*
aten__silu_backwardaten__silu_backwardB
*�B  �?��� � r
broadcast.3479	broadcast�*	
 �2    :*
aten__silu_backwardaten__silu_backward����� � r
broadcast.3476	broadcast�*	
 �2    :*
aten__silu_backwardaten__silu_backward����� � r
subtract.3477subtract�*	
 �2    :*
aten__silu_backwardaten__silu_backward������ � r
multiply.3478multiply�*	
 �2    :*
aten__silu_backwardaten__silu_backward������ � h
add.3480add�*	
 �2    :*
aten__silu_backwardaten__silu_backward������ � r
multiply.3481multiply�*	
 �2    :*
aten__silu_backwardaten__silu_backward������ � r
multiply.3482multiply�*	
 �2    :*
aten__silu_backwardaten__silu_backward������ � �
batch-norm-grad.3490batch-norm-gradD"�*	
 �2    "�*
 �2 "�*
 �2 :D
 aten__native_batch_norm_backward aten__native_batch_norm_backward�o�:����
����2�� � �
get-tuple-element.3493get-tuple-element�*
 �2 :D
 aten__native_batch_norm_backward aten__native_batch_norm_backwardh����� � �
custom-call.385custom-call�*
 �2 :�
'xla___op_TransferWithStaticRingTransfer'xla___op_TransferWithStaticRingTransferv/home/ubuntu/aws-neuron-samples/aws_neuron_venv_pytorch/lib/python3.10/site-packages/torch_xla/core/xla_op_registry.py ,x�AwsNeuronTransferWithStaticRing��.����
 �� � �
get-tuple-element.3492get-tuple-element�*
 �2 :D
 aten__native_batch_norm_backward aten__native_batch_norm_backwardh����� � �
custom-call.386custom-call�*
 �2 :�
'xla___op_TransferWithStaticRingTransfer'xla___op_TransferWithStaticRingTransferv/home/ubuntu/aws-neuron-samples/aws_neuron_venv_pytorch/lib/python3.10/site-packages/torch_xla/core/xla_op_registry.py ,x�AwsNeuronTransferWithStaticRing��.����
 �� � �
	p327.3509	parameter�*	
 �2    :$
xla__device_dataxla__device_dataH����

neff_input_namesinput327� �
get-tuple-element.3491get-tuple-element�*	
 �2    :D
 aten__native_batch_norm_backward aten__native_batch_norm_backward����� � �
convolution.3513convolution��*	
 �2    :R
'aten__convolution_backward_overrideable'aten__convolution_backward_overrideablez
(0
(0� 28HPZb �������
  �� � �
transpose.3514	transpose��*	
 �2    :R
'aten__convolution_backward_overrideable'aten__convolution_backward_overrideabler ����� � �
custom-call.387custom-call��*	
 �2    :�
'xla___op_TransferWithStaticRingTransfer'xla___op_TransferWithStaticRingTransferv/home/ubuntu/aws-neuron-samples/aws_neuron_venv_pytorch/lib/python3.10/site-packages/torch_xla/core/xla_op_registry.py ,x�AwsNeuronTransferWithStaticRing��.����
 �� � �
	p331.3531	parameter�*	
 �2    :$
xla__device_dataxla__device_dataH����

neff_input_namesinput331� z
	p330.3530	parameter�*
 �2 :$
xla__device_dataxla__device_dataH����

neff_input_namesinput330� z
	p329.3529	parameter�*
 �2 :$
xla__device_dataxla__device_dataH����

neff_input_namesinput329� �
constant.3544constant*�:D
 aten__native_batch_norm_backward aten__native_batch_norm_backwardB
*�B  �?��� � �
broadcast.3545	broadcast�*
 �2 :D
 aten__native_batch_norm_backward aten__native_batch_norm_backward����� � z
	p328.3528	parameter�*
 �2 :$
xla__device_dataxla__device_dataH����

neff_input_namesinput328� 
divide.3546divide�*
 �2 :D
 aten__native_batch_norm_backward aten__native_batch_norm_backward������ � �
multiply.3547multiply�*
 �2 :D
 aten__native_batch_norm_backward aten__native_batch_norm_backward������ � =
constant.83constant*�: B
*�Bo����6� � �
broadcast.155	broadcast�*
 �2 :D
 aten__native_batch_norm_backward aten__native_batch_norm_backward��2��6� � w
add.45add�*
 �2 :D
 aten__native_batch_norm_backward aten__native_batch_norm_backward��2���2� � z

slice.3533slice�*	
 �2    :
xla__selectxla__select����������� � �
	p332.3532	parameter�*	
 �2    :$
xla__device_dataxla__device_dataH����

neff_input_namesinput332� p
logistic.3535logistic�*	
 �2    :*
aten__silu_backwardaten__silu_backward����� � i
constant.3534constant*�:*
aten__silu_backwardaten__silu_backwardB
*�B  �?��� � r
broadcast.3539	broadcast�*	
 �2    :*
aten__silu_backwardaten__silu_backward����� � r
broadcast.3536	broadcast�*	
 �2    :*
aten__silu_backwardaten__silu_backward����� � r
subtract.3537subtract�*	
 �2    :*
aten__silu_backwardaten__silu_backward������ � r
multiply.3538multiply�*	
 �2    :*
aten__silu_backwardaten__silu_backward������ � h
add.3540add�*	
 �2    :*
aten__silu_backwardaten__silu_backward������ � r
multiply.3541multiply�*	
 �2    :*
aten__silu_backwardaten__silu_backward������ � r
multiply.3542multiply�*	
 �2    :*
aten__silu_backwardaten__silu_backward������ � �
batch-norm-grad.3550batch-norm-gradD"�*	
 �2    "�*
 �2 "�*
 �2 :D
 aten__native_batch_norm_backward aten__native_batch_norm_backward�o�:����
����2�� � �
get-tuple-element.3553get-tuple-element�*
 �2 :D
 aten__native_batch_norm_backward aten__native_batch_norm_backwardh����� � �
custom-call.388custom-call�*
 �2 :�
'xla___op_TransferWithStaticRingTransfer'xla___op_TransferWithStaticRingTransferv/home/ubuntu/aws-neuron-samples/aws_neuron_venv_pytorch/lib/python3.10/site-packages/torch_xla/core/xla_op_registry.py ,x�AwsNeuronTransferWithStaticRing��.����
 �� � �
get-tuple-element.3552get-tuple-element�*
 �2 :D
 aten__native_batch_norm_backward aten__native_batch_norm_backwardh����� � �
custom-call.389custom-call�*
 �2 :�
'xla___op_TransferWithStaticRingTransfer'xla___op_TransferWithStaticRingTransferv/home/ubuntu/aws-neuron-samples/aws_neuron_venv_pytorch/lib/python3.10/site-packages/torch_xla/core/xla_op_registry.py ,x�AwsNeuronTransferWithStaticRing��.����
 �� � �
	p334.3569	parameter�*	
 �2    :$
xla__device_dataxla__device_dataH����

neff_input_namesinput334� �
get-tuple-element.3551get-tuple-element�*	
 �2    :D
 aten__native_batch_norm_backward aten__native_batch_norm_backward����� � �
convolution.3573convolution��*	
 �2    :R
'aten__convolution_backward_overrideable'aten__convolution_backward_overrideablez
(0
(0� 28HPZb �������
  �� � �
transpose.3574	transpose��*	
 �2    :R
'aten__convolution_backward_overrideable'aten__convolution_backward_overrideabler ����� � �
custom-call.390custom-call��*	
 �2    :�
'xla___op_TransferWithStaticRingTransfer'xla___op_TransferWithStaticRingTransferv/home/ubuntu/aws-neuron-samples/aws_neuron_venv_pytorch/lib/python3.10/site-packages/torch_xla/core/xla_op_registry.py ,x�AwsNeuronTransferWithStaticRing��.����
 �� � �
	p338.3591	parameter�*	
 �2    :$
xla__device_dataxla__device_dataH����

neff_input_namesinput338� z
	p337.3590	parameter�*
 �2 :$
xla__device_dataxla__device_dataH����

neff_input_namesinput337� z
	p336.3589	parameter�*
 �2 :$
xla__device_dataxla__device_dataH����

neff_input_namesinput336� �
constant.3603constant*�:D
 aten__native_batch_norm_backward aten__native_batch_norm_backwardB
*�B  �?��� � �
broadcast.3604	broadcast�*
 �2 :D
 aten__native_batch_norm_backward aten__native_batch_norm_backward����� � z
	p335.3588	parameter�*
 �2 :$
xla__device_dataxla__device_dataH����

neff_input_namesinput335� 
divide.3605divide�*
 �2 :D
 aten__native_batch_norm_backward aten__native_batch_norm_backward������ � �
multiply.3606multiply�*
 �2 :D
 aten__native_batch_norm_backward aten__native_batch_norm_backward������ � =
constant.84constant*�: B
*�Bo����6� � �
broadcast.156	broadcast�*
 �2 :D
 aten__native_batch_norm_backward aten__native_batch_norm_backward��2��6� � w
add.46add�*
 �2 :D
 aten__native_batch_norm_backward aten__native_batch_norm_backward��2���2� � �
	p326.3508	parameter��*	
 �2    :$
xla__device_dataxla__device_dataH����

neff_input_namesinput326� �
transpose.3510	transpose��*	
 �2    :R
'aten__convolution_backward_overrideable'aten__convolution_backward_overrideabler ����� � �
convolution.3512convolution�*	
 �2    :R
'aten__convolution_backward_overrideable'aten__convolution_backward_overrideablez
(0
(0� 2 @PZb�������
  �� � �
	p339.3592	parameter�*	
 �2    :$
xla__device_dataxla__device_dataH����

neff_input_namesinput339� p
logistic.3594logistic�*	
 �2    :*
aten__silu_backwardaten__silu_backward����� � i
constant.3593constant*�:*
aten__silu_backwardaten__silu_backwardB
*�B  �?��� � r
broadcast.3598	broadcast�*	
 �2    :*
aten__silu_backwardaten__silu_backward����� � r
broadcast.3595	broadcast�*	
 �2    :*
aten__silu_backwardaten__silu_backward����� � r
subtract.3596subtract�*	
 �2    :*
aten__silu_backwardaten__silu_backward������ � r
multiply.3597multiply�*	
 �2    :*
aten__silu_backwardaten__silu_backward������ � h
add.3599add�*	
 �2    :*
aten__silu_backwardaten__silu_backward������ � r
multiply.3600multiply�*	
 �2    :*
aten__silu_backwardaten__silu_backward������ � r
multiply.3601multiply�*	
 �2    :*
aten__silu_backwardaten__silu_backward������ � �
batch-norm-grad.3609batch-norm-gradD"�*	
 �2    "�*
 �2 "�*
 �2 :D
 aten__native_batch_norm_backward aten__native_batch_norm_backward�o�:����
����2�� � �
get-tuple-element.3612get-tuple-element�*
 �2 :D
 aten__native_batch_norm_backward aten__native_batch_norm_backwardh����� � �
custom-call.391custom-call�*
 �2 :�
'xla___op_TransferWithStaticRingTransfer'xla___op_TransferWithStaticRingTransferv/home/ubuntu/aws-neuron-samples/aws_neuron_venv_pytorch/lib/python3.10/site-packages/torch_xla/core/xla_op_registry.py ,x�AwsNeuronTransferWithStaticRing��.����
 �� � �
get-tuple-element.3611get-tuple-element�*
 �2 :D
 aten__native_batch_norm_backward aten__native_batch_norm_backwardh����� � �
custom-call.392custom-call�*
 �2 :�
'xla___op_TransferWithStaticRingTransfer'xla___op_TransferWithStaticRingTransferv/home/ubuntu/aws-neuron-samples/aws_neuron_venv_pytorch/lib/python3.10/site-packages/torch_xla/core/xla_op_registry.py ,x�AwsNeuronTransferWithStaticRing��.����
 �� � �
get-tuple-element.3610get-tuple-element�*	
 �2    :D
 aten__native_batch_norm_backward aten__native_batch_norm_backward����� � �
convolution.3631convolution��*	
 �2    :R
'aten__convolution_backward_overrideable'aten__convolution_backward_overrideablez
(0
(0� 28HPZb �������
  �� � �
transpose.3632	transpose��*	
 �2    :R
'aten__convolution_backward_overrideable'aten__convolution_backward_overrideabler ����� � �
custom-call.393custom-call��*	
 �2    :�
'xla___op_TransferWithStaticRingTransfer'xla___op_TransferWithStaticRingTransferv/home/ubuntu/aws-neuron-samples/aws_neuron_venv_pytorch/lib/python3.10/site-packages/torch_xla/core/xla_op_registry.py ,x�AwsNeuronTransferWithStaticRing��.����
 �� � �
	p344.3649	parameter�*	
 �2    :$
xla__device_dataxla__device_dataH����

neff_input_namesinput344� z
	p343.3648	parameter�*
 �2 :$
xla__device_dataxla__device_dataH����

neff_input_namesinput343� z
	p342.3647	parameter�*
 �2 :$
xla__device_dataxla__device_dataH����

neff_input_namesinput342� �
constant.3668constant*�:D
 aten__native_batch_norm_backward aten__native_batch_norm_backwardB
*�B  �?��� � �
broadcast.3669	broadcast�*
 �2 :D
 aten__native_batch_norm_backward aten__native_batch_norm_backward����� � z
	p341.3646	parameter�*
 �2 :$
xla__device_dataxla__device_dataH����

neff_input_namesinput341� 
divide.3670divide�*
 �2 :D
 aten__native_batch_norm_backward aten__native_batch_norm_backward������ � �
multiply.3671multiply�*
 �2 :D
 aten__native_batch_norm_backward aten__native_batch_norm_backward������ � =
constant.85constant*�: B
*�Bo����6� � �
broadcast.158	broadcast�*
 �2 :D
 aten__native_batch_norm_backward aten__native_batch_norm_backward��2��6� � w
add.47add�*
 �2 :D
 aten__native_batch_norm_backward aten__native_batch_norm_backward��2���2� � �
	p333.3568	parameter��*	
 �2    :$
xla__device_dataxla__device_dataH����

neff_input_namesinput333� �
transpose.3570	transpose��*	
 �2    :R
'aten__convolution_backward_overrideable'aten__convolution_backward_overrideabler ����� � �
convolution.3572convolution�*	
 �2    :R
'aten__convolution_backward_overrideable'aten__convolution_backward_overrideablez
(0
(0� 2 @PZb�������
  �� � �
	p340.3627	parameter��*	
 �2    :$
xla__device_dataxla__device_dataH����

neff_input_namesinput340� �
transpose.3628	transpose��*	
 �2    :R
'aten__convolution_backward_overrideable'aten__convolution_backward_overrideabler ����� � �
convolution.3630convolution�*	
 �2    :R
'aten__convolution_backward_overrideable'aten__convolution_backward_overrideablez
(0
(0� 2 @PZb�������
  �� � T
add.3657add�*	
 �2    :
	aten__add	aten__add������ � �
	p345.3650	parameter�*	
 �2    :$
xla__device_dataxla__device_dataH����

neff_input_namesinput345� p
logistic.3659logistic�*	
 �2    :*
aten__silu_backwardaten__silu_backward����� � i
constant.3658constant*�:*
aten__silu_backwardaten__silu_backwardB
*�B  �?��� � r
broadcast.3663	broadcast�*	
 �2    :*
aten__silu_backwardaten__silu_backward����� � r
broadcast.3660	broadcast�*	
 �2    :*
aten__silu_backwardaten__silu_backward����� � r
subtract.3661subtract�*	
 �2    :*
aten__silu_backwardaten__silu_backward������ � r
multiply.3662multiply�*	
 �2    :*
aten__silu_backwardaten__silu_backward������ � h
add.3664add�*	
 �2    :*
aten__silu_backwardaten__silu_backward������ � r
multiply.3665multiply�*	
 �2    :*
aten__silu_backwardaten__silu_backward������ � r
multiply.3666multiply�*	
 �2    :*
aten__silu_backwardaten__silu_backward������ � �
batch-norm-grad.3674batch-norm-gradD"�*	
 �2    "�*
 �2 "�*
 �2 :D
 aten__native_batch_norm_backward aten__native_batch_norm_backward�o�:����
����2�� � �
get-tuple-element.3677get-tuple-element�*
 �2 :D
 aten__native_batch_norm_backward aten__native_batch_norm_backwardh����� � �
custom-call.394custom-call�*
 �2 :�
'xla___op_TransferWithStaticRingTransfer'xla___op_TransferWithStaticRingTransferv/home/ubuntu/aws-neuron-samples/aws_neuron_venv_pytorch/lib/python3.10/site-packages/torch_xla/core/xla_op_registry.py ,x�AwsNeuronTransferWithStaticRing��.����
 �� � �
get-tuple-element.3676get-tuple-element�*
 �2 :D
 aten__native_batch_norm_backward aten__native_batch_norm_backwardh����� � �
custom-call.395custom-call�*
 �2 :�
'xla___op_TransferWithStaticRingTransfer'xla___op_TransferWithStaticRingTransferv/home/ubuntu/aws-neuron-samples/aws_neuron_venv_pytorch/lib/python3.10/site-packages/torch_xla/core/xla_op_registry.py ,x�AwsNeuronTransferWithStaticRing��.����
 �� � �
	p347.3693	parameter�*	
 �2    :$
xla__device_dataxla__device_dataH����

neff_input_namesinput347� �
get-tuple-element.3675get-tuple-element�*	
 �2    :D
 aten__native_batch_norm_backward aten__native_batch_norm_backward����� � �
convolution.3697convolution��*	
 �2    :R
'aten__convolution_backward_overrideable'aten__convolution_backward_overrideablez
(0
(0� 28HPZb �������
  �� � �
transpose.3698	transpose��*	
 �2    :R
'aten__convolution_backward_overrideable'aten__convolution_backward_overrideabler ����� � �
custom-call.396custom-call��*	
 �2    :�
'xla___op_TransferWithStaticRingTransfer'xla___op_TransferWithStaticRingTransferv/home/ubuntu/aws-neuron-samples/aws_neuron_venv_pytorch/lib/python3.10/site-packages/torch_xla/core/xla_op_registry.py ,x�AwsNeuronTransferWithStaticRing��.����
 �� � �
	p351.3715	parameter�*	
 �2    :$
xla__device_dataxla__device_dataH����

neff_input_namesinput351� z
	p350.3714	parameter�*
 �2 :$
xla__device_dataxla__device_dataH����

neff_input_namesinput350� z
	p349.3713	parameter�*
 �2 :$
xla__device_dataxla__device_dataH����

neff_input_namesinput349� �
constant.3783constant*�:D
 aten__native_batch_norm_backward aten__native_batch_norm_backwardB
*�B  �?��� � �
broadcast.3784	broadcast�*
 �2 :D
 aten__native_batch_norm_backward aten__native_batch_norm_backward����� � z
	p348.3712	parameter�*
 �2 :$
xla__device_dataxla__device_dataH����

neff_input_namesinput348� 
divide.3785divide�*
 �2 :D
 aten__native_batch_norm_backward aten__native_batch_norm_backward������ � �
multiply.3786multiply�*
 �2 :D
 aten__native_batch_norm_backward aten__native_batch_norm_backward������ � =
constant.86constant*�: B
*�Bo����6� � �
broadcast.160	broadcast�*
 �2 :D
 aten__native_batch_norm_backward aten__native_batch_norm_backward��2��6� � w
add.48add�*
 �2 :D
 aten__native_batch_norm_backward aten__native_batch_norm_backward��2���2� � �
	p346.3692	parameter��*	
 �2    :$
xla__device_dataxla__device_dataH����

neff_input_namesinput346� �
transpose.3694	transpose��*	
 �2    :R
'aten__convolution_backward_overrideable'aten__convolution_backward_overrideabler ����� � �
convolution.3696convolution�*	
 �2    :R
'aten__convolution_backward_overrideable'aten__convolution_backward_overrideablez
(0
(0� 2 @PZb�������
  �� � w

slice.3769slice�*	
 �2    :
xla__selectxla__select���������� � �
	p353.3722	parameter�*	
 �2    :$
xla__device_dataxla__device_dataH����

neff_input_namesinput353� z

slice.3757slice�*	
 �2    :
xla__selectxla__select����������� � �
constant.3758constant*�:P
&aten__max_pool2d_with_indices_backward&aten__max_pool2d_with_indices_backwardB
*�B    ��� � �
select-and-scatter.3767select-and-scatter�*	
 �2    :P
&aten__max_pool2d_with_indices_backward&aten__max_pool2d_with_indices_backwardz0
(0
(0
 (0
 (0���������� � T
add.3770add�*	
 �2    :
	aten__add	aten__add������ � z

slice.3740slice�*	
 �2    :
xla__selectxla__select����������� � �
constant.3741constant*�:P
&aten__max_pool2d_with_indices_backward&aten__max_pool2d_with_indices_backwardB
*�B    ��� � �
select-and-scatter.3750select-and-scatter�*	
 �2    :P
&aten__max_pool2d_with_indices_backward&aten__max_pool2d_with_indices_backwardz0
(0
(0
	 (0
	 (0���������� � T
add.3771add�*	
 �2    :
	aten__add	aten__add������ � z

slice.3723slice�*	
 �2    :
xla__selectxla__select����������� � �
constant.3724constant*�:P
&aten__max_pool2d_with_indices_backward&aten__max_pool2d_with_indices_backwardB
*�B    ��� � �
select-and-scatter.3733select-and-scatter�*	
 �2    :P
&aten__max_pool2d_with_indices_backward&aten__max_pool2d_with_indices_backwardz0
(0
(0
 (0
 (0���������� � T
add.3772add�*	
 �2    :
	aten__add	aten__add������ � �
	p352.3716	parameter�*	
 �2    :$
xla__device_dataxla__device_dataH����

neff_input_namesinput352� p
logistic.3774logistic�*	
 �2    :*
aten__silu_backwardaten__silu_backward����� � i
constant.3773constant*�:*
aten__silu_backwardaten__silu_backwardB
*�B  �?��� � r
broadcast.3778	broadcast�*	
 �2    :*
aten__silu_backwardaten__silu_backward����� � r
broadcast.3775	broadcast�*	
 �2    :*
aten__silu_backwardaten__silu_backward����� � r
subtract.3776subtract�*	
 �2    :*
aten__silu_backwardaten__silu_backward������ � r
multiply.3777multiply�*	
 �2    :*
aten__silu_backwardaten__silu_backward������ � h
add.3779add�*	
 �2    :*
aten__silu_backwardaten__silu_backward������ � r
multiply.3780multiply�*	
 �2    :*
aten__silu_backwardaten__silu_backward������ � r
multiply.3781multiply�*	
 �2    :*
aten__silu_backwardaten__silu_backward������ � �
batch-norm-grad.3789batch-norm-gradD"�*	
 �2    "�*
 �2 "�*
 �2 :D
 aten__native_batch_norm_backward aten__native_batch_norm_backward�o�:����
����2�� � �
get-tuple-element.3792get-tuple-element�*
 �2 :D
 aten__native_batch_norm_backward aten__native_batch_norm_backwardh����� � �
custom-call.397custom-call�*
 �2 :�
'xla___op_TransferWithStaticRingTransfer'xla___op_TransferWithStaticRingTransferv/home/ubuntu/aws-neuron-samples/aws_neuron_venv_pytorch/lib/python3.10/site-packages/torch_xla/core/xla_op_registry.py ,x�AwsNeuronTransferWithStaticRing��.����
 �� � �
get-tuple-element.3791get-tuple-element�*
 �2 :D
 aten__native_batch_norm_backward aten__native_batch_norm_backwardh����� � �
custom-call.398custom-call�*
 �2 :�
'xla___op_TransferWithStaticRingTransfer'xla___op_TransferWithStaticRingTransferv/home/ubuntu/aws-neuron-samples/aws_neuron_venv_pytorch/lib/python3.10/site-packages/torch_xla/core/xla_op_registry.py ,x�AwsNeuronTransferWithStaticRing��.����
 �� � �
	p355.3808	parameter�*	
 �2    :$
xla__device_dataxla__device_dataH����

neff_input_namesinput355� �
get-tuple-element.3790get-tuple-element�*	
 �2    :D
 aten__native_batch_norm_backward aten__native_batch_norm_backward����� � �
convolution.3812convolution��*	
 �2    :R
'aten__convolution_backward_overrideable'aten__convolution_backward_overrideablez
(0
(0� 28HPZb �������
  �� � �
transpose.3813	transpose��*	
 �2    :R
'aten__convolution_backward_overrideable'aten__convolution_backward_overrideabler ����� � �
custom-call.399custom-call��*	
 �2    :�
'xla___op_TransferWithStaticRingTransfer'xla___op_TransferWithStaticRingTransferv/home/ubuntu/aws-neuron-samples/aws_neuron_venv_pytorch/lib/python3.10/site-packages/torch_xla/core/xla_op_registry.py ,x�AwsNeuronTransferWithStaticRing��.����
 �� � �
	p359.3830	parameter�*	
 �2    :$
xla__device_dataxla__device_dataH����

neff_input_namesinput359� z
	p358.3829	parameter�*
 �2 :$
xla__device_dataxla__device_dataH����

neff_input_namesinput358� z
	p357.3828	parameter�*
 �2 :$
xla__device_dataxla__device_dataH����

neff_input_namesinput357� �
constant.3842constant*�:D
 aten__native_batch_norm_backward aten__native_batch_norm_backwardB
*�B  �?��� � �
broadcast.3843	broadcast�*
 �2 :D
 aten__native_batch_norm_backward aten__native_batch_norm_backward����� � z
	p356.3827	parameter�*
 �2 :$
xla__device_dataxla__device_dataH����

neff_input_namesinput356� 
divide.3844divide�*
 �2 :D
 aten__native_batch_norm_backward aten__native_batch_norm_backward������ � �
multiply.3845multiply�*
 �2 :D
 aten__native_batch_norm_backward aten__native_batch_norm_backward������ � =
constant.87constant*�: B
*�Bo����6� � �
broadcast.167	broadcast�*
 �2 :D
 aten__native_batch_norm_backward aten__native_batch_norm_backward��2��6� � w
add.49add�*
 �2 :D
 aten__native_batch_norm_backward aten__native_batch_norm_backward��2���2� � �
	p354.3807	parameter��*	
 �2    :$
xla__device_dataxla__device_dataH����

neff_input_namesinput354� �
transpose.3809	transpose��*	
 �2    :R
'aten__convolution_backward_overrideable'aten__convolution_backward_overrideabler ����� � �
convolution.3811convolution�*	
 �2    :R
'aten__convolution_backward_overrideable'aten__convolution_backward_overrideablez
(0
(0� 2 @PZb�������
  �� � �
	p360.3831	parameter�*	
 �2    :$
xla__device_dataxla__device_dataH����

neff_input_namesinput360� p
logistic.3833logistic�*	
 �2    :*
aten__silu_backwardaten__silu_backward����� � i
constant.3832constant*�:*
aten__silu_backwardaten__silu_backwardB
*�B  �?��� � r
broadcast.3837	broadcast�*	
 �2    :*
aten__silu_backwardaten__silu_backward����� � r
broadcast.3834	broadcast�*	
 �2    :*
aten__silu_backwardaten__silu_backward����� � r
subtract.3835subtract�*	
 �2    :*
aten__silu_backwardaten__silu_backward������ � r
multiply.3836multiply�*	
 �2    :*
aten__silu_backwardaten__silu_backward������ � h
add.3838add�*	
 �2    :*
aten__silu_backwardaten__silu_backward������ � r
multiply.3839multiply�*	
 �2    :*
aten__silu_backwardaten__silu_backward������ � r
multiply.3840multiply�*	
 �2    :*
aten__silu_backwardaten__silu_backward������ � �
batch-norm-grad.3848batch-norm-gradD"�*	
 �2    "�*
 �2 "�*
 �2 :D
 aten__native_batch_norm_backward aten__native_batch_norm_backward�o�:����
����2�� � �
get-tuple-element.3851get-tuple-element�*
 �2 :D
 aten__native_batch_norm_backward aten__native_batch_norm_backwardh����� � �
custom-call.400custom-call�*
 �2 :�
'xla___op_TransferWithStaticRingTransfer'xla___op_TransferWithStaticRingTransferv/home/ubuntu/aws-neuron-samples/aws_neuron_venv_pytorch/lib/python3.10/site-packages/torch_xla/core/xla_op_registry.py ,x�AwsNeuronTransferWithStaticRing��.����
 �� � �
get-tuple-element.3850get-tuple-element�*
 �2 :D
 aten__native_batch_norm_backward aten__native_batch_norm_backwardh����� � �
custom-call.401custom-call�*
 �2 :�
'xla___op_TransferWithStaticRingTransfer'xla___op_TransferWithStaticRingTransferv/home/ubuntu/aws-neuron-samples/aws_neuron_venv_pytorch/lib/python3.10/site-packages/torch_xla/core/xla_op_registry.py ,x�AwsNeuronTransferWithStaticRing��.����
 �� � �
	p362.3867	parameter�((*	
 �2    :$
xla__device_dataxla__device_dataH����

neff_input_namesinput362� �
get-tuple-element.3849get-tuple-element�*	
 �2    :D
 aten__native_batch_norm_backward aten__native_batch_norm_backward����� � �
convolution.3871convolution��*	
 �2    :R
'aten__convolution_backward_overrideable'aten__convolution_backward_overrideablez

(0

(0� 28HPZb �������
  �� � �
transpose.3872	transpose��*	
 �2    :R
'aten__convolution_backward_overrideable'aten__convolution_backward_overrideabler ����� � �
custom-call.402custom-call��*	
 �2    :�
'xla___op_TransferWithStaticRingTransfer'xla___op_TransferWithStaticRingTransferv/home/ubuntu/aws-neuron-samples/aws_neuron_venv_pytorch/lib/python3.10/site-packages/torch_xla/core/xla_op_registry.py ,x�AwsNeuronTransferWithStaticRing��.����
 �� � �
	p366.3889	parameter�((*	
 �2    :$
xla__device_dataxla__device_dataH����

neff_input_namesinput366� z
	p365.3888	parameter�*
 �2 :$
xla__device_dataxla__device_dataH����

neff_input_namesinput365� z
	p364.3887	parameter�*
 �2 :$
xla__device_dataxla__device_dataH����

neff_input_namesinput364� �
constant.3909constant*�:D
 aten__native_batch_norm_backward aten__native_batch_norm_backwardB
*�B  �?��� � �
broadcast.3910	broadcast�*
 �2 :D
 aten__native_batch_norm_backward aten__native_batch_norm_backward����� � z
	p363.3886	parameter�*
 �2 :$
xla__device_dataxla__device_dataH����

neff_input_namesinput363� 
divide.3911divide�*
 �2 :D
 aten__native_batch_norm_backward aten__native_batch_norm_backward������ � �
multiply.3912multiply�*
 �2 :D
 aten__native_batch_norm_backward aten__native_batch_norm_backward������ � =
constant.88constant*�: B
*�Bo����6� � �
broadcast.168	broadcast�*
 �2 :D
 aten__native_batch_norm_backward aten__native_batch_norm_backward��2��6� � w
add.50add�*
 �2 :D
 aten__native_batch_norm_backward aten__native_batch_norm_backward��2���2� � z

slice.3897slice�((*	
 �2    :
xla__selectxla__select�����(�(����� � �
	p361.3866	parameter��*	
 �2    :$
xla__device_dataxla__device_dataH����

neff_input_namesinput361� �
transpose.3868	transpose��*	
 �2    :R
'aten__convolution_backward_overrideable'aten__convolution_backward_overrideabler ����� � �
reverse.3869reverse��*	
 �2    :R
'aten__convolution_backward_overrideable'aten__convolution_backward_overrideabler ����� � �
convolution.3870convolution�((*	
 �2    :R
'aten__convolution_backward_overrideable'aten__convolution_backward_overrideablez
 (0
 (0� 2 @PZb�������
  �� � T
add.3898add�((*	
 �2    :
	aten__add	aten__add������ � �
	p367.3890	parameter�((*	
 �2    :$
xla__device_dataxla__device_dataH����

neff_input_namesinput367� p
logistic.3900logistic�((*	
 �2    :*
aten__silu_backwardaten__silu_backward����� � i
constant.3899constant*�:*
aten__silu_backwardaten__silu_backwardB
*�B  �?��� � r
broadcast.3904	broadcast�((*	
 �2    :*
aten__silu_backwardaten__silu_backward����� � r
broadcast.3901	broadcast�((*	
 �2    :*
aten__silu_backwardaten__silu_backward����� � r
subtract.3902subtract�((*	
 �2    :*
aten__silu_backwardaten__silu_backward������ � r
multiply.3903multiply�((*	
 �2    :*
aten__silu_backwardaten__silu_backward������ � h
add.3905add�((*	
 �2    :*
aten__silu_backwardaten__silu_backward������ � r
multiply.3906multiply�((*	
 �2    :*
aten__silu_backwardaten__silu_backward������ � r
multiply.3907multiply�((*	
 �2    :*
aten__silu_backwardaten__silu_backward������ � �
batch-norm-grad.3915batch-norm-gradD"�((*	
 �2    "�*
 �2 "�*
 �2 :D
 aten__native_batch_norm_backward aten__native_batch_norm_backward�o�:����
����2�� � �
get-tuple-element.3918get-tuple-element�*
 �2 :D
 aten__native_batch_norm_backward aten__native_batch_norm_backwardh����� � �
custom-call.403custom-call�*
 �2 :�
'xla___op_TransferWithStaticRingTransfer'xla___op_TransferWithStaticRingTransferv/home/ubuntu/aws-neuron-samples/aws_neuron_venv_pytorch/lib/python3.10/site-packages/torch_xla/core/xla_op_registry.py ,x�AwsNeuronTransferWithStaticRing��.����
 �� � �
get-tuple-element.3917get-tuple-element�*
 �2 :D
 aten__native_batch_norm_backward aten__native_batch_norm_backwardh����� � �
custom-call.404custom-call�*
 �2 :�
'xla___op_TransferWithStaticRingTransfer'xla___op_TransferWithStaticRingTransferv/home/ubuntu/aws-neuron-samples/aws_neuron_venv_pytorch/lib/python3.10/site-packages/torch_xla/core/xla_op_registry.py ,x�AwsNeuronTransferWithStaticRing��.����
 �� � �
	p369.3934	parameter�((*	
 �2    :$
xla__device_dataxla__device_dataH����

neff_input_namesinput369� �
get-tuple-element.3916get-tuple-element�((*	
 �2    :D
 aten__native_batch_norm_backward aten__native_batch_norm_backward����� � �
convolution.3938convolution��*	
 �2    :R
'aten__convolution_backward_overrideable'aten__convolution_backward_overrideablez
((0
((0� 28HPZb �������
  �� � �
transpose.3939	transpose��*	
 �2    :R
'aten__convolution_backward_overrideable'aten__convolution_backward_overrideabler ����� � �
custom-call.405custom-call��*	
 �2    :�
'xla___op_TransferWithStaticRingTransfer'xla___op_TransferWithStaticRingTransferv/home/ubuntu/aws-neuron-samples/aws_neuron_venv_pytorch/lib/python3.10/site-packages/torch_xla/core/xla_op_registry.py ,x�AwsNeuronTransferWithStaticRing��.����
 �� � �
	p373.3956	parameter�((*	
 �2    :$
xla__device_dataxla__device_dataH����

neff_input_namesinput373� z
	p372.3955	parameter�*
 �2 :$
xla__device_dataxla__device_dataH����

neff_input_namesinput372� z
	p371.3954	parameter�*
 �2 :$
xla__device_dataxla__device_dataH����

neff_input_namesinput371� �
constant.3969constant*�:D
 aten__native_batch_norm_backward aten__native_batch_norm_backwardB
*�B  �?��� � �
broadcast.3970	broadcast�*
 �2 :D
 aten__native_batch_norm_backward aten__native_batch_norm_backward����� � z
	p370.3953	parameter�*
 �2 :$
xla__device_dataxla__device_dataH����

neff_input_namesinput370� 
divide.3971divide�*
 �2 :D
 aten__native_batch_norm_backward aten__native_batch_norm_backward������ � �
multiply.3972multiply�*
 �2 :D
 aten__native_batch_norm_backward aten__native_batch_norm_backward������ � =
constant.89constant*�: B
*�Bo����6� � �
broadcast.170	broadcast�*
 �2 :D
 aten__native_batch_norm_backward aten__native_batch_norm_backward��3��6� � w
add.51add�*
 �2 :D
 aten__native_batch_norm_backward aten__native_batch_norm_backward��3���3� � �
	p368.3933	parameter��*	
 �2    :$
xla__device_dataxla__device_dataH����

neff_input_namesinput368� �
transpose.3935	transpose��*	
 �2    :R
'aten__convolution_backward_overrideable'aten__convolution_backward_overrideabler ����� � �
convolution.3937convolution�((*	
 �2    :R
'aten__convolution_backward_overrideable'aten__convolution_backward_overrideablez
(0
(0� 2 @PZb�������
  �� � w

slice.3958slice�((*	
 �2    :
xla__selectxla__select����(�(����� � �
	p374.3957	parameter�((*	
 �2    :$
xla__device_dataxla__device_dataH����

neff_input_namesinput374� p
logistic.3960logistic�((*	
 �2    :*
aten__silu_backwardaten__silu_backward����� � i
constant.3959constant*�:*
aten__silu_backwardaten__silu_backwardB
*�B  �?��� � r
broadcast.3964	broadcast�((*	
 �2    :*
aten__silu_backwardaten__silu_backward����� � r
broadcast.3961	broadcast�((*	
 �2    :*
aten__silu_backwardaten__silu_backward����� � r
subtract.3962subtract�((*	
 �2    :*
aten__silu_backwardaten__silu_backward������ � r
multiply.3963multiply�((*	
 �2    :*
aten__silu_backwardaten__silu_backward������ � h
add.3965add�((*	
 �2    :*
aten__silu_backwardaten__silu_backward������ � r
multiply.3966multiply�((*	
 �2    :*
aten__silu_backwardaten__silu_backward������ � r
multiply.3967multiply�((*	
 �2    :*
aten__silu_backwardaten__silu_backward������ � �
batch-norm-grad.3975batch-norm-gradD"�((*	
 �2    "�*
 �2 "�*
 �2 :D
 aten__native_batch_norm_backward aten__native_batch_norm_backward�o�:����
����3�� � �
get-tuple-element.3978get-tuple-element�*
 �2 :D
 aten__native_batch_norm_backward aten__native_batch_norm_backwardh����� � �
custom-call.406custom-call�*
 �2 :�
'xla___op_TransferWithStaticRingTransfer'xla___op_TransferWithStaticRingTransferv/home/ubuntu/aws-neuron-samples/aws_neuron_venv_pytorch/lib/python3.10/site-packages/torch_xla/core/xla_op_registry.py ,x�AwsNeuronTransferWithStaticRing��.����
 �� � �
get-tuple-element.3977get-tuple-element�*
 �2 :D
 aten__native_batch_norm_backward aten__native_batch_norm_backwardh����� � �
custom-call.407custom-call�*
 �2 :�
'xla___op_TransferWithStaticRingTransfer'xla___op_TransferWithStaticRingTransferv/home/ubuntu/aws-neuron-samples/aws_neuron_venv_pytorch/lib/python3.10/site-packages/torch_xla/core/xla_op_registry.py ,x�AwsNeuronTransferWithStaticRing��.����
 �� � �
	p376.3994	parameter�((*	
 �2    :$
xla__device_dataxla__device_dataH����

neff_input_namesinput376� �
get-tuple-element.3976get-tuple-element�((*	
 �2    :D
 aten__native_batch_norm_backward aten__native_batch_norm_backward����� � �
convolution.3998convolution��*	
 �2    :R
'aten__convolution_backward_overrideable'aten__convolution_backward_overrideablez
( (0
( (0� 28HPZb �������
  �� � �
transpose.3999	transpose��*	
 �2    :R
'aten__convolution_backward_overrideable'aten__convolution_backward_overrideabler ����� � �
custom-call.408custom-call��*	
 �2    :�
'xla___op_TransferWithStaticRingTransfer'xla___op_TransferWithStaticRingTransferv/home/ubuntu/aws-neuron-samples/aws_neuron_venv_pytorch/lib/python3.10/site-packages/torch_xla/core/xla_op_registry.py ,x�AwsNeuronTransferWithStaticRing��.����
 �� � �
	p380.4016	parameter�((*	
 �2    :$
xla__device_dataxla__device_dataH����

neff_input_namesinput380� z
	p379.4015	parameter�*
 �2 :$
xla__device_dataxla__device_dataH����

neff_input_namesinput379� z
	p378.4014	parameter�*
 �2 :$
xla__device_dataxla__device_dataH����

neff_input_namesinput378� �
constant.4028constant*�:D
 aten__native_batch_norm_backward aten__native_batch_norm_backwardB
*�B  �?��� � �
broadcast.4029	broadcast�*
 �2 :D
 aten__native_batch_norm_backward aten__native_batch_norm_backward����� � z
	p377.4013	parameter�*
 �2 :$
xla__device_dataxla__device_dataH����

neff_input_namesinput377� 
divide.4030divide�*
 �2 :D
 aten__native_batch_norm_backward aten__native_batch_norm_backward������ � �
multiply.4031multiply�*
 �2 :D
 aten__native_batch_norm_backward aten__native_batch_norm_backward������ � =
constant.90constant*�: B
*�Bo����6� � �
broadcast.171	broadcast�*
 �2 :D
 aten__native_batch_norm_backward aten__native_batch_norm_backward��3��6� � w
add.52add�*
 �2 :D
 aten__native_batch_norm_backward aten__native_batch_norm_backward��3���3� � �
	p375.3993	parameter��*	
 �2    :$
xla__device_dataxla__device_dataH����

neff_input_namesinput375� �
transpose.3995	transpose��*	
 �2    :R
'aten__convolution_backward_overrideable'aten__convolution_backward_overrideabler ����� � �
reverse.3996reverse��*	
 �2    :R
'aten__convolution_backward_overrideable'aten__convolution_backward_overrideabler ����� � �
convolution.3997convolution�((*	
 �2    :R
'aten__convolution_backward_overrideable'aten__convolution_backward_overrideablez
 (0
 (0� 2 @PZb�������
  �� � �
	p381.4017	parameter�((*	
 �2    :$
xla__device_dataxla__device_dataH����

neff_input_namesinput381� p
logistic.4019logistic�((*	
 �2    :*
aten__silu_backwardaten__silu_backward����� � i
constant.4018constant*�:*
aten__silu_backwardaten__silu_backwardB
*�B  �?��� � r
broadcast.4023	broadcast�((*	
 �2    :*
aten__silu_backwardaten__silu_backward����� � r
broadcast.4020	broadcast�((*	
 �2    :*
aten__silu_backwardaten__silu_backward����� � r
subtract.4021subtract�((*	
 �2    :*
aten__silu_backwardaten__silu_backward������ � r
multiply.4022multiply�((*	
 �2    :*
aten__silu_backwardaten__silu_backward������ � h
add.4024add�((*	
 �2    :*
aten__silu_backwardaten__silu_backward������ � r
multiply.4025multiply�((*	
 �2    :*
aten__silu_backwardaten__silu_backward������ � r
multiply.4026multiply�((*	
 �2    :*
aten__silu_backwardaten__silu_backward������ � �
batch-norm-grad.4034batch-norm-gradD"�((*	
 �2    "�*
 �2 "�*
 �2 :D
 aten__native_batch_norm_backward aten__native_batch_norm_backward�o�:����
����3�� � �
get-tuple-element.4037get-tuple-element�*
 �2 :D
 aten__native_batch_norm_backward aten__native_batch_norm_backwardh����� � �
custom-call.409custom-call�*
 �2 :�
'xla___op_TransferWithStaticRingTransfer'xla___op_TransferWithStaticRingTransferv/home/ubuntu/aws-neuron-samples/aws_neuron_venv_pytorch/lib/python3.10/site-packages/torch_xla/core/xla_op_registry.py ,x�AwsNeuronTransferWithStaticRing��.����
 �� � �
get-tuple-element.4036get-tuple-element�*
 �2 :D
 aten__native_batch_norm_backward aten__native_batch_norm_backwardh����� � �
custom-call.410custom-call�*
 �2 :�
'xla___op_TransferWithStaticRingTransfer'xla___op_TransferWithStaticRingTransferv/home/ubuntu/aws-neuron-samples/aws_neuron_venv_pytorch/lib/python3.10/site-packages/torch_xla/core/xla_op_registry.py ,x�AwsNeuronTransferWithStaticRing��.����
 �� � �
	p383.4053	parameter�((*	
 �2    :$
xla__device_dataxla__device_dataH����

neff_input_namesinput383� �
get-tuple-element.4035get-tuple-element�((*	
 �2    :D
 aten__native_batch_norm_backward aten__native_batch_norm_backward����� � �
convolution.4057convolution��*	
 �2    :R
'aten__convolution_backward_overrideable'aten__convolution_backward_overrideablez
((0
((0� 28HPZb �������
  �� � �
transpose.4058	transpose��*	
 �2    :R
'aten__convolution_backward_overrideable'aten__convolution_backward_overrideabler ����� � �
custom-call.411custom-call��*	
 �2    :�
'xla___op_TransferWithStaticRingTransfer'xla___op_TransferWithStaticRingTransferv/home/ubuntu/aws-neuron-samples/aws_neuron_venv_pytorch/lib/python3.10/site-packages/torch_xla/core/xla_op_registry.py ,x�AwsNeuronTransferWithStaticRing��.����
 �� � �
	p387.4075	parameter�((*	
 �2    :$
xla__device_dataxla__device_dataH����

neff_input_namesinput387� z
	p386.4074	parameter�*
 �2 :$
xla__device_dataxla__device_dataH����

neff_input_namesinput386� z
	p385.4073	parameter�*
 �2 :$
xla__device_dataxla__device_dataH����

neff_input_namesinput385� �
constant.4094constant*�:D
 aten__native_batch_norm_backward aten__native_batch_norm_backwardB
*�B  �?��� � �
broadcast.4095	broadcast�*
 �2 :D
 aten__native_batch_norm_backward aten__native_batch_norm_backward����� � z
	p384.4072	parameter�*
 �2 :$
xla__device_dataxla__device_dataH����

neff_input_namesinput384� 
divide.4096divide�*
 �2 :D
 aten__native_batch_norm_backward aten__native_batch_norm_backward�� ���� � �
multiply.4097multiply�*
 �2 :D
 aten__native_batch_norm_backward aten__native_batch_norm_backward�� �� � � � =
constant.92constant*�: B
*�Bo����6� � �
broadcast.172	broadcast�*
 �2 :D
 aten__native_batch_norm_backward aten__native_batch_norm_backward��3��6� � w
add.53add�*
 �2 :D
 aten__native_batch_norm_backward aten__native_batch_norm_backward��3�� �3� � �
	p382.4052	parameter��*	
 �2    :$
xla__device_dataxla__device_dataH����

neff_input_namesinput382� �
transpose.4054	transpose��*	
 �2    :R
'aten__convolution_backward_overrideable'aten__convolution_backward_overrideabler ����� � �
convolution.4056convolution�((*	
 �2    :R
'aten__convolution_backward_overrideable'aten__convolution_backward_overrideablez
(0
(0� 2 @PZb�������
  �� � T
add.4083add�((*	
 �2    :
	aten__add	aten__add������ � �
	p388.4076	parameter�((*	
 �2    :$
xla__device_dataxla__device_dataH����

neff_input_namesinput388� p
logistic.4085logistic�((*	
 �2    :*
aten__silu_backwardaten__silu_backward����� � i
constant.4084constant*�:*
aten__silu_backwardaten__silu_backwardB
*�B  �?��� � r
broadcast.4089	broadcast�((*	
 �2    :*
aten__silu_backwardaten__silu_backward����� � r
broadcast.4086	broadcast�((*	
 �2    :*
aten__silu_backwardaten__silu_backward����� � r
subtract.4087subtract�((*	
 �2    :*
aten__silu_backwardaten__silu_backward������ � r
multiply.4088multiply�((*	
 �2    :*
aten__silu_backwardaten__silu_backward������ � h
add.4090add�((*	
 �2    :*
aten__silu_backwardaten__silu_backward������ � r
multiply.4091multiply�((*	
 �2    :*
aten__silu_backwardaten__silu_backward������ � r
multiply.4092multiply�((*	
 �2    :*
aten__silu_backwardaten__silu_backward������ � �
batch-norm-grad.4100batch-norm-gradD"�((*	
 �2    "�*
 �2 "�*
 �2 :D
 aten__native_batch_norm_backward aten__native_batch_norm_backward�o�:��� �
����3�� � �
get-tuple-element.4103get-tuple-element�*
 �2 :D
 aten__native_batch_norm_backward aten__native_batch_norm_backwardh�� �� � � �
custom-call.412custom-call�*
 �2 :�
'xla___op_TransferWithStaticRingTransfer'xla___op_TransferWithStaticRingTransferv/home/ubuntu/aws-neuron-samples/aws_neuron_venv_pytorch/lib/python3.10/site-packages/torch_xla/core/xla_op_registry.py ,x�AwsNeuronTransferWithStaticRing��.�� ��
 �� � �
get-tuple-element.4102get-tuple-element�*
 �2 :D
 aten__native_batch_norm_backward aten__native_batch_norm_backwardh�� �� � � �
custom-call.413custom-call�*
 �2 :�
'xla___op_TransferWithStaticRingTransfer'xla___op_TransferWithStaticRingTransferv/home/ubuntu/aws-neuron-samples/aws_neuron_venv_pytorch/lib/python3.10/site-packages/torch_xla/core/xla_op_registry.py ,x�AwsNeuronTransferWithStaticRing��.�� ��
 �� � �
	p390.4119	parameter�((*	
 �2    :$
xla__device_dataxla__device_dataH��� �

neff_input_namesinput390� �
get-tuple-element.4101get-tuple-element�((*	
 �2    :D
 aten__native_batch_norm_backward aten__native_batch_norm_backward�� �� � � �
convolution.4123convolution��*	
 �2    :R
'aten__convolution_backward_overrideable'aten__convolution_backward_overrideablez
( (0
( (0� 28HPZb �� �� � ��
  �� � �
transpose.4124	transpose��*	
 �2    :R
'aten__convolution_backward_overrideable'aten__convolution_backward_overrideabler �� �� � � �
custom-call.414custom-call��*	
 �2    :�
'xla___op_TransferWithStaticRingTransfer'xla___op_TransferWithStaticRingTransferv/home/ubuntu/aws-neuron-samples/aws_neuron_venv_pytorch/lib/python3.10/site-packages/torch_xla/core/xla_op_registry.py ,x�AwsNeuronTransferWithStaticRing��.�� ��
 �� � �
	p394.4141	parameter�((*	
 �2    :$
xla__device_dataxla__device_dataH��� �

neff_input_namesinput394� z
	p393.4140	parameter�*
 �2 :$
xla__device_dataxla__device_dataH��� �

neff_input_namesinput393� z
	p392.4139	parameter�*
 �2 :$
xla__device_dataxla__device_dataH��� �

neff_input_namesinput392� �
constant.4153constant*�:D
 aten__native_batch_norm_backward aten__native_batch_norm_backwardB
*�B  �?�� � � �
broadcast.4154	broadcast�*
 �2 :D
 aten__native_batch_norm_backward aten__native_batch_norm_backward�� �� � � z
	p391.4138	parameter�*
 �2 :$
xla__device_dataxla__device_dataH��� �

neff_input_namesinput391� 
divide.4155divide�*
 �2 :D
 aten__native_batch_norm_backward aten__native_batch_norm_backward�� �� � � � �
multiply.4156multiply�*
 �2 :D
 aten__native_batch_norm_backward aten__native_batch_norm_backward�� �� � � � =
constant.93constant*�: B
*�Bo����6� � �
broadcast.176	broadcast�*
 �2 :D
 aten__native_batch_norm_backward aten__native_batch_norm_backward��3��6� � w
add.54add�*
 �2 :D
 aten__native_batch_norm_backward aten__native_batch_norm_backward��3�� �3� � �
	p389.4118	parameter��*	
 �2    :$
xla__device_dataxla__device_dataH��� �

neff_input_namesinput389� �
transpose.4120	transpose��*	
 �2    :R
'aten__convolution_backward_overrideable'aten__convolution_backward_overrideabler �� �� � � �
reverse.4121reverse��*	
 �2    :R
'aten__convolution_backward_overrideable'aten__convolution_backward_overrideabler �� �� � � �
convolution.4122convolution�((*	
 �2    :R
'aten__convolution_backward_overrideable'aten__convolution_backward_overrideablez
 (0
 (0� 2 @PZb�� �� � ��
  �� � �
	p395.4142	parameter�((*	
 �2    :$
xla__device_dataxla__device_dataH��� �

neff_input_namesinput395� p
logistic.4144logistic�((*	
 �2    :*
aten__silu_backwardaten__silu_backward�� �� � � i
constant.4143constant*�:*
aten__silu_backwardaten__silu_backwardB
*�B  �?�� � � r
broadcast.4148	broadcast�((*	
 �2    :*
aten__silu_backwardaten__silu_backward�� �� � � r
broadcast.4145	broadcast�((*	
 �2    :*
aten__silu_backwardaten__silu_backward�� �� � � r
subtract.4146subtract�((*	
 �2    :*
aten__silu_backwardaten__silu_backward�� �� � � � r
multiply.4147multiply�((*	
 �2    :*
aten__silu_backwardaten__silu_backward�� �� � � � h
add.4149add�((*	
 �2    :*
aten__silu_backwardaten__silu_backward�� �� � � � r
multiply.4150multiply�((*	
 �2    :*
aten__silu_backwardaten__silu_backward�� �� � � � r
multiply.4151multiply�((*	
 �2    :*
aten__silu_backwardaten__silu_backward�� �� � � � �
batch-norm-grad.4159batch-norm-gradD"�((*	
 �2    "�*
 �2 "�*
 �2 :D
 aten__native_batch_norm_backward aten__native_batch_norm_backward�o�:��� �
� � � �3� � � �
get-tuple-element.4162get-tuple-element�*
 �2 :D
 aten__native_batch_norm_backward aten__native_batch_norm_backwardh�� �� � � �
custom-call.415custom-call�*
 �2 :�
'xla___op_TransferWithStaticRingTransfer'xla___op_TransferWithStaticRingTransferv/home/ubuntu/aws-neuron-samples/aws_neuron_venv_pytorch/lib/python3.10/site-packages/torch_xla/core/xla_op_registry.py ,x�AwsNeuronTransferWithStaticRing��.�� ��
 �� � �
get-tuple-element.4161get-tuple-element�*
 �2 :D
 aten__native_batch_norm_backward aten__native_batch_norm_backwardh�� �� � � �
custom-call.416custom-call�*
 �2 :�
'xla___op_TransferWithStaticRingTransfer'xla___op_TransferWithStaticRingTransferv/home/ubuntu/aws-neuron-samples/aws_neuron_venv_pytorch/lib/python3.10/site-packages/torch_xla/core/xla_op_registry.py ,x�AwsNeuronTransferWithStaticRing��.�� ��
 �� � �
	p397.4178	parameter�((*	
 �2    :$
xla__device_dataxla__device_dataH��� �

neff_input_namesinput397� �
get-tuple-element.4160get-tuple-element�((*	
 �2    :D
 aten__native_batch_norm_backward aten__native_batch_norm_backward�� �� � � �
convolution.4182convolution��*	
 �2    :R
'aten__convolution_backward_overrideable'aten__convolution_backward_overrideablez
((0
((0� 28HPZb �� �� � ��
  �� � �
transpose.4183	transpose��*	
 �2    :R
'aten__convolution_backward_overrideable'aten__convolution_backward_overrideabler �� �� � � �
custom-call.417custom-call��*	
 �2    :�
'xla___op_TransferWithStaticRingTransfer'xla___op_TransferWithStaticRingTransferv/home/ubuntu/aws-neuron-samples/aws_neuron_venv_pytorch/lib/python3.10/site-packages/torch_xla/core/xla_op_registry.py ,x�AwsNeuronTransferWithStaticRing��.�� ��
 �� � �
	p401.4200	parameter�((*	
 �2    :$
xla__device_dataxla__device_dataH��� �

neff_input_namesinput401� z
	p400.4199	parameter�*
 �2 :$
xla__device_dataxla__device_dataH��� �

neff_input_namesinput400� z
	p399.4198	parameter�*
 �2 :$
xla__device_dataxla__device_dataH��� �

neff_input_namesinput399� �
constant.4219constant*�:D
 aten__native_batch_norm_backward aten__native_batch_norm_backwardB
*�B  �?�� � � �
broadcast.4220	broadcast�*
 �2 :D
 aten__native_batch_norm_backward aten__native_batch_norm_backward�� �� � � z
	p398.4197	parameter�*
 �2 :$
xla__device_dataxla__device_dataH��� �

neff_input_namesinput398� 
divide.4221divide�*
 �2 :D
 aten__native_batch_norm_backward aten__native_batch_norm_backward�� �� � � � �
multiply.4222multiply�*
 �2 :D
 aten__native_batch_norm_backward aten__native_batch_norm_backward�� �� � � � =
constant.95constant*�: B
*�Bo����6� � �
broadcast.177	broadcast�*
 �2 :D
 aten__native_batch_norm_backward aten__native_batch_norm_backward��3��6� � w
add.55add�*
 �2 :D
 aten__native_batch_norm_backward aten__native_batch_norm_backward��3�� �3� � �
	p396.4177	parameter��*	
 �2    :$
xla__device_dataxla__device_dataH��� �

neff_input_namesinput396� �
transpose.4179	transpose��*	
 �2    :R
'aten__convolution_backward_overrideable'aten__convolution_backward_overrideabler �� �� � � �
convolution.4181convolution�((*	
 �2    :R
'aten__convolution_backward_overrideable'aten__convolution_backward_overrideablez
(0
(0� 2 @PZb�� �� � ��
  �� � T
add.4208add�((*	
 �2    :
	aten__add	aten__add�� ��� � � �
	p402.4201	parameter�((*	
 �2    :$
xla__device_dataxla__device_dataH��� �

neff_input_namesinput402� p
logistic.4210logistic�((*	
 �2    :*
aten__silu_backwardaten__silu_backward�� �� � � i
constant.4209constant*�:*
aten__silu_backwardaten__silu_backwardB
*�B  �?�� � � r
broadcast.4214	broadcast�((*	
 �2    :*
aten__silu_backwardaten__silu_backward�� �� � � r
broadcast.4211	broadcast�((*	
 �2    :*
aten__silu_backwardaten__silu_backward�� �� � � r
subtract.4212subtract�((*	
 �2    :*
aten__silu_backwardaten__silu_backward�� �� � � � r
multiply.4213multiply�((*	
 �2    :*
aten__silu_backwardaten__silu_backward�� �� � � � h
add.4215add�((*	
 �2    :*
aten__silu_backwardaten__silu_backward�� �� � � � r
multiply.4216multiply�((*	
 �2    :*
aten__silu_backwardaten__silu_backward�� �� � � � r
multiply.4217multiply�((*	
 �2    :*
aten__silu_backwardaten__silu_backward�� �� � � � �
batch-norm-grad.4225batch-norm-gradD"�((*	
 �2    "�*
 �2 "�*
 �2 :D
 aten__native_batch_norm_backward aten__native_batch_norm_backward�o�:���!�
� � � �3� � � �
get-tuple-element.4228get-tuple-element�*
 �2 :D
 aten__native_batch_norm_backward aten__native_batch_norm_backwardh��!��!� � �
custom-call.418custom-call�*
 �2 :�
'xla___op_TransferWithStaticRingTransfer'xla___op_TransferWithStaticRingTransferv/home/ubuntu/aws-neuron-samples/aws_neuron_venv_pytorch/lib/python3.10/site-packages/torch_xla/core/xla_op_registry.py ,x�AwsNeuronTransferWithStaticRing��.��!��
 �� � �
get-tuple-element.4227get-tuple-element�*
 �2 :D
 aten__native_batch_norm_backward aten__native_batch_norm_backwardh��!��!� � �
custom-call.419custom-call�*
 �2 :�
'xla___op_TransferWithStaticRingTransfer'xla___op_TransferWithStaticRingTransferv/home/ubuntu/aws-neuron-samples/aws_neuron_venv_pytorch/lib/python3.10/site-packages/torch_xla/core/xla_op_registry.py ,x�AwsNeuronTransferWithStaticRing��.��!��
 �� � �
	p404.4244	parameter�((*	
 �2    :$
xla__device_dataxla__device_dataH���!�

neff_input_namesinput404� �
get-tuple-element.4226get-tuple-element�((*	
 �2    :D
 aten__native_batch_norm_backward aten__native_batch_norm_backward��!��!� � �
convolution.4248convolution��*	
 �2    :R
'aten__convolution_backward_overrideable'aten__convolution_backward_overrideablez
( (0
( (0� 28HPZb ��!��!�!��
  �� � �
transpose.4249	transpose��*	
 �2    :R
'aten__convolution_backward_overrideable'aten__convolution_backward_overrideabler ��!��!� � �
custom-call.420custom-call��*	
 �2    :�
'xla___op_TransferWithStaticRingTransfer'xla___op_TransferWithStaticRingTransferv/home/ubuntu/aws-neuron-samples/aws_neuron_venv_pytorch/lib/python3.10/site-packages/torch_xla/core/xla_op_registry.py ,x�AwsNeuronTransferWithStaticRing��.��!��
 �� � �
	p408.4266	parameter�((*	
 �2    :$
xla__device_dataxla__device_dataH���!�

neff_input_namesinput408� z
	p407.4265	parameter�*
 �2 :$
xla__device_dataxla__device_dataH���!�

neff_input_namesinput407� z
	p406.4264	parameter�*
 �2 :$
xla__device_dataxla__device_dataH���!�

neff_input_namesinput406� �
constant.4278constant*�:D
 aten__native_batch_norm_backward aten__native_batch_norm_backwardB
*�B  �?��!� � �
broadcast.4279	broadcast�*
 �2 :D
 aten__native_batch_norm_backward aten__native_batch_norm_backward��!��!� � z
	p405.4263	parameter�*
 �2 :$
xla__device_dataxla__device_dataH���!�

neff_input_namesinput405� 
divide.4280divide�*
 �2 :D
 aten__native_batch_norm_backward aten__native_batch_norm_backward��!��!�!� � �
multiply.4281multiply�*
 �2 :D
 aten__native_batch_norm_backward aten__native_batch_norm_backward��!��!�!� � =
constant.96constant*�: B
*�Bo����6� � �
broadcast.180	broadcast�*
 �2 :D
 aten__native_batch_norm_backward aten__native_batch_norm_backward��3��6� � w
add.56add�*
 �2 :D
 aten__native_batch_norm_backward aten__native_batch_norm_backward��3��!�3� � �
	p403.4243	parameter��*	
 �2    :$
xla__device_dataxla__device_dataH���!�

neff_input_namesinput403� �
transpose.4245	transpose��*	
 �2    :R
'aten__convolution_backward_overrideable'aten__convolution_backward_overrideabler ��!��!� � �
reverse.4246reverse��*	
 �2    :R
'aten__convolution_backward_overrideable'aten__convolution_backward_overrideabler ��!��!� � �
convolution.4247convolution�((*	
 �2    :R
'aten__convolution_backward_overrideable'aten__convolution_backward_overrideablez
 (0
 (0� 2 @PZb��!��!�!��
  �� � �
	p409.4267	parameter�((*	
 �2    :$
xla__device_dataxla__device_dataH���!�

neff_input_namesinput409� p
logistic.4269logistic�((*	
 �2    :*
aten__silu_backwardaten__silu_backward��!��!� � i
constant.4268constant*�:*
aten__silu_backwardaten__silu_backwardB
*�B  �?��!� � r
broadcast.4273	broadcast�((*	
 �2    :*
aten__silu_backwardaten__silu_backward��!��!� � r
broadcast.4270	broadcast�((*	
 �2    :*
aten__silu_backwardaten__silu_backward��!��!� � r
subtract.4271subtract�((*	
 �2    :*
aten__silu_backwardaten__silu_backward��!��!�!� � r
multiply.4272multiply�((*	
 �2    :*
aten__silu_backwardaten__silu_backward��!��!�!� � h
add.4274add�((*	
 �2    :*
aten__silu_backwardaten__silu_backward��!��!�!� � r
multiply.4275multiply�((*	
 �2    :*
aten__silu_backwardaten__silu_backward��!��!�!� � r
multiply.4276multiply�((*	
 �2    :*
aten__silu_backwardaten__silu_backward��!��!�!� � �
batch-norm-grad.4284batch-norm-gradD"�((*	
 �2    "�*
 �2 "�*
 �2 :D
 aten__native_batch_norm_backward aten__native_batch_norm_backward�o�:���!�
�!�!�!�3�!� � �
get-tuple-element.4287get-tuple-element�*
 �2 :D
 aten__native_batch_norm_backward aten__native_batch_norm_backwardh��!��!� � �
custom-call.421custom-call�*
 �2 :�
'xla___op_TransferWithStaticRingTransfer'xla___op_TransferWithStaticRingTransferv/home/ubuntu/aws-neuron-samples/aws_neuron_venv_pytorch/lib/python3.10/site-packages/torch_xla/core/xla_op_registry.py ,x�AwsNeuronTransferWithStaticRing��.��!��
 �� � �
get-tuple-element.4286get-tuple-element�*
 �2 :D
 aten__native_batch_norm_backward aten__native_batch_norm_backwardh��!��!� � �
custom-call.422custom-call�*
 �2 :�
'xla___op_TransferWithStaticRingTransfer'xla___op_TransferWithStaticRingTransferv/home/ubuntu/aws-neuron-samples/aws_neuron_venv_pytorch/lib/python3.10/site-packages/torch_xla/core/xla_op_registry.py ,x�AwsNeuronTransferWithStaticRing��.��!��
 �� � �
	p411.4303	parameter�((*	
 �2    :$
xla__device_dataxla__device_dataH���!�

neff_input_namesinput411� �
get-tuple-element.4285get-tuple-element�((*	
 �2    :D
 aten__native_batch_norm_backward aten__native_batch_norm_backward��!��!� � �
convolution.4307convolution��*	
 �2    :R
'aten__convolution_backward_overrideable'aten__convolution_backward_overrideablez
((0
((0� 28HPZb ��!��!�!��
  �� � �
transpose.4308	transpose��*	
 �2    :R
'aten__convolution_backward_overrideable'aten__convolution_backward_overrideabler ��!��!� � �
custom-call.423custom-call��*	
 �2    :�
'xla___op_TransferWithStaticRingTransfer'xla___op_TransferWithStaticRingTransferv/home/ubuntu/aws-neuron-samples/aws_neuron_venv_pytorch/lib/python3.10/site-packages/torch_xla/core/xla_op_registry.py ,x�AwsNeuronTransferWithStaticRing��.��!��
 �� � �
	p415.4325	parameter�((*	
 �2    :$
xla__device_dataxla__device_dataH���!�

neff_input_namesinput415� z
	p414.4324	parameter�*
 �2 :$
xla__device_dataxla__device_dataH���!�

neff_input_namesinput414� z
	p413.4323	parameter�*
 �2 :$
xla__device_dataxla__device_dataH���!�

neff_input_namesinput413� �
constant.4338constant*�:D
 aten__native_batch_norm_backward aten__native_batch_norm_backwardB
*�B  �?��!� � �
broadcast.4339	broadcast�*
 �2 :D
 aten__native_batch_norm_backward aten__native_batch_norm_backward��!��!� � z
	p412.4322	parameter�*
 �2 :$
xla__device_dataxla__device_dataH���!�

neff_input_namesinput412� 
divide.4340divide�*
 �2 :D
 aten__native_batch_norm_backward aten__native_batch_norm_backward��!��!�!� � �
multiply.4341multiply�*
 �2 :D
 aten__native_batch_norm_backward aten__native_batch_norm_backward��!��!�!� � =
constant.98constant*�: B
*�Bo����6� � �
broadcast.183	broadcast�*
 �2 :D
 aten__native_batch_norm_backward aten__native_batch_norm_backward��3��6� � w
add.57add�*
 �2 :D
 aten__native_batch_norm_backward aten__native_batch_norm_backward��3��!�3� � z

slice.4327slice�((*	
 �2    :
xla__selectxla__select�����(�(��!��� � �
	p416.4326	parameter�((*	
 �2    :$
xla__device_dataxla__device_dataH���!�

neff_input_namesinput416� p
logistic.4329logistic�((*	
 �2    :*
aten__silu_backwardaten__silu_backward��!��!� � i
constant.4328constant*�:*
aten__silu_backwardaten__silu_backwardB
*�B  �?��!� � r
broadcast.4333	broadcast�((*	
 �2    :*
aten__silu_backwardaten__silu_backward��!��!� � r
broadcast.4330	broadcast�((*	
 �2    :*
aten__silu_backwardaten__silu_backward��!��!� � r
subtract.4331subtract�((*	
 �2    :*
aten__silu_backwardaten__silu_backward��!��!�!� � r
multiply.4332multiply�((*	
 �2    :*
aten__silu_backwardaten__silu_backward��!��!�!� � h
add.4334add�((*	
 �2    :*
aten__silu_backwardaten__silu_backward��!��!�!� � r
multiply.4335multiply�((*	
 �2    :*
aten__silu_backwardaten__silu_backward��!��!�!� � r
multiply.4336multiply�((*	
 �2    :*
aten__silu_backwardaten__silu_backward��!��!�!� � �
batch-norm-grad.4344batch-norm-gradD"�((*	
 �2    "�*
 �2 "�*
 �2 :D
 aten__native_batch_norm_backward aten__native_batch_norm_backward�o�:���!�
�!�!�!�3�!� � �
get-tuple-element.4347get-tuple-element�*
 �2 :D
 aten__native_batch_norm_backward aten__native_batch_norm_backwardh��!��!� � �
custom-call.424custom-call�*
 �2 :�
'xla___op_TransferWithStaticRingTransfer'xla___op_TransferWithStaticRingTransferv/home/ubuntu/aws-neuron-samples/aws_neuron_venv_pytorch/lib/python3.10/site-packages/torch_xla/core/xla_op_registry.py ,x�AwsNeuronTransferWithStaticRing��.��!��
 �� � �
get-tuple-element.4346get-tuple-element�*
 �2 :D
 aten__native_batch_norm_backward aten__native_batch_norm_backwardh��!��!� � �
custom-call.425custom-call�*
 �2 :�
'xla___op_TransferWithStaticRingTransfer'xla___op_TransferWithStaticRingTransferv/home/ubuntu/aws-neuron-samples/aws_neuron_venv_pytorch/lib/python3.10/site-packages/torch_xla/core/xla_op_registry.py ,x�AwsNeuronTransferWithStaticRing��.��!��
 �� � �
	p418.4363	parameter�((*	
 �2    :$
xla__device_dataxla__device_dataH���"�

neff_input_namesinput418� �
get-tuple-element.4345get-tuple-element�((*	
 �2    :D
 aten__native_batch_norm_backward aten__native_batch_norm_backward��!��!� � �
convolution.4367convolution��*	
 �2    :R
'aten__convolution_backward_overrideable'aten__convolution_backward_overrideablez
((0
((0� 28HPZb ��"��"�!��
  �� � �
transpose.4368	transpose��*	
 �2    :R
'aten__convolution_backward_overrideable'aten__convolution_backward_overrideabler ��"��"� � �
custom-call.426custom-call��*	
 �2    :�
'xla___op_TransferWithStaticRingTransfer'xla___op_TransferWithStaticRingTransferv/home/ubuntu/aws-neuron-samples/aws_neuron_venv_pytorch/lib/python3.10/site-packages/torch_xla/core/xla_op_registry.py ,x�AwsNeuronTransferWithStaticRing��.��"��
 �� � �
	p422.4385	parameter�((*	
 �2    :$
xla__device_dataxla__device_dataH���"�

neff_input_namesinput422� z
	p421.4384	parameter�*
 �2 :$
xla__device_dataxla__device_dataH���"�

neff_input_namesinput421� z
	p420.4383	parameter�*
 �2 :$
xla__device_dataxla__device_dataH���"�

neff_input_namesinput420� �
constant.4404constant*�:D
 aten__native_batch_norm_backward aten__native_batch_norm_backwardB
*�B  �?��"� � �
broadcast.4405	broadcast�*
 �2 :D
 aten__native_batch_norm_backward aten__native_batch_norm_backward��"��"� � z
	p419.4382	parameter�*
 �2 :$
xla__device_dataxla__device_dataH���"�

neff_input_namesinput419� 
divide.4406divide�*
 �2 :D
 aten__native_batch_norm_backward aten__native_batch_norm_backward��"��"�"� � �
multiply.4407multiply�*
 �2 :D
 aten__native_batch_norm_backward aten__native_batch_norm_backward��"��"�"� � =
constant.99constant*�: B
*�Bo����6� � �
broadcast.184	broadcast�*
 �2 :D
 aten__native_batch_norm_backward aten__native_batch_norm_backward��3��6� � w
add.58add�*
 �2 :D
 aten__native_batch_norm_backward aten__native_batch_norm_backward��3��"�3� � �
	p410.4302	parameter��*	
 �2    :$
xla__device_dataxla__device_dataH���!�

neff_input_namesinput410� �
transpose.4304	transpose��*	
 �2    :R
'aten__convolution_backward_overrideable'aten__convolution_backward_overrideabler ��!��!� � �
convolution.4306convolution�((*	
 �2    :R
'aten__convolution_backward_overrideable'aten__convolution_backward_overrideablez
(0
(0� 2 @PZb��!��!�!��
  �� � T
add.4393add�((*	
 �2    :
	aten__add	aten__add��"�� �!� � �
	p423.4386	parameter�((*	
 �2    :$
xla__device_dataxla__device_dataH���"�

neff_input_namesinput423� p
logistic.4395logistic�((*	
 �2    :*
aten__silu_backwardaten__silu_backward��"��"� � i
constant.4394constant*�:*
aten__silu_backwardaten__silu_backwardB
*�B  �?��"� � r
broadcast.4399	broadcast�((*	
 �2    :*
aten__silu_backwardaten__silu_backward��"��"� � r
broadcast.4396	broadcast�((*	
 �2    :*
aten__silu_backwardaten__silu_backward��"��"� � r
subtract.4397subtract�((*	
 �2    :*
aten__silu_backwardaten__silu_backward��"��"�"� � r
multiply.4398multiply�((*	
 �2    :*
aten__silu_backwardaten__silu_backward��"��"�"� � h
add.4400add�((*	
 �2    :*
aten__silu_backwardaten__silu_backward��"��"�"� � r
multiply.4401multiply�((*	
 �2    :*
aten__silu_backwardaten__silu_backward��"��"�"� � r
multiply.4402multiply�((*	
 �2    :*
aten__silu_backwardaten__silu_backward��"��"�"� � �
batch-norm-grad.4410batch-norm-gradD"�((*	
 �2    "�*
 �2 "�*
 �2 :D
 aten__native_batch_norm_backward aten__native_batch_norm_backward�o�:���"�
�"�"�"�3�"� � �
get-tuple-element.4413get-tuple-element�*
 �2 :D
 aten__native_batch_norm_backward aten__native_batch_norm_backwardh��"��"� � �
custom-call.427custom-call�*
 �2 :�
'xla___op_TransferWithStaticRingTransfer'xla___op_TransferWithStaticRingTransferv/home/ubuntu/aws-neuron-samples/aws_neuron_venv_pytorch/lib/python3.10/site-packages/torch_xla/core/xla_op_registry.py ,x�AwsNeuronTransferWithStaticRing��.��"��
 �� � �
get-tuple-element.4412get-tuple-element�*
 �2 :D
 aten__native_batch_norm_backward aten__native_batch_norm_backwardh��"��"� � �
custom-call.428custom-call�*
 �2 :�
'xla___op_TransferWithStaticRingTransfer'xla___op_TransferWithStaticRingTransferv/home/ubuntu/aws-neuron-samples/aws_neuron_venv_pytorch/lib/python3.10/site-packages/torch_xla/core/xla_op_registry.py ,x�AwsNeuronTransferWithStaticRing��.��"��
 �� � �
get-tuple-element.4411get-tuple-element�((*	
 �2    :D
 aten__native_batch_norm_backward aten__native_batch_norm_backward��"��"� � �
convolution.4432convolution��*	
 �2    :R
'aten__convolution_backward_overrideable'aten__convolution_backward_overrideablez
((0
((0� 28HPZb ��"��"�"��
  �� � �
transpose.4433	transpose��*	
 �2    :R
'aten__convolution_backward_overrideable'aten__convolution_backward_overrideabler ��"��"� � �
custom-call.429custom-call��*	
 �2    :�
'xla___op_TransferWithStaticRingTransfer'xla___op_TransferWithStaticRingTransferv/home/ubuntu/aws-neuron-samples/aws_neuron_venv_pytorch/lib/python3.10/site-packages/torch_xla/core/xla_op_registry.py ,x�AwsNeuronTransferWithStaticRing��.��"��
 �� � �
	p428.4450	parameter�((*	
 �2    :$
xla__device_dataxla__device_dataH���"�

neff_input_namesinput428� z
	p427.4449	parameter�*
 �2 :$
xla__device_dataxla__device_dataH���"�

neff_input_namesinput427� z
	p426.4448	parameter�*
 �2 :$
xla__device_dataxla__device_dataH���"�

neff_input_namesinput426� �
constant.4469constant*�:D
 aten__native_batch_norm_backward aten__native_batch_norm_backwardB
*�B  �?��"� � �
broadcast.4470	broadcast�*
 �2 :D
 aten__native_batch_norm_backward aten__native_batch_norm_backward��"��"� � z
	p425.4447	parameter�*
 �2 :$
xla__device_dataxla__device_dataH���"�

neff_input_namesinput425� 
divide.4471divide�*
 �2 :D
 aten__native_batch_norm_backward aten__native_batch_norm_backward��"��"�"� � �
multiply.4472multiply�*
 �2 :D
 aten__native_batch_norm_backward aten__native_batch_norm_backward��"��"�"� � >
constant.100constant*�: B
*�Bo����6� � �
broadcast.187	broadcast�*
 �2 :D
 aten__native_batch_norm_backward aten__native_batch_norm_backward��3��6� � w
add.59add�*
 �2 :D
 aten__native_batch_norm_backward aten__native_batch_norm_backward��3��"�3� � �
	p417.4362	parameter��*	
 �2    :$
xla__device_dataxla__device_dataH���"�

neff_input_namesinput417� �
transpose.4364	transpose��*	
 �2    :R
'aten__convolution_backward_overrideable'aten__convolution_backward_overrideabler ��"��"� � �
convolution.4366convolution�((*	
 �2    :R
'aten__convolution_backward_overrideable'aten__convolution_backward_overrideablez
(0
(0� 2 @PZb��"��!�"��
  �� � �
	p424.4428	parameter��*	
 �2    :$
xla__device_dataxla__device_dataH���"�

neff_input_namesinput424� �
transpose.4429	transpose��*	
 �2    :R
'aten__convolution_backward_overrideable'aten__convolution_backward_overrideabler ��"��"� � �
convolution.4431convolution�((*	
 �2    :R
'aten__convolution_backward_overrideable'aten__convolution_backward_overrideablez
(0
(0� 2 @PZb��"��"�"��
  �� � T
add.4458add�((*	
 �2    :
	aten__add	aten__add��"��"�"� � �
	p429.4451	parameter�((*	
 �2    :$
xla__device_dataxla__device_dataH���"�

neff_input_namesinput429� p
logistic.4460logistic�((*	
 �2    :*
aten__silu_backwardaten__silu_backward��"��"� � i
constant.4459constant*�:*
aten__silu_backwardaten__silu_backwardB
*�B  �?��"� � r
broadcast.4464	broadcast�((*	
 �2    :*
aten__silu_backwardaten__silu_backward��"��"� � r
broadcast.4461	broadcast�((*	
 �2    :*
aten__silu_backwardaten__silu_backward��"��"� � r
subtract.4462subtract�((*	
 �2    :*
aten__silu_backwardaten__silu_backward��"��"�"� � r
multiply.4463multiply�((*	
 �2    :*
aten__silu_backwardaten__silu_backward��"��"�"� � h
add.4465add�((*	
 �2    :*
aten__silu_backwardaten__silu_backward��"��"�"� � r
multiply.4466multiply�((*	
 �2    :*
aten__silu_backwardaten__silu_backward��"��"�"� � r
multiply.4467multiply�((*	
 �2    :*
aten__silu_backwardaten__silu_backward��"��"�"� � �
batch-norm-grad.4475batch-norm-gradD"�((*	
 �2    "�*
 �2 "�*
 �2 :D
 aten__native_batch_norm_backward aten__native_batch_norm_backward�o�:���"�
�"�"�"�3�"� � �
get-tuple-element.4478get-tuple-element�*
 �2 :D
 aten__native_batch_norm_backward aten__native_batch_norm_backwardh��"��"� � �
custom-call.430custom-call�*
 �2 :�
'xla___op_TransferWithStaticRingTransfer'xla___op_TransferWithStaticRingTransferv/home/ubuntu/aws-neuron-samples/aws_neuron_venv_pytorch/lib/python3.10/site-packages/torch_xla/core/xla_op_registry.py ,x�AwsNeuronTransferWithStaticRing��.��"��
 �� � �
get-tuple-element.4477get-tuple-element�*
 �2 :D
 aten__native_batch_norm_backward aten__native_batch_norm_backwardh��"��"� � �
custom-call.431custom-call�*
 �2 :�
'xla___op_TransferWithStaticRingTransfer'xla___op_TransferWithStaticRingTransferv/home/ubuntu/aws-neuron-samples/aws_neuron_venv_pytorch/lib/python3.10/site-packages/torch_xla/core/xla_op_registry.py ,x�AwsNeuronTransferWithStaticRing��.��"��
 �� � �
	p431.4494	parameter�PP*	
 �2    :$
xla__device_dataxla__device_dataH���#�

neff_input_namesinput431� �
get-tuple-element.4476get-tuple-element�((*	
 �2    :D
 aten__native_batch_norm_backward aten__native_batch_norm_backward��"��"� � �
convolution.4498convolution��*	
 �2    :R
'aten__convolution_backward_overrideable'aten__convolution_backward_overrideablez

((0

((0� 28HPZb ��#��#�"��
  �� � �
transpose.4499	transpose��*	
 �2    :R
'aten__convolution_backward_overrideable'aten__convolution_backward_overrideabler ��#��#� � �
custom-call.432custom-call��*	
 �2    :�
'xla___op_TransferWithStaticRingTransfer'xla___op_TransferWithStaticRingTransferv/home/ubuntu/aws-neuron-samples/aws_neuron_venv_pytorch/lib/python3.10/site-packages/torch_xla/core/xla_op_registry.py ,x�AwsNeuronTransferWithStaticRing��.��#��
 �� � �
	p435.4516	parameter�PP*	
 �2    :$
xla__device_dataxla__device_dataH���#�

neff_input_namesinput435� z
	p434.4515	parameter�*
 �2 :$
xla__device_dataxla__device_dataH���#�

neff_input_namesinput434� z
	p433.4514	parameter�*
 �2 :$
xla__device_dataxla__device_dataH���#�

neff_input_namesinput433� �
constant.4536constant*�:D
 aten__native_batch_norm_backward aten__native_batch_norm_backwardB
*�B  �?��#� � �
broadcast.4537	broadcast�*
 �2 :D
 aten__native_batch_norm_backward aten__native_batch_norm_backward��#��#� � z
	p432.4513	parameter�*
 �2 :$
xla__device_dataxla__device_dataH���#�

neff_input_namesinput432� 
divide.4538divide�*
 �2 :D
 aten__native_batch_norm_backward aten__native_batch_norm_backward��#��#�#� � �
multiply.4539multiply�*
 �2 :D
 aten__native_batch_norm_backward aten__native_batch_norm_backward��#��#�#� � >
constant.101constant*�: B
*�Bo����6� � �
broadcast.191	broadcast�*
 �2 :D
 aten__native_batch_norm_backward aten__native_batch_norm_backward��3��6� � w
add.60add�*
 �2 :D
 aten__native_batch_norm_backward aten__native_batch_norm_backward��3��#�3� � z

slice.4524slice�PP*	
 �2    :
xla__selectxla__select�����P�P��#��� � �
	p430.4493	parameter��*	
 �2    :$
xla__device_dataxla__device_dataH���#�

neff_input_namesinput430� �
transpose.4495	transpose��*	
 �2    :R
'aten__convolution_backward_overrideable'aten__convolution_backward_overrideabler ��#��#� � �
reverse.4496reverse��*	
 �2    :R
'aten__convolution_backward_overrideable'aten__convolution_backward_overrideabler ��#��#� � �
convolution.4497convolution�PP*	
 �2    :R
'aten__convolution_backward_overrideable'aten__convolution_backward_overrideablez
 (0
 (0� 2 @PZb��#��"�#��
  �� � T
add.4525add�PP*	
 �2    :
	aten__add	aten__add��#��#�#� � �
	p436.4517	parameter�PP*	
 �2    :$
xla__device_dataxla__device_dataH���#�

neff_input_namesinput436� p
logistic.4527logistic�PP*	
 �2    :*
aten__silu_backwardaten__silu_backward��#��#� � i
constant.4526constant*�:*
aten__silu_backwardaten__silu_backwardB
*�B  �?��#� � r
broadcast.4531	broadcast�PP*	
 �2    :*
aten__silu_backwardaten__silu_backward��#��#� � r
broadcast.4528	broadcast�PP*	
 �2    :*
aten__silu_backwardaten__silu_backward��#��#� � r
subtract.4529subtract�PP*	
 �2    :*
aten__silu_backwardaten__silu_backward��#��#�#� � r
multiply.4530multiply�PP*	
 �2    :*
aten__silu_backwardaten__silu_backward��#��#�#� � h
add.4532add�PP*	
 �2    :*
aten__silu_backwardaten__silu_backward��#��#�#� � r
multiply.4533multiply�PP*	
 �2    :*
aten__silu_backwardaten__silu_backward��#��#�#� � r
multiply.4534multiply�PP*	
 �2    :*
aten__silu_backwardaten__silu_backward��#��#�#� � �
batch-norm-grad.4542batch-norm-gradD"�PP*	
 �2    "�*
 �2 "�*
 �2 :D
 aten__native_batch_norm_backward aten__native_batch_norm_backward�o�:���#�
�#�#�#�3�#� � �
get-tuple-element.4545get-tuple-element�*
 �2 :D
 aten__native_batch_norm_backward aten__native_batch_norm_backwardh��#��#� � �
custom-call.433custom-call�*
 �2 :�
'xla___op_TransferWithStaticRingTransfer'xla___op_TransferWithStaticRingTransferv/home/ubuntu/aws-neuron-samples/aws_neuron_venv_pytorch/lib/python3.10/site-packages/torch_xla/core/xla_op_registry.py ,x�AwsNeuronTransferWithStaticRing��.��#��
 �� � �
get-tuple-element.4544get-tuple-element�*
 �2 :D
 aten__native_batch_norm_backward aten__native_batch_norm_backwardh��#��#� � �
custom-call.434custom-call�*
 �2 :�
'xla___op_TransferWithStaticRingTransfer'xla___op_TransferWithStaticRingTransferv/home/ubuntu/aws-neuron-samples/aws_neuron_venv_pytorch/lib/python3.10/site-packages/torch_xla/core/xla_op_registry.py ,x�AwsNeuronTransferWithStaticRing��.��#��
 �� � �
	p438.4561	parameter�PP*	
 �2    :$
xla__device_dataxla__device_dataH���#�

neff_input_namesinput438� �
get-tuple-element.4543get-tuple-element�PP*	
 �2    :D
 aten__native_batch_norm_backward aten__native_batch_norm_backward��#��#� � �
convolution.4565convolution��*	
 �2    :R
'aten__convolution_backward_overrideable'aten__convolution_backward_overrideablez
P(0
P(0� 28HPZb ��#��#�#��
  �� � �
transpose.4566	transpose��*	
 �2    :R
'aten__convolution_backward_overrideable'aten__convolution_backward_overrideabler ��#��#� � �
custom-call.435custom-call��*	
 �2    :�
'xla___op_TransferWithStaticRingTransfer'xla___op_TransferWithStaticRingTransferv/home/ubuntu/aws-neuron-samples/aws_neuron_venv_pytorch/lib/python3.10/site-packages/torch_xla/core/xla_op_registry.py ,x�AwsNeuronTransferWithStaticRing��.��#��
 �� � �
	p442.4583	parameter@PP*	
 �2    :$
xla__device_dataxla__device_dataH���#�

neff_input_namesinput442� y
	p441.4582	parameter@*
 �2 :$
xla__device_dataxla__device_dataH���#�

neff_input_namesinput441� y
	p440.4581	parameter@*
 �2 :$
xla__device_dataxla__device_dataH���#�

neff_input_namesinput440� �
constant.4596constant*�:D
 aten__native_batch_norm_backward aten__native_batch_norm_backwardB
*�B  �?��#� � �
broadcast.4597	broadcast@*
 �2 :D
 aten__native_batch_norm_backward aten__native_batch_norm_backward��#��#� � y
	p439.4580	parameter@*
 �2 :$
xla__device_dataxla__device_dataH���#�

neff_input_namesinput439� ~
divide.4598divide@*
 �2 :D
 aten__native_batch_norm_backward aten__native_batch_norm_backward��#��#�#� � �
multiply.4599multiply@*
 �2 :D
 aten__native_batch_norm_backward aten__native_batch_norm_backward��#��#�#� � >
constant.103constant*�: B
*�Bo����6� � �
broadcast.193	broadcast@*
 �2 :D
 aten__native_batch_norm_backward aten__native_batch_norm_backward��3��6� � v
add.61add@*
 �2 :D
 aten__native_batch_norm_backward aten__native_batch_norm_backward��3��#�3� � �
	p437.4560	parameter��*	
 �2    :$
xla__device_dataxla__device_dataH���#�

neff_input_namesinput437� �
transpose.4562	transpose��*	
 �2    :R
'aten__convolution_backward_overrideable'aten__convolution_backward_overrideabler ��#��#� � �
convolution.4564convolution�PP*	
 �2    :R
'aten__convolution_backward_overrideable'aten__convolution_backward_overrideablez
(0
(0� 2 @PZb��#��#�#��
  �� � u

slice.4585slice@PP*	
 �2    :
xla__selectxla__select��@�P�P��#��#� � �
	p443.4584	parameter@PP*	
 �2    :$
xla__device_dataxla__device_dataH���#�

neff_input_namesinput443� o
logistic.4587logistic@PP*	
 �2    :*
aten__silu_backwardaten__silu_backward��#��#� � i
constant.4586constant*�:*
aten__silu_backwardaten__silu_backwardB
*�B  �?��#� � q
broadcast.4591	broadcast@PP*	
 �2    :*
aten__silu_backwardaten__silu_backward��#��#� � q
broadcast.4588	broadcast@PP*	
 �2    :*
aten__silu_backwardaten__silu_backward��#��#� � q
subtract.4589subtract@PP*	
 �2    :*
aten__silu_backwardaten__silu_backward��#��#�#� � q
multiply.4590multiply@PP*	
 �2    :*
aten__silu_backwardaten__silu_backward��#��#�#� � g
add.4592add@PP*	
 �2    :*
aten__silu_backwardaten__silu_backward��#��#�#� � q
multiply.4593multiply@PP*	
 �2    :*
aten__silu_backwardaten__silu_backward��#��#�#� � q
multiply.4594multiply@PP*	
 �2    :*
aten__silu_backwardaten__silu_backward��#��#�#� � �
batch-norm-grad.4602batch-norm-gradA"@PP*	
 �2    "@*
 �2 "@*
 �2 :D
 aten__native_batch_norm_backward aten__native_batch_norm_backward�o�:���#�
�#�#�#�3�#� � �
get-tuple-element.4605get-tuple-element@*
 �2 :D
 aten__native_batch_norm_backward aten__native_batch_norm_backwardh��#��#� � �
custom-call.436custom-call@*
 �2 :�
'xla___op_TransferWithStaticRingTransfer'xla___op_TransferWithStaticRingTransferv/home/ubuntu/aws-neuron-samples/aws_neuron_venv_pytorch/lib/python3.10/site-packages/torch_xla/core/xla_op_registry.py ,x�AwsNeuronTransferWithStaticRing��.��#��
 �� � �
get-tuple-element.4604get-tuple-element@*
 �2 :D
 aten__native_batch_norm_backward aten__native_batch_norm_backwardh��#��#� � �
custom-call.437custom-call@*
 �2 :�
'xla___op_TransferWithStaticRingTransfer'xla___op_TransferWithStaticRingTransferv/home/ubuntu/aws-neuron-samples/aws_neuron_venv_pytorch/lib/python3.10/site-packages/torch_xla/core/xla_op_registry.py ,x�AwsNeuronTransferWithStaticRing��.��#��
 �� � �
	p445.4621	parameter@PP*	
 �2    :$
xla__device_dataxla__device_dataH���$�

neff_input_namesinput445� �
get-tuple-element.4603get-tuple-element@PP*	
 �2    :D
 aten__native_batch_norm_backward aten__native_batch_norm_backward��#��#� � �
convolution.4625convolution@@*	
 �2    :R
'aten__convolution_backward_overrideable'aten__convolution_backward_overrideablez
P (0
P (0� 28HPZb ��$��$�#��
  �� � �
transpose.4626	transpose@@*	
 �2    :R
'aten__convolution_backward_overrideable'aten__convolution_backward_overrideabler ��$��$� � �
custom-call.438custom-call@@*	
 �2    :�
'xla___op_TransferWithStaticRingTransfer'xla___op_TransferWithStaticRingTransferv/home/ubuntu/aws-neuron-samples/aws_neuron_venv_pytorch/lib/python3.10/site-packages/torch_xla/core/xla_op_registry.py ,x�AwsNeuronTransferWithStaticRing��.��$��
 �� � �
	p449.4643	parameter@PP*	
 �2    :$
xla__device_dataxla__device_dataH���$�

neff_input_namesinput449� y
	p448.4642	parameter@*
 �2 :$
xla__device_dataxla__device_dataH���$�

neff_input_namesinput448� y
	p447.4641	parameter@*
 �2 :$
xla__device_dataxla__device_dataH���$�

neff_input_namesinput447� �
constant.4655constant*�:D
 aten__native_batch_norm_backward aten__native_batch_norm_backwardB
*�B  �?��$� � �
broadcast.4656	broadcast@*
 �2 :D
 aten__native_batch_norm_backward aten__native_batch_norm_backward��$��$� � y
	p446.4640	parameter@*
 �2 :$
xla__device_dataxla__device_dataH���$�

neff_input_namesinput446� ~
divide.4657divide@*
 �2 :D
 aten__native_batch_norm_backward aten__native_batch_norm_backward��$��$�$� � �
multiply.4658multiply@*
 �2 :D
 aten__native_batch_norm_backward aten__native_batch_norm_backward��$��$�$� � >
constant.106constant*�: B
*�Bo����6� � �
broadcast.194	broadcast@*
 �2 :D
 aten__native_batch_norm_backward aten__native_batch_norm_backward��3��6� � v
add.62add@*
 �2 :D
 aten__native_batch_norm_backward aten__native_batch_norm_backward��3��$�3� � �
	p444.4620	parameter@@*	
 �2    :$
xla__device_dataxla__device_dataH���$�

neff_input_namesinput444� �
transpose.4622	transpose@@*	
 �2    :R
'aten__convolution_backward_overrideable'aten__convolution_backward_overrideabler ��$��$� � �
reverse.4623reverse@@*	
 �2    :R
'aten__convolution_backward_overrideable'aten__convolution_backward_overrideabler ��$��$� � �
convolution.4624convolution@PP*	
 �2    :R
'aten__convolution_backward_overrideable'aten__convolution_backward_overrideablez
 (0
 (0� 2 @PZb��$��#�$��
  �� � �
	p450.4644	parameter@PP*	
 �2    :$
xla__device_dataxla__device_dataH���$�

neff_input_namesinput450� o
logistic.4646logistic@PP*	
 �2    :*
aten__silu_backwardaten__silu_backward��$��$� � i
constant.4645constant*�:*
aten__silu_backwardaten__silu_backwardB
*�B  �?��$� � q
broadcast.4650	broadcast@PP*	
 �2    :*
aten__silu_backwardaten__silu_backward��$��$� � q
broadcast.4647	broadcast@PP*	
 �2    :*
aten__silu_backwardaten__silu_backward��$��$� � q
subtract.4648subtract@PP*	
 �2    :*
aten__silu_backwardaten__silu_backward��$��$�$� � q
multiply.4649multiply@PP*	
 �2    :*
aten__silu_backwardaten__silu_backward��$��$�$� � g
add.4651add@PP*	
 �2    :*
aten__silu_backwardaten__silu_backward��$��$�$� � q
multiply.4652multiply@PP*	
 �2    :*
aten__silu_backwardaten__silu_backward��$��$�$� � q
multiply.4653multiply@PP*	
 �2    :*
aten__silu_backwardaten__silu_backward��$��$�$� � �
batch-norm-grad.4661batch-norm-gradA"@PP*	
 �2    "@*
 �2 "@*
 �2 :D
 aten__native_batch_norm_backward aten__native_batch_norm_backward�o�:���$�
�$�$�$�3�$� � �
get-tuple-element.4664get-tuple-element@*
 �2 :D
 aten__native_batch_norm_backward aten__native_batch_norm_backwardh��$��$� � �
custom-call.439custom-call@*
 �2 :�
'xla___op_TransferWithStaticRingTransfer'xla___op_TransferWithStaticRingTransferv/home/ubuntu/aws-neuron-samples/aws_neuron_venv_pytorch/lib/python3.10/site-packages/torch_xla/core/xla_op_registry.py ,x�AwsNeuronTransferWithStaticRing��.��$��
 �� � �
get-tuple-element.4663get-tuple-element@*
 �2 :D
 aten__native_batch_norm_backward aten__native_batch_norm_backwardh��$��$� � �
custom-call.440custom-call@*
 �2 :�
'xla___op_TransferWithStaticRingTransfer'xla___op_TransferWithStaticRingTransferv/home/ubuntu/aws-neuron-samples/aws_neuron_venv_pytorch/lib/python3.10/site-packages/torch_xla/core/xla_op_registry.py ,x�AwsNeuronTransferWithStaticRing��.��$��
 �� � �
	p452.4680	parameter@PP*	
 �2    :$
xla__device_dataxla__device_dataH���$�

neff_input_namesinput452� �
get-tuple-element.4662get-tuple-element@PP*	
 �2    :D
 aten__native_batch_norm_backward aten__native_batch_norm_backward��$��$� � �
convolution.4684convolution@@*	
 �2    :R
'aten__convolution_backward_overrideable'aten__convolution_backward_overrideablez
P(0
P(0� 28HPZb ��$��$�$��
  �� � �
transpose.4685	transpose@@*	
 �2    :R
'aten__convolution_backward_overrideable'aten__convolution_backward_overrideabler ��$��$� � �
custom-call.441custom-call@@*	
 �2    :�
'xla___op_TransferWithStaticRingTransfer'xla___op_TransferWithStaticRingTransferv/home/ubuntu/aws-neuron-samples/aws_neuron_venv_pytorch/lib/python3.10/site-packages/torch_xla/core/xla_op_registry.py ,x�AwsNeuronTransferWithStaticRing��.��$��
 �� � �
	p456.4702	parameter@PP*	
 �2    :$
xla__device_dataxla__device_dataH���$�

neff_input_namesinput456� y
	p455.4701	parameter@*
 �2 :$
xla__device_dataxla__device_dataH���$�

neff_input_namesinput455� y
	p454.4700	parameter@*
 �2 :$
xla__device_dataxla__device_dataH���$�

neff_input_namesinput454� �
constant.4721constant*�:D
 aten__native_batch_norm_backward aten__native_batch_norm_backwardB
*�B  �?��$� � �
broadcast.4722	broadcast@*
 �2 :D
 aten__native_batch_norm_backward aten__native_batch_norm_backward��$��$� � y
	p453.4699	parameter@*
 �2 :$
xla__device_dataxla__device_dataH���$�

neff_input_namesinput453� ~
divide.4723divide@*
 �2 :D
 aten__native_batch_norm_backward aten__native_batch_norm_backward��$��$�$� � �
multiply.4724multiply@*
 �2 :D
 aten__native_batch_norm_backward aten__native_batch_norm_backward��$��$�$� � >
constant.107constant*�: B
*�Bo����6� � �
broadcast.195	broadcast@*
 �2 :D
 aten__native_batch_norm_backward aten__native_batch_norm_backward��3��6� � v
add.63add@*
 �2 :D
 aten__native_batch_norm_backward aten__native_batch_norm_backward��3��$�3� � �
	p451.4679	parameter@@*	
 �2    :$
xla__device_dataxla__device_dataH���$�

neff_input_namesinput451� �
transpose.4681	transpose@@*	
 �2    :R
'aten__convolution_backward_overrideable'aten__convolution_backward_overrideabler ��$��$� � �
convolution.4683convolution@PP*	
 �2    :R
'aten__convolution_backward_overrideable'aten__convolution_backward_overrideablez
(0
(0� 2 @PZb��$��$�$��
  �� � S
add.4710add@PP*	
 �2    :
	aten__add	aten__add��$��#�$� � �
	p457.4703	parameter@PP*	
 �2    :$
xla__device_dataxla__device_dataH���$�

neff_input_namesinput457� o
logistic.4712logistic@PP*	
 �2    :*
aten__silu_backwardaten__silu_backward��$��$� � i
constant.4711constant*�:*
aten__silu_backwardaten__silu_backwardB
*�B  �?��$� � q
broadcast.4716	broadcast@PP*	
 �2    :*
aten__silu_backwardaten__silu_backward��$��$� � q
broadcast.4713	broadcast@PP*	
 �2    :*
aten__silu_backwardaten__silu_backward��$��$� � q
subtract.4714subtract@PP*	
 �2    :*
aten__silu_backwardaten__silu_backward��$��$�$� � q
multiply.4715multiply@PP*	
 �2    :*
aten__silu_backwardaten__silu_backward��$��$�$� � g
add.4717add@PP*	
 �2    :*
aten__silu_backwardaten__silu_backward��$��$�$� � q
multiply.4718multiply@PP*	
 �2    :*
aten__silu_backwardaten__silu_backward��$��$�$� � q
multiply.4719multiply@PP*	
 �2    :*
aten__silu_backwardaten__silu_backward��$��$�$� � �
batch-norm-grad.4727batch-norm-gradA"@PP*	
 �2    "@*
 �2 "@*
 �2 :D
 aten__native_batch_norm_backward aten__native_batch_norm_backward�o�:���$�
�$�$�$�3�$� � �
get-tuple-element.4730get-tuple-element@*
 �2 :D
 aten__native_batch_norm_backward aten__native_batch_norm_backwardh��$��$� � �
custom-call.442custom-call@*
 �2 :�
'xla___op_TransferWithStaticRingTransfer'xla___op_TransferWithStaticRingTransferv/home/ubuntu/aws-neuron-samples/aws_neuron_venv_pytorch/lib/python3.10/site-packages/torch_xla/core/xla_op_registry.py ,x�AwsNeuronTransferWithStaticRing��.��$��
 �� � �
get-tuple-element.4729get-tuple-element@*
 �2 :D
 aten__native_batch_norm_backward aten__native_batch_norm_backwardh��$��$� � �
custom-call.443custom-call@*
 �2 :�
'xla___op_TransferWithStaticRingTransfer'xla___op_TransferWithStaticRingTransferv/home/ubuntu/aws-neuron-samples/aws_neuron_venv_pytorch/lib/python3.10/site-packages/torch_xla/core/xla_op_registry.py ,x�AwsNeuronTransferWithStaticRing��.��$��
 �� � �
	p459.4746	parameter@PP*	
 �2    :$
xla__device_dataxla__device_dataH���%�

neff_input_namesinput459� �
get-tuple-element.4728get-tuple-element@PP*	
 �2    :D
 aten__native_batch_norm_backward aten__native_batch_norm_backward��$��$� � �
convolution.4750convolution@@*	
 �2    :R
'aten__convolution_backward_overrideable'aten__convolution_backward_overrideablez
P (0
P (0� 28HPZb ��%��%�$��
  �� � �
transpose.4751	transpose@@*	
 �2    :R
'aten__convolution_backward_overrideable'aten__convolution_backward_overrideabler ��%��%� � �
custom-call.444custom-call@@*	
 �2    :�
'xla___op_TransferWithStaticRingTransfer'xla___op_TransferWithStaticRingTransferv/home/ubuntu/aws-neuron-samples/aws_neuron_venv_pytorch/lib/python3.10/site-packages/torch_xla/core/xla_op_registry.py ,x�AwsNeuronTransferWithStaticRing��.��%��
 �� � �
	p463.4768	parameter@PP*	
 �2    :$
xla__device_dataxla__device_dataH���%�

neff_input_namesinput463� y
	p462.4767	parameter@*
 �2 :$
xla__device_dataxla__device_dataH���%�

neff_input_namesinput462� y
	p461.4766	parameter@*
 �2 :$
xla__device_dataxla__device_dataH���%�

neff_input_namesinput461� �
constant.4780constant*�:D
 aten__native_batch_norm_backward aten__native_batch_norm_backwardB
*�B  �?��%� � �
broadcast.4781	broadcast@*
 �2 :D
 aten__native_batch_norm_backward aten__native_batch_norm_backward��%��%� � y
	p460.4765	parameter@*
 �2 :$
xla__device_dataxla__device_dataH���%�

neff_input_namesinput460� ~
divide.4782divide@*
 �2 :D
 aten__native_batch_norm_backward aten__native_batch_norm_backward��%��%�%� � �
multiply.4783multiply@*
 �2 :D
 aten__native_batch_norm_backward aten__native_batch_norm_backward��%��%�%� � >
constant.108constant*�: B
*�Bo����6� � �
broadcast.197	broadcast@*
 �2 :D
 aten__native_batch_norm_backward aten__native_batch_norm_backward��3��6� � v
add.64add@*
 �2 :D
 aten__native_batch_norm_backward aten__native_batch_norm_backward��3��%�3� � �
	p458.4745	parameter@@*	
 �2    :$
xla__device_dataxla__device_dataH���%�

neff_input_namesinput458� �
transpose.4747	transpose@@*	
 �2    :R
'aten__convolution_backward_overrideable'aten__convolution_backward_overrideabler ��%��%� � �
reverse.4748reverse@@*	
 �2    :R
'aten__convolution_backward_overrideable'aten__convolution_backward_overrideabler ��%��%� � �
convolution.4749convolution@PP*	
 �2    :R
'aten__convolution_backward_overrideable'aten__convolution_backward_overrideablez
 (0
 (0� 2 @PZb��%��$�%��
  �� � �
	p464.4769	parameter@PP*	
 �2    :$
xla__device_dataxla__device_dataH���%�

neff_input_namesinput464� o
logistic.4771logistic@PP*	
 �2    :*
aten__silu_backwardaten__silu_backward��%��%� � i
constant.4770constant*�:*
aten__silu_backwardaten__silu_backwardB
*�B  �?��%� � q
broadcast.4775	broadcast@PP*	
 �2    :*
aten__silu_backwardaten__silu_backward��%��%� � q
broadcast.4772	broadcast@PP*	
 �2    :*
aten__silu_backwardaten__silu_backward��%��%� � q
subtract.4773subtract@PP*	
 �2    :*
aten__silu_backwardaten__silu_backward��%��%�%� � q
multiply.4774multiply@PP*	
 �2    :*
aten__silu_backwardaten__silu_backward��%��%�%� � g
add.4776add@PP*	
 �2    :*
aten__silu_backwardaten__silu_backward��%��%�%� � q
multiply.4777multiply@PP*	
 �2    :*
aten__silu_backwardaten__silu_backward��%��%�%� � q
multiply.4778multiply@PP*	
 �2    :*
aten__silu_backwardaten__silu_backward��%��%�%� � �
batch-norm-grad.4786batch-norm-gradA"@PP*	
 �2    "@*
 �2 "@*
 �2 :D
 aten__native_batch_norm_backward aten__native_batch_norm_backward�o�:���%�
�%�%�%�3�%� � �
get-tuple-element.4789get-tuple-element@*
 �2 :D
 aten__native_batch_norm_backward aten__native_batch_norm_backwardh��%��%� � �
custom-call.445custom-call@*
 �2 :�
'xla___op_TransferWithStaticRingTransfer'xla___op_TransferWithStaticRingTransferv/home/ubuntu/aws-neuron-samples/aws_neuron_venv_pytorch/lib/python3.10/site-packages/torch_xla/core/xla_op_registry.py ,x�AwsNeuronTransferWithStaticRing��.��%��
 �� � �
get-tuple-element.4788get-tuple-element@*
 �2 :D
 aten__native_batch_norm_backward aten__native_batch_norm_backwardh��%��%� � �
custom-call.446custom-call@*
 �2 :�
'xla___op_TransferWithStaticRingTransfer'xla___op_TransferWithStaticRingTransferv/home/ubuntu/aws-neuron-samples/aws_neuron_venv_pytorch/lib/python3.10/site-packages/torch_xla/core/xla_op_registry.py ,x�AwsNeuronTransferWithStaticRing��/��%��
 �� � �
	p466.4805	parameter@PP*	
 �2    :$
xla__device_dataxla__device_dataH���%�

neff_input_namesinput466� �
get-tuple-element.4787get-tuple-element@PP*	
 �2    :D
 aten__native_batch_norm_backward aten__native_batch_norm_backward��%��%� � �
convolution.4809convolution@@*	
 �2    :R
'aten__convolution_backward_overrideable'aten__convolution_backward_overrideablez
P(0
P(0� 28HPZb ��%��%�%��
  �� � �
transpose.4810	transpose@@*	
 �2    :R
'aten__convolution_backward_overrideable'aten__convolution_backward_overrideabler ��%��%� � �
custom-call.447custom-call@@*	
 �2    :�
'xla___op_TransferWithStaticRingTransfer'xla___op_TransferWithStaticRingTransferv/home/ubuntu/aws-neuron-samples/aws_neuron_venv_pytorch/lib/python3.10/site-packages/torch_xla/core/xla_op_registry.py ,x�AwsNeuronTransferWithStaticRing��/��%��
 �� � �
	p470.4827	parameter@PP*	
 �2    :$
xla__device_dataxla__device_dataH���%�

neff_input_namesinput470� y
	p469.4826	parameter@*
 �2 :$
xla__device_dataxla__device_dataH���%�

neff_input_namesinput469� y
	p468.4825	parameter@*
 �2 :$
xla__device_dataxla__device_dataH���%�

neff_input_namesinput468� �
constant.4846constant*�:D
 aten__native_batch_norm_backward aten__native_batch_norm_backwardB
*�B  �?��%� � �
broadcast.4847	broadcast@*
 �2 :D
 aten__native_batch_norm_backward aten__native_batch_norm_backward��%��%� � y
	p467.4824	parameter@*
 �2 :$
xla__device_dataxla__device_dataH���%�

neff_input_namesinput467� ~
divide.4848divide@*
 �2 :D
 aten__native_batch_norm_backward aten__native_batch_norm_backward��%��%�%� � �
multiply.4849multiply@*
 �2 :D
 aten__native_batch_norm_backward aten__native_batch_norm_backward��%��%�%� � >
constant.110constant*�: B
*�Bo����6� � �
broadcast.199	broadcast@*
 �2 :D
 aten__native_batch_norm_backward aten__native_batch_norm_backward��3��6� � v
add.65add@*
 �2 :D
 aten__native_batch_norm_backward aten__native_batch_norm_backward��3��%�3� � �
	p465.4804	parameter@@*	
 �2    :$
xla__device_dataxla__device_dataH���%�

neff_input_namesinput465� �
transpose.4806	transpose@@*	
 �2    :R
'aten__convolution_backward_overrideable'aten__convolution_backward_overrideabler ��%��%� � �
convolution.4808convolution@PP*	
 �2    :R
'aten__convolution_backward_overrideable'aten__convolution_backward_overrideablez
(0
(0� 2 @PZb��%��%�%��
  �� � S
add.4835add@PP*	
 �2    :
	aten__add	aten__add��%��$�%� � �
	p471.4828	parameter@PP*	
 �2    :$
xla__device_dataxla__device_dataH���%�

neff_input_namesinput471� o
logistic.4837logistic@PP*	
 �2    :*
aten__silu_backwardaten__silu_backward��%��%� � i
constant.4836constant*�:*
aten__silu_backwardaten__silu_backwardB
*�B  �?��%� � q
broadcast.4841	broadcast@PP*	
 �2    :*
aten__silu_backwardaten__silu_backward��%��%� � q
broadcast.4838	broadcast@PP*	
 �2    :*
aten__silu_backwardaten__silu_backward��%��%� � q
subtract.4839subtract@PP*	
 �2    :*
aten__silu_backwardaten__silu_backward��%��%�%� � q
multiply.4840multiply@PP*	
 �2    :*
aten__silu_backwardaten__silu_backward��%��%�%� � g
add.4842add@PP*	
 �2    :*
aten__silu_backwardaten__silu_backward��%��%�%� � q
multiply.4843multiply@PP*	
 �2    :*
aten__silu_backwardaten__silu_backward��%��%�%� � q
multiply.4844multiply@PP*	
 �2    :*
aten__silu_backwardaten__silu_backward��%��%�%� � �
batch-norm-grad.4852batch-norm-gradA"@PP*	
 �2    "@*
 �2 "@*
 �2 :D
 aten__native_batch_norm_backward aten__native_batch_norm_backward�o�:���%�
�%�%�%�3�%� � �
get-tuple-element.4855get-tuple-element@*
 �2 :D
 aten__native_batch_norm_backward aten__native_batch_norm_backwardh��%��%� � �
custom-call.448custom-call@*
 �2 :�
'xla___op_TransferWithStaticRingTransfer'xla___op_TransferWithStaticRingTransferv/home/ubuntu/aws-neuron-samples/aws_neuron_venv_pytorch/lib/python3.10/site-packages/torch_xla/core/xla_op_registry.py ,x�AwsNeuronTransferWithStaticRing��/��%��
 �� � �
get-tuple-element.4854get-tuple-element@*
 �2 :D
 aten__native_batch_norm_backward aten__native_batch_norm_backwardh��%��%� � �
custom-call.449custom-call@*
 �2 :�
'xla___op_TransferWithStaticRingTransfer'xla___op_TransferWithStaticRingTransferv/home/ubuntu/aws-neuron-samples/aws_neuron_venv_pytorch/lib/python3.10/site-packages/torch_xla/core/xla_op_registry.py ,x�AwsNeuronTransferWithStaticRing��/��%��
 �� � �
	p473.4871	parameter@PP*	
 �2    :$
xla__device_dataxla__device_dataH���&�

neff_input_namesinput473� �
get-tuple-element.4853get-tuple-element@PP*	
 �2    :D
 aten__native_batch_norm_backward aten__native_batch_norm_backward��%��%� � �
convolution.4875convolution@@*	
 �2    :R
'aten__convolution_backward_overrideable'aten__convolution_backward_overrideablez
P (0
P (0� 28HPZb ��&��&�%��
  �� � �
transpose.4876	transpose@@*	
 �2    :R
'aten__convolution_backward_overrideable'aten__convolution_backward_overrideabler ��&��&� � �
custom-call.450custom-call@@*	
 �2    :�
'xla___op_TransferWithStaticRingTransfer'xla___op_TransferWithStaticRingTransferv/home/ubuntu/aws-neuron-samples/aws_neuron_venv_pytorch/lib/python3.10/site-packages/torch_xla/core/xla_op_registry.py ,x�AwsNeuronTransferWithStaticRing��/��&��
 �� � �
	p477.4893	parameter@PP*	
 �2    :$
xla__device_dataxla__device_dataH���&�

neff_input_namesinput477� y
	p476.4892	parameter@*
 �2 :$
xla__device_dataxla__device_dataH���&�

neff_input_namesinput476� y
	p475.4891	parameter@*
 �2 :$
xla__device_dataxla__device_dataH���&�

neff_input_namesinput475� �
constant.4905constant*�:D
 aten__native_batch_norm_backward aten__native_batch_norm_backwardB
*�B  �?��&� � �
broadcast.4906	broadcast@*
 �2 :D
 aten__native_batch_norm_backward aten__native_batch_norm_backward��&��&� � y
	p474.4890	parameter@*
 �2 :$
xla__device_dataxla__device_dataH���&�

neff_input_namesinput474� ~
divide.4907divide@*
 �2 :D
 aten__native_batch_norm_backward aten__native_batch_norm_backward��&��&�&� � �
multiply.4908multiply@*
 �2 :D
 aten__native_batch_norm_backward aten__native_batch_norm_backward��&��&�&� � >
constant.111constant*�: B
*�Bo����6� � �
broadcast.202	broadcast@*
 �2 :D
 aten__native_batch_norm_backward aten__native_batch_norm_backward��3��6� � v
add.66add@*
 �2 :D
 aten__native_batch_norm_backward aten__native_batch_norm_backward��3��&�3� � �
	p472.4870	parameter@@*	
 �2    :$
xla__device_dataxla__device_dataH���&�

neff_input_namesinput472� �
transpose.4872	transpose@@*	
 �2    :R
'aten__convolution_backward_overrideable'aten__convolution_backward_overrideabler ��&��&� � �
reverse.4873reverse@@*	
 �2    :R
'aten__convolution_backward_overrideable'aten__convolution_backward_overrideabler ��&��&� � �
convolution.4874convolution@PP*	
 �2    :R
'aten__convolution_backward_overrideable'aten__convolution_backward_overrideablez
 (0
 (0� 2 @PZb��&��%�&��
  �� � �
	p478.4894	parameter@PP*	
 �2    :$
xla__device_dataxla__device_dataH���&�

neff_input_namesinput478� o
logistic.4896logistic@PP*	
 �2    :*
aten__silu_backwardaten__silu_backward��&��&� � i
constant.4895constant*�:*
aten__silu_backwardaten__silu_backwardB
*�B  �?��&� � q
broadcast.4900	broadcast@PP*	
 �2    :*
aten__silu_backwardaten__silu_backward��&��&� � q
broadcast.4897	broadcast@PP*	
 �2    :*
aten__silu_backwardaten__silu_backward��&��&� � q
subtract.4898subtract@PP*	
 �2    :*
aten__silu_backwardaten__silu_backward��&��&�&� � q
multiply.4899multiply@PP*	
 �2    :*
aten__silu_backwardaten__silu_backward��&��&�&� � g
add.4901add@PP*	
 �2    :*
aten__silu_backwardaten__silu_backward��&��&�&� � q
multiply.4902multiply@PP*	
 �2    :*
aten__silu_backwardaten__silu_backward��&��&�&� � q
multiply.4903multiply@PP*	
 �2    :*
aten__silu_backwardaten__silu_backward��&��&�&� � �
batch-norm-grad.4911batch-norm-gradA"@PP*	
 �2    "@*
 �2 "@*
 �2 :D
 aten__native_batch_norm_backward aten__native_batch_norm_backward�o�:���&�
�&�&�&�3�&� � �
get-tuple-element.4914get-tuple-element@*
 �2 :D
 aten__native_batch_norm_backward aten__native_batch_norm_backwardh��&��&� � �
custom-call.451custom-call@*
 �2 :�
'xla___op_TransferWithStaticRingTransfer'xla___op_TransferWithStaticRingTransferv/home/ubuntu/aws-neuron-samples/aws_neuron_venv_pytorch/lib/python3.10/site-packages/torch_xla/core/xla_op_registry.py ,x�AwsNeuronTransferWithStaticRing��/��&��
 �� � �
get-tuple-element.4913get-tuple-element@*
 �2 :D
 aten__native_batch_norm_backward aten__native_batch_norm_backwardh��&��&� � �
custom-call.452custom-call@*
 �2 :�
'xla___op_TransferWithStaticRingTransfer'xla___op_TransferWithStaticRingTransferv/home/ubuntu/aws-neuron-samples/aws_neuron_venv_pytorch/lib/python3.10/site-packages/torch_xla/core/xla_op_registry.py ,x�AwsNeuronTransferWithStaticRing��/��&��
 �� � �
	p480.4930	parameter@PP*	
 �2    :$
xla__device_dataxla__device_dataH���&�

neff_input_namesinput480� �
get-tuple-element.4912get-tuple-element@PP*	
 �2    :D
 aten__native_batch_norm_backward aten__native_batch_norm_backward��&��&� � �
convolution.4934convolution@@*	
 �2    :R
'aten__convolution_backward_overrideable'aten__convolution_backward_overrideablez
P(0
P(0� 28HPZb ��&��&�&��
  �� � �
transpose.4935	transpose@@*	
 �2    :R
'aten__convolution_backward_overrideable'aten__convolution_backward_overrideabler ��&��&� � �
custom-call.453custom-call@@*	
 �2    :�
'xla___op_TransferWithStaticRingTransfer'xla___op_TransferWithStaticRingTransferv/home/ubuntu/aws-neuron-samples/aws_neuron_venv_pytorch/lib/python3.10/site-packages/torch_xla/core/xla_op_registry.py ,x�AwsNeuronTransferWithStaticRing��/��&��
 �� � �
	p484.4952	parameter@PP*	
 �2    :$
xla__device_dataxla__device_dataH���&�

neff_input_namesinput484� y
	p483.4951	parameter@*
 �2 :$
xla__device_dataxla__device_dataH���&�

neff_input_namesinput483� y
	p482.4950	parameter@*
 �2 :$
xla__device_dataxla__device_dataH���&�

neff_input_namesinput482� �
constant.4965constant*�:D
 aten__native_batch_norm_backward aten__native_batch_norm_backwardB
*�B  �?��&� � �
broadcast.4966	broadcast@*
 �2 :D
 aten__native_batch_norm_backward aten__native_batch_norm_backward��&��&� � y
	p481.4949	parameter@*
 �2 :$
xla__device_dataxla__device_dataH���&�

neff_input_namesinput481� ~
divide.4967divide@*
 �2 :D
 aten__native_batch_norm_backward aten__native_batch_norm_backward��&��&�&� � �
multiply.4968multiply@*
 �2 :D
 aten__native_batch_norm_backward aten__native_batch_norm_backward��&��&�&� � >
constant.112constant*�: B
*�Bo����6� � �
broadcast.203	broadcast@*
 �2 :D
 aten__native_batch_norm_backward aten__native_batch_norm_backward��3��6� � v
add.67add@*
 �2 :D
 aten__native_batch_norm_backward aten__native_batch_norm_backward��3��&�3� � x

slice.4954slice@PP*	
 �2    :
xla__selectxla__select��@��P�P��&��#� � �
	p485.4953	parameter@PP*	
 �2    :$
xla__device_dataxla__device_dataH���&�

neff_input_namesinput485� o
logistic.4956logistic@PP*	
 �2    :*
aten__silu_backwardaten__silu_backward��&��&� � i
constant.4955constant*�:*
aten__silu_backwardaten__silu_backwardB
*�B  �?��&� � q
broadcast.4960	broadcast@PP*	
 �2    :*
aten__silu_backwardaten__silu_backward��&��&� � q
broadcast.4957	broadcast@PP*	
 �2    :*
aten__silu_backwardaten__silu_backward��&��&� � q
subtract.4958subtract@PP*	
 �2    :*
aten__silu_backwardaten__silu_backward��&��&�&� � q
multiply.4959multiply@PP*	
 �2    :*
aten__silu_backwardaten__silu_backward��&��&�&� � g
add.4961add@PP*	
 �2    :*
aten__silu_backwardaten__silu_backward��&��&�&� � q
multiply.4962multiply@PP*	
 �2    :*
aten__silu_backwardaten__silu_backward��&��&�&� � q
multiply.4963multiply@PP*	
 �2    :*
aten__silu_backwardaten__silu_backward��&��&�&� � �
batch-norm-grad.4971batch-norm-gradA"@PP*	
 �2    "@*
 �2 "@*
 �2 :D
 aten__native_batch_norm_backward aten__native_batch_norm_backward�o�:���&�
�&�&�&�3�&� � �
get-tuple-element.4974get-tuple-element@*
 �2 :D
 aten__native_batch_norm_backward aten__native_batch_norm_backwardh��&��&� � �
custom-call.454custom-call@*
 �2 :�
'xla___op_TransferWithStaticRingTransfer'xla___op_TransferWithStaticRingTransferv/home/ubuntu/aws-neuron-samples/aws_neuron_venv_pytorch/lib/python3.10/site-packages/torch_xla/core/xla_op_registry.py ,x�AwsNeuronTransferWithStaticRing��/��&��
 �� � �
get-tuple-element.4973get-tuple-element@*
 �2 :D
 aten__native_batch_norm_backward aten__native_batch_norm_backwardh��&��&� � �
custom-call.455custom-call@*
 �2 :�
'xla___op_TransferWithStaticRingTransfer'xla___op_TransferWithStaticRingTransferv/home/ubuntu/aws-neuron-samples/aws_neuron_venv_pytorch/lib/python3.10/site-packages/torch_xla/core/xla_op_registry.py ,x�AwsNeuronTransferWithStaticRing��/��&��
 �� � �
	p487.4990	parameter�PP*	
 �2    :$
xla__device_dataxla__device_dataH���&�

neff_input_namesinput487� �
get-tuple-element.4972get-tuple-element@PP*	
 �2    :D
 aten__native_batch_norm_backward aten__native_batch_norm_backward��&��&� � �
convolution.4994convolution�@*	
 �2    :R
'aten__convolution_backward_overrideable'aten__convolution_backward_overrideablez
P(0
P(0� 28HPZb ��'��&�&��
  �� � �
transpose.4995	transpose@�*	
 �2    :R
'aten__convolution_backward_overrideable'aten__convolution_backward_overrideabler ��'��'� � �
custom-call.456custom-call@�*	
 �2    :�
'xla___op_TransferWithStaticRingTransfer'xla___op_TransferWithStaticRingTransferv/home/ubuntu/aws-neuron-samples/aws_neuron_venv_pytorch/lib/python3.10/site-packages/torch_xla/core/xla_op_registry.py ,x�AwsNeuronTransferWithStaticRing��/��'��
 �� � �
	p491.5012	parameter@PP*	
 �2    :$
xla__device_dataxla__device_dataH���'�

neff_input_namesinput491� y
	p490.5011	parameter@*
 �2 :$
xla__device_dataxla__device_dataH���'�

neff_input_namesinput490� y
	p489.5010	parameter@*
 �2 :$
xla__device_dataxla__device_dataH���'�

neff_input_namesinput489� �
constant.5031constant*�:D
 aten__native_batch_norm_backward aten__native_batch_norm_backwardB
*�B  �?��'� � �
broadcast.5032	broadcast@*
 �2 :D
 aten__native_batch_norm_backward aten__native_batch_norm_backward��'��'� � y
	p488.5009	parameter@*
 �2 :$
xla__device_dataxla__device_dataH���'�

neff_input_namesinput488� ~
divide.5033divide@*
 �2 :D
 aten__native_batch_norm_backward aten__native_batch_norm_backward��'��'�'� � �
multiply.5034multiply@*
 �2 :D
 aten__native_batch_norm_backward aten__native_batch_norm_backward��'��'�'� � >
constant.113constant*�: B
*�Bo����6� � �
broadcast.204	broadcast@*
 �2 :D
 aten__native_batch_norm_backward aten__native_batch_norm_backward��3��6� � v
add.68add@*
 �2 :D
 aten__native_batch_norm_backward aten__native_batch_norm_backward��3��'�3� � �
	p479.4929	parameter@@*	
 �2    :$
xla__device_dataxla__device_dataH���&�

neff_input_namesinput479� �
transpose.4931	transpose@@*	
 �2    :R
'aten__convolution_backward_overrideable'aten__convolution_backward_overrideabler ��&��&� � �
convolution.4933convolution@PP*	
 �2    :R
'aten__convolution_backward_overrideable'aten__convolution_backward_overrideablez
(0
(0� 2 @PZb��&��&�&��
  �� � S
add.5020add@PP*	
 �2    :
	aten__add	aten__add��'��%�&� � �
	p492.5013	parameter@PP*	
 �2    :$
xla__device_dataxla__device_dataH���'�

neff_input_namesinput492� o
logistic.5022logistic@PP*	
 �2    :*
aten__silu_backwardaten__silu_backward��'��'� � i
constant.5021constant*�:*
aten__silu_backwardaten__silu_backwardB
*�B  �?��'� � q
broadcast.5026	broadcast@PP*	
 �2    :*
aten__silu_backwardaten__silu_backward��'��'� � q
broadcast.5023	broadcast@PP*	
 �2    :*
aten__silu_backwardaten__silu_backward��'��'� � q
subtract.5024subtract@PP*	
 �2    :*
aten__silu_backwardaten__silu_backward��'��'�'� � q
multiply.5025multiply@PP*	
 �2    :*
aten__silu_backwardaten__silu_backward��'��'�'� � g
add.5027add@PP*	
 �2    :*
aten__silu_backwardaten__silu_backward��'��'�'� � q
multiply.5028multiply@PP*	
 �2    :*
aten__silu_backwardaten__silu_backward��'��'�'� � q
multiply.5029multiply@PP*	
 �2    :*
aten__silu_backwardaten__silu_backward��'��'�'� � �
batch-norm-grad.5037batch-norm-gradA"@PP*	
 �2    "@*
 �2 "@*
 �2 :D
 aten__native_batch_norm_backward aten__native_batch_norm_backward�o�:���'�
�'�'�'�3�'� � �
get-tuple-element.5040get-tuple-element@*
 �2 :D
 aten__native_batch_norm_backward aten__native_batch_norm_backwardh��'��'� � �
custom-call.457custom-call@*
 �2 :�
'xla___op_TransferWithStaticRingTransfer'xla___op_TransferWithStaticRingTransferv/home/ubuntu/aws-neuron-samples/aws_neuron_venv_pytorch/lib/python3.10/site-packages/torch_xla/core/xla_op_registry.py ,x�AwsNeuronTransferWithStaticRing��/��'��
 �� � �
get-tuple-element.5039get-tuple-element@*
 �2 :D
 aten__native_batch_norm_backward aten__native_batch_norm_backwardh��'��'� � �
custom-call.458custom-call@*
 �2 :�
'xla___op_TransferWithStaticRingTransfer'xla___op_TransferWithStaticRingTransferv/home/ubuntu/aws-neuron-samples/aws_neuron_venv_pytorch/lib/python3.10/site-packages/torch_xla/core/xla_op_registry.py ,x�AwsNeuronTransferWithStaticRing��/��'��
 �� � �
get-tuple-element.5038get-tuple-element@PP*	
 �2    :D
 aten__native_batch_norm_backward aten__native_batch_norm_backward��'��'� � �
convolution.5059convolution�@*	
 �2    :R
'aten__convolution_backward_overrideable'aten__convolution_backward_overrideablez
P(0
P(0� 28HPZb ��'��&�'��
  �� � �
transpose.5060	transpose@�*	
 �2    :R
'aten__convolution_backward_overrideable'aten__convolution_backward_overrideabler ��'��'� � �
custom-call.459custom-call@�*	
 �2    :�
'xla___op_TransferWithStaticRingTransfer'xla___op_TransferWithStaticRingTransferv/home/ubuntu/aws-neuron-samples/aws_neuron_venv_pytorch/lib/python3.10/site-packages/torch_xla/core/xla_op_registry.py ,x�AwsNeuronTransferWithStaticRing��/��'��
 �� � �
	p497.5077	parameter�PP*	
 �2    :$
xla__device_dataxla__device_dataH���'�

neff_input_namesinput497� z
	p496.5076	parameter�*
 �2 :$
xla__device_dataxla__device_dataH���'�

neff_input_namesinput496� z
	p495.5075	parameter�*
 �2 :$
xla__device_dataxla__device_dataH���'�

neff_input_namesinput495� �
constant.5096constant*�:D
 aten__native_batch_norm_backward aten__native_batch_norm_backwardB
*�B  �?��'� � �
broadcast.5097	broadcast�*
 �2 :D
 aten__native_batch_norm_backward aten__native_batch_norm_backward��'��'� � z
	p494.5074	parameter�*
 �2 :$
xla__device_dataxla__device_dataH���'�

neff_input_namesinput494� 
divide.5098divide�*
 �2 :D
 aten__native_batch_norm_backward aten__native_batch_norm_backward��'��'�'� � �
multiply.5099multiply�*
 �2 :D
 aten__native_batch_norm_backward aten__native_batch_norm_backward��'��'�'� � >
constant.114constant*�: B
*�Bo����6� � �
broadcast.208	broadcast�*
 �2 :D
 aten__native_batch_norm_backward aten__native_batch_norm_backward��3��6� � w
add.69add�*
 �2 :D
 aten__native_batch_norm_backward aten__native_batch_norm_backward��3��'�3� � �
	p486.4989	parameter@�*	
 �2    :$
xla__device_dataxla__device_dataH���&�

neff_input_namesinput486� �
transpose.4991	transpose�@*	
 �2    :R
'aten__convolution_backward_overrideable'aten__convolution_backward_overrideabler ��&��&� � �
convolution.4993convolution�PP*	
 �2    :R
'aten__convolution_backward_overrideable'aten__convolution_backward_overrideablez
(0
(0� 2 @PZb��'��&�&��
  �� � �
	p493.5055	parameter@�*	
 �2    :$
xla__device_dataxla__device_dataH���'�

neff_input_namesinput493� �
transpose.5056	transpose�@*	
 �2    :R
'aten__convolution_backward_overrideable'aten__convolution_backward_overrideabler ��'��'� � �
convolution.5058convolution�PP*	
 �2    :R
'aten__convolution_backward_overrideable'aten__convolution_backward_overrideablez
(0
(0� 2 @PZb��'��'�'��
  �� � T
add.5085add�PP*	
 �2    :
	aten__add	aten__add��'��'�'� � �
	p498.5078	parameter�PP*	
 �2    :$
xla__device_dataxla__device_dataH���'�

neff_input_namesinput498� p
logistic.5087logistic�PP*	
 �2    :*
aten__silu_backwardaten__silu_backward��'��'� � i
constant.5086constant*�:*
aten__silu_backwardaten__silu_backwardB
*�B  �?��'� � r
broadcast.5091	broadcast�PP*	
 �2    :*
aten__silu_backwardaten__silu_backward��'��'� � r
broadcast.5088	broadcast�PP*	
 �2    :*
aten__silu_backwardaten__silu_backward��'��'� � r
subtract.5089subtract�PP*	
 �2    :*
aten__silu_backwardaten__silu_backward��'��'�'� � r
multiply.5090multiply�PP*	
 �2    :*
aten__silu_backwardaten__silu_backward��'��'�'� � h
add.5092add�PP*	
 �2    :*
aten__silu_backwardaten__silu_backward��'��'�'� � r
multiply.5093multiply�PP*	
 �2    :*
aten__silu_backwardaten__silu_backward��'��'�'� � r
multiply.5094multiply�PP*	
 �2    :*
aten__silu_backwardaten__silu_backward��'��'�'� � �
batch-norm-grad.5102batch-norm-gradD"�PP*	
 �2    "�*
 �2 "�*
 �2 :D
 aten__native_batch_norm_backward aten__native_batch_norm_backward�o�:���'�
�'�'�'�3�'� � �
get-tuple-element.5105get-tuple-element�*
 �2 :D
 aten__native_batch_norm_backward aten__native_batch_norm_backwardh��'��'� � �
custom-call.460custom-call�*
 �2 :�
'xla___op_TransferWithStaticRingTransfer'xla___op_TransferWithStaticRingTransferv/home/ubuntu/aws-neuron-samples/aws_neuron_venv_pytorch/lib/python3.10/site-packages/torch_xla/core/xla_op_registry.py ,x�AwsNeuronTransferWithStaticRing��/��'��
 �� � �
get-tuple-element.5104get-tuple-element�*
 �2 :D
 aten__native_batch_norm_backward aten__native_batch_norm_backwardh��'��'� � �
custom-call.461custom-call�*
 �2 :�
'xla___op_TransferWithStaticRingTransfer'xla___op_TransferWithStaticRingTransferv/home/ubuntu/aws-neuron-samples/aws_neuron_venv_pytorch/lib/python3.10/site-packages/torch_xla/core/xla_op_registry.py ,x�AwsNeuronTransferWithStaticRing��/��'��
 �� � �
	p500.5121	parameter@��*	
 �2    :$
xla__device_dataxla__device_dataH���(�

neff_input_namesinput500� �
get-tuple-element.5103get-tuple-element�PP*	
 �2    :D
 aten__native_batch_norm_backward aten__native_batch_norm_backward��'��'� � �
convolution.5125convolution@�*	
 �2    :R
'aten__convolution_backward_overrideable'aten__convolution_backward_overrideablez

P(0

P(0� 28HPZb ��(��(�'��
  �� � �
transpose.5126	transpose�@*	
 �2    :R
'aten__convolution_backward_overrideable'aten__convolution_backward_overrideabler ��(��(� � �
custom-call.462custom-call�@*	
 �2    :�
'xla___op_TransferWithStaticRingTransfer'xla___op_TransferWithStaticRingTransferv/home/ubuntu/aws-neuron-samples/aws_neuron_venv_pytorch/lib/python3.10/site-packages/torch_xla/core/xla_op_registry.py ,x�AwsNeuronTransferWithStaticRing��/��(��
 �� � �
	p504.5143	parameter@��*	
 �2    :$
xla__device_dataxla__device_dataH���(�

neff_input_namesinput504� y
	p503.5142	parameter@*
 �2 :$
xla__device_dataxla__device_dataH���(�

neff_input_namesinput503� y
	p502.5141	parameter@*
 �2 :$
xla__device_dataxla__device_dataH���(�

neff_input_namesinput502� �
constant.5155constant*�:D
 aten__native_batch_norm_backward aten__native_batch_norm_backwardB
*�B  �?��(� � �
broadcast.5156	broadcast@*
 �2 :D
 aten__native_batch_norm_backward aten__native_batch_norm_backward��(��(� � y
	p501.5140	parameter@*
 �2 :$
xla__device_dataxla__device_dataH���(�

neff_input_namesinput501� ~
divide.5157divide@*
 �2 :D
 aten__native_batch_norm_backward aten__native_batch_norm_backward��(��(�(� � �
multiply.5158multiply@*
 �2 :D
 aten__native_batch_norm_backward aten__native_batch_norm_backward��(��(�(� � >
constant.116constant*�: B
*�Bo����6� � �
broadcast.211	broadcast@*
 �2 :D
 aten__native_batch_norm_backward aten__native_batch_norm_backward��3��6� � v
add.70add@*
 �2 :D
 aten__native_batch_norm_backward aten__native_batch_norm_backward��3��(�3� � �
	p499.5120	parameter�@*	
 �2    :$
xla__device_dataxla__device_dataH���(�

neff_input_namesinput499� �
transpose.5122	transpose@�*	
 �2    :R
'aten__convolution_backward_overrideable'aten__convolution_backward_overrideabler ��(��(� � �
reverse.5123reverse@�*	
 �2    :R
'aten__convolution_backward_overrideable'aten__convolution_backward_overrideabler ��(��(� � �
convolution.5124convolution@��*	
 �2    :R
'aten__convolution_backward_overrideable'aten__convolution_backward_overrideablez
 (0
 (0� 2 @PZb��(��'�(��
  �� � �
	p505.5144	parameter@��*	
 �2    :$
xla__device_dataxla__device_dataH���(�

neff_input_namesinput505� q
logistic.5146logistic@��*	
 �2    :*
aten__silu_backwardaten__silu_backward��(��(� � i
constant.5145constant*�:*
aten__silu_backwardaten__silu_backwardB
*�B  �?��(� � s
broadcast.5150	broadcast@��*	
 �2    :*
aten__silu_backwardaten__silu_backward��(��(� � s
broadcast.5147	broadcast@��*	
 �2    :*
aten__silu_backwardaten__silu_backward��(��(� � s
subtract.5148subtract@��*	
 �2    :*
aten__silu_backwardaten__silu_backward��(��(�(� � s
multiply.5149multiply@��*	
 �2    :*
aten__silu_backwardaten__silu_backward��(��(�(� � i
add.5151add@��*	
 �2    :*
aten__silu_backwardaten__silu_backward��(��(�(� � s
multiply.5152multiply@��*	
 �2    :*
aten__silu_backwardaten__silu_backward��(��(�(� � s
multiply.5153multiply@��*	
 �2    :*
aten__silu_backwardaten__silu_backward��(��(�(� � �
batch-norm-grad.5161batch-norm-gradC"@��*	
 �2    "@*
 �2 "@*
 �2 :D
 aten__native_batch_norm_backward aten__native_batch_norm_backward�o�:���(�
�(�(�(�3�(� � �
get-tuple-element.5164get-tuple-element@*
 �2 :D
 aten__native_batch_norm_backward aten__native_batch_norm_backwardh��(��(� � �
custom-call.463custom-call@*
 �2 :�
'xla___op_TransferWithStaticRingTransfer'xla___op_TransferWithStaticRingTransferv/home/ubuntu/aws-neuron-samples/aws_neuron_venv_pytorch/lib/python3.10/site-packages/torch_xla/core/xla_op_registry.py ,x�AwsNeuronTransferWithStaticRing��/��(��
 �� � �
get-tuple-element.5163get-tuple-element@*
 �2 :D
 aten__native_batch_norm_backward aten__native_batch_norm_backwardh��(��(� � �
custom-call.464custom-call@*
 �2 :�
'xla___op_TransferWithStaticRingTransfer'xla___op_TransferWithStaticRingTransferv/home/ubuntu/aws-neuron-samples/aws_neuron_venv_pytorch/lib/python3.10/site-packages/torch_xla/core/xla_op_registry.py ,x�AwsNeuronTransferWithStaticRing��/��(��
 �� � �
	p507.5180	parameter@��*	
 �2    :$
xla__device_dataxla__device_dataH���(�

neff_input_namesinput507� �
get-tuple-element.5162get-tuple-element@��*	
 �2    :D
 aten__native_batch_norm_backward aten__native_batch_norm_backward��(��(� � �
convolution.5184convolution@@*	
 �2    :R
'aten__convolution_backward_overrideable'aten__convolution_backward_overrideablez
	�(0
	�(0� 28HPZb ��(��(�(��
  �� � �
transpose.5185	transpose@@*	
 �2    :R
'aten__convolution_backward_overrideable'aten__convolution_backward_overrideabler ��(��(� � �
custom-call.465custom-call@@*	
 �2    :�
'xla___op_TransferWithStaticRingTransfer'xla___op_TransferWithStaticRingTransferv/home/ubuntu/aws-neuron-samples/aws_neuron_venv_pytorch/lib/python3.10/site-packages/torch_xla/core/xla_op_registry.py ,x�AwsNeuronTransferWithStaticRing��/��(��
 �� � �
	p511.5202	parameter ��*	
 �2    :$
xla__device_dataxla__device_dataH���(�

neff_input_namesinput511� y
	p510.5201	parameter *
 �2 :$
xla__device_dataxla__device_dataH���(�

neff_input_namesinput510� y
	p509.5200	parameter *
 �2 :$
xla__device_dataxla__device_dataH���(�

neff_input_namesinput509� �
constant.5215constant*�:D
 aten__native_batch_norm_backward aten__native_batch_norm_backwardB
*�B  �?��(� � �
broadcast.5216	broadcast *
 �2 :D
 aten__native_batch_norm_backward aten__native_batch_norm_backward��(��(� � y
	p508.5199	parameter *
 �2 :$
xla__device_dataxla__device_dataH���(�

neff_input_namesinput508� ~
divide.5217divide *
 �2 :D
 aten__native_batch_norm_backward aten__native_batch_norm_backward��(��(�(� � �
multiply.5218multiply *
 �2 :D
 aten__native_batch_norm_backward aten__native_batch_norm_backward��(��(�(� � >
constant.117constant*�: B
*�Bo����6� � �
broadcast.213	broadcast *
 �2 :D
 aten__native_batch_norm_backward aten__native_batch_norm_backward��3��6� � v
add.71add *
 �2 :D
 aten__native_batch_norm_backward aten__native_batch_norm_backward��3��(�3� � �
	p506.5179	parameter@@*	
 �2    :$
xla__device_dataxla__device_dataH���(�

neff_input_namesinput506� �
transpose.5181	transpose@@*	
 �2    :R
'aten__convolution_backward_overrideable'aten__convolution_backward_overrideabler ��(��(� � �
convolution.5183convolution@��*	
 �2    :R
'aten__convolution_backward_overrideable'aten__convolution_backward_overrideablez
(0
(0� 2 @PZb��(��(�(��
  �� � y

slice.5204slice ��*	
 �2    :
xla__selectxla__select�� ������(��(� � �
	p512.5203	parameter ��*	
 �2    :$
xla__device_dataxla__device_dataH���(�

neff_input_namesinput512� q
logistic.5206logistic ��*	
 �2    :*
aten__silu_backwardaten__silu_backward��(��(� � i
constant.5205constant*�:*
aten__silu_backwardaten__silu_backwardB
*�B  �?��(� � s
broadcast.5210	broadcast ��*	
 �2    :*
aten__silu_backwardaten__silu_backward��(��(� � s
broadcast.5207	broadcast ��*	
 �2    :*
aten__silu_backwardaten__silu_backward��(��(� � s
subtract.5208subtract ��*	
 �2    :*
aten__silu_backwardaten__silu_backward��(��(�(� � s
multiply.5209multiply ��*	
 �2    :*
aten__silu_backwardaten__silu_backward��(��(�(� � i
add.5211add ��*	
 �2    :*
aten__silu_backwardaten__silu_backward��(��(�(� � s
multiply.5212multiply ��*	
 �2    :*
aten__silu_backwardaten__silu_backward��(��(�(� � s
multiply.5213multiply ��*	
 �2    :*
aten__silu_backwardaten__silu_backward��(��(�(� � �
batch-norm-grad.5221batch-norm-gradC" ��*	
 �2    " *
 �2 " *
 �2 :D
 aten__native_batch_norm_backward aten__native_batch_norm_backward�o�:���(�
�(�(�(�3�(� � �
get-tuple-element.5224get-tuple-element *
 �2 :D
 aten__native_batch_norm_backward aten__native_batch_norm_backwardh��(��(� � �
custom-call.466custom-call *
 �2 :�
'xla___op_TransferWithStaticRingTransfer'xla___op_TransferWithStaticRingTransferv/home/ubuntu/aws-neuron-samples/aws_neuron_venv_pytorch/lib/python3.10/site-packages/torch_xla/core/xla_op_registry.py ,x�AwsNeuronTransferWithStaticRing��/��(��
 �� � �
get-tuple-element.5223get-tuple-element *
 �2 :D
 aten__native_batch_norm_backward aten__native_batch_norm_backwardh��(��(� � �
custom-call.467custom-call *
 �2 :�
'xla___op_TransferWithStaticRingTransfer'xla___op_TransferWithStaticRingTransferv/home/ubuntu/aws-neuron-samples/aws_neuron_venv_pytorch/lib/python3.10/site-packages/torch_xla/core/xla_op_registry.py ,x�AwsNeuronTransferWithStaticRing��/��(��
 �� � �
	p514.5240	parameter ��*	
 �2    :$
xla__device_dataxla__device_dataH���(�

neff_input_namesinput514� �
get-tuple-element.5222get-tuple-element ��*	
 �2    :D
 aten__native_batch_norm_backward aten__native_batch_norm_backward��(��(� � �
convolution.5244convolution  *	
 �2    :R
'aten__convolution_backward_overrideable'aten__convolution_backward_overrideablez
� (0
� (0� 28HPZb ��(��(�(��
  �� � �
transpose.5245	transpose  *	
 �2    :R
'aten__convolution_backward_overrideable'aten__convolution_backward_overrideabler ��(��(� � �
custom-call.468custom-call  *	
 �2    :�
'xla___op_TransferWithStaticRingTransfer'xla___op_TransferWithStaticRingTransferv/home/ubuntu/aws-neuron-samples/aws_neuron_venv_pytorch/lib/python3.10/site-packages/torch_xla/core/xla_op_registry.py ,x�AwsNeuronTransferWithStaticRing��/��(��
 �� � �
	p518.5262	parameter ��*	
 �2    :$
xla__device_dataxla__device_dataH���)�

neff_input_namesinput518� y
	p517.5261	parameter *
 �2 :$
xla__device_dataxla__device_dataH���)�

neff_input_namesinput517� y
	p516.5260	parameter *
 �2 :$
xla__device_dataxla__device_dataH���)�

neff_input_namesinput516� �
constant.5274constant*�:D
 aten__native_batch_norm_backward aten__native_batch_norm_backwardB
*�B  �?��)� � �
broadcast.5275	broadcast *
 �2 :D
 aten__native_batch_norm_backward aten__native_batch_norm_backward��)��)� � y
	p515.5259	parameter *
 �2 :$
xla__device_dataxla__device_dataH���)�

neff_input_namesinput515� ~
divide.5276divide *
 �2 :D
 aten__native_batch_norm_backward aten__native_batch_norm_backward��)��)�)� � �
multiply.5277multiply *
 �2 :D
 aten__native_batch_norm_backward aten__native_batch_norm_backward��)��)�)� � >
constant.118constant*�: B
*�Bo����6� � �
broadcast.214	broadcast *
 �2 :D
 aten__native_batch_norm_backward aten__native_batch_norm_backward��3��6� � v
add.72add *
 �2 :D
 aten__native_batch_norm_backward aten__native_batch_norm_backward��3��)�3� � �
	p513.5239	parameter  *	
 �2    :$
xla__device_dataxla__device_dataH���(�

neff_input_namesinput513� �
transpose.5241	transpose  *	
 �2    :R
'aten__convolution_backward_overrideable'aten__convolution_backward_overrideabler ��(��(� � �
reverse.5242reverse  *	
 �2    :R
'aten__convolution_backward_overrideable'aten__convolution_backward_overrideabler ��(��(� � �
convolution.5243convolution ��*	
 �2    :R
'aten__convolution_backward_overrideable'aten__convolution_backward_overrideablez
 (0
 (0� 2 @PZb��(��(�(��
  �� � �
	p519.5263	parameter ��*	
 �2    :$
xla__device_dataxla__device_dataH���)�

neff_input_namesinput519� q
logistic.5265logistic ��*	
 �2    :*
aten__silu_backwardaten__silu_backward��)��)� � i
constant.5264constant*�:*
aten__silu_backwardaten__silu_backwardB
*�B  �?��)� � s
broadcast.5269	broadcast ��*	
 �2    :*
aten__silu_backwardaten__silu_backward��)��)� � s
broadcast.5266	broadcast ��*	
 �2    :*
aten__silu_backwardaten__silu_backward��)��)� � s
subtract.5267subtract ��*	
 �2    :*
aten__silu_backwardaten__silu_backward��)��)�)� � s
multiply.5268multiply ��*	
 �2    :*
aten__silu_backwardaten__silu_backward��)��)�)� � i
add.5270add ��*	
 �2    :*
aten__silu_backwardaten__silu_backward��)��)�)� � s
multiply.5271multiply ��*	
 �2    :*
aten__silu_backwardaten__silu_backward��)��)�)� � s
multiply.5272multiply ��*	
 �2    :*
aten__silu_backwardaten__silu_backward��)��(�)� � �
batch-norm-grad.5280batch-norm-gradC" ��*	
 �2    " *
 �2 " *
 �2 :D
 aten__native_batch_norm_backward aten__native_batch_norm_backward�o�:���)�
�)�)�)�3�)� � �
get-tuple-element.5283get-tuple-element *
 �2 :D
 aten__native_batch_norm_backward aten__native_batch_norm_backwardh��)��)� � �
custom-call.469custom-call *
 �2 :�
'xla___op_TransferWithStaticRingTransfer'xla___op_TransferWithStaticRingTransferv/home/ubuntu/aws-neuron-samples/aws_neuron_venv_pytorch/lib/python3.10/site-packages/torch_xla/core/xla_op_registry.py ,x�AwsNeuronTransferWithStaticRing��/��)��
 �� � �
get-tuple-element.5282get-tuple-element *
 �2 :D
 aten__native_batch_norm_backward aten__native_batch_norm_backwardh��)��)� � �
custom-call.470custom-call *
 �2 :�
'xla___op_TransferWithStaticRingTransfer'xla___op_TransferWithStaticRingTransferv/home/ubuntu/aws-neuron-samples/aws_neuron_venv_pytorch/lib/python3.10/site-packages/torch_xla/core/xla_op_registry.py ,x�AwsNeuronTransferWithStaticRing��/��)��
 �� � �
	p521.5299	parameter ��*	
 �2    :$
xla__device_dataxla__device_dataH���)�

neff_input_namesinput521� �
get-tuple-element.5281get-tuple-element ��*	
 �2    :D
 aten__native_batch_norm_backward aten__native_batch_norm_backward��)��)� � �
convolution.5303convolution  *	
 �2    :R
'aten__convolution_backward_overrideable'aten__convolution_backward_overrideablez
	�(0
	�(0� 28HPZb ��)��)�)��
  �� � �
transpose.5304	transpose  *	
 �2    :R
'aten__convolution_backward_overrideable'aten__convolution_backward_overrideabler ��)��)� � �
custom-call.471custom-call  *	
 �2    :�
'xla___op_TransferWithStaticRingTransfer'xla___op_TransferWithStaticRingTransferv/home/ubuntu/aws-neuron-samples/aws_neuron_venv_pytorch/lib/python3.10/site-packages/torch_xla/core/xla_op_registry.py ,x�AwsNeuronTransferWithStaticRing��/��)��
 �� � �
	p525.5321	parameter ��*	
 �2    :$
xla__device_dataxla__device_dataH���)�

neff_input_namesinput525� y
	p524.5320	parameter *
 �2 :$
xla__device_dataxla__device_dataH���)�

neff_input_namesinput524� y
	p523.5319	parameter *
 �2 :$
xla__device_dataxla__device_dataH���)�

neff_input_namesinput523� �
constant.5334constant*�:D
 aten__native_batch_norm_backward aten__native_batch_norm_backwardB
*�B  �?��)� � �
broadcast.5335	broadcast *
 �2 :D
 aten__native_batch_norm_backward aten__native_batch_norm_backward��)��)� � y
	p522.5318	parameter *
 �2 :$
xla__device_dataxla__device_dataH���)�

neff_input_namesinput522� ~
divide.5336divide *
 �2 :D
 aten__native_batch_norm_backward aten__native_batch_norm_backward��)��)�)� � �
multiply.5337multiply *
 �2 :D
 aten__native_batch_norm_backward aten__native_batch_norm_backward��)��)�)� � >
constant.119constant*�: B
*�Bo����6� � �
broadcast.215	broadcast *
 �2 :D
 aten__native_batch_norm_backward aten__native_batch_norm_backward��3��6� � v
add.73add *
 �2 :D
 aten__native_batch_norm_backward aten__native_batch_norm_backward��3��)�3� � {

slice.5323slice ��*	
 �2    :
xla__selectxla__select�� @������)��(� � �
	p526.5322	parameter ��*	
 �2    :$
xla__device_dataxla__device_dataH���)�

neff_input_namesinput526� q
logistic.5325logistic ��*	
 �2    :*
aten__silu_backwardaten__silu_backward��)��)� � i
constant.5324constant*�:*
aten__silu_backwardaten__silu_backwardB
*�B  �?��)� � s
broadcast.5329	broadcast ��*	
 �2    :*
aten__silu_backwardaten__silu_backward��)��)� � s
broadcast.5326	broadcast ��*	
 �2    :*
aten__silu_backwardaten__silu_backward��)��)� � s
subtract.5327subtract ��*	
 �2    :*
aten__silu_backwardaten__silu_backward��)��)�)� � s
multiply.5328multiply ��*	
 �2    :*
aten__silu_backwardaten__silu_backward��)��)�)� � i
add.5330add ��*	
 �2    :*
aten__silu_backwardaten__silu_backward��)��)�)� � s
multiply.5331multiply ��*	
 �2    :*
aten__silu_backwardaten__silu_backward��)��)�)� � s
multiply.5332multiply ��*	
 �2    :*
aten__silu_backwardaten__silu_backward��)��)�)� � �
batch-norm-grad.5340batch-norm-gradC" ��*	
 �2    " *
 �2 " *
 �2 :D
 aten__native_batch_norm_backward aten__native_batch_norm_backward�o�:���)�
�)�)�)�3�)� � �
get-tuple-element.5343get-tuple-element *
 �2 :D
 aten__native_batch_norm_backward aten__native_batch_norm_backwardh��)��)� � �
custom-call.472custom-call *
 �2 :�
'xla___op_TransferWithStaticRingTransfer'xla___op_TransferWithStaticRingTransferv/home/ubuntu/aws-neuron-samples/aws_neuron_venv_pytorch/lib/python3.10/site-packages/torch_xla/core/xla_op_registry.py ,x�AwsNeuronTransferWithStaticRing��/��)��
 �� � �
get-tuple-element.5342get-tuple-element *
 �2 :D
 aten__native_batch_norm_backward aten__native_batch_norm_backwardh��)��)� � �
custom-call.473custom-call *
 �2 :�
'xla___op_TransferWithStaticRingTransfer'xla___op_TransferWithStaticRingTransferv/home/ubuntu/aws-neuron-samples/aws_neuron_venv_pytorch/lib/python3.10/site-packages/torch_xla/core/xla_op_registry.py ,x�AwsNeuronTransferWithStaticRing��/��)��
 �� � �
	p528.5359	parameter@��*	
 �2    :$
xla__device_dataxla__device_dataH���)�

neff_input_namesinput528� �
get-tuple-element.5341get-tuple-element ��*	
 �2    :D
 aten__native_batch_norm_backward aten__native_batch_norm_backward��)��)� � �
convolution.5363convolution@ *	
 �2    :R
'aten__convolution_backward_overrideable'aten__convolution_backward_overrideablez
	�(0
	�(0� 28HPZb ��)��)�)��
  �� � �
transpose.5364	transpose @*	
 �2    :R
'aten__convolution_backward_overrideable'aten__convolution_backward_overrideabler ��)��)� � �
custom-call.474custom-call @*	
 �2    :�
'xla___op_TransferWithStaticRingTransfer'xla___op_TransferWithStaticRingTransferv/home/ubuntu/aws-neuron-samples/aws_neuron_venv_pytorch/lib/python3.10/site-packages/torch_xla/core/xla_op_registry.py ,x�AwsNeuronTransferWithStaticRing��/��)��
 �� � �
	p532.5381	parameter ��*	
 �2    :$
xla__device_dataxla__device_dataH���*�

neff_input_namesinput532� y
	p531.5380	parameter *
 �2 :$
xla__device_dataxla__device_dataH���*�

neff_input_namesinput531� y
	p530.5379	parameter *
 �2 :$
xla__device_dataxla__device_dataH���*�

neff_input_namesinput530� �
constant.5400constant*�:D
 aten__native_batch_norm_backward aten__native_batch_norm_backwardB
*�B  �?��*� � �
broadcast.5401	broadcast *
 �2 :D
 aten__native_batch_norm_backward aten__native_batch_norm_backward��*��*� � y
	p529.5378	parameter *
 �2 :$
xla__device_dataxla__device_dataH���*�

neff_input_namesinput529� ~
divide.5402divide *
 �2 :D
 aten__native_batch_norm_backward aten__native_batch_norm_backward��*��*�*� � �
multiply.5403multiply *
 �2 :D
 aten__native_batch_norm_backward aten__native_batch_norm_backward��*��*�*� � >
constant.120constant*�: B
*�Bo����6� � �
broadcast.217	broadcast *
 �2 :D
 aten__native_batch_norm_backward aten__native_batch_norm_backward��3��6� � v
add.74add *
 �2 :D
 aten__native_batch_norm_backward aten__native_batch_norm_backward��3��*�3� � �
	p520.5298	parameter  *	
 �2    :$
xla__device_dataxla__device_dataH���)�

neff_input_namesinput520� �
transpose.5300	transpose  *	
 �2    :R
'aten__convolution_backward_overrideable'aten__convolution_backward_overrideabler ��)��)� � �
convolution.5302convolution ��*	
 �2    :R
'aten__convolution_backward_overrideable'aten__convolution_backward_overrideablez
(0
(0� 2 @PZb��)��)�)��
  �� � U
add.5389add ��*	
 �2    :
	aten__add	aten__add��*��(�)� � �
	p533.5382	parameter ��*	
 �2    :$
xla__device_dataxla__device_dataH���*�

neff_input_namesinput533� q
logistic.5391logistic ��*	
 �2    :*
aten__silu_backwardaten__silu_backward��*��*� � i
constant.5390constant*�:*
aten__silu_backwardaten__silu_backwardB
*�B  �?��*� � s
broadcast.5395	broadcast ��*	
 �2    :*
aten__silu_backwardaten__silu_backward��*��*� � s
broadcast.5392	broadcast ��*	
 �2    :*
aten__silu_backwardaten__silu_backward��*��*� � s
subtract.5393subtract ��*	
 �2    :*
aten__silu_backwardaten__silu_backward��*��*�*� � s
multiply.5394multiply ��*	
 �2    :*
aten__silu_backwardaten__silu_backward��*��*�*� � i
add.5396add ��*	
 �2    :*
aten__silu_backwardaten__silu_backward��*��*�*� � s
multiply.5397multiply ��*	
 �2    :*
aten__silu_backwardaten__silu_backward��*��*�*� � s
multiply.5398multiply ��*	
 �2    :*
aten__silu_backwardaten__silu_backward��*��*�*� � �
batch-norm-grad.5406batch-norm-gradC" ��*	
 �2    " *
 �2 " *
 �2 :D
 aten__native_batch_norm_backward aten__native_batch_norm_backward�o�:���*�
�*�*�*�3�*� � �
get-tuple-element.5409get-tuple-element *
 �2 :D
 aten__native_batch_norm_backward aten__native_batch_norm_backwardh��*��*� � �
custom-call.475custom-call *
 �2 :�
'xla___op_TransferWithStaticRingTransfer'xla___op_TransferWithStaticRingTransferv/home/ubuntu/aws-neuron-samples/aws_neuron_venv_pytorch/lib/python3.10/site-packages/torch_xla/core/xla_op_registry.py ,x�AwsNeuronTransferWithStaticRing��/��*��
 �� � �
get-tuple-element.5408get-tuple-element *
 �2 :D
 aten__native_batch_norm_backward aten__native_batch_norm_backwardh��*��*� � �
custom-call.476custom-call *
 �2 :�
'xla___op_TransferWithStaticRingTransfer'xla___op_TransferWithStaticRingTransferv/home/ubuntu/aws-neuron-samples/aws_neuron_venv_pytorch/lib/python3.10/site-packages/torch_xla/core/xla_op_registry.py ,x�AwsNeuronTransferWithStaticRing��/��*��
 �� � �
get-tuple-element.5407get-tuple-element ��*	
 �2    :D
 aten__native_batch_norm_backward aten__native_batch_norm_backward��*��*� � �
convolution.5428convolution@ *	
 �2    :R
'aten__convolution_backward_overrideable'aten__convolution_backward_overrideablez
	�(0
	�(0� 28HPZb ��*��)�*��
  �� � �
transpose.5429	transpose @*	
 �2    :R
'aten__convolution_backward_overrideable'aten__convolution_backward_overrideabler ��*��*� � �
custom-call.477custom-call @*	
 �2    :�
'xla___op_TransferWithStaticRingTransfer'xla___op_TransferWithStaticRingTransferv/home/ubuntu/aws-neuron-samples/aws_neuron_venv_pytorch/lib/python3.10/site-packages/torch_xla/core/xla_op_registry.py ,x�AwsNeuronTransferWithStaticRing��/��*��
 �� � �
	p538.5446	parameter@��*	
 �2    :$
xla__device_dataxla__device_dataH���*�

neff_input_namesinput538� y
	p537.5445	parameter@*
 �2 :$
xla__device_dataxla__device_dataH���*�

neff_input_namesinput537� y
	p536.5444	parameter@*
 �2 :$
xla__device_dataxla__device_dataH���*�

neff_input_namesinput536� �
constant.5465constant*�:D
 aten__native_batch_norm_backward aten__native_batch_norm_backwardB
*�B  �?��*� � �
broadcast.5466	broadcast@*
 �2 :D
 aten__native_batch_norm_backward aten__native_batch_norm_backward��*��*� � y
	p535.5443	parameter@*
 �2 :$
xla__device_dataxla__device_dataH���*�

neff_input_namesinput535� ~
divide.5467divide@*
 �2 :D
 aten__native_batch_norm_backward aten__native_batch_norm_backward��*��*�*� � �
multiply.5468multiply@*
 �2 :D
 aten__native_batch_norm_backward aten__native_batch_norm_backward��*��*�*� � >
constant.121constant*�: B
*�Bo����6� � �
broadcast.221	broadcast@*
 �2 :D
 aten__native_batch_norm_backward aten__native_batch_norm_backward��3��6� � v
add.75add@*
 �2 :D
 aten__native_batch_norm_backward aten__native_batch_norm_backward��3��*�3� � �
	p527.5358	parameter @*	
 �2    :$
xla__device_dataxla__device_dataH���)�

neff_input_namesinput527� �
transpose.5360	transpose@ *	
 �2    :R
'aten__convolution_backward_overrideable'aten__convolution_backward_overrideabler ��)��)� � �
convolution.5362convolution@��*	
 �2    :R
'aten__convolution_backward_overrideable'aten__convolution_backward_overrideablez
(0
(0� 2 @PZb��)��)�)��
  �� � �
	p534.5424	parameter @*	
 �2    :$
xla__device_dataxla__device_dataH���*�

neff_input_namesinput534� �
transpose.5425	transpose@ *	
 �2    :R
'aten__convolution_backward_overrideable'aten__convolution_backward_overrideabler ��*��*� � �
convolution.5427convolution@��*	
 �2    :R
'aten__convolution_backward_overrideable'aten__convolution_backward_overrideablez
(0
(0� 2 @PZb��*��*�*��
  �� � U
add.5454add@��*	
 �2    :
	aten__add	aten__add��*��)�*� � �
	p539.5447	parameter@��*	
 �2    :$
xla__device_dataxla__device_dataH���*�

neff_input_namesinput539� q
logistic.5456logistic@��*	
 �2    :*
aten__silu_backwardaten__silu_backward��*��*� � i
constant.5455constant*�:*
aten__silu_backwardaten__silu_backwardB
*�B  �?��*� � s
broadcast.5460	broadcast@��*	
 �2    :*
aten__silu_backwardaten__silu_backward��*��*� � s
broadcast.5457	broadcast@��*	
 �2    :*
aten__silu_backwardaten__silu_backward��*��*� � s
subtract.5458subtract@��*	
 �2    :*
aten__silu_backwardaten__silu_backward��*��*�*� � s
multiply.5459multiply@��*	
 �2    :*
aten__silu_backwardaten__silu_backward��*��*�*� � i
add.5461add@��*	
 �2    :*
aten__silu_backwardaten__silu_backward��*��*�*� � s
multiply.5462multiply@��*	
 �2    :*
aten__silu_backwardaten__silu_backward��*��*�*� � s
multiply.5463multiply@��*	
 �2    :*
aten__silu_backwardaten__silu_backward��*��*�*� � �
batch-norm-grad.5471batch-norm-gradC"@��*	
 �2    "@*
 �2 "@*
 �2 :D
 aten__native_batch_norm_backward aten__native_batch_norm_backward�o�:���*�
�*�*�*�3�*� � �
get-tuple-element.5474get-tuple-element@*
 �2 :D
 aten__native_batch_norm_backward aten__native_batch_norm_backwardh��*��*� � �
custom-call.478custom-call@*
 �2 :�
'xla___op_TransferWithStaticRingTransfer'xla___op_TransferWithStaticRingTransferv/home/ubuntu/aws-neuron-samples/aws_neuron_venv_pytorch/lib/python3.10/site-packages/torch_xla/core/xla_op_registry.py ,x�AwsNeuronTransferWithStaticRing��/��*��
 �� � �
get-tuple-element.5473get-tuple-element@*
 �2 :D
 aten__native_batch_norm_backward aten__native_batch_norm_backwardh��*��*� � �
custom-call.479custom-call@*
 �2 :�
'xla___op_TransferWithStaticRingTransfer'xla___op_TransferWithStaticRingTransferv/home/ubuntu/aws-neuron-samples/aws_neuron_venv_pytorch/lib/python3.10/site-packages/torch_xla/core/xla_op_registry.py ,x�AwsNeuronTransferWithStaticRing��/��*��
 �� � �
	p541.5490	parameter ��*	
 �2    :$
xla__device_dataxla__device_dataH���*�

neff_input_namesinput541� �
get-tuple-element.5472get-tuple-element@��*	
 �2    :D
 aten__native_batch_norm_backward aten__native_batch_norm_backward��*��*� � �
convolution.5494convolution @*	
 �2    :R
'aten__convolution_backward_overrideable'aten__convolution_backward_overrideablez
�(0
�(0� 28HPZb ��*��*�*��
  �� � �
transpose.5495	transpose@ *	
 �2    :R
'aten__convolution_backward_overrideable'aten__convolution_backward_overrideabler ��*��*� � �
custom-call.480custom-call@ *	
 �2    :�
'xla___op_TransferWithStaticRingTransfer'xla___op_TransferWithStaticRingTransferv/home/ubuntu/aws-neuron-samples/aws_neuron_venv_pytorch/lib/python3.10/site-packages/torch_xla/core/xla_op_registry.py ,x�AwsNeuronTransferWithStaticRing��/��*��
 �� � �
	p545.5512	parameter ��*	
 �2    :$
xla__device_dataxla__device_dataH���+�

neff_input_namesinput545� y
	p544.5511	parameter *
 �2 :$
xla__device_dataxla__device_dataH���+�

neff_input_namesinput544� y
	p543.5510	parameter *
 �2 :$
xla__device_dataxla__device_dataH���+�

neff_input_namesinput543� �
constant.5524constant*�:D
 aten__native_batch_norm_backward aten__native_batch_norm_backwardB
*�B  �?��+� � �
broadcast.5525	broadcast *
 �2 :D
 aten__native_batch_norm_backward aten__native_batch_norm_backward��+��+� � y
	p542.5509	parameter *
 �2 :$
xla__device_dataxla__device_dataH���+�

neff_input_namesinput542� ~
divide.5526divide *
 �2 :D
 aten__native_batch_norm_backward aten__native_batch_norm_backward��+��+�+� � �
multiply.5527multiply *
 �2 :D
 aten__native_batch_norm_backward aten__native_batch_norm_backward��+��+�+� � >
constant.122constant*�: B
*�Bo����6� � �
broadcast.223	broadcast *
 �2 :D
 aten__native_batch_norm_backward aten__native_batch_norm_backward��3��6� � v
add.76add *
 �2 :D
 aten__native_batch_norm_backward aten__native_batch_norm_backward��3��+�3� � �
	p540.5489	parameter@ *	
 �2    :$
xla__device_dataxla__device_dataH���*�

neff_input_namesinput540� �
transpose.5491	transpose @*	
 �2    :R
'aten__convolution_backward_overrideable'aten__convolution_backward_overrideabler ��*��*� � �
reverse.5492reverse @*	
 �2    :R
'aten__convolution_backward_overrideable'aten__convolution_backward_overrideabler ��*��*� � �
convolution.5493convolution ��*	
 �2    :R
'aten__convolution_backward_overrideable'aten__convolution_backward_overrideablez
 (0
 (0� 2 @PZb��*��*�*��
  �� � �
	p546.5513	parameter ��*	
 �2    :$
xla__device_dataxla__device_dataH���+�

neff_input_namesinput546� q
logistic.5515logistic ��*	
 �2    :*
aten__silu_backwardaten__silu_backward��+��+� � i
constant.5514constant*�:*
aten__silu_backwardaten__silu_backwardB
*�B  �?��+� � s
broadcast.5519	broadcast ��*	
 �2    :*
aten__silu_backwardaten__silu_backward��+��+� � s
broadcast.5516	broadcast ��*	
 �2    :*
aten__silu_backwardaten__silu_backward��+��+� � s
subtract.5517subtract ��*	
 �2    :*
aten__silu_backwardaten__silu_backward��+��+�+� � s
multiply.5518multiply ��*	
 �2    :*
aten__silu_backwardaten__silu_backward��+��+�+� � i
add.5520add ��*	
 �2    :*
aten__silu_backwardaten__silu_backward��+��+�+� � s
multiply.5521multiply ��*	
 �2    :*
aten__silu_backwardaten__silu_backward��+��+�+� � s
multiply.5522multiply ��*	
 �2    :*
aten__silu_backwardaten__silu_backward��+��*�+� � �
batch-norm-grad.5530batch-norm-gradC" ��*	
 �2    " *
 �2 " *
 �2 :D
 aten__native_batch_norm_backward aten__native_batch_norm_backward�o�:���+�
�+�+�+�3�+� � �
get-tuple-element.5533get-tuple-element *
 �2 :D
 aten__native_batch_norm_backward aten__native_batch_norm_backwardh��+��+� � �
custom-call.481custom-call *
 �2 :�
'xla___op_TransferWithStaticRingTransfer'xla___op_TransferWithStaticRingTransferv/home/ubuntu/aws-neuron-samples/aws_neuron_venv_pytorch/lib/python3.10/site-packages/torch_xla/core/xla_op_registry.py ,x�AwsNeuronTransferWithStaticRing��/��+��
 �� � �
get-tuple-element.5532get-tuple-element *
 �2 :D
 aten__native_batch_norm_backward aten__native_batch_norm_backwardh��+��+� � �
custom-call.482custom-call *
 �2 :�
'xla___op_TransferWithStaticRingTransfer'xla___op_TransferWithStaticRingTransferv/home/ubuntu/aws-neuron-samples/aws_neuron_venv_pytorch/lib/python3.10/site-packages/torch_xla/core/xla_op_registry.py ,x�AwsNeuronTransferWithStaticRing��/��+��
 �� � �
	p548.5549	parameter��*	
 �2    :$
xla__device_dataxla__device_dataH���+�

neff_input_namesinput548� �
get-tuple-element.5531get-tuple-element ��*	
 �2    :D
 aten__native_batch_norm_backward aten__native_batch_norm_backward��+��+� � �
convolution.5553convolution *	
 �2    :R
'aten__convolution_backward_overrideable'aten__convolution_backward_overrideablez
� (0
� (0� 28HPZb ��+��+�+��
  �� � �
transpose.5554	transpose *	
 �2    :R
'aten__convolution_backward_overrideable'aten__convolution_backward_overrideabler ��+��+� � �
custom-call.483custom-call *	
 �2    :�
'xla___op_TransferWithStaticRingTransfer'xla___op_TransferWithStaticRingTransferv/home/ubuntu/aws-neuron-samples/aws_neuron_venv_pytorch/lib/python3.10/site-packages/torch_xla/core/xla_op_registry.py ,x�AwsNeuronTransferWithStaticRing��/��+��
 �� � �?

tuple.5568tuple�)"*�"*�"*
 �2 "�*	
 �2    "*
 �2 "�*	
 �2    "�*
 �2 "�*
 �2 "��*	
 �2    "�*
 �2 "�*
 �2 "��*	
 �2    "P*
 �2 "P�*	
 �2    "�*
 �2 "�*
 �2 "��*	
 �2    "�*
 �2 "�*
 �2 "��*	
 �2    "�*
 �2 "�*
 �2 "��*	
 �2    "*
 �2 "�*	
 �2    "*
 �2 "�*	
 �2    "�*
 �2 "�*
 �2 "��*	
 �2    "�*
 �2 "�*
 �2 "��*	
 �2    "P*
 �2 "P�*	
 �2    "�*
 �2 "�*
 �2 "��*	
 �2    "�*
 �2 "�*
 �2 "��*	
 �2    "�*
 �2 "�*
 �2 "��*	
 �2    "*
 �2 "�*	
 �2    "*
 �2 "�*	
 �2    "�*
 �2 "�*
 �2 "��*	
 �2    "�*
 �2 "�*
 �2 "��*	
 �2    "P*
 �2 "P�*	
 �2    "�*
 �2 "�*
 �2 "��*	
 �2    "�*
 �2 "�*
 �2 "��*	
 �2    "�*
 �2 "�*
 �2 "��*	
 �2    "�*
 �2 "�*
 �2 "��*	
 �2    "�*
 �2 "�*
 �2 "��*	
 �2    "�*
 �2 "�*
 �2 "��*	
 �2    "�*
 �2 "�*
 �2 "��*	
 �2    "�*
 �2 "�*
 �2 "��*	
 �2    "�*
 �2 "�*
 �2 "��*	
 �2    "�*
 �2 "�*
 �2 "��*	
 �2    "�*
 �2 "�*
 �2 "��*	
 �2    "�*
 �2 "�*
 �2 "��*	
 �2    "�*
 �2 "�*
 �2 "��*	
 �2    "�*
 �2 "�*
 �2 "��*	
 �2    "�*
 �2 "�*
 �2 "��*	
 �2    "�*
 �2 "�*
 �2 "��*	
 �2    "@*
 �2 "@*
 �2 "@@*	
 �2    "@*
 �2 "@*
 �2 "@@*	
 �2    "@*
 �2 "@*
 �2 "@�*	
 �2    "@*
 �2 "@*
 �2 "@�*	
 �2    "�*
 �2 "�*
 �2 "��*	
 �2    "�*
 �2 "�*
 �2 "��*	
 �2    "�*
 �2 "�*
 �2 "��*	
 �2    "�*
 �2 "�*
 �2 "��*	
 �2    "�*
 �2 "�*
 �2 "��*	
 �2    "�*
 �2 "�*
 �2 "��*	
 �2    "�*
 �2 "�*
 �2 "��*	
 �2    "�*
 �2 "�*
 �2 "��*	
 �2    "�*
 �2 "�*
 �2 "��*	
 �2    "�*
 �2 "�*
 �2 "��*	
 �2    "�*
 �2 "�*
 �2 "��*	
 �2    "�*
 �2 "�*
 �2 "��*	
 �2    "�*
 �2 "�*
 �2 "��*	
 �2    "�*
 �2 "�*
 �2 "��*	
 �2    "�*
 �2 "�*
 �2 "��*	
 �2    "�*
 �2 "�*
 �2 "��*	
 �2    "�*
 �2 "�*
 �2 "��*	
 �2    "�*
 �2 "�*
 �2 "��*	
 �2    "�*
 �2 "�*
 �2 "��*	
 �2    "�*
 �2 "�*
 �2 "��*	
 �2    "�*
 �2 "�*
 �2 "��*	
 �2    "�*
 �2 "�*
 �2 "��*	
 �2    "�*
 �2 "�*
 �2 "��*	
 �2    "�*
 �2 "�*
 �2 "��*	
 �2    "�*
 �2 "�*
 �2 "��*	
 �2    "�*
 �2 "�*
 �2 "��*	
 �2    "@*
 �2 "@*
 �2 "@@*	
 �2    "@*
 �2 "@*
 �2 "@@*	
 �2    "@*
 �2 "@*
 �2 "@@*	
 �2    "@*
 �2 "@*
 �2 "@@*	
 �2    "@*
 �2 "@*
 �2 "@@*	
 �2    "@*
 �2 "@*
 �2 "@@*	
 �2    "@*
 �2 "@*
 �2 "@�*	
 �2    "@*
 �2 "@*
 �2 "@�*	
 �2    "�*
 �2 "�*
 �2 "�@*	
 �2    "@*
 �2 "@*
 �2 "@@*	
 �2    " *
 �2 " *
 �2 "  *	
 �2    " *
 �2 " *
 �2 "  *	
 �2    " *
 �2 " *
 �2 " @*	
 �2    " *
 �2 " *
 �2 " @*	
 �2    "@*
 �2 "@*
 �2 "@ *	
 �2    " *
 �2 " *
 �2 " *	
 �2    : ��+���-�-�-�-�-�-�-�-�-�-�-�-�-�-�-�-�-�-�-�-�-�-�-�-�-�-�-�-�-�-�-�-�-�-�-�-�-�-�-�-�-�-�-�-�-�-�-�-�-�-�-�-�-�-�-�-�-�-�-�-�-�-�-�-�-�-�-�-�-�-�-�-�-�-�.�.�.�.�.�.�.�.�.�.�.�.�.�.�.�.�.�.�.�.�.�.�.�.�.�.�.�.�.�.�.�.�.�.�.�.�.�.�.�.�.�.�.�.�.�.�.�.�.�.�.�.�.�.�.�.�.�.�.�.�.�.�.�.�.�.�.�.�.�.�.�.�.�.�.�.�.�.�.�.�.�.�.�.�.�.�.�.�.�.�.�.�.�.�.�.�.�.�.�.�.�.�.�.�.�.�.�.�.�.�.�.�.�.�.�.�.�.�.�.�.�.�.�.�.�.�.�.�/�/�/�/�/�/�/�/�/�/�/�/�/�/�/�/�/�/�/�/�/�/�/�/�/�/�/�/�/�/�/�/�/�/�/�/�/�/��
�
neff_output_names�output0,output1,output2,output3,output4,output5,output6,output7,output8,output9,output10,output11,output12,output13,output14,output15,output16,output17,output18,output19,output20,output21,output22,output23,output24,output25,output26,output27,output28,output29,output30,output31,output32,output33,output34,output35,output36,output37,output38,output39,output40,output41,output42,output43,output44,output45,output46,output47,output48,output49,output50,output51,output52,output53,output54,output55,output56,output57,output58,output59,output60,output61,output62,output63,output64,output65,output66,output67,output68,output69,output70,output71,output72,output73,output74,output75,output76,output77,output78,output79,output80,output81,output82,output83,output84,output85,output86,output87,output88,output89,output90,output91,output92,output93,output94,output95,output96,output97,output98,output99,output100,output101,output102,output103,output104,output105,output106,output107,output108,output109,output110,output111,output112,output113,output114,output115,output116,output117,output118,output119,output120,output121,output122,output123,output124,output125,output126,output127,output128,output129,output130,output131,output132,output133,output134,output135,output136,output137,output138,output139,output140,output141,output142,output143,output144,output145,output146,output147,output148,output149,output150,output151,output152,output153,output154,output155,output156,output157,output158,output159,output160,output161,output162,output163,output164,output165,output166,output167,output168,output169,output170,output171,output172,output173,output174,output175,output176,output177,output178,output179,output180,output181,output182,output183,output184,output185,output186,output187,output188,output189,output190,output191,output192,output193,output194,output195,output196,output197,output198,output199,output200,output201,output202,output203,output204,output205,output206,output207,output208,output209,output210,output211,output212,output213,output214,output215,output216,output217,output218,output219,output220,output221,output222,output223,output224,output225,output226,output227,output228,output229,output230,output231,output232,output233,output234,output235,output236,output237,output238,output239,output240,output241� "��
*�
�*	
 �2    
�*	
 �2    
*�
�*
 �2   
*
 �2  
*
 �2  
*
 �2  
*
 �2 
*
 �2 
*
 �2 
*
 �2 
*�
*
 �2 
*�
*
 �2  
*
 �2  
*
 �2  
*
 �2 
*
 �2  
*�
�A*
 �2  
�A*
 �2  
P*
 �2  
P*
 �2  
*
 �2  
�*	
 �2    
�*
 �2 
�*
 �2 
�*
 �2 
�*	
 �2    
�*	
 �2    
��*	
 �2    
�*	
 �2    
�*
 �2 
�*
 �2 
�*
 �2 
�*	
 �2    
�*	
 �2    
��*	
 �2    
�*	
 �2    
P�*	
 �2    
�*	
 �2    
�*
 �2 
�*
 �2 
�*
 �2 
�*	
 �2    
�*	
 �2    
��*	
 �2    
�*	
 �2    
�*
 �2 
�*
 �2 
�*
 �2 
�*	
 �2    
�*	
 �2    
��*	
 �2    
�*
 �2 
�*
 �2 
�*
 �2 
�*	
 �2    
�*	
 �2    
��*	
 �2    
�*	
 �2    
�*	
 �2    
�((*	
 �2    
*�
�*
 �2   
�*	
 �2    
�*
 �2 
�*
 �2 
�*
 �2 
�((*	
 �2    
�((*	
 �2    
��*	
 �2    
�((*	
 �2    
�*
 �2 
�*
 �2 
�*
 �2 
�((*	
 �2    
�((*	
 �2    
��*	
 �2    
�((*	
 �2    
P�*	
 �2    
�((*	
 �2    
�*
 �2 
�*
 �2 
�*
 �2 
�((*	
 �2    
�((*	
 �2    
��*	
 �2    
�((*	
 �2    
�*
 �2 
�*
 �2 
�*
 �2 
�((*	
 �2    
�((*	
 �2    
��*	
 �2    
�*
 �2 
�*
 �2 
�*
 �2 
�((*	
 �2    
�((*	
 �2    
��*	
 �2    
�((*	
 �2    
�*	
 �2    
�PP*	
 �2    
*�
�2*
 �2   
�*	
 �2    
�*
 �2 
�*
 �2 
�*
 �2 
�PP*	
 �2    
�PP*	
 �2    
��*	
 �2    
�PP*	
 �2    
�*
 �2 
�*
 �2 
�*
 �2 
�PP*	
 �2    
�PP*	
 �2    
��*	
 �2    
�PP*	
 �2    
P�*	
 �2    
�PP*	
 �2    
�*
 �2 
�*
 �2 
�*
 �2 
�PP*	
 �2    
�PP*	
 �2    
��*	
 �2    
�PP*	
 �2    
�*
 �2 
�*
 �2 
�*
 �2 
�PP*	
 �2    
�PP*	
 �2    
��*	
 �2    
�*
 �2 
�*
 �2 
�*
 �2 
�PP*	
 �2    
�PP*	
 �2    
��*	
 �2    
�PP*	
 �2    
�*
 �2 
�*
 �2 
�*
 �2 
�*	
 �2    
�*	
 �2    
��*	
 �2    
�*	
 �2    
�*
 �2 
�*
 �2 
�*
 �2 
�*	
 �2    
�*	
 �2    
��*	
 �2    
�*	
 �2    
�*
 �2 
�*
 �2 
�*
 �2 
�*	
 �2    
�*	
 �2    
��*	
 �2    
�*	
 �2    
�*
 �2 
�*
 �2 
�*
 �2 
�*	
 �2    
�*	
 �2    
��*	
 �2    
�*	
 �2    
�*
 �2 
�*
 �2 
�*
 �2 
�*	
 �2    
�*	
 �2    
��*	
 �2    
�*
 �2 
�*
 �2 
�*
 �2 
�*	
 �2    
�*	
 �2    
��*	
 �2    
�*
 �2 
�*
 �2 
�*
 �2 
�((*	
 �2    
�((*	
 �2    
��*	
 �2    
�((*	
 �2    
�*
 �2 
�*
 �2 
�*
 �2 
�((*	
 �2    
�((*	
 �2    
��*	
 �2    
�((*	
 �2    
�*
 �2 
�*
 �2 
�*
 �2 
�((*	
 �2    
�((*	
 �2    
��*	
 �2    
�((*	
 �2    
�*
 �2 
�*
 �2 
�*
 �2 
�((*	
 �2    
�((*	
 �2    
��*	
 �2    
�((*	
 �2    
�*
 �2 
�*
 �2 
�*
 �2 
�((*	
 �2    
�((*	
 �2    
��*	
 �2    
�*
 �2 
�*
 �2 
�*
 �2 
�((*	
 �2    
�((*	
 �2    
��*	
 �2    
�*
 �2 
�*
 �2 
�*
 �2 
�PP*	
 �2    
�PP*	
 �2    
��*	
 �2    
�PP*	
 �2    
@*
 �2 
@*
 �2 
@*
 �2 
@PP*	
 �2    
@PP*	
 �2    
@@*	
 �2    
@PP*	
 �2    
@*
 �2 
@*
 �2 
@*
 �2 
@PP*	
 �2    
@PP*	
 �2    
@@*	
 �2    
@PP*	
 �2    
@*
 �2 
@*
 �2 
@*
 �2 
@PP*	
 �2    
@PP*	
 �2    
@�*	
 �2    
�PP*	
 �2    
@*
 �2 
@*
 �2 
@*
 �2 
@PP*	
 �2    
@PP*	
 �2    
@�*	
 �2    
�*
 �2 
�*
 �2 
�*
 �2 
�((*	
 �2    
�((*	
 �2    
��*	
 �2    
�((*	
 �2    
�*
 �2 
�*
 �2 
�*
 �2 
�((*	
 �2    
�((*	
 �2    
��*	
 �2    
�((*	
 �2    
�*
 �2 
�*
 �2 
�*
 �2 
�((*	
 �2    
�((*	
 �2    
��*	
 �2    
�((*	
 �2    
�*
 �2 
�*
 �2 
�*
 �2 
�((*	
 �2    
�((*	
 �2    
��*	
 �2    
�((*	
 �2    
�*
 �2 
�*
 �2 
�*
 �2 
�((*	
 �2    
�((*	
 �2    
��*	
 �2    
�((*	
 �2    
�*
 �2 
�*
 �2 
�*
 �2 
�((*	
 �2    
�((*	
 �2    
��*	
 �2    
�*
 �2 
�*
 �2 
�*
 �2 
�*	
 �2    
�*	
 �2    
��*	
 �2    
�*	
 �2    
�*
 �2 
�*
 �2 
�*
 �2 
�*	
 �2    
�*	
 �2    
��*	
 �2    
�*	
 �2    
�*
 �2 
�*
 �2 
�*
 �2 
�*	
 �2    
�*	
 �2    
��*	
 �2    
�*	
 �2    
�*
 �2 
�*
 �2 
�*
 �2 
�*	
 �2    
�*	
 �2    
��*	
 �2    
�*	
 �2    
�*
 �2 
�*
 �2 
�*
 �2 
�*	
 �2    
�*	
 �2    
��*	
 �2    
�*	
 �2    
�*
 �2 
�*
 �2 
�*
 �2 
�*	
 �2    
�*	
 �2    
��*	
 �2    
�*
 �2 
�*
 �2 
�*
 �2 
�*	
 �2    
�*	
 �2    
��*	
 �2    
�*	
 �2    
�*
 �2 
�*
 �2 
�*
 �2 
�*	
 �2    
�*	
 �2    
�*	
 �2    
��*	
 �2    
�*	
 �2    
�*
 �2 
�*
 �2 
�*
 �2 
�*	
 �2    
�*	
 �2    
��*	
 �2    
�((*	
 �2    
�*
 �2 
�*
 �2 
�*
 �2 
�((*	
 �2    
�((*	
 �2    
��*	
 �2    
�((*	
 �2    
�*
 �2 
�*
 �2 
�*
 �2 
�((*	
 �2    
�((*	
 �2    
��*	
 �2    
�((*	
 �2    
�*
 �2 
�*
 �2 
�*
 �2 
�((*	
 �2    
�((*	
 �2    
��*	
 �2    
�((*	
 �2    
�*
 �2 
�*
 �2 
�*
 �2 
�((*	
 �2    
�((*	
 �2    
��*	
 �2    
�((*	
 �2    
�*
 �2 
�*
 �2 
�*
 �2 
�((*	
 �2    
�((*	
 �2    
��*	
 �2    
�((*	
 �2    
�*
 �2 
�*
 �2 
�*
 �2 
�((*	
 �2    
�((*	
 �2    
��*	
 �2    
�((*	
 �2    
�*
 �2 
�*
 �2 
�*
 �2 
�((*	
 �2    
�((*	
 �2    
��*	
 �2    
�((*	
 �2    
�*
 �2 
�*
 �2 
�*
 �2 
�((*	
 �2    
�((*	
 �2    
��*	
 �2    
�((*	
 �2    
�*
 �2 
�*
 �2 
�*
 �2 
�((*	
 �2    
�((*	
 �2    
��*	
 �2    
�*
 �2 
�*
 �2 
�*
 �2 
�((*	
 �2    
�((*	
 �2    
��*	
 �2    
�PP*	
 �2    
�*
 �2 
�*
 �2 
�*
 �2 
�PP*	
 �2    
�PP*	
 �2    
��*	
 �2    
�PP*	
 �2    
@*
 �2 
@*
 �2 
@*
 �2 
@PP*	
 �2    
@PP*	
 �2    
@@*	
 �2    
@PP*	
 �2    
@*
 �2 
@*
 �2 
@*
 �2 
@PP*	
 �2    
@PP*	
 �2    
@@*	
 �2    
@PP*	
 �2    
@*
 �2 
@*
 �2 
@*
 �2 
@PP*	
 �2    
@PP*	
 �2    
@@*	
 �2    
@PP*	
 �2    
@*
 �2 
@*
 �2 
@*
 �2 
@PP*	
 �2    
@PP*	
 �2    
@@*	
 �2    
@PP*	
 �2    
@*
 �2 
@*
 �2 
@*
 �2 
@PP*	
 �2    
@PP*	
 �2    
@@*	
 �2    
@PP*	
 �2    
@*
 �2 
@*
 �2 
@*
 �2 
@PP*	
 �2    
@PP*	
 �2    
@@*	
 �2    
@PP*	
 �2    
@*
 �2 
@*
 �2 
@*
 �2 
@PP*	
 �2    
@PP*	
 �2    
@�*	
 �2    
�PP*	
 �2    
@*
 �2 
@*
 �2 
@*
 �2 
@PP*	
 �2    
@PP*	
 �2    
@�*	
 �2    
�*
 �2 
�*
 �2 
�*
 �2 
�PP*	
 �2    
�PP*	
 �2    
�@*	
 �2    
@��*	
 �2    
@*
 �2 
@*
 �2 
@*
 �2 
@��*	
 �2    
@��*	
 �2    
@@*	
 �2    
@��*	
 �2    
 *
 �2 
 *
 �2 
 *
 �2 
 ��*	
 �2    
 ��*	
 �2    
  *	
 �2    
 ��*	
 �2    
 *
 �2 
 *
 �2 
 *
 �2 
 ��*	
 �2    
 ��*	
 �2    
  *	
 �2    
 ��*	
 �2    
 *
 �2 
 *
 �2 
 *
 �2 
 ��*	
 �2    
 ��*	
 �2    
 @*	
 �2    
@��*	
 �2    
 *
 �2 
 *
 �2 
 *
 �2 
 ��*	
 �2    
 ��*	
 �2    
 @*	
 �2    
@*
 �2 
@*
 �2 
@*
 �2 
@��*	
 �2    
@��*	
 �2    
@ *	
 �2    
 ��*	
 �2    
 *
 �2 
 *
 �2 
 *
 �2 
 ��*	
 �2    
 ��*	
 �2    
 *	
 �2    
��*	
 �2    �)"*�"*�"*
 �2 "�*	
 �2    "*
 �2 "�*	
 �2    "�*
 �2 "�*
 �2 "��*	
 �2    "�*
 �2 "�*
 �2 "��*	
 �2    "P*
 �2 "P�*	
 �2    "�*
 �2 "�*
 �2 "��*	
 �2    "�*
 �2 "�*
 �2 "��*	
 �2    "�*
 �2 "�*
 �2 "��*	
 �2    "*
 �2 "�*	
 �2    "*
 �2 "�*	
 �2    "�*
 �2 "�*
 �2 "��*	
 �2    "�*
 �2 "�*
 �2 "��*	
 �2    "P*
 �2 "P�*	
 �2    "�*
 �2 "�*
 �2 "��*	
 �2    "�*
 �2 "�*
 �2 "��*	
 �2    "�*
 �2 "�*
 �2 "��*	
 �2    "*
 �2 "�*	
 �2    "*
 �2 "�*	
 �2    "�*
 �2 "�*
 �2 "��*	
 �2    "�*
 �2 "�*
 �2 "��*	
 �2    "P*
 �2 "P�*	
 �2    "�*
 �2 "�*
 �2 "��*	
 �2    "�*
 �2 "�*
 �2 "��*	
 �2    "�*
 �2 "�*
 �2 "��*	
 �2    "�*
 �2 "�*
 �2 "��*	
 �2    "�*
 �2 "�*
 �2 "��*	
 �2    "�*
 �2 "�*
 �2 "��*	
 �2    "�*
 �2 "�*
 �2 "��*	
 �2    "�*
 �2 "�*
 �2 "��*	
 �2    "�*
 �2 "�*
 �2 "��*	
 �2    "�*
 �2 "�*
 �2 "��*	
 �2    "�*
 �2 "�*
 �2 "��*	
 �2    "�*
 �2 "�*
 �2 "��*	
 �2    "�*
 �2 "�*
 �2 "��*	
 �2    "�*
 �2 "�*
 �2 "��*	
 �2    "�*
 �2 "�*
 �2 "��*	
 �2    "�*
 �2 "�*
 �2 "��*	
 �2    "@*
 �2 "@*
 �2 "@@*	
 �2    "@*
 �2 "@*
 �2 "@@*	
 �2    "@*
 �2 "@*
 �2 "@�*	
 �2    "@*
 �2 "@*
 �2 "@�*	
 �2    "�*
 �2 "�*
 �2 "��*	
 �2    "�*
 �2 "�*
 �2 "��*	
 �2    "�*
 �2 "�*
 �2 "��*	
 �2    "�*
 �2 "�*
 �2 "��*	
 �2    "�*
 �2 "�*
 �2 "��*	
 �2    "�*
 �2 "�*
 �2 "��*	
 �2    "�*
 �2 "�*
 �2 "��*	
 �2    "�*
 �2 "�*
 �2 "��*	
 �2    "�*
 �2 "�*
 �2 "��*	
 �2    "�*
 �2 "�*
 �2 "��*	
 �2    "�*
 �2 "�*
 �2 "��*	
 �2    "�*
 �2 "�*
 �2 "��*	
 �2    "�*
 �2 "�*
 �2 "��*	
 �2    "�*
 �2 "�*
 �2 "��*	
 �2    "�*
 �2 "�*
 �2 "��*	
 �2    "�*
 �2 "�*
 �2 "��*	
 �2    "�*
 �2 "�*
 �2 "��*	
 �2    "�*
 �2 "�*
 �2 "��*	
 �2    "�*
 �2 "�*
 �2 "��*	
 �2    "�*
 �2 "�*
 �2 "��*	
 �2    "�*
 �2 "�*
 �2 "��*	
 �2    "�*
 �2 "�*
 �2 "��*	
 �2    "�*
 �2 "�*
 �2 "��*	
 �2    "�*
 �2 "�*
 �2 "��*	
 �2    "�*
 �2 "�*
 �2 "��*	
 �2    "�*
 �2 "�*
 �2 "��*	
 �2    "@*
 �2 "@*
 �2 "@@*	
 �2    "@*
 �2 "@*
 �2 "@@*	
 �2    "@*
 �2 "@*
 �2 "@@*	
 �2    "@*
 �2 "@*
 �2 "@@*	
 �2    "@*
 �2 "@*
 �2 "@@*	
 �2    "@*
 �2 "@*
 �2 "@@*	
 �2    "@*
 �2 "@*
 �2 "@�*	
 �2    "@*
 �2 "@*
 �2 "@�*	
 �2    "�*
 �2 "�*
 �2 "�@*	
 �2    "@*
 �2 "@*
 �2 "@@*	
 �2    " *
 �2 " *
 �2 "  *	
 �2    " *
 �2 " *
 �2 "  *	
 �2    " *
 �2 " *
 �2 " @*	
 �2    " *
 �2 " *
 �2 " @*	
 �2    "@*
 �2 "@*
 �2 "@ *	
 �2    " *
 �2 " *
 �2 " *	
 �2    p0.1p1.5p2.6p3.12p4.18p5.29p6.30p7.32p8.33p9.39p10.45p11.46p12.49p13.59p14.62p15.146p16.147p17.202p18.203p19.235p20.237p21.300p22.302p23.340p24.342p25.346p26.509p27.536p28.537p29.538p30.539p31.540p32.582p33.583p34.602p35.603p36.604p37.605p38.606p39.641p40.642p41.661p42.662p43.689p44.690p45.691p46.692p47.693p48.728p49.729p50.748p51.749p52.750p53.751p54.752p55.787p56.806p57.807p58.808p59.809p60.810p61.852p62.853p63.872p64.873p65.879p66.885p67.999p68.1026p69.1027p70.1028p71.1029p72.1030p73.1072p74.1073p75.1092p76.1093p77.1094p78.1095p79.1096p80.1131p81.1132p82.1151p83.1152p84.1179p85.1180p86.1181p87.1182p88.1183p89.1218p90.1219p91.1238p92.1239p93.1240p94.1241p95.1242p96.1277p97.1296p98.1297p99.1298	p100.1299	p101.1300	p102.1342	p103.1343	p104.1362	p105.1363	p106.1369	p107.1375	p108.1489	p109.1516	p110.1517	p111.1518	p112.1519	p113.1520	p114.1562	p115.1563	p116.1582	p117.1583	p118.1584	p119.1585	p120.1586	p121.1621	p122.1622	p123.1641	p124.1642	p125.1669	p126.1670	p127.1671	p128.1672	p129.1673	p130.1708	p131.1709	p132.1728	p133.1729	p134.1730	p135.1731	p136.1732	p137.1767	p138.1786	p139.1787	p140.1788	p141.1789	p142.1790	p143.1832	p144.1833	p145.1852	p146.1853	p147.1854	p148.1855	p149.1856	p150.1891	p151.1892	p152.1911	p153.1912	p154.1913	p155.1914	p156.1915	p157.1951	p158.1952	p159.1971	p160.1972	p161.1973	p162.1974	p163.1975	p164.2010	p165.2011	p166.2030	p167.2031	p168.2032	p169.2033	p170.2034	p171.2070	p172.2071	p173.2090	p174.2091	p175.2092	p176.2093	p177.2094	p178.2129	p179.2148	p180.2149	p181.2150	p182.2151	p183.2152	p184.2195	p185.2214	p186.2215	p187.2216	p188.2217	p189.2218	p190.2260	p191.2261	p192.2280	p193.2281	p194.2282	p195.2283	p196.2284	p197.2320	p198.2321	p199.2340	p200.2341	p201.2342	p202.2343	p203.2344	p204.2379	p205.2380	p206.2399	p207.2400	p208.2401	p209.2402	p210.2403	p211.2439	p212.2440	p213.2459	p214.2460	p215.2461	p216.2462	p217.2463	p218.2498	p219.2517	p220.2518	p221.2519	p222.2520	p223.2521	p224.2564	p225.2583	p226.2584	p227.2585	p228.2586	p229.2587	p230.2629	p231.2630	p232.2649	p233.2650	p234.2651	p235.2652	p236.2653	p237.2689	p238.2690	p239.2709	p240.2710	p241.2711	p242.2712	p243.2713	p244.2748	p245.2749	p246.2768	p247.2769	p248.2770	p249.2771	p250.2772	p251.2808	p252.2809	p253.2828	p254.2829	p255.2830	p256.2831	p257.2832	p258.2867	p259.2886	p260.2887	p261.2888	p262.2889	p263.2890	p264.2950	p265.2951	p266.2970	p267.2971	p268.2972	p269.2973	p270.2974	p271.3009	p272.3010	p273.3029	p274.3030	p275.3031	p276.3032	p277.3033	p278.3069	p279.3070	p280.3089	p281.3090	p282.3091	p283.3092	p284.3093	p285.3128	p286.3129	p287.3148	p288.3149	p289.3150	p290.3151	p291.3152	p292.3188	p293.3189	p294.3208	p295.3209	p296.3210	p297.3211	p298.3212	p299.3247	p300.3266	p301.3267	p302.3268	p303.3269	p304.3270	p305.3330	p306.3331	p307.3350	p308.3351	p309.3352	p310.3353	p311.3354	p312.3389	p313.3390	p314.3409	p315.3410	p316.3411	p317.3412	p318.3413	p319.3449	p320.3450	p321.3469	p322.3470	p323.3471	p324.3472	p325.3473	p326.3508	p327.3509	p328.3528	p329.3529	p330.3530	p331.3531	p332.3532	p333.3568	p334.3569	p335.3588	p336.3589	p337.3590	p338.3591	p339.3592	p340.3627	p341.3646	p342.3647	p343.3648	p344.3649	p345.3650	p346.3692	p347.3693	p348.3712	p349.3713	p350.3714	p351.3715	p352.3716	p353.3722	p354.3807	p355.3808	p356.3827	p357.3828	p358.3829	p359.3830	p360.3831	p361.3866	p362.3867	p363.3886	p364.3887	p365.3888	p366.3889	p367.3890	p368.3933	p369.3934	p370.3953	p371.3954	p372.3955	p373.3956	p374.3957	p375.3993	p376.3994	p377.4013	p378.4014	p379.4015	p380.4016	p381.4017	p382.4052	p383.4053	p384.4072	p385.4073	p386.4074	p387.4075	p388.4076	p389.4118	p390.4119	p391.4138	p392.4139	p393.4140	p394.4141	p395.4142	p396.4177	p397.4178	p398.4197	p399.4198	p400.4199	p401.4200	p402.4201	p403.4243	p404.4244	p405.4263	p406.4264	p407.4265	p408.4266	p409.4267	p410.4302	p411.4303	p412.4322	p413.4323	p414.4324	p415.4325	p416.4326	p417.4362	p418.4363	p419.4382	p420.4383	p421.4384	p422.4385	p423.4386	p424.4428	p425.4447	p426.4448	p427.4449	p428.4450	p429.4451	p430.4493	p431.4494	p432.4513	p433.4514	p434.4515	p435.4516	p436.4517	p437.4560	p438.4561	p439.4580	p440.4581	p441.4582	p442.4583	p443.4584	p444.4620	p445.4621	p446.4640	p447.4641	p448.4642	p449.4643	p450.4644	p451.4679	p452.4680	p453.4699	p454.4700	p455.4701	p456.4702	p457.4703	p458.4745	p459.4746	p460.4765	p461.4766	p462.4767	p463.4768	p464.4769	p465.4804	p466.4805	p467.4824	p468.4825	p469.4826	p470.4827	p471.4828	p472.4870	p473.4871	p474.4890	p475.4891	p476.4892	p477.4893	p478.4894	p479.4929	p480.4930	p481.4949	p482.4950	p483.4951	p484.4952	p485.4953	p486.4989	p487.4990	p488.5009	p489.5010	p490.5011	p491.5012	p492.5013	p493.5055	p494.5074	p495.5075	p496.5076	p497.5077	p498.5078	p499.5120	p500.5121	p501.5140	p502.5141	p503.5142	p504.5143	p505.5144	p506.5179	p507.5180	p508.5199	p509.5200	p510.5201	p511.5202	p512.5203	p513.5239	p514.5240	p515.5259	p516.5260	p517.5261	p518.5262	p519.5263	p520.5298	p521.5299	p522.5318	p523.5319	p524.5320	p525.5321	p526.5322	p527.5358	p528.5359	p529.5378	p530.5379	p531.5380	p532.5381	p533.5382	p534.5424	p535.5443	p536.5444	p537.5445	p538.5446	p539.5447	p540.5489	p541.5490	p542.5509	p543.5510	p544.5511	p545.5512	p546.5513	p547.5548	p548.5549(�+0�+"��
*�
�*	
 �2    
�*	
 �2    
*�
�*
 �2   
*
 �2  
*
 �2  
*
 �2  
*
 �2 
*
 �2 
*
 �2 
*
 �2 
*�
*
 �2 
*�
*
 �2  
*
 �2  
*
 �2  
*
 �2 
*
 �2  
*�
�A*
 �2  
�A*
 �2  
P*
 �2  
P*
 �2  
*
 �2  
�*	
 �2    
�*
 �2 
�*
 �2 
�*
 �2 
�*	
 �2    
�*	
 �2    
��*	
 �2    
�*	
 �2    
�*
 �2 
�*
 �2 
�*
 �2 
�*	
 �2    
�*	
 �2    
��*	
 �2    
�*	
 �2    
P�*	
 �2    
�*	
 �2    
�*
 �2 
�*
 �2 
�*
 �2 
�*	
 �2    
�*	
 �2    
��*	
 �2    
�*	
 �2    
�*
 �2 
�*
 �2 
�*
 �2 
�*	
 �2    
�*	
 �2    
��*	
 �2    
�*
 �2 
�*
 �2 
�*
 �2 
�*	
 �2    
�*	
 �2    
��*	
 �2    
�*	
 �2    
�*	
 �2    
�((*	
 �2    
*�
�*
 �2   
�*	
 �2    
�*
 �2 
�*
 �2 
�*
 �2 
�((*	
 �2    
�((*	
 �2    
��*	
 �2    
�((*	
 �2    
�*
 �2 
�*
 �2 
�*
 �2 
�((*	
 �2    
�((*	
 �2    
��*	
 �2    
�((*	
 �2    
P�*	
 �2    
�((*	
 �2    
�*
 �2 
�*
 �2 
�*
 �2 
�((*	
 �2    
�((*	
 �2    
��*	
 �2    
�((*	
 �2    
�*
 �2 
�*
 �2 
�*
 �2 
�((*	
 �2    
�((*	
 �2    
��*	
 �2    
�*
 �2 
�*
 �2 
�*
 �2 
�((*	
 �2    
�((*	
 �2    
��*	
 �2    
�((*	
 �2    
�*	
 �2    
�PP*	
 �2    
*�
�2*
 �2   
�*	
 �2    
�*
 �2 
�*
 �2 
�*
 �2 
�PP*	
 �2    
�PP*	
 �2    
��*	
 �2    
�PP*	
 �2    
�*
 �2 
�*
 �2 
�*
 �2 
�PP*	
 �2    
�PP*	
 �2    
��*	
 �2    
�PP*	
 �2    
P�*	
 �2    
�PP*	
 �2    
�*
 �2 
�*
 �2 
�*
 �2 
�PP*	
 �2    
�PP*	
 �2    
��*	
 �2    
�PP*	
 �2    
�*
 �2 
�*
 �2 
�*
 �2 
�PP*	
 �2    
�PP*	
 �2    
��*	
 �2    
�*
 �2 
�*
 �2 
�*
 �2 
�PP*	
 �2    
�PP*	
 �2    
��*	
 �2    
�PP*	
 �2    
�*
 �2 
�*
 �2 
�*
 �2 
�*	
 �2    
�*	
 �2    
��*	
 �2    
�*	
 �2    
�*
 �2 
�*
 �2 
�*
 �2 
�*	
 �2    
�*	
 �2    
��*	
 �2    
�*	
 �2    
�*
 �2 
�*
 �2 
�*
 �2 
�*	
 �2    
�*	
 �2    
��*	
 �2    
�*	
 �2    
�*
 �2 
�*
 �2 
�*
 �2 
�*	
 �2    
�*	
 �2    
��*	
 �2    
�*	
 �2    
�*
 �2 
�*
 �2 
�*
 �2 
�*	
 �2    
�*	
 �2    
��*	
 �2    
�*
 �2 
�*
 �2 
�*
 �2 
�*	
 �2    
�*	
 �2    
��*	
 �2    
�*
 �2 
�*
 �2 
�*
 �2 
�((*	
 �2    
�((*	
 �2    
��*	
 �2    
�((*	
 �2    
�*
 �2 
�*
 �2 
�*
 �2 
�((*	
 �2    
�((*	
 �2    
��*	
 �2    
�((*	
 �2    
�*
 �2 
�*
 �2 
�*
 �2 
�((*	
 �2    
�((*	
 �2    
��*	
 �2    
�((*	
 �2    
�*
 �2 
�*
 �2 
�*
 �2 
�((*	
 �2    
�((*	
 �2    
��*	
 �2    
�((*	
 �2    
�*
 �2 
�*
 �2 
�*
 �2 
�((*	
 �2    
�((*	
 �2    
��*	
 �2    
�*
 �2 
�*
 �2 
�*
 �2 
�((*	
 �2    
�((*	
 �2    
��*	
 �2    
�*
 �2 
�*
 �2 
�*
 �2 
�PP*	
 �2    
�PP*	
 �2    
��*	
 �2    
�PP*	
 �2    
@*
 �2 
@*
 �2 
@*
 �2 
@PP*	
 �2    
@PP*	
 �2    
@@*	
 �2    
@PP*	
 �2    
@*
 �2 
@*
 �2 
@*
 �2 
@PP*	
 �2    
@PP*	
 �2    
@@*	
 �2    
@PP*	
 �2    
@*
 �2 
@*
 �2 
@*
 �2 
@PP*	
 �2    
@PP*	
 �2    
@�*	
 �2    
�PP*	
 �2    
@*
 �2 
@*
 �2 
@*
 �2 
@PP*	
 �2    
@PP*	
 �2    
@�*	
 �2    
�*
 �2 
�*
 �2 
�*
 �2 
�((*	
 �2    
�((*	
 �2    
��*	
 �2    
�((*	
 �2    
�*
 �2 
�*
 �2 
�*
 �2 
�((*	
 �2    
�((*	
 �2    
��*	
 �2    
�((*	
 �2    
�*
 �2 
�*
 �2 
�*
 �2 
�((*	
 �2    
�((*	
 �2    
��*	
 �2    
�((*	
 �2    
�*
 �2 
�*
 �2 
�*
 �2 
�((*	
 �2    
�((*	
 �2    
��*	
 �2    
�((*	
 �2    
�*
 �2 
�*
 �2 
�*
 �2 
�((*	
 �2    
�((*	
 �2    
��*	
 �2    
�((*	
 �2    
�*
 �2 
�*
 �2 
�*
 �2 
�((*	
 �2    
�((*	
 �2    
��*	
 �2    
�*
 �2 
�*
 �2 
�*
 �2 
�*	
 �2    
�*	
 �2    
��*	
 �2    
�*	
 �2    
�*
 �2 
�*
 �2 
�*
 �2 
�*	
 �2    
�*	
 �2    
��*	
 �2    
�*	
 �2    
�*
 �2 
�*
 �2 
�*
 �2 
�*	
 �2    
�*	
 �2    
��*	
 �2    
�*	
 �2    
�*
 �2 
�*
 �2 
�*
 �2 
�*	
 �2    
�*	
 �2    
��*	
 �2    
�*	
 �2    
�*
 �2 
�*
 �2 
�*
 �2 
�*	
 �2    
�*	
 �2    
��*	
 �2    
�*	
 �2    
�*
 �2 
�*
 �2 
�*
 �2 
�*	
 �2    
�*	
 �2    
��*	
 �2    
�*
 �2 
�*
 �2 
�*
 �2 
�*	
 �2    
�*	
 �2    
��*	
 �2    
�*	
 �2    
�*
 �2 
�*
 �2 
�*
 �2 
�*	
 �2    
�*	
 �2    
�*	
 �2    
��*	
 �2    
�*	
 �2    
�*
 �2 
�*
 �2 
�*
 �2 
�*	
 �2    
�*	
 �2    
��*	
 �2    
�((*	
 �2    
�*
 �2 
�*
 �2 
�*
 �2 
�((*	
 �2    
�((*	
 �2    
��*	
 �2    
�((*	
 �2    
�*
 �2 
�*
 �2 
�*
 �2 
�((*	
 �2    
�((*	
 �2    
��*	
 �2    
�((*	
 �2    
�*
 �2 
�*
 �2 
�*
 �2 
�((*	
 �2    
�((*	
 �2    
��*	
 �2    
�((*	
 �2    
�*
 �2 
�*
 �2 
�*
 �2 
�((*	
 �2    
�((*	
 �2    
��*	
 �2    
�((*	
 �2    
�*
 �2 
�*
 �2 
�*
 �2 
�((*	
 �2    
�((*	
 �2    
��*	
 �2    
�((*	
 �2    
�*
 �2 
�*
 �2 
�*
 �2 
�((*	
 �2    
�((*	
 �2    
��*	
 �2    
�((*	
 �2    
�*
 �2 
�*
 �2 
�*
 �2 
�((*	
 �2    
�((*	
 �2    
��*	
 �2    
�((*	
 �2    
�*
 �2 
�*
 �2 
�*
 �2 
�((*	
 �2    
�((*	
 �2    
��*	
 �2    
�((*	
 �2    
�*
 �2 
�*
 �2 
�*
 �2 
�((*	
 �2    
�((*	
 �2    
��*	
 �2    
�*
 �2 
�*
 �2 
�*
 �2 
�((*	
 �2    
�((*	
 �2    
��*	
 �2    
�PP*	
 �2    
�*
 �2 
�*
 �2 
�*
 �2 
�PP*	
 �2    
�PP*	
 �2    
��*	
 �2    
�PP*	
 �2    
@*
 �2 
@*
 �2 
@*
 �2 
@PP*	
 �2    
@PP*	
 �2    
@@*	
 �2    
@PP*	
 �2    
@*
 �2 
@*
 �2 
@*
 �2 
@PP*	
 �2    
@PP*	
 �2    
@@*	
 �2    
@PP*	
 �2    
@*
 �2 
@*
 �2 
@*
 �2 
@PP*	
 �2    
@PP*	
 �2    
@@*	
 �2    
@PP*	
 �2    
@*
 �2 
@*
 �2 
@*
 �2 
@PP*	
 �2    
@PP*	
 �2    
@@*	
 �2    
@PP*	
 �2    
@*
 �2 
@*
 �2 
@*
 �2 
@PP*	
 �2    
@PP*	
 �2    
@@*	
 �2    
@PP*	
 �2    
@*
 �2 
@*
 �2 
@*
 �2 
@PP*	
 �2    
@PP*	
 �2    
@@*	
 �2    
@PP*	
 �2    
@*
 �2 
@*
 �2 
@*
 �2 
@PP*	
 �2    
@PP*	
 �2    
@�*	
 �2    
�PP*	
 �2    
@*
 �2 
@*
 �2 
@*
 �2 
@PP*	
 �2    
@PP*	
 �2    
@�*	
 �2    
�*
 �2 
�*
 �2 
�*
 �2 
�PP*	
 �2    
�PP*	
 �2    
�@*	
 �2    
@��*	
 �2    
@*
 �2 
@*
 �2 
@*
 �2 
@��*	
 �2    
@��*	
 �2    
@@*	
 �2    
@��*	
 �2    
 *
 �2 
 *
 �2 
 *
 �2 
 ��*	
 �2    
 ��*	
 �2    
  *	
 �2    
 ��*	
 �2    
 *
 �2 
 *
 �2 
 *
 �2 
 ��*	
 �2    
 ��*	
 �2    
  *	
 �2    
 ��*	
 �2    
 *
 �2 
 *
 �2 
 *
 �2 
 ��*	
 �2    
 ��*	
 �2    
 @*	
 �2    
@��*	
 �2    
 *
 �2 
 *
 �2 
 *
 �2 
 ��*	
 �2    
 ��*	
 �2    
 @*	
 �2    
@*
 �2 
@*
 �2 
@*
 �2 
@��*	
 �2    
@��*	
 �2    
@ *	
 �2    
 ��*	
 �2    
 *
 �2 
 *
 �2 
 *
 �2 
 ��*	
 �2    
 ��*	
 �2    
 *	
 �2    
��*	
 �2    �)"*�"*�"*
 �2 "�*	
 �2    "*
 �2 "�*	
 �2    "�*
 �2 "�*
 �2 "��*	
 �2    "�*
 �2 "�*
 �2 "��*	
 �2    "P*
 �2 "P�*	
 �2    "�*
 �2 "�*
 �2 "��*	
 �2    "�*
 �2 "�*
 �2 "��*	
 �2    "�*
 �2 "�*
 �2 "��*	
 �2    "*
 �2 "�*	
 �2    "*
 �2 "�*	
 �2    "�*
 �2 "�*
 �2 "��*	
 �2    "�*
 �2 "�*
 �2 "��*	
 �2    "P*
 �2 "P�*	
 �2    "�*
 �2 "�*
 �2 "��*	
 �2    "�*
 �2 "�*
 �2 "��*	
 �2    "�*
 �2 "�*
 �2 "��*	
 �2    "*
 �2 "�*	
 �2    "*
 �2 "�*	
 �2    "�*
 �2 "�*
 �2 "��*	
 �2    "�*
 �2 "�*
 �2 "��*	
 �2    "P*
 �2 "P�*	
 �2    "�*
 �2 "�*
 �2 "��*	
 �2    "�*
 �2 "�*
 �2 "��*	
 �2    "�*
 �2 "�*
 �2 "��*	
 �2    "�*
 �2 "�*
 �2 "��*	
 �2    "�*
 �2 "�*
 �2 "��*	
 �2    "�*
 �2 "�*
 �2 "��*	
 �2    "�*
 �2 "�*
 �2 "��*	
 �2    "�*
 �2 "�*
 �2 "��*	
 �2    "�*
 �2 "�*
 �2 "��*	
 �2    "�*
 �2 "�*
 �2 "��*	
 �2    "�*
 �2 "�*
 �2 "��*	
 �2    "�*
 �2 "�*
 �2 "��*	
 �2    "�*
 �2 "�*
 �2 "��*	
 �2    "�*
 �2 "�*
 �2 "��*	
 �2    "�*
 �2 "�*
 �2 "��*	
 �2    "�*
 �2 "�*
 �2 "��*	
 �2    "@*
 �2 "@*
 �2 "@@*	
 �2    "@*
 �2 "@*
 �2 "@@*	
 �2    "@*
 �2 "@*
 �2 "@�*	
 �2    "@*
 �2 "@*
 �2 "@�*	
 �2    "�*
 �2 "�*
 �2 "��*	
 �2    "�*
 �2 "�*
 �2 "��*	
 �2    "�*
 �2 "�*
 �2 "��*	
 �2    "�*
 �2 "�*
 �2 "��*	
 �2    "�*
 �2 "�*
 �2 "��*	
 �2    "�*
 �2 "�*
 �2 "��*	
 �2    "�*
 �2 "�*
 �2 "��*	
 �2    "�*
 �2 "�*
 �2 "��*	
 �2    "�*
 �2 "�*
 �2 "��*	
 �2    "�*
 �2 "�*
 �2 "��*	
 �2    "�*
 �2 "�*
 �2 "��*	
 �2    "�*
 �2 "�*
 �2 "��*	
 �2    "�*
 �2 "�*
 �2 "��*	
 �2    "�*
 �2 "�*
 �2 "��*	
 �2    "�*
 �2 "�*
 �2 "��*	
 �2    "�*
 �2 "�*
 �2 "��*	
 �2    "�*
 �2 "�*
 �2 "��*	
 �2    "�*
 �2 "�*
 �2 "��*	
 �2    "�*
 �2 "�*
 �2 "��*	
 �2    "�*
 �2 "�*
 �2 "��*	
 �2    "�*
 �2 "�*
 �2 "��*	
 �2    "�*
 �2 "�*
 �2 "��*	
 �2    "�*
 �2 "�*
 �2 "��*	
 �2    "�*
 �2 "�*
 �2 "��*	
 �2    "�*
 �2 "�*
 �2 "��*	
 �2    "�*
 �2 "�*
 �2 "��*	
 �2    "@*
 �2 "@*
 �2 "@@*	
 �2    "@*
 �2 "@*
 �2 "@@*	
 �2    "@*
 �2 "@*
 �2 "@@*	
 �2    "@*
 �2 "@*
 �2 "@@*	
 �2    "@*
 �2 "@*
 �2 "@@*	
 �2    "@*
 �2 "@*
 �2 "@@*	
 �2    "@*
 �2 "@*
 �2 "@�*	
 �2    "@*
 �2 "@*
 �2 "@�*	
 �2    "�*
 �2 "�*
 �2 "�@*	
 �2    "@*
 �2 "@*
 �2 "@@*	
 �2    " *
 �2 " *
 �2 "  *	
 �2    " *
 �2 " *
 �2 "  *	
 �2    " *
 �2 " *
 �2 " @*	
 �2    " *
 �2 " *
 �2 " @*	
 �2    "@*
 �2 "@*
 �2 "@ *	
 �2    " *
 �2 " *
 �2 " *	
 �2    p0p1p2p3p4p5p6p7p8p9p10p11p12p13p14p15p16p17p18p19p20p21p22p23p24p25p26p27p28p29p30p31p32p33p34p35p36p37p38p39p40p41p42p43p44p45p46p47p48p49p50p51p52p53p54p55p56p57p58p59p60p61p62p63p64p65p66p67p68p69p70p71p72p73p74p75p76p77p78p79p80p81p82p83p84p85p86p87p88p89p90p91p92p93p94p95p96p97p98p99p100p101p102p103p104p105p106p107p108p109p110p111p112p113p114p115p116p117p118p119p120p121p122p123p124p125p126p127p128p129p130p131p132p133p134p135p136p137p138p139p140p141p142p143p144p145p146p147p148p149p150p151p152p153p154p155p156p157p158p159p160p161p162p163p164p165p166p167p168p169p170p171p172p173p174p175p176p177p178p179p180p181p182p183p184p185p186p187p188p189p190p191p192p193p194p195p196p197p198p199p200p201p202p203p204p205p206p207p208p209p210p211p212p213p214p215p216p217p218p219p220p221p222p223p224p225p226p227p228p229p230p231p232p233p234p235p236p237p238p239p240p241p242p243p244p245p246p247p248p249p250p251p252p253p254p255p256p257p258p259p260p261p262p263p264p265p266p267p268p269p270p271p272p273p274p275p276p277p278p279p280p281p282p283p284p285p286p287p288p289p290p291p292p293p294p295p296p297p298p299p300p301p302p303p304p305p306p307p308p309p310p311p312p313p314p315p316p317p318p319p320p321p322p323p324p325p326p327p328p329p330p331p332p333p334p335p336p337p338p339p340p341p342p343p344p345p346p347p348p349p350p351p352p353p354p355p356p357p358p359p360p361p362p363p364p365p366p367p368p369p370p371p372p373p374p375p376p377p378p379p380p381p382p383p384p385p386p387p388p389p390p391p392p393p394p395p396p397p398p399p400p401p402p403p404p405p406p407p408p409p410p411p412p413p414p415p416p417p418p419p420p421p422p423p424p425p426p427p428p429p430p431p432p433p434p435p436p437p438p439p440p441p442p443p444p445p446p447p448p449p450p451p452p453p454p455p456p457p458p459p460p461p462p463p464p465p466p467p468p469p470p471p472p473p474p475p476p477p478p479p480p481p482p483p484p485p486p487p488p489p490p491p492p493p494p495p496p497p498p499p500p501p502p503p504p505p506p507p508p509p510p511p512p513p514p515p516p517p518p519p520p521p522p523p524p525p526p527p528p529p530p531p532p533p534p535p536p537p538p539p540p541p542p543p544p545p546p547p548(T0�+B z	
 ��
/usr/lib/python3.10/runpy.py
U/home/ubuntu/aws-neuron-samples/torch-neuronx/training/neuron-adoption/tools/train.py
[/home/ubuntu/aws-neuron-samples/torch-neuronx/training/neuron-adoption/yolox/core/launch.py
~/home/ubuntu/aws-neuron-samples/aws_neuron_venv_pytorch/lib/python3.10/site-packages/loguru-0.7.2-py3.10.egg/loguru/_logger.py
\/home/ubuntu/aws-neuron-samples/torch-neuronx/training/neuron-adoption/yolox/core/trainer.py
r/home/ubuntu/aws-neuron-samples/aws_neuron_venv_pytorch/lib/python3.10/site-packages/torch/cuda/amp/grad_scaler.py
e/home/ubuntu/aws-neuron-samples/aws_neuron_venv_pytorch/lib/python3.10/site-packages/torch/_tensor.py
o/home/ubuntu/aws-neuron-samples/aws_neuron_venv_pytorch/lib/python3.10/site-packages/torch/autograd/__init__.py
o/home/ubuntu/aws-neuron-samples/aws_neuron_venv_pytorch/lib/python3.10/site-packages/torch/autograd/function.py
s/home/ubuntu/aws-neuron-samples/aws_neuron_venv_pytorch/lib/python3.10/site-packages/torch_neuronx/xla_impl/hint.py
v/home/ubuntu/aws-neuron-samples/aws_neuron_venv_pytorch/lib/python3.10/site-packages/torch_xla/core/xla_op_registry.py
r/home/ubuntu/aws-neuron-samples/aws_neuron_venv_pytorch/lib/python3.10/site-packages/torch_neuronx/xla_impl/ops.py_run_module_as_main	_run_code<module>launchcatch_wrappermaintraintrain_in_epochtrain_in_itertrain_one_iterscale_lazy_init_scale_growth_trackerbackward_make_gradsapply__call__�V�b�	vS]	g
������	�
P,�	"""""""""	"
	"
"""
"""""""� � 