то
Я│
8
Const
output"dtype"
valuetensor"
dtypetype

NoOp
│
PartitionedCall
args2Tin
output2Tout"
Tin
list(type)("
Tout
list(type)("	
ffunc"
configstring "
config_protostring "
executor_typestring 
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
@
ReadVariableOp
resource
value"dtype"
dtypetypeѕ
Й
StatefulPartitionedCall
args2Tin
output2Tout"
Tin
list(type)("
Tout
list(type)("	
ffunc"
configstring "
config_protostring "
executor_typestring ѕ
q
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshapeѕ"serve*2.2.02v2.2.0-rc4-8-g2b96f3662b8╦з
д
%QNetwork/EncodingNetwork/dense/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:d*6
shared_name'%QNetwork/EncodingNetwork/dense/kernel
Ъ
9QNetwork/EncodingNetwork/dense/kernel/Read/ReadVariableOpReadVariableOp%QNetwork/EncodingNetwork/dense/kernel*
_output_shapes

:d*
dtype0
ъ
#QNetwork/EncodingNetwork/dense/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:d*4
shared_name%#QNetwork/EncodingNetwork/dense/bias
Ќ
7QNetwork/EncodingNetwork/dense/bias/Read/ReadVariableOpReadVariableOp#QNetwork/EncodingNetwork/dense/bias*
_output_shapes
:d*
dtype0
і
QNetwork/dense_1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:d*(
shared_nameQNetwork/dense_1/kernel
Ѓ
+QNetwork/dense_1/kernel/Read/ReadVariableOpReadVariableOpQNetwork/dense_1/kernel*
_output_shapes

:d*
dtype0
ѓ
QNetwork/dense_1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*&
shared_nameQNetwork/dense_1/bias
{
)QNetwork/dense_1/bias/Read/ReadVariableOpReadVariableOpQNetwork/dense_1/bias*
_output_shapes
:*
dtype0
P
ConstConst*
_output_shapes
: *
dtype0*
valueB :
         

NoOpNoOp
М
Const_1Const"/device:CPU:0*
_output_shapes
: *
dtype0*ї
valueѓB  BЭ
:
_wrapped_policy
model_variables

signatures


_q_network

0
1
2
3
 
t
	_encoder

_q_value_layer
	variables
trainable_variables
regularization_losses
	keras_api
ge
VARIABLE_VALUE%QNetwork/EncodingNetwork/dense/kernel,model_variables/0/.ATTRIBUTES/VARIABLE_VALUE
ec
VARIABLE_VALUE#QNetwork/EncodingNetwork/dense/bias,model_variables/1/.ATTRIBUTES/VARIABLE_VALUE
YW
VARIABLE_VALUEQNetwork/dense_1/kernel,model_variables/2/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEQNetwork/dense_1/bias,model_variables/3/.ATTRIBUTES/VARIABLE_VALUE
n
_postprocessing_layers
	variables
trainable_variables
regularization_losses
	keras_api
h

kernel
bias
	variables
trainable_variables
regularization_losses
	keras_api

0
1
2
3

0
1
2
3
 
Г

layers
layer_regularization_losses
	variables
non_trainable_variables
trainable_variables
layer_metrics
metrics
regularization_losses

0
1

0
1

0
1
 
Г

layers
 layer_regularization_losses
	variables
!non_trainable_variables
trainable_variables
"layer_metrics
#metrics
regularization_losses

0
1

0
1
 
Г

$layers
%layer_regularization_losses
	variables
&non_trainable_variables
trainable_variables
'layer_metrics
(metrics
regularization_losses

	0

1
 
 
 
 
R
)	variables
*trainable_variables
+regularization_losses
,	keras_api
h

kernel
bias
-	variables
.trainable_variables
/regularization_losses
0	keras_api

0
1
 
 
 
 
 
 
 
 
 
 
 
 
Г

1layers
2layer_regularization_losses
)	variables
3non_trainable_variables
*trainable_variables
4layer_metrics
5metrics
+regularization_losses

0
1

0
1
 
Г

6layers
7layer_regularization_losses
-	variables
8non_trainable_variables
.trainable_variables
9layer_metrics
:metrics
/regularization_losses
 
 
 
 
 
 
 
 
 
 
l
action_0/discountPlaceholder*#
_output_shapes
:         *
dtype0*
shape:         
w
action_0/observationPlaceholder*'
_output_shapes
:         *
dtype0*
shape:         
j
action_0/rewardPlaceholder*#
_output_shapes
:         *
dtype0*
shape:         
m
action_0/step_typePlaceholder*#
_output_shapes
:         *
dtype0*
shape:         
╦
StatefulPartitionedCallStatefulPartitionedCallaction_0/discountaction_0/observationaction_0/rewardaction_0/step_type%QNetwork/EncodingNetwork/dense/kernel#QNetwork/EncodingNetwork/dense/biasQNetwork/dense_1/kernelQNetwork/dense_1/bias*
Tin

2*
Tout
2	*#
_output_shapes
:         *&
_read_only_resource_inputs
*-
config_proto

CPU

GPU2*0J 8*-
f(R&
$__inference_signature_wrapper_735090
]
get_initial_state_batch_sizePlaceholder*
_output_shapes
: *
dtype0*
shape: 
┌
PartitionedCallPartitionedCallget_initial_state_batch_size*
Tin
2*

Tout
 * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU2*0J 8*-
f(R&
$__inference_signature_wrapper_735102
Я
PartitionedCall_1PartitionedCallConst*
Tin
2*
Tout
2*
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU2*0J 8*-
f(R&
$__inference_signature_wrapper_735117
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
╬
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename9QNetwork/EncodingNetwork/dense/kernel/Read/ReadVariableOp7QNetwork/EncodingNetwork/dense/bias/Read/ReadVariableOp+QNetwork/dense_1/kernel/Read/ReadVariableOp)QNetwork/dense_1/bias/Read/ReadVariableOpConst_1*
Tin

2*
Tout
2*
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU2*0J 8*(
f#R!
__inference__traced_save_735178
э
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filename%QNetwork/EncodingNetwork/dense/kernel#QNetwork/EncodingNetwork/dense/biasQNetwork/dense_1/kernelQNetwork/dense_1/bias*
Tin	
2*
Tout
2*
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU2*0J 8*+
f&R$
"__inference__traced_restore_735202ц┼
У 
Ч
__inference__traced_save_735178
file_prefixD
@savev2_qnetwork_encodingnetwork_dense_kernel_read_readvariableopB
>savev2_qnetwork_encodingnetwork_dense_bias_read_readvariableop6
2savev2_qnetwork_dense_1_kernel_read_readvariableop4
0savev2_qnetwork_dense_1_bias_read_readvariableop
savev2_1_const_1

identity_1ѕбMergeV2CheckpointsбSaveV2бSaveV2_1Ј
StaticRegexFullMatchStaticRegexFullMatchfile_prefix"/device:CPU:**
_output_shapes
: *
pattern
^s3://.*2
StaticRegexFullMatchc
ConstConst"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B.part2
ConstЇ
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*<
value3B1 B+_temp_7f8e92195ccd4ab6b209b52e53b9cca3/part2	
Const_1І
SelectSelectStaticRegexFullMatch:output:0Const:output:0Const_1:output:0"/device:CPU:**
T0*
_output_shapes
: 2
Selectt

StringJoin
StringJoinfile_prefixSelect:output:0"/device:CPU:**
N*
_output_shapes
: 2

StringJoinZ

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :2

num_shards
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : 2
ShardedFilename/shardд
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: 2
ShardedFilename╗
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*═
value├B└B,model_variables/0/.ATTRIBUTES/VARIABLE_VALUEB,model_variables/1/.ATTRIBUTES/VARIABLE_VALUEB,model_variables/2/.ATTRIBUTES/VARIABLE_VALUEB,model_variables/3/.ATTRIBUTES/VARIABLE_VALUE2
SaveV2/tensor_namesљ
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*
valueBB B B B 2
SaveV2/shape_and_slicesЌ
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0@savev2_qnetwork_encodingnetwork_dense_kernel_read_readvariableop>savev2_qnetwork_encodingnetwork_dense_bias_read_readvariableop2savev2_qnetwork_dense_1_kernel_read_readvariableop0savev2_qnetwork_dense_1_bias_read_readvariableop"/device:CPU:0*
_output_shapes
 *
dtypes
22
SaveV2Ѓ
ShardedFilename_1/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B :2
ShardedFilename_1/shardг
ShardedFilename_1ShardedFilenameStringJoin:output:0 ShardedFilename_1/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: 2
ShardedFilename_1б
SaveV2_1/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*1
value(B&B_CHECKPOINTABLE_OBJECT_GRAPH2
SaveV2_1/tensor_namesј
SaveV2_1/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*
valueB
B 2
SaveV2_1/shape_and_slicesЛ
SaveV2_1SaveV2ShardedFilename_1:filename:0SaveV2_1/tensor_names:output:0"SaveV2_1/shape_and_slices:output:0savev2_1_const_1^SaveV2"/device:CPU:0*
_output_shapes
 *
dtypes
22

SaveV2_1с
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0ShardedFilename_1:filename:0^SaveV2	^SaveV2_1"/device:CPU:0*
N*
T0*
_output_shapes
:2(
&MergeV2Checkpoints/checkpoint_prefixesг
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix	^SaveV2_1"/device:CPU:0*
_output_shapes
 2
MergeV2Checkpointsr
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: 2

IdentityЂ

Identity_1IdentityIdentity:output:0^MergeV2Checkpoints^SaveV2	^SaveV2_1*
T0*
_output_shapes
: 2

Identity_1"!

identity_1Identity_1:output:0*7
_input_shapes&
$: :d:d:d:: 2(
MergeV2CheckpointsMergeV2Checkpoints2
SaveV2SaveV22
SaveV2_1SaveV2_1:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:$ 

_output_shapes

:d: 

_output_shapes
:d:$ 

_output_shapes

:d: 

_output_shapes
::

_output_shapes
: 
╝
8
__inference_<lambda>_734969
unknown
identityJ
IdentityIdentityunknown*
T0*
_output_shapes
: 2

Identity"
identityIdentity:output:0*
_input_shapes
: 
Њ
6
$__inference_get_initial_state_735096

batch_size*
_input_shapes
: :B >

_output_shapes
: 
$
_user_specified_name
batch_size
Ї
<
*__inference_function_with_signature_735097

batch_sizeы
PartitionedCallPartitionedCall
batch_size*
Tin
2*

Tout
 *
_output_shapes
 * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU2*0J 8*-
f(R&
$__inference_get_initial_state_7350962
PartitionedCall*
_input_shapes
: :B >

_output_shapes
: 
$
_user_specified_name
batch_size
м
A
$__inference_signature_wrapper_735117
unknown
identityщ
PartitionedCallPartitionedCallunknown*
Tin
2*
Tout
2*
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU2*0J 8*3
f.R,
*__inference_function_with_signature_7351092
PartitionedCall[
IdentityIdentityPartitionedCall:output:0*
T0*
_output_shapes
: 2

Identity"
identityIdentity:output:0*
_input_shapes
: 
│

╔
(__inference_polymorphic_action_fn_735039
	step_type

reward
discount
observation
unknown
	unknown_0
	unknown_1
	unknown_2
identity	ѕбStatefulPartitionedCallт
StatefulPartitionedCallStatefulPartitionedCall	step_typerewarddiscountobservationunknown	unknown_0	unknown_1	unknown_2*
Tin

2*
Tout
2	*#
_output_shapes
:         *&
_read_only_resource_inputs
*-
config_proto

CPU

GPU2*0J 8*"
fR
__inference_action_7350282
StatefulPartitionedCallі
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0	*#
_output_shapes
:         2

Identity"
identityIdentity:output:0*c
_input_shapesR
P:         :         :         :         ::::22
StatefulPartitionedCallStatefulPartitionedCall:N J
#
_output_shapes
:         
#
_user_specified_name	step_type:KG
#
_output_shapes
:         
 
_user_specified_namereward:MI
#
_output_shapes
:         
"
_user_specified_name
discount:TP
'
_output_shapes
:         
%
_user_specified_nameobservation:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
Ї
6
$__inference_signature_wrapper_735102

batch_sizeэ
PartitionedCallPartitionedCall
batch_size*
Tin
2*

Tout
 *
_output_shapes
 * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU2*0J 8*3
f.R,
*__inference_function_with_signature_7350972
PartitionedCall*
_input_shapes
: :B >

_output_shapes
: 
$
_user_specified_name
batch_size
╚

┼
$__inference_signature_wrapper_735090
discount
observation

reward
	step_type
unknown
	unknown_0
	unknown_1
	unknown_2
identity	ѕбStatefulPartitionedCallШ
StatefulPartitionedCallStatefulPartitionedCall	step_typerewarddiscountobservationunknown	unknown_0	unknown_1	unknown_2*
Tin

2*
Tout
2	*#
_output_shapes
:         *&
_read_only_resource_inputs
*-
config_proto

CPU

GPU2*0J 8*3
f.R,
*__inference_function_with_signature_7350722
StatefulPartitionedCallі
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0	*#
_output_shapes
:         2

Identity"
identityIdentity:output:0*c
_input_shapesR
P:         :         :         :         ::::22
StatefulPartitionedCallStatefulPartitionedCall:O K
#
_output_shapes
:         
$
_user_specified_name
0/discount:VR
'
_output_shapes
:         
'
_user_specified_name0/observation:MI
#
_output_shapes
:         
"
_user_specified_name
0/reward:PL
#
_output_shapes
:         
%
_user_specified_name0/step_type:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
╠

╦
*__inference_function_with_signature_735072
	step_type

reward
discount
observation
unknown
	unknown_0
	unknown_1
	unknown_2
identity	ѕбStatefulPartitionedCallЗ
StatefulPartitionedCallStatefulPartitionedCall	step_typerewarddiscountobservationunknown	unknown_0	unknown_1	unknown_2*
Tin

2*
Tout
2	*#
_output_shapes
:         *&
_read_only_resource_inputs
*-
config_proto

CPU

GPU2*0J 8*1
f,R*
(__inference_polymorphic_action_fn_7350612
StatefulPartitionedCallі
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0	*#
_output_shapes
:         2

Identity"
identityIdentity:output:0*c
_input_shapesR
P:         :         :         :         ::::22
StatefulPartitionedCallStatefulPartitionedCall:P L
#
_output_shapes
:         
%
_user_specified_name0/step_type:MI
#
_output_shapes
:         
"
_user_specified_name
0/reward:OK
#
_output_shapes
:         
$
_user_specified_name
0/discount:VR
'
_output_shapes
:         
'
_user_specified_name0/observation:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
Њ
6
$__inference_get_initial_state_734963

batch_size*
_input_shapes
: :B >

_output_shapes
: 
$
_user_specified_name
batch_size
┼

Л
(__inference_polymorphic_action_fn_735061
	time_step
time_step_1
time_step_2
time_step_3
unknown
	unknown_0
	unknown_1
	unknown_2
identity	ѕбStatefulPartitionedCallь
StatefulPartitionedCallStatefulPartitionedCall	time_steptime_step_1time_step_2time_step_3unknown	unknown_0	unknown_1	unknown_2*
Tin

2*
Tout
2	*#
_output_shapes
:         *&
_read_only_resource_inputs
*-
config_proto

CPU

GPU2*0J 8*"
fR
__inference_action_7350282
StatefulPartitionedCallі
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0	*#
_output_shapes
:         2

Identity"
identityIdentity:output:0*c
_input_shapesR
P:         :         :         :         ::::22
StatefulPartitionedCallStatefulPartitionedCall:N J
#
_output_shapes
:         
#
_user_specified_name	time_step:NJ
#
_output_shapes
:         
#
_user_specified_name	time_step:NJ
#
_output_shapes
:         
#
_user_specified_name	time_step:RN
'
_output_shapes
:         
#
_user_specified_name	time_step:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
╔
G
*__inference_function_with_signature_735109
unknown
identityЖ
PartitionedCallPartitionedCallunknown*
Tin
2*
Tout
2*
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU2*0J 8*$
fR
__inference_<lambda>_7349692
PartitionedCall[
IdentityIdentityPartitionedCall:output:0*
T0*
_output_shapes
: 2

Identity"
identityIdentity:output:0*
_input_shapes
: 
Ф
ы
(__inference_polymorphic_action_fn_735133
time_step_step_type
time_step_reward
time_step_discount
time_step_observation
unknown
	unknown_0
	unknown_1
	unknown_2
identity	ѕбStatefulPartitionedCallЇ
StatefulPartitionedCallStatefulPartitionedCalltime_step_step_typetime_step_rewardtime_step_discounttime_step_observationunknown	unknown_0	unknown_1	unknown_2*
Tin

2*
Tout
2	*#
_output_shapes
:         *&
_read_only_resource_inputs
*-
config_proto

CPU

GPU2*0J 8*"
fR
__inference_action_7350282
StatefulPartitionedCallі
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0	*#
_output_shapes
:         2

Identity"
identityIdentity:output:0*c
_input_shapesR
P:         :         :         :         ::::22
StatefulPartitionedCallStatefulPartitionedCall:X T
#
_output_shapes
:         
-
_user_specified_nametime_step/step_type:UQ
#
_output_shapes
:         
*
_user_specified_nametime_step/reward:WS
#
_output_shapes
:         
,
_user_specified_nametime_step/discount:^Z
'
_output_shapes
:         
/
_user_specified_nametime_step/observation:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
▄A
Я
__inference_action_735028
	time_step
time_step_1
time_step_2
time_step_3A
=qnetwork_encodingnetwork_dense_matmul_readvariableop_resourceB
>qnetwork_encodingnetwork_dense_biasadd_readvariableop_resource3
/qnetwork_dense_1_matmul_readvariableop_resource4
0qnetwork_dense_1_biasadd_readvariableop_resource
identity	ѕА
&QNetwork/EncodingNetwork/flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2(
&QNetwork/EncodingNetwork/flatten/Const¤
(QNetwork/EncodingNetwork/flatten/ReshapeReshapetime_step_3/QNetwork/EncodingNetwork/flatten/Const:output:0*
T0*'
_output_shapes
:         2*
(QNetwork/EncodingNetwork/flatten/ReshapeЖ
4QNetwork/EncodingNetwork/dense/MatMul/ReadVariableOpReadVariableOp=qnetwork_encodingnetwork_dense_matmul_readvariableop_resource*
_output_shapes

:d*
dtype026
4QNetwork/EncodingNetwork/dense/MatMul/ReadVariableOpч
%QNetwork/EncodingNetwork/dense/MatMulMatMul1QNetwork/EncodingNetwork/flatten/Reshape:output:0<QNetwork/EncodingNetwork/dense/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         d2'
%QNetwork/EncodingNetwork/dense/MatMulж
5QNetwork/EncodingNetwork/dense/BiasAdd/ReadVariableOpReadVariableOp>qnetwork_encodingnetwork_dense_biasadd_readvariableop_resource*
_output_shapes
:d*
dtype027
5QNetwork/EncodingNetwork/dense/BiasAdd/ReadVariableOp§
&QNetwork/EncodingNetwork/dense/BiasAddBiasAdd/QNetwork/EncodingNetwork/dense/MatMul:product:0=QNetwork/EncodingNetwork/dense/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         d2(
&QNetwork/EncodingNetwork/dense/BiasAddх
#QNetwork/EncodingNetwork/dense/ReluRelu/QNetwork/EncodingNetwork/dense/BiasAdd:output:0*
T0*'
_output_shapes
:         d2%
#QNetwork/EncodingNetwork/dense/Relu└
&QNetwork/dense_1/MatMul/ReadVariableOpReadVariableOp/qnetwork_dense_1_matmul_readvariableop_resource*
_output_shapes

:d*
dtype02(
&QNetwork/dense_1/MatMul/ReadVariableOpЛ
QNetwork/dense_1/MatMulMatMul1QNetwork/EncodingNetwork/dense/Relu:activations:0.QNetwork/dense_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2
QNetwork/dense_1/MatMul┐
'QNetwork/dense_1/BiasAdd/ReadVariableOpReadVariableOp0qnetwork_dense_1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02)
'QNetwork/dense_1/BiasAdd/ReadVariableOp┼
QNetwork/dense_1/BiasAddBiasAdd!QNetwork/dense_1/MatMul:product:0/QNetwork/dense_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2
QNetwork/dense_1/BiasAddБ
*ShiftedCategorical_1/mode/ArgMax/dimensionConst*
_output_shapes
: *
dtype0*
valueB :
         2,
*ShiftedCategorical_1/mode/ArgMax/dimensionн
 ShiftedCategorical_1/mode/ArgMaxArgMax!QNetwork/dense_1/BiasAdd:output:03ShiftedCategorical_1/mode/ArgMax/dimension:output:0*
T0*#
_output_shapes
:         2"
 ShiftedCategorical_1/mode/ArgMaxP
add/yConst*
_output_shapes
: *
dtype0	*
value	B	 R 2
add/y|
addAddV2)ShiftedCategorical_1/mode/ArgMax:output:0add/y:output:0*
T0	*#
_output_shapes
:         2
addj
Deterministic/atolConst*
_output_shapes
: *
dtype0	*
value	B	 R 2
Deterministic/atolj
Deterministic/rtolConst*
_output_shapes
: *
dtype0	*
value	B	 R 2
Deterministic/rtolЉ
%Deterministic_1/sample/sample_shape/xConst*
_output_shapes
: *
dtype0*
valueB 2'
%Deterministic_1/sample/sample_shape/x┤
#Deterministic_1/sample/sample_shapeCast.Deterministic_1/sample/sample_shape/x:output:0*

DstT0*

SrcT0*
_output_shapes
: 2%
#Deterministic_1/sample/sample_shapes
Deterministic_1/sample/ShapeShapeadd:z:0*
T0	*
_output_shapes
:2
Deterministic_1/sample/ShapeЃ
Deterministic_1/sample/Shape_1Const*
_output_shapes
: *
dtype0*
valueB 2 
Deterministic_1/sample/Shape_1Ѓ
Deterministic_1/sample/Shape_2Const*
_output_shapes
: *
dtype0*
valueB 2 
Deterministic_1/sample/Shape_2╔
$Deterministic_1/sample/BroadcastArgsBroadcastArgs'Deterministic_1/sample/Shape_1:output:0'Deterministic_1/sample/Shape_2:output:0*
_output_shapes
: 2&
$Deterministic_1/sample/BroadcastArgs¤
&Deterministic_1/sample/BroadcastArgs_1BroadcastArgs%Deterministic_1/sample/Shape:output:0)Deterministic_1/sample/BroadcastArgs:r0:0*
_output_shapes
:2(
&Deterministic_1/sample/BroadcastArgs_1
Deterministic_1/sample/ConstConst*
_output_shapes
: *
dtype0*
valueB 2
Deterministic_1/sample/Constџ
&Deterministic_1/sample/concat/values_0Const*
_output_shapes
:*
dtype0*
valueB:2(
&Deterministic_1/sample/concat/values_0і
"Deterministic_1/sample/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2$
"Deterministic_1/sample/concat/axisф
Deterministic_1/sample/concatConcatV2/Deterministic_1/sample/concat/values_0:output:0+Deterministic_1/sample/BroadcastArgs_1:r0:0%Deterministic_1/sample/Const:output:0+Deterministic_1/sample/concat/axis:output:0*
N*
T0*
_output_shapes
:2
Deterministic_1/sample/concat║
"Deterministic_1/sample/BroadcastToBroadcastToadd:z:0&Deterministic_1/sample/concat:output:0*
T0	*'
_output_shapes
:         2$
"Deterministic_1/sample/BroadcastToЏ
Deterministic_1/sample/Shape_3Shape+Deterministic_1/sample/BroadcastTo:output:0*
T0	*
_output_shapes
:2 
Deterministic_1/sample/Shape_3б
*Deterministic_1/sample/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:2,
*Deterministic_1/sample/strided_slice/stackд
,Deterministic_1/sample/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2.
,Deterministic_1/sample/strided_slice/stack_1д
,Deterministic_1/sample/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2.
,Deterministic_1/sample/strided_slice/stack_2Ж
$Deterministic_1/sample/strided_sliceStridedSlice'Deterministic_1/sample/Shape_3:output:03Deterministic_1/sample/strided_slice/stack:output:05Deterministic_1/sample/strided_slice/stack_1:output:05Deterministic_1/sample/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*
end_mask2&
$Deterministic_1/sample/strided_sliceј
$Deterministic_1/sample/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2&
$Deterministic_1/sample/concat_1/axisЃ
Deterministic_1/sample/concat_1ConcatV2'Deterministic_1/sample/sample_shape:y:0-Deterministic_1/sample/strided_slice:output:0-Deterministic_1/sample/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2!
Deterministic_1/sample/concat_1л
Deterministic_1/sample/ReshapeReshape+Deterministic_1/sample/BroadcastTo:output:0(Deterministic_1/sample/concat_1:output:0*
T0	*#
_output_shapes
:         2 
Deterministic_1/sample/Reshapet
clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0	*
value	B	 R2
clip_by_value/Minimum/y▓
clip_by_value/MinimumMinimum'Deterministic_1/sample/Reshape:output:0 clip_by_value/Minimum/y:output:0*
T0	*#
_output_shapes
:         2
clip_by_value/Minimumd
clip_by_value/yConst*
_output_shapes
: *
dtype0	*
value	B	 R 2
clip_by_value/yї
clip_by_valueMaximumclip_by_value/Minimum:z:0clip_by_value/y:output:0*
T0	*#
_output_shapes
:         2
clip_by_valuea
IdentityIdentityclip_by_value:z:0*
T0	*#
_output_shapes
:         2

Identity"
identityIdentity:output:0*c
_input_shapesR
P:         :         :         :         :::::N J
#
_output_shapes
:         
#
_user_specified_name	time_step:NJ
#
_output_shapes
:         
#
_user_specified_name	time_step:NJ
#
_output_shapes
:         
#
_user_specified_name	time_step:RN
'
_output_shapes
:         
#
_user_specified_name	time_step:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
т
і
"__inference__traced_restore_735202
file_prefix:
6assignvariableop_qnetwork_encodingnetwork_dense_kernel:
6assignvariableop_1_qnetwork_encodingnetwork_dense_bias.
*assignvariableop_2_qnetwork_dense_1_kernel,
(assignvariableop_3_qnetwork_dense_1_bias

identity_5ѕбAssignVariableOpбAssignVariableOp_1бAssignVariableOp_2бAssignVariableOp_3б	RestoreV2бRestoreV2_1┴
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*═
value├B└B,model_variables/0/.ATTRIBUTES/VARIABLE_VALUEB,model_variables/1/.ATTRIBUTES/VARIABLE_VALUEB,model_variables/2/.ATTRIBUTES/VARIABLE_VALUEB,model_variables/3/.ATTRIBUTES/VARIABLE_VALUE2
RestoreV2/tensor_namesќ
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*
valueBB B B B 2
RestoreV2/shape_and_slices┐
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*$
_output_shapes
::::*
dtypes
22
	RestoreV2X
IdentityIdentityRestoreV2:tensors:0*
T0*
_output_shapes
:2

Identityд
AssignVariableOpAssignVariableOp6assignvariableop_qnetwork_encodingnetwork_dense_kernelIdentity:output:0*
_output_shapes
 *
dtype02
AssignVariableOp\

Identity_1IdentityRestoreV2:tensors:1*
T0*
_output_shapes
:2

Identity_1г
AssignVariableOp_1AssignVariableOp6assignvariableop_1_qnetwork_encodingnetwork_dense_biasIdentity_1:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_1\

Identity_2IdentityRestoreV2:tensors:2*
T0*
_output_shapes
:2

Identity_2а
AssignVariableOp_2AssignVariableOp*assignvariableop_2_qnetwork_dense_1_kernelIdentity_2:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_2\

Identity_3IdentityRestoreV2:tensors:3*
T0*
_output_shapes
:2

Identity_3ъ
AssignVariableOp_3AssignVariableOp(assignvariableop_3_qnetwork_dense_1_biasIdentity_3:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_3е
RestoreV2_1/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*1
value(B&B_CHECKPOINTABLE_OBJECT_GRAPH2
RestoreV2_1/tensor_namesћ
RestoreV2_1/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*
valueB
B 2
RestoreV2_1/shape_and_slices─
RestoreV2_1	RestoreV2file_prefix!RestoreV2_1/tensor_names:output:0%RestoreV2_1/shape_and_slices:output:0
^RestoreV2"/device:CPU:0*
_output_shapes
:*
dtypes
22
RestoreV2_19
NoOpNoOp"/device:CPU:0*
_output_shapes
 2
NoOp║

Identity_4Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_2^AssignVariableOp_3^NoOp"/device:CPU:0*
T0*
_output_shapes
: 2

Identity_4к

Identity_5IdentityIdentity_4:output:0^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_2^AssignVariableOp_3
^RestoreV2^RestoreV2_1*
T0*
_output_shapes
: 2

Identity_5"!

identity_5Identity_5:output:0*%
_input_shapes
: ::::2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12(
AssignVariableOp_2AssignVariableOp_22(
AssignVariableOp_3AssignVariableOp_32
	RestoreV2	RestoreV22
RestoreV2_1RestoreV2_1:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: "»L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*┐
action┤
4

0/discount&
action_0/discount:0         
>
0/observation-
action_0/observation:0         
0
0/reward$
action_0/reward:0         
6
0/step_type'
action_0/step_type:0         6
action,
StatefulPartitionedCall:0	         tensorflow/serving/predict*e
get_initial_stateP
2

batch_size$
get_initial_state_batch_size:0 tensorflow/serving/predict*R
get_train_step@"
int32
PartitionedCall_1:0 tensorflow/serving/predict:ёX
І
_wrapped_policy
model_variables

signatures

;action
<get_initial_state
=
train_step"
_generic_user_object
.

_q_network"
_generic_user_object
<
0
1
2
3"
trackable_list_wrapper
N

>action
?get_initial_state
@get_train_step"
signature_map
ф
	_encoder

_q_value_layer
	variables
trainable_variables
regularization_losses
	keras_api
*A&call_and_return_all_conditional_losses
B__call__"щ
_tf_keras_networkП{"class_name": "QNetwork", "name": "QNetwork", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "config": {"layer was saved without config": true}, "is_graph_network": false}
7:5d2%QNetwork/EncodingNetwork/dense/kernel
1:/d2#QNetwork/EncodingNetwork/dense/bias
):'d2QNetwork/dense_1/kernel
#:!2QNetwork/dense_1/bias
▓
_postprocessing_layers
	variables
trainable_variables
regularization_losses
	keras_api
*C&call_and_return_all_conditional_losses
D__call__"Є
_tf_keras_networkв{"class_name": "EncodingNetwork", "name": "EncodingNetwork", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "config": {"layer was saved without config": true}, "is_graph_network": false}
д

kernel
bias
	variables
trainable_variables
regularization_losses
	keras_api
*E&call_and_return_all_conditional_losses
F__call__"Ђ
_tf_keras_layerу{"class_name": "Dense", "name": "dense_1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "config": {"name": "dense_1", "trainable": true, "dtype": "float32", "units": 2, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "RandomUniform", "config": {"minval": -0.03, "maxval": 0.03, "seed": null, "dtype": "float32"}}, "bias_initializer": {"class_name": "Constant", "config": {"value": -0.2, "dtype": "float32"}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 100}}}, "build_input_shape": {"class_name": "TensorShape", "items": [0, 100]}}
<
0
1
2
3"
trackable_list_wrapper
<
0
1
2
3"
trackable_list_wrapper
 "
trackable_list_wrapper
Г

layers
layer_regularization_losses
	variables
non_trainable_variables
trainable_variables
layer_metrics
metrics
regularization_losses
B__call__
*A&call_and_return_all_conditional_losses
&A"call_and_return_conditional_losses"
_generic_user_object
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
Г

layers
 layer_regularization_losses
	variables
!non_trainable_variables
trainable_variables
"layer_metrics
#metrics
regularization_losses
D__call__
*C&call_and_return_all_conditional_losses
&C"call_and_return_conditional_losses"
_generic_user_object
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
Г

$layers
%layer_regularization_losses
	variables
&non_trainable_variables
trainable_variables
'layer_metrics
(metrics
regularization_losses
F__call__
*E&call_and_return_all_conditional_losses
&E"call_and_return_conditional_losses"
_generic_user_object
.
	0

1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
┐
)	variables
*trainable_variables
+regularization_losses
,	keras_api
*G&call_and_return_all_conditional_losses
H__call__"░
_tf_keras_layerќ{"class_name": "Flatten", "name": "flatten", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "config": {"name": "flatten", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 1, "axes": {}}}}
Ъ

kernel
bias
-	variables
.trainable_variables
/regularization_losses
0	keras_api
*I&call_and_return_all_conditional_losses
J__call__"Щ
_tf_keras_layerЯ{"class_name": "Dense", "name": "dense", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "config": {"name": "dense", "trainable": true, "dtype": "float32", "units": 100, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "VarianceScaling", "config": {"scale": 2.0, "mode": "fan_in", "distribution": "truncated_normal", "seed": null, "dtype": "float32"}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 4}}}, "build_input_shape": {"class_name": "TensorShape", "items": [0, 4]}}
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
Г

1layers
2layer_regularization_losses
)	variables
3non_trainable_variables
*trainable_variables
4layer_metrics
5metrics
+regularization_losses
H__call__
*G&call_and_return_all_conditional_losses
&G"call_and_return_conditional_losses"
_generic_user_object
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
Г

6layers
7layer_regularization_losses
-	variables
8non_trainable_variables
.trainable_variables
9layer_metrics
:metrics
/regularization_losses
J__call__
*I&call_and_return_all_conditional_losses
&I"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
І2ѕ
(__inference_polymorphic_action_fn_735039
(__inference_polymorphic_action_fn_735133▒
ф▓д
FullArgSpec(
args џ
j	time_step
jpolicy_state
varargs
 
varkw
 
defaultsб
б 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
м2¤
$__inference_get_initial_state_734963д
Ю▓Ў
FullArgSpec!
argsџ
jself
j
batch_size
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
B
__inference_<lambda>_734969
ZBX
$__inference_signature_wrapper_735090
0/discount0/observation0/reward0/step_type
6B4
$__inference_signature_wrapper_735102
batch_size
(B&
$__inference_signature_wrapper_735117
Т2сЯ
О▓М
FullArgSpecL
argsDџA
jself
jobservation
j	step_type
jnetwork_state

jtraining
varargs
 
varkw
 
defaultsџ

 
б 
p 

kwonlyargsџ 
kwonlydefaultsф 
annotationsф *
 
Т2сЯ
О▓М
FullArgSpecL
argsDџA
jself
jobservation
j	step_type
jnetwork_state

jtraining
varargs
 
varkw
 
defaultsџ

 
б 
p 

kwonlyargsџ 
kwonlydefaultsф 
annotationsф *
 
Т2сЯ
О▓М
FullArgSpecL
argsDџA
jself
jobservation
j	step_type
jnetwork_state

jtraining
varargs
 
varkw
 
defaultsџ

 
б 
p 

kwonlyargsџ 
kwonlydefaultsф 
annotationsф *
 
Т2сЯ
О▓М
FullArgSpecL
argsDџA
jself
jobservation
j	step_type
jnetwork_state

jtraining
varargs
 
varkw
 
defaultsџ

 
б 
p 

kwonlyargsџ 
kwonlydefaultsф 
annotationsф *
 
е2Цб
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
е2Цб
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
е2Цб
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
е2Цб
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
е2Цб
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
е2Цб
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
	J
Const:
__inference_<lambda>_734969Kб

б 
ф "і Q
$__inference_get_initial_state_734963)"б
б
і

batch_size 
ф "б У
(__inference_polymorphic_action_fn_735039╗яб┌
мб╬
к▓┬
TimeStep,
	step_typeі
	step_type         &
rewardі
reward         *
discountі
discount         4
observation%і"
observation         
б 
ф "R▓O

PolicyStep&
actionі
action         	
stateб 
infoб љ
(__inference_polymorphic_action_fn_735133сєбѓ
ЩбШ
Ь▓Ж
TimeStep6
	step_type)і&
time_step/step_type         0
reward&і#
time_step/reward         4
discount(і%
time_step/discount         >
observation/і,
time_step/observation         
б 
ф "R▓O

PolicyStep&
actionі
action         	
stateб 
infoб и
$__inference_signature_wrapper_735090јпбн
б 
╠ф╚
.

0/discount і

0/discount         
8
0/observation'і$
0/observation         
*
0/rewardі
0/reward         
0
0/step_type!і
0/step_type         "+ф(
&
actionі
action         	_
$__inference_signature_wrapper_73510270б-
б 
&ф#
!

batch_sizeі

batch_size "ф X
$__inference_signature_wrapper_7351170Kб

б 
ф "ф

int32і
int32 