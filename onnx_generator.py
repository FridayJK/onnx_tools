import os, sys, json
import numpy as np
import onnx
from onnx import helper, TensorProto, AttributeProto, GraphProto


def make_initializer_tensor(blob) -> TensorProto:
    type_ = None
    if(len(blob["data_f"])):
        type_ = TensorProto.DataType.FLOAT
        value = np.array(blob["data_f"], dtype=np.float32)
    if(len(blob["data_i"])):
        type_ = TensorProto.DataType.INT32
        value = np.array(blob["data_i"], dtype=np.int32)
        print(f'---------------------{blob["name"]} is INT---------------------')
    value = value.reshape(blob["shape"])

    tensor = helper.make_tensor(
        name=blob["name"],
        data_type=type_,
        dims=blob["shape"],
        vals=value.tobytes(),
        raw=True
    )
    return tensor

def get_attri(op_args, attri_name):
    for arg in op_args:
        if(arg.name == attri_name):
            return arg
    return 0

def make_shape_infer(onnx_maker, out_blob):
    onnx_graph = helper.make_graph(nodes=onnx_maker.node_graph, name="shape_infer_tmp", inputs=[onnx_maker.inputs["input"]], outputs=[out_blob], initializer=onnx_maker.w_blobs.values())
    onnx_model = helper.make_model(onnx_graph)
    onnx_model = onnx.shape_inference.infer_shapes(onnx_model)

    return [dim.dim_value for dim in onnx_model.graph.output[0].type.tensor_type.shape.dim]


class OnnxMaker:
    def __init__(self):
        self.funs_map = {}
        self.w_blobs    = {}
        self.inputs   = {}
        self.value_info = {}
        self.infer_blob_idx = 1
        self.node_graph = []
    
    def register(self, func):
        self.funs_map[func.__name__.replace("make_", "")] = func


onnx_maker = OnnxMaker()

def make_output_tensor(output_list):
    out_tensor = []
    for output in output_list:
        out_tensor.append(helper.make_tensor_value_info(output, TensorProto.FLOAT, None))
    return out_tensor


@onnx_maker.register
def make_Input(op):
    input = helper.make_tensor_value_info(op["name"], TensorProto.FLOAT, list(op["arg"][0].v_i))

    return input, []

@onnx_maker.register
def make_Scale(op):
    if(op["name"]=="raw_data"):    #preprocess
        scale_w = np.array(list(op["arg"][3].v_f), dtype=np.float32)
        bias_w = np.array(list(op["arg"][4].v_f), dtype=np.float32)
        scale_tensor = helper.make_tensor(name=op["arg"][3].name, data_type=TensorProto.DataType.FLOAT, dims=[1,3,1,1], vals=scale_w.tobytes(), raw=True)
        bias_tensor = helper.make_tensor(name=op["arg"][4].name, data_type=TensorProto.DataType.FLOAT, dims=[1,3,1,1], vals=bias_w.tobytes(), raw=True)

        scale_out_name = str(onnx_maker.infer_blob_idx)
        onnx_maker.infer_blob_idx += 1
        bias_out_name = op["top"][0]
        mul_name = "Mul_" + str(onnx_maker.infer_blob_idx)
        onnx_maker.infer_blob_idx += 1
        add_name = "Add_" + str(onnx_maker.infer_blob_idx)
        
        node_scale = helper.make_node("Mul", inputs=["input", op["arg"][3].name], outputs=[scale_out_name], name=mul_name)
        node_bias  = helper.make_node("Add", inputs=[scale_out_name, op["arg"][4].name], outputs=[bias_out_name], name=add_name)

        onnx_maker.w_blobs[op["arg"][3].name] = scale_tensor
        onnx_maker.w_blobs[op["arg"][4].name] = bias_tensor
    else:
        exit(-1)
        print("Done")

    bias_out  = helper.make_tensor_value_info(bias_out_name, TensorProto.FLOAT, None)
    return [node_scale, node_bias], [bias_out]

@onnx_maker.register
def make_Conv(op):
    output = helper.make_tensor_value_info(op["top"][0], TensorProto.FLOAT, None)

    # parse attributes
    arg_kernel   = get_attri(op["arg"], "kernel_size")
    arg_strides  = get_attri(op["arg"], "stride")
    arg_dilation = get_attri(op["arg"], "dilation")
    arg_group    = get_attri(op["arg"], "group")
    arg_pads     = get_attri(op["arg"], "pad")
    arg_relu     = get_attri(op["arg"], "type")

    if(isinstance(arg_dilation.s_i, int)):      # dilations
        dilations_shape = [arg_dilation.s_i]*2
    else:
        dilations_shape = list(arg_dilation.v_i)
    pads_shape_tron = list(arg_pads.v_i)
    pads_shape_onnx = [pads_shape_tron[0], pads_shape_tron[0], pads_shape_tron[1], pads_shape_tron[1]]

    if(arg_relu):
        if(arg_relu.s_i==1):
            conv_out_name = op["top"][0] + "_" + str(onnx_maker.infer_blob_idx)
            onnx_maker.infer_blob_idx += 1
            node_conv = helper.make_node("Conv", inputs=op["bottom"], outputs=[conv_out_name], kernel_shape=list(arg_kernel.v_i), strides=list(arg_strides.v_i), dilations=dilations_shape, group=arg_group.s_i, pads=pads_shape_onnx, name=op["name"])
            node_activate = helper.make_node("Relu", inputs=[conv_out_name], outputs=op["top"], name="Relu_"+op["name"])
            return [node_conv, node_activate], [output]
        else:
            print("error")
            exit(0)
    else:
        node_conv = helper.make_node("Conv", inputs=op["bottom"], outputs=[op["top"][0]], kernel_shape=list(arg_kernel.v_i), strides=list(arg_strides.v_i), dilations=dilations_shape, group=arg_group.s_i, pads=pads_shape_onnx, name=op["name"])
        return [node_conv], [output]
    
    return [node_conv], [output]

@onnx_maker.register
def make_Activate(op):
    activate_types = ["PRelu", "Relu", "Leaky", "Sigmoid", "SoftPlus", "Tanh", "Relu6", "HardSwish", "Gelu", "Mish", "HardSigmoid"]
    
    if(op["arg"][0].s_i == 6):     # https://github.com/onnx/onnx/issues/3262, http://www.xavierdupre.fr/app/mlprodict/helpsphinx/onnxops/onnx__Clip.html
        min_name = "Clip_"+str(onnx_maker.infer_blob_idx)
        onnx_maker.infer_blob_idx+=1
        max_name = "Clip_"+str(onnx_maker.infer_blob_idx)
        onnx_maker.infer_blob_idx+=1
        min_val = np.array([0],dtype=np.float32)
        max_val = np.array([6],dtype=np.float32)
        min_tensor = helper.make_tensor(name=min_name, data_type=TensorProto.DataType.FLOAT, dims=min_val.shape, vals=min_val)
        max_tensor = helper.make_tensor(name=max_name, data_type=TensorProto.DataType.FLOAT, dims=max_val.shape, vals=max_val)
        # min_ = helper.make_empty_tensor_value_info(min_name,)
        node_activate = helper.make_node("Clip", inputs=[op["bottom"][0], min_name, max_name], outputs=[op["top"][0]], name=op["name"])

        onnx_maker.w_blobs[min_name] = min_tensor
        onnx_maker.w_blobs[max_name] = max_tensor
    else:
        activate_type = activate_types[op["arg"][0].s_i]
        node_activate = helper.make_node(activate_type, inputs=op["bottom"], outputs=op["top"], name=op["name"])

    output = helper.make_tensor_value_info(op["top"][0], TensorProto.FLOAT, None)
    return [node_activate], [output]


@onnx_maker.register
def make_Binary(op):
    bin_ops = ['Add', 'Sub', 'Mul', 'Div', 'Pow', 'Max', 'Min']

    # check output name
    if(op["top"][0] == op["bottom"][0]):
        for node in onnx_maker.node_graph:
            if(node.output[0] == op["top"][0]):
                node.output[0] = node.output[0] + "_" + str(onnx_maker.infer_blob_idx)
                onnx_maker.infer_blob_idx += 1
                op["bottom"][0] = node.output[0]

    if(op["arg"][0].s_i<5):
        node_binary = helper.make_node(bin_ops[op["arg"][0].s_i], inputs=op["bottom"], outputs=[op["top"][0]], name=op["name"])
    else:
        scalar_val = np.array([op["arg"][1].s_f],dtype=np.float32)
        scalar_val_name = "Scalar_"+str(onnx_maker.infer_blob_idx)
        onnx_maker.infer_blob_idx += 1
        scalar_val_tensor = helper.make_tensor(name=scalar_val_name, data_type=TensorProto.DataType.FLOAT, dims=scalar_val.shape, vals=scalar_val)
        node_binary = helper.make_node(bin_ops[op["arg"][0].s_i], inputs=op["bottom"] + [scalar_val_name], outputs=[op["top"][0]], name=op["name"])
        onnx_maker.w_blobs[scalar_val_name] = scalar_val_tensor

    bin_output = helper.make_tensor_value_info(op["top"][0], TensorProto.FLOAT, None)
    return [node_binary], [bin_output]


@onnx_maker.register
def make_BatchNorm(op_batch_norm, op_scale):
    esp_name = "Batch_esp_"+str(onnx_maker.infer_blob_idx)
    onnx_maker.infer_blob_idx+=1
    esp_val = np.array([op_batch_norm["arg"][1].s_f],dtype=np.float32)
    esp_tensor = helper.make_tensor(name=esp_name, data_type=TensorProto.DataType.FLOAT, dims=esp_val.shape, vals=esp_val)

    # node_batchNorm = helper.make_node("BatchNormalization", inputs=op_batch_norm["bottom"][:1] + op_scale["bottom"][1:] + op_batch_norm["bottom"][1:], outputs=op_scale["top"][0], epsilon=op_batch_norm["arg"][1].s_f, name=op_batch_norm["name"])
    node_batchNorm = helper.make_node("BatchNormalization", inputs=op_batch_norm["bottom"][:1] + op_scale["bottom"][1:] + op_batch_norm["bottom"][1:], outputs=op_scale["top"], epsilon=op_batch_norm["arg"][1].s_f, name=op_batch_norm["name"])


    output = helper.make_tensor_value_info(op_scale["top"][0], TensorProto.FLOAT, None)
    return [node_batchNorm], [output]


@onnx_maker.register
def make_Pooling(op):                    # only support 'MaxPool', 'AveragePool'
    pool_ops = ['MaxPool', 'AveragePool', 'GlobalMaxPool', 'GlobalAveragePool']  
    
    node_type = pool_ops[op["arg"][0].s_i]

    kernel_shapes= list(op["arg"][1].v_i)
    pad_shape = [op["arg"][3].v_i[0], op["arg"][3].v_i[0], op["arg"][3].v_i[1], op["arg"][3].v_i[1]]
    stride_shape = [op["arg"][2].v_i[0], op["arg"][2].v_i[1]]
    ceil_mode_val = op["arg"][5].s_i
    
    node_pool = helper.make_node(node_type, inputs=op["bottom"], outputs=op["top"], ceil_mode=ceil_mode_val, kernel_shape=kernel_shapes, pads=pad_shape, strides=stride_shape, name=op["name"])

    output = helper.make_tensor_value_info(op["top"][0], TensorProto.FLOAT, None)
    return [node_pool], [output]

@onnx_maker.register
def make_Reshape(op):
    shape_tensor_name = "shape_"+str(onnx_maker.infer_blob_idx)
    onnx_maker.infer_blob_idx+=1
    shape_np = np.array(list(op["arg"][0].v_i),dtype=np.int64)
    shape_tensor = helper.make_tensor(name=shape_tensor_name, data_type=TensorProto.DataType.INT64, dims=shape_np.shape, vals=shape_np)

    node_reshape = helper.make_node("Reshape", inputs=op["bottom"] + [shape_tensor_name], outputs=op["top"], name=op["name"])

    onnx_maker.w_blobs[shape_tensor_name] = shape_tensor

    output = helper.make_tensor_value_info(op["top"][0], TensorProto.FLOAT, None)
    return [node_reshape], [output]


@onnx_maker.register
def make_Connected(op):
    alpha_gemm = np.array([1],dtype=np.float32)
    beta_gemm  = np.array([1],dtype=np.float32)
    trans_B    = np.array([1],dtype=np.float32)

    node_gemm = helper.make_node("Gemm", inputs=op["bottom"], outputs=op["top"], transB=op["arg"][2].s_i, name=op["name"])

    output = helper.make_tensor_value_info(op["top"][0], TensorProto.FLOAT, None)
    return [node_gemm], [output]

@onnx_maker.register
def make_Softmax(op):
    axis = op["arg"][0].s_i

    node_softmax = helper.make_node("Softmax", inputs=op["bottom"], outputs=op["top"], axis=axis, name=op["name"])

    output = helper.make_tensor_value_info(op["top"][0], TensorProto.FLOAT, None)
    return [node_softmax], [output]

@onnx_maker.register
def make_Concat(op):
    axis = op["arg"][0].s_i

    node_concat = helper.make_node("Concat", inputs=op["bottom"], outputs=op["top"], axis=axis, name=op["name"])

    output = helper.make_tensor_value_info(op["top"][0], TensorProto.FLOAT, None)
    return [node_concat], [output]


@onnx_maker.register
def make_Resize(op):
    resize_types = ["nearest", "bilinear"]

    # shape infer 
    input = helper.make_tensor_value_info(op["bottom"][0], TensorProto.FLOAT, None)
    input_shape = make_shape_infer(onnx_maker, input)
    

    # make scales tensor, make roi tensor
    if(len(list(op["arg"][0].v_i)) == 2):   #use scale
        size_ = list(op["arg"][0].v_i)
        scale_np = np.array([1, 1, int(size_[0]/input_shape[2]), int(size_[1]/input_shape[3])],dtype=np.float32)
    elif(op["arg"][0].s_i != 0):
        size_ = [op["arg"][0].s_i]*2
        exit(-1)
    else:
        if(len(list(op["arg"][1].v_f)) == 2):
            scale_ = list(op["arg"][1].v_f)
        elif(op["arg"][1].s_f != 0):
            scale_ = [op["arg"][1].s_f]*2
        scale_np = np.array([1, 1, scale_[0], scale_[1]],dtype=np.float32) 

    scales_tensor_name = "Scales_" + str(onnx_maker.infer_blob_idx)
    onnx_maker.infer_blob_idx += 1
    scales_tensor = helper.make_tensor(name=scales_tensor_name, data_type=TensorProto.DataType.FLOAT, dims=scale_np.shape, vals=scale_np)

    roi_np = np.empty([0], dtype=np.float32)
    roi_tensor_name =  "Roi_" + str(onnx_maker.infer_blob_idx)
    onnx_maker.infer_blob_idx += 1
    roi_tensor = helper.make_tensor(name=roi_tensor_name, data_type=TensorProto.DataType.FLOAT, dims=roi_np.shape, vals=roi_np)

    # make node 
    node_resize = helper.make_node("Resize", inputs=op["bottom"] + [roi_tensor_name, scales_tensor_name], outputs=op["top"], mode=resize_types[op["arg"][2].s_i], nearest_mode="floor", coordinate_transformation_mode="asymmetric", cubic_coeff_a=-0.75, name=op["name"])
    # node_resize = helper.make_node("Resize", inputs=op["bottom"] + [scales_tensor_name], outputs=op["top"], mode=resize_types[op["arg"][2].s_i], nearest_mode="floor", coordinate_transformation_mode="asymmetric", cubic_coeff_a=-0.75, name=op["name"])

    onnx_maker.w_blobs[roi_tensor_name] = roi_tensor
    onnx_maker.w_blobs[scales_tensor_name] = scales_tensor

    output = helper.make_tensor_value_info(op["top"][0], TensorProto.FLOAT, None)
    return [node_resize], [output]

@onnx_maker.register
def make_Slice(op):
    tensor_slice_points = []
    tensor_slice_point_names = []
    # shape infer 
    input = helper.make_tensor_value_info(op["bottom"][0], TensorProto.FLOAT, None)
    input_shape = make_shape_infer(onnx_maker, input)

    axis = op["arg"][0].s_i
    if(len(op["arg"])==2):
        slice_point = list(op["arg"][1].v_i)
    elif(len(op["arg"])==1):
        out_num = len(op["top"])
        ave_len = input_shape[axis]//out_num
        slice_point = [ave_len*i for i in range(1,out_num)]
    
    # make slice point tensor
    axis_np = np.array([axis],dtype=np.int64)
    axis_tensor_name = "Slice_axis_"+str(onnx_maker.infer_blob_idx)
    onnx_maker.infer_blob_idx += 1
    axis_tensor = helper.make_tensor(name=axis_tensor_name, data_type=TensorProto.DataType.INT64, dims=axis_np.shape, vals=axis_np)
    slice_point = [0] + slice_point + input_shape[axis:axis+1]
    for i, point in enumerate(slice_point):
        tensor_slice_point_name = "Slice_point"+str(i)+"_"+str(onnx_maker.infer_blob_idx)
        onnx_maker.infer_blob_idx += 1
        tensor_slice_point_names.append(tensor_slice_point_name)

        slice_point_np = np.array([point],dtype=np.int64)
        tensor_slice_points.append(helper.make_tensor(name=tensor_slice_point_name, data_type=TensorProto.DataType.INT64, dims=slice_point_np.shape, vals=slice_point_np))

    # make node
    node_slices = []
    out_list    = []
    for i, top_ in enumerate(op["top"]):
        out_list.append(helper.make_tensor_value_info(top_, TensorProto.FLOAT, None))
        node_slices.append(helper.make_node(op["type"], inputs=op["bottom"]+tensor_slice_point_names[i:(i+2)]+[axis_tensor_name], outputs=[top_], name=op["name"]+"_"+str(i)))
        
    # add init blob
    for i, name in enumerate(tensor_slice_point_names):
        onnx_maker.w_blobs[name] = tensor_slice_points[i]
    onnx_maker.w_blobs[axis_tensor_name] = axis_tensor

    return node_slices, out_list

@onnx_maker.register
def make_Unary(op):
    unary_types = ["Abs", "Square", "Sqrt", "Log", "Exp", "Sin", "Cos", "Tan", "Asin", "Acos", "Atan", "Floor", "Ceil", "Neg", "Reciprocal"]
    unary_type  = unary_types[op["arg"][0].s_i]

    node_unary = helper.make_node(unary_type, inputs=op["bottom"], outputs=op["top"], name=op["name"])

    out = helper.make_tensor_value_info(op["top"][0], TensorProto.FLOAT, None)
    return [node_unary], [out]


@onnx_maker.register
def make_Permute(op):
    permutation = op["arg"][0].v_i
    node_transpose = helper.make_node("Transpose", inputs=op["bottom"], outputs=op["top"], perm=permutation, name=op["name"])

    out = helper.make_tensor_value_info(op["top"][0], TensorProto.FLOAT, None)
    return [node_transpose], [out]

@onnx_maker.register
def make_Unsqueeze(op):
    axes_np = np.array(list(op["arg"][0].v_i), dtype=np.int32)
    axes_tensor_name =  "Axes_" + str(onnx_maker.infer_blob_idx)
    onnx_maker.infer_blob_idx += 1
    axes_tensor = helper.make_tensor(name=axes_tensor_name, data_type=TensorProto.DataType.INT64, dims=axes_np.shape, vals=axes_np)
    
    node_unsqueeze = helper.make_node("Unsqueeze", inputs=op["bottom"] + [axes_tensor_name], outputs=op["top"], name=op["name"])

    onnx_maker.w_blobs[axes_tensor_name] = axes_tensor

    out = helper.make_tensor_value_info(op["top"][0], TensorProto.FLOAT, None)
    return [node_unsqueeze], [out]
