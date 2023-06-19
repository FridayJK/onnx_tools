import os, sys, json
import numpy as np
import onnx
import onnxruntime




# onnx_name = "/mnt/data/sm_models/model_20230519/platedetection_model_v3.1.0.onnx"
onnx_name = "/mnt/data/sm_models/model_20230519/20211103-tkzx-experiment19_ckpt_epoch_56_acc1_81.40_norm.onnx"
onnx_name_out = "20211103-tkzx-experiment19_ckpt_epoch_56_acc1_81.40_norm.onnx"



# onnx.shape_inference.infer_shapes_path(onnx_name, onnx_name_out)
onnx_model = onnx.load_model(onnx_name)

onnx_model.graph.input[0].type.tensor_type.shape.dim[0].dim_param = "dim_value"
onnx_model.graph.input[0].type.tensor_type.shape.dim[0].dim_value=1

onnx_model.graph.output[0].type.tensor_type.shape.dim[0].dim_param = "dim_value"
onnx_model.graph.output[0].type.tensor_type.shape.dim[0].dim_value=1

onnx_mode_out = onnx.shape_inference.infer_shapes(onnx_model)

onnx.save(onnx_mode_out, onnx_name_out)

onnx.checker.check_model(onnx_model)

print("Done")



