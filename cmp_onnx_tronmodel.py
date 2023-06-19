
import os, sys, json
import numpy as np
import onnx
from onnx import helper, TensorProto, AttributeProto, GraphProto

import onnxruntime
import pyshadow



if __name__ == "__main__":
    # onnx_file = "test_zjk.onnx"
    # onnx_file_out = "test_zjk_cheliang.onnx"
    onnx_file_out = "det_test_shigong.onnx"

    onnx_model = onnx.load_model(onnx_file_out)

    intermediate_layer_value_info = helper.ValueInfoProto()
    intermediate_layer_value_info.name = "op_434_0"
    onnx_model.graph.output.extend([intermediate_layer_value_info])

    intermediate_layer_value_info = helper.ValueInfoProto()
    intermediate_layer_value_info.name = "op_497_0"
    onnx_model.graph.output.extend([intermediate_layer_value_info])

    sess = onnxruntime.InferenceSession(onnx_model.SerializeToString())
    input_name = sess.get_inputs()[0].name
    input_shape= sess.get_inputs()[0].shape
    np_data = np.random.random(input_shape).astype(np.float32)
    output_list = [x.name for x in sess.get_outputs()]

    outputs_onnx = sess.run(output_list, {input_name : np_data})
# -------------------------------------------------------------
    onnx_file_out2 = "det_test_shigong_op13.onnx"
    onnx_model2 = onnx.load_model(onnx_file_out2)

    intermediate_layer_value_info = helper.ValueInfoProto()
    intermediate_layer_value_info.name = "op_434_0"
    onnx_model2.graph.output.extend([intermediate_layer_value_info])

    intermediate_layer_value_info = helper.ValueInfoProto()
    intermediate_layer_value_info.name = "op_497_0"
    onnx_model2.graph.output.extend([intermediate_layer_value_info])

    sess2 = onnxruntime.InferenceSession(onnx_model2.SerializeToString())
    input_name2 = sess2.get_inputs()[0].name
    input_shape2= sess2.get_inputs()[0].shape
    output_list2 = [x.name for x in sess2.get_outputs()]

    outputs_onnx2 = sess2.run(output_list2, {input_name2 : np_data})

    # tronmodel
    tronmodel_file = "./models/construction_sign_detection_v0.1.4.tronmodel"
    meta_network = pyshadow.MetaNetwork(tronmodel_file)
    net = meta_network.load_model(
        net_id=0,
        device_id=0,
        backend_type='Native',
        # max_batch_size=1,
        # use_fp16=True,
        # use_int8=True,
        # debug = True
        # calibrate_data_reader=None,
        # calibrate_data_reader=Reader("/home/jun/Workspace/datasets/person/person_1000.txt")
    )

    net.forward(data_map={net.inputs()[0]: np_data})
    outputs_tron = []
    for output in net.outputs():
        outputs_tron.append(net.get_blob(name=output))

    # outputs_tron = net.get_blob(name=net.outputs()[0])


    print("Done")

