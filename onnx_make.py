import os, sys, json
import numpy as np
import onnx
from onnx import helper, TensorProto, AttributeProto, GraphProto
import onnxruntime

import onnx_generator
import tronmodel_parse



def make_input():
    return 0

def make_conv():
    return 0


def make_onnx(tron_dict):

    tron_graph = tron_dict["network"]
    tron_graph_ops   = tron_graph["op"]
    tron_graph_ops_name = tron_graph["op_name"]
    tron_graph_blobs = tron_graph["blob"]

    graph_input = {}
    graph_output = {}
    value_info   = {}
    onnx_node_graph = []

    # make tensor 
    graph_tensors = {}
    for i, blob in enumerate(tron_graph_blobs):
        graph_tensors[blob["name"]] = onnx_generator.make_initializer_tensor(blob)
    onnx_generator.onnx_maker.w_blobs = graph_tensors
    onnx_generator.onnx_maker.inputs = graph_input
    onnx_generator.onnx_maker.node_graph = onnx_node_graph
    

    # make input
    for i, op in enumerate(tron_graph_ops):
        if(op["name"] == "op_436"):
            print("pause")
        # parse BN
        if(op["type"] == "BatchNorm" and tron_graph_ops[i+1]["type"] == "Scale"):
            op_tmp = op
            continue
        if(op["type"] == "Scale" and tron_graph_ops[i-1]["type"] == "BatchNorm"):
            node_, out_ = onnx_generator.onnx_maker.funs_map["BatchNorm"](op_tmp, op)
            for node_s in node_:
                onnx_node_graph.append(node_s)
            continue

        node_, out_ = onnx_generator.onnx_maker.funs_map[op["type"]](op)
        if(op["type"]=="Input"):
            graph_input[op["name"]] = node_
        else:
            for node_s in node_:
                onnx_node_graph.append(node_s)

    # make output
    for arg in tron_dict["network"]["arg"]:
        if(arg.name=="out_blob"):
            network_outblob = tron_dict["network"]["arg"][1].v_s
            break
    output_tensor_list = onnx_generator.make_output_tensor(network_outblob)
        

    # onnx_graph = helper.make_graph(nodes=onnx_node_graph, name="test_zjk", inputs=[graph_input["input"]], outputs=out_, initializer=onnx_generator.onnx_maker.w_blobs.values())
    # onnx_graph = helper.make_graph(nodes=onnx_node_graph, name="test_zjk", inputs=[graph_input["input"]], outputs=output_tensor_list, initializer=onnx_generator.onnx_maker.w_blobs.values(), value_info=onnx_generator.onnx_maker.value_info.values())
    onnx_graph = helper.make_graph(nodes=onnx_node_graph, name="test_zjk", inputs=[graph_input["input"]], outputs=output_tensor_list, initializer=onnx_generator.onnx_maker.w_blobs.values())
    opset_imports = [helper.make_operatorsetid('', 13)]
    onnx_model = helper.make_model(onnx_graph, opset_imports=opset_imports)

    onnx_model = onnx.shape_inference.infer_shapes(onnx_model)

    onnx.checker.check_model(onnx_model)
    onnx.save(onnx_model, 'det_test_shigong_op13.onnx')
    # onnx.save(onnx_model, 'test_zjk1.onnx')

    return 0








if __name__ == "__main__":
    # tron_model_file = "./models/vehicle_attr_v1.4.1.tronmodel"
    tron_model_file = "./models/construction_sign_detection_v0.1.4.tronmodel"
    # onnx_model_file = "test.onnx"
    
    tron_model_dict = tronmodel_parse.tronmodel_parse(tron_model_file)

    # make onnx mdoel
    onnx_model = make_onnx(tron_model_dict)
    # onnx.checker.check_model(onnx_model)
    # print(onnx_model)

    # onnx.save(onnx_model, onnx_model_file)


