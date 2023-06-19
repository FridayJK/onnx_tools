import os, sys, json
import numpy as np

import src.proto.src.proto.tron_pb2 as tron_pb2





def serialize():
    return 0


def parse_Argument(arg):
    arg_dict = {"name":arg.name, "s_f":arg.s_f, "s_i":arg.s_i, "s_s":arg.s_s, "v_f":[], "v_i":[], "v_s":[]}
    basic_repeated = ["v_f", "v_i", "v_s"]
    for arr in basic_repeated:
        for v_ in eval(f'arg.{arr}'):
            arg_dict[f'{arr}'].append(v_)

    # [arg_dict[f'{arr}'].append(v_) for arr in basic_repeated for v_ in eval(f'arg.{arr}') if len(arr!=[])]
    return arg_dict


def tron_deserialize(tron_model_message):
    metaNet_info = {}
    metaNet_info["name"] = tron_model_message.name
    metaNet_info["model_info"] = {"project":tron_model_message.model_info.project, "version":tron_model_message.model_info.version, "method":tron_model_message.model_info.method}
    metaNet_info["arg"] = []

    for i, arg in enumerate(tron_model_message.arg):
        metaNet_info["arg"].append(arg)
        arg_ = parse_Argument(arg)

        # metaNet_info["arg"].append({"name":arg.name, "s_f":arg.s_f, "s_i":arg.s_i, "s_s":arg.s_s, "v_f":[], "v_i":[], "v_s":[]})
        # basic_repeated = ["v_f", "v_i", "v_s"]
        # [metaNet_info["arg"][i][f'{arr}'].append(v_) for arr in basic_repeated for v_ in eval(f'arg.{arr}')]
    
    # parse NetParam
    netGraph = {}
    for net_ in tron_model_message.network:
        netGraph["name"] = net_.name
        netGraph["arg"]  = []
        for arg in net_.arg:
            netGraph["arg"].append(arg)
            arg_ = parse_Argument(arg)
            print(f'net.arg: {arg_}')
        # op
        netGraph["op"] = []
        netGraph["op_name"] = {}
        for i, op_ in enumerate(net_.op):
            node_ = {}
            node_["name"] = op_.name
            node_["type"] = op_.type
            node_["top"]  = []
            node_["bottom"] = []
            node_["arg"] = []
            for top_ in op_.top:
                node_["top"].append(top_)
            for bottom_ in op_.bottom:
                node_["bottom"].append(bottom_)
            for arg_ in op_.arg:
                node_["arg"].append(arg_)
                # print(f'{op_.type}.arg: {arg_}')
            print(f'node: {node_}')
            netGraph["op_name"][op_.name] = i
            netGraph["op"].append(node_)
        # weight
        netGraph["blob"] = []
        netGraph["blob_name"] = {}
        for i, blob_ in enumerate(net_.blob):
            weight_ = {}
            weight_["name"] = blob_.name
            weight_["type"] = blob_.type
            weight_["shape"] = list(blob_.shape)
            print(f'blob.{blob_.name}:{weight_}')
            weight_["data_f"] = list(blob_.data_f)
            # print(f'blob.{blob_.name}:{weight_}')
            weight_["data_i"] = list(blob_.data_i)
            weight_["data_b"] = list(blob_.data_b)
            netGraph["blob_name"][blob_.name] = i
            netGraph["blob"].append(weight_)

    metaNet_info["network"] = netGraph

    return metaNet_info




def tronmodel_parse(file_name):
    with open(file_name, "rb") as f:
        tron_model_message = tron_pb2.MetaNetParam().FromString(f.read())

    metaNet_info = tron_deserialize(tron_model_message)

    return metaNet_info

if __name__ == "__main__":
    tron_model_file = "./models/vehicle_attr_v1.4.1.tronmodel"

    with open(tron_model_file, "rb") as f:
        tron_model_message = tron_pb2.MetaNetParam().FromString(f.read())

    metaNet_info = tron_deserialize(tron_model_message)

    print("Done!")

