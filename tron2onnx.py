import os, sys, json
import numpy as np
import onnx

import tronmodel_parse
import onnx_make










if __name__ == "__main__":
    tron_model_file = "./models/vehicle_attr_v1.4.1.tronmodel"
    onnx_model_file = "test.onnx"
    
    tron_model_dict = tronmodel_parse(tron_model_file)

    onnx_model = make_onnx(tron_model_dict)

    # save model
    onnx.checker.check_model(onnx_model)
    print(onnx_model)

    onnx.save(onnx_model, onnx_model_file)