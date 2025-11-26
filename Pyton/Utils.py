from skl2onnx import to_onnx
from onnx2json import convert
import pickle
import json
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
import numpy as np

def one_hot_encoding(Y):
    enc = OneHotEncoder(sparse_output=False, categories=[[0,1,2,3,4,5,6,7,8,9,10]])
    YEnc =enc.fit_transform(Y.reshape(-1,1))
    #YEnc = YEnc.T
    return YEnc

def one_hot_encoding2(Y):
    enc = OneHotEncoder(sparse_output=False, categories=[[0,1,2,3,4]])
    YEnc =enc.fit_transform(Y.reshape(-1,1))
    #YEnc = YEnc.T
    return YEnc

def cleanData(data):
    encoded_inicio = []   # one-hot de las NEIGHBORHOOD

    # One-hot para las NEIGHBORHOOD
    for i in range(4):
        encoded = one_hot_encoding(data[:, i])
        encoded_inicio.append(encoded)

    # Columnas intermedias sin tocar 
    medio = data[:, 4:-1].astype(float)
    

    # One-hot de action
    encoded_final = one_hot_encoding2(data[:, -1])

    # Concatenar manteniendo el orden 
    final_data = np.concatenate(
        encoded_inicio + [medio] + [encoded_final],
        axis=1
    )

    return final_data
    

def load_data(file):
    data = pd.read_csv(file)
    data = np.array(data)
    print(data.shape)
    data = cleanData(data)
    return data

def ExportONNX_JSON_TO_Custom(onnx_json,mlp):
    graphDic = onnx_json["graph"]
    initializer = graphDic["initializer"]
    s= "num_layers:"+str(mlp.n_layers_)+"\n"
    index = 0
    parameterIndex = 0
    for parameter in initializer:
        name = parameter["name"]
        print("Capa ",name)
        if name != "classes" and name != "shape_tensor":
            print("procesando ",name)
            s += "parameter:"+str(parameterIndex)+"\n"
            print(parameter["dims"])
            s += "dims:"+str(parameter["dims"])+"\n"
            print(parameter["name"])
            s += "name:"+str(parameter["name"])+"\n"
            print(parameter["doubleData"])
            s += "values:"+str(parameter["doubleData"])+"\n"
            index = index + 1
            parameterIndex = index // 2
        else:
            print("Esta capa no es interesante ",name)
    return s

def ExportAllformatsMLPSKlearn(mlp,X,picklefileName,onixFileName,jsonFileName,customFileName):
    with open(picklefileName,'wb') as f:
        pickle.dump(mlp,f)
    
    onx = to_onnx(mlp, X[:1])
    with open(onixFileName, "wb") as f:
        f.write(onx.SerializeToString())
    
    onnx_json = convert(input_onnx_file_path=onixFileName,output_json_path=jsonFileName,json_indent=2)
    
    customFormat = ExportONNX_JSON_TO_Custom(onnx_json,mlp)
    with open(customFileName, 'w') as f:
        f.write(customFormat)
