from skl2onnx import to_onnx
from onnx2json import convert
import pickle
import json
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
import numpy as np

def load_data(file):
    data = pd.read_csv(file)
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

def WriteStandardScaler(file,mean,var):
    line = ""
    for i in range(0,len(mean)-1):
        line = line + str(mean[i]) + ","
    line = line + str(mean[len(mean)-1])+ "\n"
    for i in range(0,len(var)-1):
        line = line + str(var[i]) + ","
    line = line + str(var[len(var)-1])+ "\n"
    with open(file, 'w') as f:
        f.write(line)

def confusionMatrix(P,Y):
    TP=0
    FP=0
    TN=0
    FN=0
    for x in range(0,np.size(P)):
        if P[x] == Y[x] and Y[x] == 0:
            TP += 1
        elif P[x] == Y[x] and Y[x] != 0:
            TN += 1
        elif Y[x] == 0:
            FN += 1
        else:
            FP += 1

    cm=np.array([[TP,FP],[FN,TN]])
    return cm

def precission(CM):
    pr = CM[0][0]/(CM[0][0]+CM[0][1])
    return pr

def recall(CM):
    rc = CM[0][0]/(CM[0][0]+CM[1][0])
    return rc

def F1_Score(pr,rc):
    f1 = 2*((pr*rc)/(pr+rc))
    return f1
