from MLP import MLP, target_gradient, costNN, MLP_backprop_predict
from Utils import load_data, ExportONNX_JSON_TO_Custom, ExportAllformatsMLPSKlearn
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier



def main():
#TO-DO: calculate a testing a prediction and cost.
    print("Main program")
    data = load_data('./Datos/TodosJuntos.csv')
    print(data)
    print(data.shape)
    
    
main()