from MLP import MLP, target_gradient, costNN, MLP_backprop_predict
from Utils import load_data, ExportONNX_JSON_TO_Custom, ExportAllformatsMLPSKlearn
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler



def main():
#TO-DO: calculate a testing a prediction and cost.
    print("Main program")
    data = load_data('./Datos/TankTraining_clean_OHE.csv')
    print(data)
    print(data.shape)

    import numpy as np
    x = data.drop(columns=data.columns[-1]).to_numpy()

    # Etiquetas
    y = data["action_int"].to_numpy()
    print(x)
    print(x.shape)

    #separar datos para entrenar
    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        x, y,
        test_size=0.33,
        random_state=0,
        stratify=y
    )
    #normalizar datos
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train_raw)
    X_test = scaler.transform(X_test_raw)

    print("Train:", X_train.shape, " Test:", X_test.shape)
    
    
main()