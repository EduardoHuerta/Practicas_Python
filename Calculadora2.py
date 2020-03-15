import pandas as pd 
import numpy as np
import random

def get_random_ops(rows=100):
    data = []
    for i in range(0,rows):
        a = random.randint(1,100)
        b = random.randint(1,100)
        suma , resta, multiplicacion, division = random.choice([
            [1,0,0,0],
            [0,1,0,0],
            [0,0,1,0],
            [0,0,0,1],
        ])

        if suma == 1: y = a + b
        if resta == 1: y = a - b
        if multiplicacion == 1: y = a*b
        if division == 1: y = a/b
            
        data.append({
            "a": a,
            "b": b,
            "suma": suma,
            "resta": resta,
            "multi": multiplicacion,
            "division": division,
            "y": round(y,2)
        })
    return data        

data = pd.DataFrame(get_random_ops(25000))
data[["a","b","suma", "resta", "multi", "division", "y"]].head()

from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split

X_train,X_test, y_train, y_test =  train_test_split(
    data[["a","b","suma", "resta", "multi","division"]], data["y"],
    test_size = 0.30, random_state = 42
) 

model = MLPRegressor(
    max_iter = 800,
)
model.fit(X_train,y_train)
MLPRegressor(activation='relu', alpha=0.001, batch_size='auto',beta_1= 0.9,
        beta_2 = 0.999, early_stopping= False, epsilon= 1e-08,
        hidden_layer_sizes= (100,), learning_rate = 'constant',
        learning_rate_init=0.001, max_iter=800, momentum=0.9,
        nesterovs_momentum=True,power_t=0.5, random_state=None,
        shuffle= True, solver="adam", tol=0.0001, validation_fraction=0.1,
        verbose=False, warm_start=False)

print(X_test.iloc[3000])        
print(y_test.iloc[3000])        
print(model.predict([X_test.iloc[3000]]))

predict = model.predict(X_test)
print("Predicts: %s" % list(predict[:5]))



