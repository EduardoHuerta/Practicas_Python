import numpy as np

class NeuralNetwork():
    
    def __init__(self):
        # siembra para generación de números aleatorios 
        np.random.seed(1)
        
        #convertir pesos a una matriz 3 por 1 con valores de -1 a 1 y media de 0
        self.synaptic_weights = 2 * np.random.random((3, 1)) - 1

    def sigmoid(self, x):
        #aplicando la función sigmoid 
        return 1 / (1 + np.exp(-x))

    def sigmoid_derivative(self, x):
        #computación derivada de la función Sigmoide
        return x * (1 - x)

    def train(self, training_inputs, training_outputs, training_iterations):
        
        #entrenamiento del modelo para hacer predicciones precisas mientras se ajustan los pesos continuamente
        for iteration in range(training_iterations):
            ##siphon los datos de entrenamiento a través de la neurona
            output = self.think(training_inputs)

            #computing error rate for back-propagation
            error = training_outputs - output
            
            #performing weight adjustments
            adjustments = np.dot(training_inputs.T, error * self.sigmoid_derivative(output))

            self.synaptic_weights += adjustments

    def think(self, inputs):
        #passing the inputs via the neuron to get output   
        #converting values to floats
        
        inputs = inputs.astype(float)
        output = self.sigmoid(np.dot(inputs, self.synaptic_weights))
        return output


if __name__ == "__main__":

    #initializing the neuron class
    neural_network = NeuralNetwork()

    print("Iniciamos con los pesos generados aleatoriamente: ")
    print(neural_network.synaptic_weights)

    #training data consisting of 4 examples--3 input values and 1 output
    training_inputs = np.array([[0,0,1],
                                [1,1,1],
                                [1,0,1],
                                [0,1,1]])

    training_outputs = np.array([[0,1,1,0]]).T

    #training taking place
    neural_network.train(training_inputs, training_outputs, 15000)

    print("Finalizando los pesos del final del entrenamiento: ")
    print(neural_network.synaptic_weights)

    user_input_one = str(input("Ingrese primer dato: "))
    user_input_two = str(input("Ingrese segundo dato: "))
    user_input_three = str(input("Ingrese tercer dato: "))
    
    print("Considerando una nueva situacion: ", user_input_one, user_input_two, user_input_three)
    print("Nuevo dato de salida: ")
    print(neural_network.think(np.array([user_input_one, user_input_two, user_input_three])))
    print("Wow, Lo logramos!")