import numpy as np

class NeuralNetwork:
    def __init__(self, input_nodes, hidden_nodes, output_nodes, learning_rate):
        # Set number of nodes in input, hidden and output layers.
        self.input_nodes = input_nodes
        self.hidden_nodes = hidden_nodes
        self.output_nodes = output_nodes

        # Initialize weights
        self.weights_input_to_hidden = np.random.normal(0.0, self.hidden_nodes**-0.5,
                                                (self.hidden_nodes, self.input_nodes))

        self.weights_hidden_to_output = np.random.normal(0.0, self.output_nodes**-0.5,
                                                (self.output_nodes, self.hidden_nodes))

        self.learning_rate = learning_rate

        #### Set this to your implemented sigmoid function ####
        # TODO: Activation function is the sigmoid function
        self.activation_function = np.vectorize(lambda x: 1/float(1 + np.exp(-x)))

    def train(self, inputs_list, targets_list):

        # Convert inputs list to 2d array
        inputs = np.array(inputs_list, ndmin = 2).T
        targets = np.array(targets_list, ndmin = 2).T

        #### Implement the forward pass here ####
        ### Forward pass ###

        # TODO: Hidden layer
        hidden_inputs = np.dot(self.weights_input_to_hidden, inputs) # signals into hidden layer
        hidden_outputs = self.activation_function(hidden_inputs) # signals from hidden layer

        # TODO: Output layer
        final_inputs = np.dot(self.weights_hidden_to_output, hidden_outputs) # signals into final output layer
        final_outputs = final_inputs # signals from final output layer

        #### Implement the backward pass here ####
        ### Backward pass ###

        # TODO: Output error
        output_errors = (targets - final_outputs) # Output layer error is the difference between desired target and actual output.
        output_grad = 1 # Because f(x) = x has its derivative f'(x) = 1

        # TODO: Backpropagated error
        hidden_errors = (output_errors) * self.weights_hidden_to_output # errors propagated to the hidden layer
        hidden_grad = (hidden_outputs * (1-hidden_outputs)).T # hidden layer gradients

        # TODO: Update the weights
        print(output_errors.shape)
        print(hidden_outputs.shape)
        print((output_grad * output_errors).shape)
        print()
        self.weights_hidden_to_output += self.learning_rate * (np.dot(hidden_outputs, (output_grad * output_errors).T).T) # update hidden-to-output weights with gradient descent step
        print(self.weights_hidden_to_output.shape)
        print(hidden_grad.shape)+
        print(hidden_errors.shape)
        print(inputs.shape)
        print((hidden_grad * hidden_errors).shape)
        # self.weights_input_to_hidden += self.learning_rate * (np.dot(inputs, hidden_grad * hidden_errors).T) # update input-to-hidden weights with gradient descent step
        self.weights_input_to_hidden += self.learning_rate * (np.dot(inputs, hidden_grad * hidden_errors.T).T) # update input-to-hidden weights with gradient descent step

    def run(self, inputs_list):
        # Run a forward pass through the network
        inputs = np.array(inputs_list, ndmin=2).T

        #### Implement the forward pass here ####
        # TODO: Hidden layer
        hidden_inputs = np.dot(self.weights_input_to_hidden, inputs) # signals into hidden layer
        hidden_outputs = self.activation_function(hidden_inputs) # signals from hidden layer

        # TODO: Output layer
        final_inputs = np.dot(self.weights_hidden_to_output, hidden_outputs) # signals into final output layer
        final_outputs = final_inputs # signals from final output layer

        return final_outputs
