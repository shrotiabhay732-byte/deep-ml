
import numpy as np

class LSTM:
    def __init__(self, input_size, hidden_size):
        self.input_size = input_size
        self.hidden_size = hidden_size

        # Initialize weights
        self.Wf = np.random.randn(hidden_size, input_size + hidden_size)
        self.Wi = np.random.randn(hidden_size, input_size + hidden_size)
        self.Wc = np.random.randn(hidden_size, input_size + hidden_size)
        self.Wo = np.random.randn(hidden_size, input_size + hidden_size)

        # Initialize biases
        self.bf = np.zeros((hidden_size, 1))
        self.bi = np.zeros((hidden_size, 1))
        self.bc = np.zeros((hidden_size, 1))
        self.bo = np.zeros((hidden_size, 1))

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def forward(self, x, initial_hidden_state, initial_cell_state):

        h = initial_hidden_state
        c = initial_cell_state

        hidden_states = []

        for t in range(len(x)):

            # Current input
            xt = x[t].reshape(-1, 1)

            # Concatenate previous hidden state and input
            combined = np.vstack((h, xt))

            # Four gates
            f = self.sigmoid(self.Wf @ combined + self.bf)

            i = self.sigmoid(self.Wi @ combined + self.bi)

            g = np.tanh(self.Wc @ combined + self.bc)

            o = self.sigmoid(self.Wo @ combined + self.bo)

            # Update cell state
            c = f * c + i * g

            # Update hidden state
            h = o * np.tanh(c)

            hidden_states.append(h.copy())

        return np.array(hidden_states), h, c
