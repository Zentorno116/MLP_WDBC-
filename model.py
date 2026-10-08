import numpy as np

def sigmoid(z):
    z = np.clip(z, -500, 500)
    return 1.0 / (1.0 + np.exp(-z))

def d_sigmoid(a):
    return a * (1.0 - a)

def relu(z):
    return np.maximum(0.0, z)

def d_relu(a):
    return (a > 0.0).astype(float)

def tanh(z):
    return np.tanh(z)

def d_tanh(a):
    return 1.0 - a ** 2

def softmax(z):
    shift_z = z - np.max(z, axis=1, keepdims=True)
    exp_z = np.exp(shift_z)
    return exp_z / np.sum(exp_z, axis=1, keepdims=True)

class MLP:
    def __init__(self, layer_sizes, activation="sigmoid"):
        self.layer_sizes = layer_sizes
        self.activation_name = activation
        self.weights = []
        self.biases = []
        self.mean = None
        self.std = None
        self._init_weights()

    def _init_weights(self):
        for i in range(len(self.layer_sizes) - 1):
            din = self.layer_sizes[i]
            dout = self.layer_sizes[i + 1]
            limit = np.sqrt(6.0 / din)
            w = np.random.uniform(-limit, limit, (din, dout))
            b = np.zeros((1, dout))
            self.weights.append(w)
            self.biases.append(b)

    def forward(self, x):
        a = x
        self.zs = []
        self.acts = [x]
        for i in range(len(self.weights) - 1):
            z = a @ self.weights[i] + self.biases[i]
            self.zs.append(z)
            if self.activation_name == "relu":
                a = relu(z)
            elif self.activation_name == "tanh":
                a = tanh(z)
            else:
                a = sigmoid(z)
            self.acts.append(a)
        z_out = a @ self.weights[-1] + self.biases[-1]
        self.zs.append(z_out)
        a_out = softmax(z_out)
        self.acts.append(a_out)
        return a_out

    def backward(self, y_true, y_pred, lr):
        m = y_true.shape[0]
        dz = y_pred - y_true
        for i in reversed(range(len(self.weights))):
            a_prev = self.acts[i]
            dw = (a_prev.T @ dz) / m
            db = np.sum(dz, axis=0, keepdims=True) / m
            if i > 0:
                da = dz @ self.weights[i].T
                if self.activation_name == "relu":
                    dz = da * d_relu(self.acts[i])
                elif self.activation_name == "tanh":
                    dz = da * d_tanh(self.acts[i])
                else:
                    dz = da * d_sigmoid(self.acts[i])
            self.weights[i] -= lr * dw
            self.biases[i] -= lr * db

    def compute_loss(self, y_true, y_pred):
        eps = 1e-15
        clipped = np.clip(y_pred, eps, 1.0 - eps)
        return -np.mean(np.sum(y_true * np.log(clipped), axis=1))

    def compute_acc(self, y_true, y_pred):
        return np.mean(np.argmax(y_pred, axis=1) == np.argmax(y_true, axis=1))

    def save(self, path):
        data = {
            "layer_sizes": self.layer_sizes,
            "activation": self.activation_name,
            "mean": self.mean,
            "std": self.std,
            "weights": self.weights,
            "biases": self.biases
        }
        np.save(path, data)

    @classmethod
    def load(cls, path):
        if not path.endswith(".npy"):
            path += ".npy"
        data = np.load(path, allow_pickle=True).item()
        model = cls(data["layer_sizes"], activation=data["activation"])
        model.mean = data["mean"]
        model.std = data["std"]
        model.weights = data["weights"]
        model.biases = data["biases"]
        return model
