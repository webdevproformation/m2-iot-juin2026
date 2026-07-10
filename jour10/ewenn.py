"""
Modele de classification multiclasse pour le dataset "spirales"
Implementation NumPy pure (equivalent Adam + reseau dense + ReLU)
"""

import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)

# --------------------------------------------------
# 1. Generation du dataset "spirales"
# --------------------------------------------------
N = 100   # nombre de points par classe
D = 2     # dimensionnalite
K = 3     # nombre de classes

X = np.zeros((N*K, D))
y = np.zeros(N*K, dtype='uint8')

for j in range(K):
    ix = range(N*j, N*(j+1))
    r = np.linspace(0.0, 1, N)                              # rayon
    t = np.linspace(j*4, (j+1)*4, N) + np.random.randn(N)*0.2  # angle theta
    X[ix] = np.c_[r*np.sin(t), r*np.cos(t)]
    y[ix] = j

plt.scatter(X[:, 0], X[:, 1], c=y, s=40, cmap=plt.cm.Spectral)
plt.title("Dataset spirales")
plt.show()

# --------------------------------------------------
# 2. Split train / test
# --------------------------------------------------
idx = np.random.permutation(N*K)
X, y = X[idx], y[idx]
split = int(0.8 * len(X))
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

# --------------------------------------------------
# 3. Modele : combinaison de couches lineaires + ReLU (non-lineaire)
# --------------------------------------------------
class SpiralNet:
    def __init__(self, in_dim=2, hidden=64, out_dim=3):
        self.W1 = np.random.randn(in_dim, hidden) * np.sqrt(2.0/in_dim)
        self.b1 = np.zeros(hidden)
        self.W2 = np.random.randn(hidden, hidden) * np.sqrt(2.0/hidden)
        self.b2 = np.zeros(hidden)
        self.W3 = np.random.randn(hidden, out_dim) * np.sqrt(2.0/hidden)
        self.b3 = np.zeros(out_dim)

        self.params = [self.W1, self.b1, self.W2, self.b2, self.W3, self.b3]
        self.m = [np.zeros_like(p) for p in self.params]  # Adam: 1er moment
        self.v = [np.zeros_like(p) for p in self.params]  # Adam: 2e moment
        self.t = 0

    def forward(self, X):
        self.z1 = X @ self.W1 + self.b1
        self.a1 = np.maximum(0, self.z1)          # ReLU
        self.z2 = self.a1 @ self.W2 + self.b2
        self.a2 = np.maximum(0, self.z2)          # ReLU
        self.z3 = self.a2 @ self.W3 + self.b3     # logits (sortie lineaire)
        return self.z3

    def softmax(self, logits):
        ex = np.exp(logits - logits.max(axis=1, keepdims=True))
        return ex / ex.sum(axis=1, keepdims=True)

    # --------------------------------------------------
    # 4. Perte multiclasse : cross-entropy (softmax + NLL)
    # --------------------------------------------------
    def loss_and_grad(self, X, y):
        n = X.shape[0]
        logits = self.forward(X)
        probs = self.softmax(logits)
        correct_logprobs = -np.log(probs[range(n), y] + 1e-9)
        loss = correct_logprobs.mean()

        # Backpropagation manuelle
        dlogits = probs.copy()
        dlogits[range(n), y] -= 1
        dlogits /= n

        dW3 = self.a2.T @ dlogits
        db3 = dlogits.sum(axis=0)

        da2 = dlogits @ self.W3.T
        dz2 = da2 * (self.z2 > 0)
        dW2 = self.a1.T @ dz2
        db2 = dz2.sum(axis=0)

        da1 = dz2 @ self.W2.T
        dz1 = da1 * (self.z1 > 0)
        dW1 = X.T @ dz1
        db1 = dz1.sum(axis=0)

        return loss, [dW1, db1, dW2, db2, dW3, db3]

    # --------------------------------------------------
    # 5. Optimiseur Adam (extension optionnelle demandee)
    # --------------------------------------------------
    def adam_step(self, grads, lr=0.01, beta1=0.9, beta2=0.999, eps=1e-8):
        self.t += 1
        for i, (p, g) in enumerate(zip(self.params, grads)):
            self.m[i] = beta1*self.m[i] + (1-beta1)*g
            self.v[i] = beta2*self.v[i] + (1-beta2)*(g**2)
            mhat = self.m[i] / (1 - beta1**self.t)
            vhat = self.v[i] / (1 - beta2**self.t)
            p -= lr * mhat / (np.sqrt(vhat) + eps)

    def predict(self, X):
        probs = self.softmax(self.forward(X))
        return np.argmax(probs, axis=1)


def accuracy(y_true, y_pred):
    return (y_true == y_pred).mean() * 100


# --------------------------------------------------
# 6. Boucle d'entrainement / test
# --------------------------------------------------
model = SpiralNet(in_dim=2, hidden=64, out_dim=3)
lr = 0.01
epochs = 3000

for epoch in range(epochs):
    loss, grads = model.loss_and_grad(X_train, y_train)
    model.adam_step(grads, lr=lr)

    if epoch % 500 == 0 or epoch == epochs - 1:
        tr_acc = accuracy(y_train, model.predict(X_train))
        te_acc = accuracy(y_test, model.predict(X_test))
        print(f"Epoch {epoch}: loss={loss:.4f}, train_acc={tr_acc:.2f}%, test_acc={te_acc:.2f}%")

print(f"\nPrecision finale sur le test set: {accuracy(y_test, model.predict(X_test)):.2f}%")


# --------------------------------------------------
# 7. Frontieres de decision
# --------------------------------------------------
def plot_decision_boundary(model, X, y):
    x_min, x_max = X[:, 0].min() - 0.1, X[:, 0].max() + 0.1
    y_min, y_max = X[:, 1].min() - 0.1, X[:, 1].max() + 0.1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 300),
                          np.linspace(y_min, y_max, 300))
    grid = np.c_[xx.ravel(), yy.ravel()]
    preds = model.predict(grid).reshape(xx.shape)

    plt.contourf(xx, yy, preds, cmap=plt.cm.Spectral, alpha=0.5)
    plt.scatter(X[:, 0], X[:, 1], c=y, s=30, cmap=plt.cm.Spectral, edgecolors='k')
    plt.title("Frontieres de decision - Spirales")
    plt.show()

plot_decision_boundary(model, X_test, y_test)