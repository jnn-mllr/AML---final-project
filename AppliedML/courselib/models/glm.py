import numpy as np
from .base import TrainableModel

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

class LogisticRegression(TrainableModel):
    """
    Binary logistic regression model with optional regularization.

    Parameters:
        - w: Initial weights (array-like)
        - b: Initial bias (scalar)
        - optimizer: Optimizer object (e.g., GDOptimizer)
        - penalty: One of {"none", "ridge", "lasso"}
        - lam: Regularization strength (default: 0.0)
    """
    
    def __init__(self, w, b, optimizer, penalty="none", lam=0.0):
        super().__init__(optimizer)
        self.w = np.array(w, dtype=float)
        self.b = np.array(b, dtype=float)
        self.penalty = penalty
        self.lam = float(lam)
        

    def loss_grad(self, X, y):
        residual = self.decision_function(X) - y
        grad_w = X.T @ residual / len(X)
        grad_b = np.mean(residual)

        # Add regularization if specified
        if self.penalty == "ridge":
            grad_w += self.lam * self.w
        elif self.penalty == "lasso":
            grad_w += self.lam * np.sign(self.w)

        return {"w": grad_w, "b": grad_b}
    
    def decision_function(self, X):
        return sigmoid(X @ self.w + self.b)
    
    def _get_params(self):
        """
        Return model parameters as a dict for the optimizer.
        """
        return {"w": self.w, "b": self.b}

    def __call__(self, X):
        return np.where(self.decision_function(X) >= 0.5, 1, 0)
    
    # new functions for training process to vis loss during train/test 
    def loss(self, X, y):
        """
        Computes the loss for the given data.
        """
        # prevent log(0)
        epsilon = 1e-15
        y_hat = np.clip(self.decision_function(X), epsilon, 1 - epsilon)
        # Binary cross-entropy loss
        bce_loss = -np.mean(y * np.log(y_hat) + (1 - y) * np.log(1 - y_hat))

        # regularization terms
        reg_term = 0.0
        if self.penalty == "ridge":
            reg_term = 0.5 * self.lam * np.sum(self.w**2) / len(X)
        elif self.penalty == "lasso":
            reg_term = self.lam * np.sum(np.abs(self.w)) / len(X)
        return bce_loss + reg_term