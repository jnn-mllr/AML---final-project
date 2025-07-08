import numpy as np

class Optimizer:
    """
    Base optimizer class.
    """

    def __init__(self, learning_rate=0.01):
        self.learning_rate = learning_rate

    def update(self, params, grads):
        """
        Update parameters based on gradients.
        This method should be overridden by subclasses.

        Parameters:
        - params: list or dict of parameters (e.g., weights)
        - grads: list or dict of gradients (same structure as params)
        """
        raise NotImplementedError("`update` must be implemented by the subclass.")


class GDOptimizer(Optimizer):
    """
    Gradient descent optimizer with optional learning rate schedule.

    Parameters:
    - learning_rate (float): Initial learning rate
    - schedule_fn (callable): Function(step) → new_learning_rate
    """

    def __init__(self, learning_rate=0.01, schedule_fn=None):
        super().__init__(learning_rate)
        self.schedule_fn = schedule_fn
        self.step = 0

    def update(self, params, grads):
        if self.schedule_fn is not None:
            self.step += 1
            self.learning_rate = self.schedule_fn(self.step)

        for key in params:
            np.subtract(params[key], self.learning_rate * grads[key], out=params[key])


class AdamOptimizer(Optimizer):
    """
    Adam optimizer.

    Parameters:
    - learning_rate (float): Learning rate
    - beta1 (float): Exponential decay rate for the first moment estimates
    - beta2 (float): Exponential decay rate for the second moment estimates
    - epsilon (float): Small constant for numerical stability
    """

    def __init__(self, learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8):
        super().__init__(learning_rate)
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.m = None  # First moment vector
        self.v = None  # Second moment vector
        self.t = 0     # Timestep

    def update(self, params, grads):
        """
        Update parameters using the Adam optimization algorithm.
        """
        self.t += 1

        if self.m is None:
            self.m = {key: np.zeros_like(value) for key, value in params.items()}
            self.v = {key: np.zeros_like(value) for key, value in params.items()}

        for key in params:
            # Update biased first moment estimate
            self.m[key] = self.beta1 * self.m[key] + (1 - self.beta1) * grads[key]
            
            # Update biased second moment estimate
            self.v[key] = self.beta2 * self.v[key] + (1 - self.beta2) * np.square(grads[key])
            
            # Compute bias-corrected first moment estimate
            m_hat = self.m[key] / (1 - self.beta1 ** self.t)
            
            # Compute bias-corrected second moment estimate
            v_hat = self.v[key] / (1 - self.beta2 ** self.t)
            
            # Update parameters
            update_value = self.learning_rate * m_hat / (np.sqrt(v_hat) + self.epsilon)
            params[key] -= update_value

class SGDOptimizer(Optimizer):
    """
    Stochastic Gradient Descent (SGD) optimizer with optional momentum.

    Parameters:
    - learning_rate (float): The learning rate.
    - momentum (float): The momentum factor. Must be in [0, 1].
                      Defaults to 0, which is standard SGD.
    """

    def __init__(self, learning_rate=0.01, momentum=0.0):
        super().__init__(learning_rate)
        if not (0.0 <= momentum <= 1.0):
            raise ValueError("Momentum must be between 0.0 and 1.0.")
        self.momentum = momentum
        self.velocity = None

    def update(self, params, grads):
        """
        Update parameters using SGD with momentum.
        """
        if self.momentum > 0 and self.velocity is None:
            # Initialize velocity with the same shape as parameters, filled with zeros
            self.velocity = {key: np.zeros_like(value) for key, value in params.items()}

        for key in params:
            if self.momentum > 0:
                # Update velocity: v = momentum * v - learning_rate * grad
                self.velocity[key] = self.momentum * self.velocity[key] - self.learning_rate * grads[key]
                # Update parameters: p = p + v
                params[key] += self.velocity[key]
            else:
                # Standard SGD update
                params[key] -= self.learning_rate * grads[key]

