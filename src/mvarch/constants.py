"""
Global Constants:

DEBUG, PROGRESS_ITERATIONS, LEARNING_RATE, INITIAL_DECAY, MAX_CLAMP, MIN_CLAMP, and EPS
"""

DEBUG = False

# Maximum number of optimization iterations to
# perfrom without reporting progress
PROGRESS_ITERATIONS = 20

# Learning rate to use in PyTorch.
LEARNING_RATE = 1.0

# For first-order AR models, choose initial values with this decay
# rate (AR term).
INITIAL_DECAY = 0.10

# Clamp output of ARMA models at these values
# because coefficients can be unstable during optimization
MAX_CLAMP = 1e10
MIN_CLAMP = -MAX_CLAMP

# Relative EPS (should be slightly above the machien epsilon)
EPS = 1e-6
