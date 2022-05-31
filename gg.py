from numpy import exp, array, random, dot
from typing import Final

from sys import setrecursionlimit
setrecursionlimit(7000)


def train(count, training_set_inputs, synaptic_weights, training_set_outputs):
    if count < 0:
        return synaptic_weights
    OUTPUT: Final[int] = 1 / (1 + exp(-(dot(training_set_inputs, synaptic_weights))))
    NEW_WEIGHTS: Final[array] = synaptic_weights + dot(training_set_inputs.T,
                                                       (training_set_outputs - OUTPUT) * OUTPUT * (1 - OUTPUT))
    return train(count - 1, training_set_inputs, NEW_WEIGHTS, training_set_outputs)


if __name__ == "__main__":
    TRAINING_SET_INPUTS: Final[array] = array([[0, 0, 1],
                                               [1, 1, 1],
                                               [1, 0, 1],
                                               [0, 1, 1]])
    TRAINING_SET_OUTPUTS: Final[array] = array([[0, 1, 1, 0]]).T
    random.seed(1)
    INITIAL_SYNAPTIC_WEIGHTS: Final[int] = 2 * random.random((3, 1)) - 1
    SYNAPTIC_WEIGHTS: Final[array] = train(6000, TRAINING_SET_INPUTS, INITIAL_SYNAPTIC_WEIGHTS, TRAINING_SET_OUTPUTS)
    print(1 / (1 + exp(-(dot(array([1, 0, 0]), SYNAPTIC_WEIGHTS)))))

