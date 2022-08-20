# Standard Python
import logging
from typing import Callable

# Common packages
import torch


def optimize(
    optim: torch.optim.Optimizer, closure: Callable[[], float], label: str = ""
) -> None:
    """
    This is a wrapper around optim.step() that higher level monitoring.
    Arguments:
       optim: torch.optim.Optimizer - optimizer to use.
       closure: a "closure" that evaluates the objective function
                and the Pytorch optimizer closure() conventions
                which include 1) zeroing the gradient; 2) evaluating
                the objective; 3) back-propagating the derivative
                informaiton; 4) returning the objective value.

    Returns: Nothing

    """
    best_loss = float("inf")
    done = False
    logging.info(f"Starting {label}" + (" " if label else "") + "optimization")
    while not done:
        optim.step(closure)
        current_loss = closure()
        logging.info(f"\tCurrent loss: {current_loss:.4f}")
        if float(current_loss) < best_loss:
            best_loss = current_loss
        else:
            logging.info("Finished")
            done = True
