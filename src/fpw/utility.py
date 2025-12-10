import abc
from docstring_inheritance import GoogleDocstringInheritanceInitMeta
from typing import Union, Tuple, Generator
from types import MethodType

import warnings


import numpy as np
import scipy as sp
import torch

import os
import sys
import pickle

# import emcee

def _as_generator(r: Union[np.float64, Generator]):
    if isinstance(r, Generator):
        return r
    elif isinstance(r, float):

        def rgen():
            while True:
                yield r

        return rgen()
    else:
        raise RuntimeError(f"Type of relaxation/regularization ({r}) not supported")

class _Meta(abc.ABC, GoogleDocstringInheritanceInitMeta):
    pass


