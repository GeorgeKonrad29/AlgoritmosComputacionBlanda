
import functools
import time
import math
import numpy as np
import matplotlib.pyplot as plt
import tsplib95

from deap import base, creator, tools , algorithms
from typing import List
tsplib95.load("berlin52.tsp")

@dataclass
class GAConfig:
  pop_size: int = 300
  ngen: int = 400
  cxpb: float = 0.9
  mutpb: float = 0.2
  tournsize: int = 3
  elite: int = 1
  seed: int = 123