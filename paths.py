import os
import subprocess
import sys

from itertools import combinations, count, permutations, product
from more_itertools.more import distinct_permutations, distinct_combinations
from multiset import Multiset as multiset
from numpy import array, floor, gcd, reshape