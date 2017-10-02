from ipyparallel import Client
import numpy as np

#ipcluster start -n 4

rc = Client()
dview = rc[:]

@dview.parallel(block=True)
def pmul(a, b):
    return a, b

a = np.random.random((4, 8))

res = pmul(a, a)