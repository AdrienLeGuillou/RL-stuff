from multiprocessing import Pool
import numpy as np

def f(x, y):
    return np.array([x*x, y*y])

if __name__ == '__main__':
    with Pool(4) as p:
        a = p.starmap(f, np.ndindex((3,4)))
        print(np.array(a))