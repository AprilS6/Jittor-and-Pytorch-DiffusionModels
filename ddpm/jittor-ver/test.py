import jittor as jt
import numpy as np
from prepared_data import load_data
import math

class Test:
    def __init__(self, n_channels):
        self.n_channels = n_channels
    
    def test(self, t: jt.Var):
        half_dim = self.n_channels // 8 # dim为Positional Embedding的维度
        emb = jt.exp(jt.arange(half_dim) * -(math.log(10000) / (half_dim - 1)))
        emb = t[:, None] * emb[None, :]
        emb = jt.cat([emb.sin(), emb.cos()], dim=1)
        return emb

def main():
    test = Test(4)
    t = jt.int32(10)
    emb = test.test(t)
    print(emb)

if __name__ == '__main__':
    main()