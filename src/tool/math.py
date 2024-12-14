import numpy as np



def cnk(n,k):
    if k < 0 or k > n:
        print("input error!")
        return
    ret = np.float64(1.0)
    index = 0
    while(index < k):
        ret /= (k - index)
        ret *= (n - index)
        index += 1
    return ret


def binomial_distribution(target_quat,sample_size,probability):
    return cnk(sample_size,target_quat) * (probability ** target_quat)*((1 - probability) ** (sample_size - target_quat))