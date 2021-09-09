# Utilities for noise generation, clamping etc
import torch
import time
import numpy as np
from torch.autograd import Variable


def make_noise(batch_size, latent_size, scale, fixed_seed=None):
    noise_t = torch.FloatTensor(batch_size, latent_size * scale * scale)
    if fixed_seed is not None:
        seed(fixed_seed)
    noise_t.normal_(0, 1)
    noise = Variable(noise_t).cuda()
    result = clamp_to_unit_sphere(noise, scale**2)
    if fixed_seed is not None:
        seed(int(time.time()))
    return result


def seed(val=42):
    np.random.seed(val)
    torch.manual_seed(val)
    torch.cuda.manual_seed(val)


# TODO: Merge this with the more fully-featured make_noise()
def gen_noise(K, latent_size):
    noise = torch.zeros((K, latent_size))
    noise.normal_(0, 1)
    noise = clamp_to_unit_sphere(noise)
    return noise


def clamp_to_unit_sphere(x, components=1):
    # If components=4, then we normalize each quarter of x independently
    # Useful for the latent spaces of fully-convolutional networks
    batch_size, latent_size = x.shape
    #print("z의 shape: {}".format(x.shape))
    # latent_size는 마지막 결과중 하나의 채널x높이x너비.
    # encoder의 components는 마지막 결과 중 하나의 높이x너비.
    latent_subspaces = []
    for i in range(components):
        step = latent_size // components # encoder에서 사용할 때에는 분류할 클래스 수.
        left, right = step * i, step * (i+1)
        subspace = x[:, left:right].clone() # step(right-left)씩 만큼
        norm = torch.norm(subspace, p=2, dim=1) # 2-norm, dim=1 행마다 정규화
        subspace = subspace / norm.expand(1, -1).t()  # + epsilon # norm의 차원을 나눌수 있게 차원을 하나 늘려준다(expand). t()는 매트랩의 x' 처럼 차원을 뒤바꿔주는 함수
        latent_subspaces.append(subspace)
    # Join the normalized pieces back together # 행방향(수평방향, dim=1)으로 값들을 붙여(cat) 출력
    return torch.cat(latent_subspaces, dim=1)

