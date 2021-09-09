import torch
from torch import autograd

def calc_gradient_penalty(batch_size, netD, real_data, fake_data, penalty_lambda=10.0):
    
    alpha = torch.FloatTensor(batch_size,1,1,1).uniform_(0,1)
    alpha = alpha.expand(batch_size, real_data.size(1), real_data.size(2), real_data.size(3))
    alpha = alpha.cuda()

    # Traditional WGAN-GP
    interpolates = alpha * real_data + (1 - alpha) * fake_data
    #인터폴레이트 사이즈 torch.Size([64, 3, 32, 32])
    
    # Possibly more reasonable
    #interpolates = torch.cat([real_data, fake_data])
    interpolates = interpolates.cuda()
    interpolates = autograd.Variable(interpolates, requires_grad=True)

    disc_interpolates = netD(interpolates)

    ones = torch.ones(disc_interpolates.size()).cuda()
    gradients = autograd.grad(
            outputs=disc_interpolates, 
            inputs=interpolates, 
            grad_outputs=ones, 
            create_graph=True, 
            retain_graph=True)[0]

    penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * penalty_lambda
    return penalty
