import time
import os
import torch
import random
import torch.nn.functional as F
from torch.autograd import Variable

from vector import make_noise
from dataloader import FlexibleCustomDataloader
import imutil
from logutil import TimeSeries

from gradient_penalty_wgan import calc_gradient_penalty
from torchvision import utils
from tensorboard_logger import Logger

log = TimeSeries('Training WGAN-GP')

def to_np(x):
    return x.data.cpu().numpy()

def real_images(image_channels, images, number_of_images):
    if (image_channels == 3):
        return to_np(images.view(-1, image_channels, 32, 32)[:number_of_images])
    else:
        return to_np(images.view(-1, 32, 32)[:number_of_images])

def generate_img(netGen, image_channels, z, number_of_images):
    samples = netGen(z).data.cpu().numpy()[:number_of_images]
    generated_images = []
    for sample in samples:
        if image_channels == 3:
            generated_images.append(sample.reshape(image_channels, 32, 32))
        else:
            generated_images.append(sample.reshape(32, 32))
    return generated_images

def train_wgan_gp(networks, optimizers, dataloader, number_of_images, logger=None, epoch=None, total_epoch=None,**options):
    for net in networks.values():
        net.train()    

    # t_begin = time.time()

    netD = networks['discriminator']
    netG = networks['generator']
    optimizerD = optimizers['discriminator']
    optimizerG = optimizers['generator']
    result_dir = options['result_dir']
    batch_size = options['batch_size']
    latent_size = options['latent_size']
    discriminator_iter = options['discriminator_iter']
    
    one = torch.tensor(1, dtype=torch.float).cuda()
    mone = one * -1

    for p in netD.parameters():
        p.requires_grad = True

    d_loss_real = 0
    d_loss_fake = 0
    Wasserstein_D = 0
    # Train Critic: max E[critic(real)] - E[critic(fake)]
    # equivalent to minimizing the negative of that
    for d_iter in range(discriminator_iter):
        print(f"Train discriminator part({d_iter+1}/{discriminator_iter})")
        for i, (images, _) in enumerate(dataloader):
            images = Variable(images, requires_grad=False).cuda()
            cur_batch_size = images.shape[0]

            netD.zero_grad()

            z = torch.rand((cur_batch_size, 100, 1, 1))
            z = Variable(z).cuda()

            # Train discriminator
            # WGAN - Training discriminator more iterations than generator
            # Train with real images
            d_loss_real = netD(images)
            d_loss_real = d_loss_real.mean()
            log.collect('Discriminator Real', d_loss_real)
            d_loss_real.backward(mone)
            
            # Train with fake images
            fake_images = netG(z)
            d_loss_fake = netD(fake_images)
            d_loss_fake = d_loss_fake.mean()
            log.collect('Discriminator Fake', d_loss_fake)
            d_loss_fake.backward(one)

            # Train with gradient penalty
            gradient_penalty = calc_gradient_penalty(cur_batch_size, netD, images.data, fake_images.data)
            log.collect('Gradient Penalty', gradient_penalty)
            gradient_penalty.backward()

            d_loss = d_loss_fake - d_loss_real + gradient_penalty
            Wasserstein_D = d_loss_real - d_loss_fake
            optimizerD.step()
            # print(f'  Discriminator iteration: {d_iter}/{discriminator_iter}, loss_fake: {d_loss_fake}, loss_real: {d_loss_real}')
            log.print_every()

    # Generator update
    print("Train generator part")
    for i, (images, _) in enumerate(dataloader):
        
        images = Variable(images, requires_grad=False).cuda()
        cur_batch_size = images.shape[0]

        for p in netD.parameters():
            p.requires_grad = False    
        
        netG.zero_grad()
        # train generator
        # compute loss with fake images
        z = torch.randn((cur_batch_size, 100, 1, 1))
        z = Variable(z).cuda()
        fake_images = netG(z)
        g_loss = netD(fake_images)
        g_loss = g_loss.mean()
        g_loss.backward(mone)
        log.collect('Generator Sampled', g_loss)
        g_cost = -g_loss
        optimizerG.step()
        # print(f'Generator iteration: {epoch}/{total_epoch}, g_loss: {g_loss}')

        log.print_every()
    
    # Test generator train result
    if epoch % 10 == 0:
        if not os.path.exists('training_result_images/'):
            os.makedirs('training_result_images/')
            
        # Denormalize images and save them in grid 8x8
        z = torch.randn(800, 100, 1, 1)
        z = Variable(z).cuda()

        samples = netG(z)
        samples = samples.mul(0.5).add(0.5)
        samples = samples.data.cpu()[:64]
        grid = utils.make_grid(samples)
        utils.save_image(grid, 'training_result_images/img_generatori_iter_{}.png'.format(str(epoch).zfill(3)))

        # ============ TensorBoard logging ============#
        # (1) Log the scalar values
        info = {
            'Wasserstein distance': Wasserstein_D.data,
            'Loss D': d_loss.data,
            'Loss G': g_cost.data,
            'Loss D Real': d_loss_real.data,
            'Loss D Fake': d_loss_fake.data
        }

        for tag, value in info.items():
            logger.scalar_summary(tag, value, epoch + 1)

        # (3) Log the images
        info = {
            'real_images': real_images(images.size(1), images, number_of_images),
            'generated_images': generate_img(netG, images.size(1), z, number_of_images)
        }

        for tag, log_images in info.items():
            logger.image_summary(tag, log_images, epoch + 1)
    

    return True

def train_wgan_gp_pre(networks, optimizers, dataloader, number_of_images, logger=None, epoch=None, total_epoch=None,**options):
    for net in networks.values():
        net.train()    

    # t_begin = time.time()

    netD = networks['discriminator']
    netG = networks['generator']
    optimizerD = optimizers['discriminator']
    optimizerG = optimizers['generator']
    result_dir = options['result_dir']
    batch_size = options['batch_size']
    latent_size = options['latent_size']
    discriminator_iter = options['discriminator_iter']
    
    one = torch.tensor(1, dtype=torch.float).cuda()
    mone = one * -1

    for i, (images, _) in enumerate(dataloader):
        images = Variable(images, requires_grad=False).cuda()
        cur_batch_size = images.shape[0]

        for p in netD.parameters():
            p.requires_grad = True

        d_loss_real = 0
        d_loss_fake = 0
        Wasserstein_D = 0
        # Train Critic: max E[critic(real)] - E[critic(fake)]
        # equivalent to minimizing the negative of that
        for d_iter in range(discriminator_iter):
            netD.zero_grad()

            z = torch.rand((cur_batch_size, 100, 1, 1))
            z = Variable(z).cuda()

            # Train discriminator
            # WGAN - Training discriminator more iterations than generator
            # Train with real images
            d_loss_real = netD(images)
            d_loss_real = d_loss_real.mean()
            log.collect('Discriminator Real', d_loss_real)
            d_loss_real.backward(mone)
            
            # Train with fake images
            fake_images = netG(z)
            d_loss_fake = netD(fake_images)
            d_loss_fake = d_loss_fake.mean()
            log.collect('Discriminator Fake', d_loss_fake)
            d_loss_fake.backward(one)

            # Train with gradient penalty
            gradient_penalty = calc_gradient_penalty(cur_batch_size, netD, images.data, fake_images.data)
            log.collect('Gradient Penalty', gradient_penalty)
            gradient_penalty.backward()

            d_loss = d_loss_fake - d_loss_real + gradient_penalty
            Wasserstein_D = d_loss_real - d_loss_fake
            optimizerD.step()
            # print(f'  Discriminator iteration: {d_iter}/{discriminator_iter}, loss_fake: {d_loss_fake}, loss_real: {d_loss_real}')
            

        # Generator update
        for p in netD.parameters():
            p.requires_grad = False    
        
        netG.zero_grad()
        # train generator
        # compute loss with fake images
        z = torch.randn((cur_batch_size, 100, 1, 1))
        z = Variable(z).cuda()
        fake_images = netG(z)
        g_loss = netD(fake_images)
        g_loss = g_loss.mean()
        g_loss.backward(mone)
        log.collect('Generator Sampled', g_loss)
        g_cost = -g_loss
        optimizerG.step()
        # print(f'Generator iteration: {epoch}/{total_epoch}, g_loss: {g_loss}')

        log.print_every()

        if epoch % 100 == 0:
            if not os.path.exists('training_result_images/'):
                os.makedirs('training_result_images/')
                
            # Denormalize images and save them in grid 8x8
            z = torch.randn(800, 100, 1, 1)
            z = Variable(z).cuda()

            samples = netG(z)
            samples = samples.mul(0.5).add(0.5)
            samples = samples.data.cpu()[:64]
            grid = utils.make_grid(samples)
            utils.save_image(grid, 'training_result_images/img_generatori_iter_{}.png'.format(str(epoch).zfill(3)))

            # Testing
            # time = time.time() - t_begin
            # #print("Real Inception score: {}".format(inception_score))
            # print("Generator iter: {}".format(g_iter))
            # print("Time {}".format(time))

            # ============ TensorBoard logging ============#
            # (1) Log the scalar values
            info = {
                'Wasserstein distance': Wasserstein_D.data,
                'Loss D': d_loss.data,
                'Loss G': g_cost.data,
                'Loss D Real': d_loss_real.data,
                'Loss D Fake': d_loss_fake.data
            }

            for tag, value in info.items():
                logger.scalar_summary(tag, value, epoch + 1)

            # (3) Log the images
            info = {
                'real_images': real_images(images.size(1), images, number_of_images),
                'generated_images': generate_img(netG, images.size(1), z, number_of_images)
            }

            for tag, log_images in info.items():
                logger.image_summary(tag, log_images, epoch + 1)
    
    return True

def train_wresnet_classifier(networks, optimizers, dataloader, epoch=None, **options):
    networks['classifier'].train()
    networks['generator'].eval()
    print(f"Generator train mode: {networks['generator'].training} / Classifier train mode: {networks['classifier'].training}")

    netG = networks['generator']
    optimizersG = optimizers['generator']
    netC = networks['classifier']
    optimizersC = optimizers['classifier']
    result_dir = options['result_dir']
    batch_size = options['batch_size']

    for i, (images, _) in enumerate(dataloader):
        images = Variable(images, requires_grad=False).cuda()
        cur_batch_size = images.shape[0]

        z = torch.rand((cur_batch_size, 100, 1, 1))
        z = Variable(z).cuda()

        if not os.path.exists('check_generator_images/'):
            os.makedirs('check_generator_images/')

        samples = netG(z)
        samples = samples.mul(0.5).add(0.5)
        samples = samples.data.cpu()[:64]
        grid = utils.make_grid(samples)
        utils.save_image(grid, 'check_generator_images/img_generatori_iter_{}.png'.format(str(epoch).zfill(3)))
        print("asdsad")
        exit()


    return True

# def train_pre_wgan_gp(networks, optimizers, dataloader, epoch=None, **options):
#     for net in networks.values():
#         net.train()    

#     netCritic = networks['critic']
#     netG = networks['generator']
#     optimizerCritic = optimizers['critic']
#     optimizerG = optimizers['generator']
#     result_dir = options['result_dir']
#     batch_size = options['batch_size']
#     latent_size = options['latent_size']
#     critic_iter = options['critic_iter']

#     for i, (images, _) in enumerate(dataloader):
#         images = Variable(images, requires_grad=False)
#         cur_batch_size = images.shape[0]

#         # Train Critic: max E[critic(real)] - E[critic(fake)]
#         # equivalent to minimizing the negative of that
#         for _ in range(critic_iter):
#             noise = torch.randn(cur_batch_size, latent_size, 1, 1).to("cuda")
#             fake = netG(noise)
#             critic_real = netCritic(images).reshape(-1)
#             critic_fake = netCritic(fake).reshape(-1)
#             gp = calc_gradient_penalty(netCritic, images.data, fake.data )

#     return True
