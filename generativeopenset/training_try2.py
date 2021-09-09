import time
import os
import torch
import random
import torch.nn.functional as F
from torch.autograd import Variable
from torch.random import initial_seed

from vector import make_noise
from dataloader import FlexibleCustomDataloader
import imutil
from logutil import TimeSeries
from pprint import pprint

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


def get_infinite_batches(data_loader):
    while True:
        for i, (images, _) in enumerate(data_loader):
            yield images

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

    data = get_infinite_batches(dataloader)
    
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
        netD.zero_grad()

        images = data.__next__()
        cur_batch_size = images.shape[0]    
        if (images.size()[0] != batch_size):
            continue

        images = Variable(images, requires_grad=False).cuda()

        z = torch.randn((cur_batch_size, 100, 1, 1))
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

def custom_one_hot_embedding(labels, num_classes):
    """Embedding labels to one-hot form.

    Args:
      labels: (LongTensor) class labels, sized [N,].
      num_classes: (int) number of classes.

    Returns:
      (tensor) encoded labels, sized [N, #classes].
    """
    y = torch.eye(num_classes)
    customOneHot = y[labels]
    customOneHot[customOneHot==0] = -1
    return customOneHot

def transform_img(concated_Imgs, trans_Idx, transNum, cur_batch_size):
    Rot_Idx = torch.ceil(torch.true_divide(trans_Idx, 2)) # 0.=0 / 1.=90 / 2.=180/ 3.=270
    flip_Idx = torch.remainder(trans_Idx, 2) # 0=no flip / 1=flip

    totalTransNum = transNum*cur_batch_size
    transedImgs = torch.zeros(totalTransNum , concated_Imgs.size(1), concated_Imgs.size(2), concated_Imgs.size(3))
    
    for img_idx, data in enumerate(concated_Imgs):
        initialImg = data
        # flipConditioned_Idx = torch.nonzero(flip_Idx[img_idx]==1, as_tuple=True)
        for idx in range(4):
            rotConditioned_Idx = torch.nonzero(Rot_Idx[img_idx]==idx+1, as_tuple=True)

            for _, rotCon_idx in enumerate(rotConditioned_Idx[0]):
                # rotation
                transedImgs[img_idx*transNum+rotCon_idx, :,:,:] = torch.rot90(initialImg, idx, [1, 2])
                # flip (left/right)
                if flip_Idx[img_idx][rotCon_idx] == 1:
                    transedImgs[img_idx*transNum+rotCon_idx, :,:,:] = torch.flip(transedImgs[img_idx*transNum+rotCon_idx, :,:,:], [1,])

    return transedImgs

def one_select_transform_img(concated_Imgs, trans_Idx, transNum):
    Rot_Idx = torch.ceil(torch.true_divide(trans_Idx, 2)) # 0.=0 / 1.=90 / 2.=180/ 3.=270
    flip_Idx = torch.remainder(trans_Idx, 2) # 0=no flip / 1=flip

    transedImgs = torch.zeros((transNum, concated_Imgs.size(0), concated_Imgs.size(1), concated_Imgs.size(2)))    
    initialImg = concated_Imgs

    for idx, _ in enumerate(trans_Idx[0]):
        transedImgs[idx,:,:,:] = torch.rot90(initialImg, int(Rot_Idx[0][idx]), [1, 2])

        if flip_Idx[0][idx] == 1:
            transedImgs[idx,:,:,:] = torch.flip(transedImgs[idx,:,:,:], [1, ])

    return transedImgs

def train_wresnet_classifier(networks, optimizers, dataloader, epoch=None, **options):
    networks['classifier'].train()
    networks['SSclassifier'].train()
    networks['generator'].eval()
    print(f"Generator train mode: {networks['generator'].training} / Classifier train mode: {networks['classifier'].training}")

    netG = networks['generator']
    # optimizersG = optimizers['generator']

    netC = networks['classifier']
    optimizerC = optimizers['classifier']

    netSSC = networks['SSclassifier']
    optimizerSSC = optimizers['SSclassifier']

    result_dir = options['result_dir']
    batch_size = options['batch_size']

    for i, (images, class_labels) in enumerate(dataloader):
        images = Variable(images, requires_grad=False).cuda()
        labels = Variable(class_labels, requires_grad=False).cuda()
        cur_batch_size = images.shape[0]

        # Create concated image(gen image + basic image), ch6
        z = torch.randn((cur_batch_size, 100, 1, 1))
        z = Variable(z).cuda()

        samples = netG(z)
        samples = samples.mul(0.5).add(0.5)
        concatedImgs = torch.zeros((64, 6, 32, 32)).cuda()
        concatedImgs[:, 0:6:2, :, :] = images
        concatedImgs[:, 1:6:2, :, :] = samples
        
        # Basic image(ch 6) classifier
        netC.zero_grad()
        classifier_logits = netC(concatedImgs)
        errC =  F.softplus(classifier_logits * -labels).mean()
        errC.backward()
        log.collect('Basic Classifier Loss', errC)

        optimizerC.step()

        ## Semi-supervised train
        # transform images
        # try 1-아직. all image transformed
        # transIdx = torch.randint(1, 9, (cur_batch_size, options['transNum']))
        # transformedImgs = transform_img(concatedImgs, transIdx, options['transNum'], cur_batch_size).cuda()

        # SSlabel = transIdx.view([-1])-1 # 1~8 -> 0~7
        # SSlabel = custom_one_hot_embedding(SSlabel, 8).cuda()

        # try 2. 1장 씩 transformed
        meanSSC = torch.zeros(concatedImgs.size(0))
        for err_idx, selectImage in enumerate(concatedImgs):
            netSSC.zero_grad()
            transIdx = torch.randint(1, 9, (1, options['transNum']))
            transformedImgs = one_select_transform_img(selectImage, transIdx,  options['transNum']).cuda()
            SSlabel = custom_one_hot_embedding(transIdx-1, 8).cuda()

            SSclassifier_logits = netSSC(transformedImgs)
            SSclassifier_logits = Variable(SSclassifier_logits.data, requires_grad=True)
            SSerrC = F.softplus(SSclassifier_logits * -SSlabel).mean()
            meanSSC[err_idx] = SSerrC
            SSerrC.backward(retain_graph=True)
            log.collect('Semi-supervised Classifier Loss', SSerrC)
            optimizerSSC.step()
        meanSSC = torch.mean(meanSSC)

        netC.zero_grad()
        total_errC = options['alpha1']*errC + options['alpha2']*meanSSC
        total_errC = Variable(total_errC.data, requires_grad=True)
        total_errC.backward()

        # generate image check
        # if not os.path.exists('check_generator_images/'):
        #    os.makedirs('check_generator_images/')
        # samples = samples.mul(0.5).add(0.5)
        # samples = samples.data.cpu()[:64]
        # grid = utils.make_grid(samples)
        # utils.save_image(grid, 'check_generator_images/img_generatori_iter_{}.png'.format(str(epoch).zfill(3)))


        # Keep track of accuracy on positive-labeled examples for monitoring
        z = torch.randn((cur_batch_size, 100, 1, 1))
        z = Variable(z).cuda()

        samples = netG(z)
        samples = samples.mul(0.5).add(0.5)
        concatedImgs = torch.zeros((64, 6, 32, 32)).cuda()
        concatedImgs[:, 0:6:2, :, :] = images
        concatedImgs[:, 1:6:2, :, :] = samples

        log.collect_prediction('Classifier Accuracy', netC(concatedImgs), labels)
        log.print_every()
        
    return True
