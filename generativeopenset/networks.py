import os
import network_definitions
import network_define_wgan
import wgan_gp_network
import torch
from torch import optim
from torch import nn
from imutil import ensure_directory_exists
import os.path

def build_WResNet_basic_network(num_classes, epoch=None, **options):

    networks = {}
    
    # ClassifierClass = network_definitions.VGG_basic
    # networks['classifier'] = ClassifierClass('VGG11', basic_num_classes=num_classes)
    ClassifierClass = network_definitions.Wide_Basic_ResNet
    networks['classifier'] = ClassifierClass(num_classes=num_classes, depth=28, widen_factor=10, dropout_rate=0.3, stride=10)
    # networks['classifier'] = ClassifierClass(num_classes=num_classes, depth=40, widen_factor=4, dropout_rate=0.3, stride=10)

    # load pre trained basic Classifier 
    # pth = get_pth_by_epoch(options['result_dir'], 'classifier', epoch)
    pth = get_basic_pth_by_epoch(options['result_dir'], 'classifier', epoch)

    if pth:
        print("Loading {} from checkpoint {}".format('classifier', pth))
        networks['classifier'].load_state_dict(torch.load(pth))
    else:
        print("Using randomly-initialized weights for {}".format('classifier'))

    return networks

def build_VGG_basic_network(num_classes, epoch=None, **options):

    networks = {}
    
    ClassifierClass = network_definitions.VGG_basic
    networks['classifier'] = ClassifierClass('VGG11', basic_num_classes=num_classes)

    # load pre trained basic Classifier 
    # pth = get_pth_by_epoch(options['result_dir'], 'classifier', epoch)
    pth = get_basic_pth_by_epoch(options['result_dir'], 'classifier', epoch)

    if pth:
        print("Loading {} from checkpoint {}".format('classifier', pth))
        networks['classifier'].load_state_dict(torch.load(pth))
    else:
        print("Using randomly-initialized weights for {}".format('classifier'))

    return networks

def build_VGG_semi_sup_network_multibranch(num_classes, epoch=None, **options):

    networks = {}
    
    GeneratorClass = network_define_wgan.Generator
    networks['generator'] = GeneratorClass(channels=3)

    ClassifierClass = network_definitions.VGG_multibranch
    networks['classifier'] = ClassifierClass('VGG16', basic_num_classes=num_classes, SS_num_classes=8, depth=28, widen_factor=10, dropout_rate=0.3, stride=10)

    # load trained generative parameters, epoch=1000
    checkpoint_path = os.path.join(options['result_dir'], 'checkpoints_import/')
    ensure_directory_exists(checkpoint_path)
    files = os.listdir(checkpoint_path)
    # suffix = 'generator_epoch_{:04d}.pth'.format(options['gen_epoch'])
    suffix = 'generator_epoch_{:06d}.pth'.format(options['gen_epoch'])
    gen_file = os.path.join(checkpoint_path, suffix)
    networks['generator'].load_state_dict(torch.load(gen_file))

    # load pre trained basic Classifier 
    pth = get_pth_by_epoch(options['result_dir'], 'classifier', epoch)

    if pth:
        print("Loading {} from checkpoint {}".format('classifier', pth))
        networks['classifier'].load_state_dict(torch.load(pth))
    else:
        print("Using randomly-initialized weights for {}".format('classifier'))

    return networks

def build_semi_sup_network_multibranch(num_classes, epoch=None, **options):

    networks = {}
    
    GeneratorClass = network_define_wgan.Generator
    networks['generator'] = GeneratorClass(channels=3)

    ClassifierClass = network_definitions.Wide_ResNet_multibranch
    networks['classifier'] = ClassifierClass(basic_num_classes=num_classes, SS_num_classes=8, depth=28, widen_factor=10, dropout_rate=0.3, stride=10)

    # load trained generative parameters, epoch=1000
    checkpoint_path = os.path.join(options['result_dir'], 'checkpoints_import/')
    ensure_directory_exists(checkpoint_path)
    files = os.listdir(checkpoint_path)
    # suffix = 'generator_epoch_{:04d}.pth'.format(options['gen_epoch'])
    suffix = 'generator_epoch_{:06d}.pth'.format(options['gen_epoch'])
    gen_file = os.path.join(checkpoint_path, suffix)
    networks['generator'].load_state_dict(torch.load(gen_file))

    # load pre trained basic Classifier 
    pth = get_pth_by_epoch(options['result_dir'], 'classifier', epoch)

    if pth:
        print("Loading {} from checkpoint {}".format('classifier', pth))
        networks['classifier'].load_state_dict(torch.load(pth))
    else:
        print("Using randomly-initialized weights for {}".format('classifier'))

    return networks


def build_semi_sup_network(num_classes, epoch=None, **options):
    networks = {}
    
    GeneratorClass = network_define_wgan.Generator
    networks['generator'] = GeneratorClass(channels=3)

    ClassifierClass = network_definitions.Wide_ResNet
    networks['classifier'] = ClassifierClass(num_classes=num_classes, depth=28, widen_factor=10, dropout_rate=0.3, stride=10)
    
    SSClassifierClass = network_definitions.Wide_ResNet
    networks['SSclassifier'] = SSClassifierClass(num_classes=8, batch_size=64, depth=28, widen_factor=10, dropout_rate=0.3, stride=10)

    # load trained generative parameters, epoch=1000
    checkpoint_path = os.path.join(options['result_dir'], 'checkpoints_import/')
    ensure_directory_exists(checkpoint_path)
    files = os.listdir(checkpoint_path)
    suffix = 'generator_epoch_{:04d}.pth'.format(options['gen_epoch'])
    gen_file = os.path.join(checkpoint_path, suffix)
    networks['generator'].load_state_dict(torch.load(gen_file))

    # load pre trained basic Classifier 
    pth = get_pth_by_epoch(options['result_dir'], 'classifier', epoch)
    if pth:
        print("Loading {} from checkpoint {}".format('classifier', pth))
        networks['classifier'].load_state_dict(torch.load(pth))
    else:
        print("Using randomly-initialized weights for {}".format('classifier'))

    # load pre trained Semi-supervised Classifier 
    pth = get_pth_by_epoch(options['result_dir'], 'SSclassifier', epoch)
    if pth:
        print("Loading {} from checkpoint {}".format('SSclassifier', pth))
        networks['SSclassifier'].load_state_dict(torch.load(pth))
    else:
        print("Using randomly-initialized weights for {}".format('SSclassifier'))

    return networks

def build_wgan_gp_exact_network(num_classes, epoch=None, **options):
    networks = {}

    GeneratorClass = network_define_wgan.Generator
    networks['generator'] = GeneratorClass(channels=3)

    DiscriminatorClass = network_define_wgan.Discriminator
    networks['discriminator'] = DiscriminatorClass(channels=3)

    # 만일 미리 학습했던 것이 있다면 불러오기
    for net_name in networks:
        pth = get_pth_by_epoch(options['result_dir'], net_name, epoch)
        if pth:
            print("Loading {} from checkpoint {}".format(net_name, pth))
            networks[net_name].load_state_dict(torch.load(pth))
        else:
            print("Using randomly-initialized weights for {}".format(net_name))

    return networks

def build_wgan_gp_network(num_classes, epoch=None, **options):
    networks = {}

    CriticClass = wgan_gp_network.Critic
    networks['critic'] = CriticClass(num_classes=num_classes)

    GeneratorClass = wgan_gp_network.Generator
    networks['generator'] = GeneratorClass(num_classes=num_classes)

    # 만일 미리 학습했던 것이 있다면 불러오기
    for net_name in networks:
        pth = get_pth_by_epoch(options['result_dir'], net_name, epoch)
        if pth:
            print("Loading {} from checkpoint {}".format(net_name, pth))
            networks[net_name].load_state_dict(torch.load(pth))
        else:
            print("Using randomly-initialized weights for {}".format(net_name))
    return networks



def build_networks_semisuper_gen(num_classes, epoch=None, latent_size=10, batch_size=64, **options):
    networks = {}

    # EncoderClass = network_definitions.encoder32
    # networks['encoder'] = EncoderClass(latent_size=latent_size)

    GeneratorClass = network_definitions.generator32
    networks['generator'] = GeneratorClass(latent_size=latent_size)

    DiscrimClass = network_definitions.multiclassDiscriminator32
    networks['discriminator'] = DiscrimClass(num_classes=num_classes, latent_size=latent_size)

    # ClassifierClass = network_definitions.Wide_ResNet
    # networks['classifier'] = ClassifierClass(num_classes=num_classes, depth=28, widen_factor=10, dropout_rate=0.3, stride=10)


    # 만일 미리 학습했던 것이 있다면 불러오기
    for net_name in networks:
        pth = get_pth_by_epoch(options['result_dir'], net_name, epoch)
        if pth:
            print("Loading {} from checkpoint {}".format(net_name, pth))
            networks[net_name].load_state_dict(torch.load(pth))
        else:
            print("Using randomly-initialized weights for {}".format(net_name))
    return networks



def build_networks(num_classes, epoch=None, latent_size=10, batch_size=64, **options):
    networks = {}

    EncoderClass = network_definitions.encoder32
    networks['encoder'] = EncoderClass(latent_size=latent_size)

    GeneratorClass = network_definitions.generator32
    networks['generator'] = GeneratorClass(latent_size=latent_size)

    DiscrimClass = network_definitions.multiclassDiscriminator32
    networks['discriminator'] = DiscrimClass(num_classes=num_classes, latent_size=latent_size)

    ClassifierClass = network_definitions.classifier32
    networks['classifier_k'] = ClassifierClass(num_classes=num_classes, latent_size=latent_size)
    networks['classifier_kplusone'] = ClassifierClass(num_classes=num_classes, latent_size=latent_size)

    # 만일 미리 학습했던 것이 있다면 불러오기
    for net_name in networks:
        pth = get_pth_by_epoch(options['result_dir'], net_name, epoch)
        if pth:
            print("Loading {} from checkpoint {}".format(net_name, pth))
            networks[net_name].load_state_dict(torch.load(pth))
        else:
            print("Using randomly-initialized weights for {}".format(net_name))
    return networks


def get_network_class(name):
    if type(name) is not str or not hasattr(network_definitions, name):
        print("Error: could not construct network '{}'".format(name))
        print("Available networks are:")
        for net_name in dir(network_definitions):
            classobj = getattr(network_definitions, net_name)
            if type(classobj) is type and issubclass(classobj, nn.Module):
                print('\t' + net_name)
    return getattr(network_definitions, name)

def save_classification_networks(networks, epoch, result_dir):
    for name in networks:
        if name == 'generator':
            continue
        weights = networks[name].state_dict()
        filename = '{}/checkpoints/{}_epoch_{:04d}.pth'.format(result_dir, name, epoch)
        ensure_directory_exists(filename)
        torch.save(weights, filename)
        # old_filename = '{}/checkpoints/{}_epoch_{:04d}.pth'.format(result_dir, name, epoch-2)
        # if os.path.isfile(old_filename):
        #     os.remove(old_filename)

def save_basic_classification_networks(networks, epoch, result_dir):
    for name in networks:
        if name == 'generator':
            continue
        weights = networks[name].state_dict()
        filename = '{}/basic_checkpoints/{}_epoch_{:04d}.pth'.format(result_dir, name, epoch)
        ensure_directory_exists(filename)
        torch.save(weights, filename)
        # old_filename = '{}/checkpoints/{}_epoch_{:04d}.pth'.format(result_dir, name, epoch-2)
        # if os.path.isfile(old_filename):
        #     os.remove(old_filename)


def save_networks(networks, epoch, result_dir):
    for name in networks:
        weights = networks[name].state_dict()
        filename = '{}/checkpoints/{}_epoch_{:06d}.pth'.format(result_dir, name, epoch)
        ensure_directory_exists(filename)
        torch.save(weights, filename)

        # Delete the old file to prevent memory dump.
        old_filename = '{}/checkpoints/{}_epoch_{:06d}.pth'.format(result_dir, name, epoch-2)
        if os.path.isfile(old_filename):
            if (epoch-2)%1000 != 0:
                os.remove(old_filename)

def get_optimizers_wgan(networks, lr=.001, beta1=.5, beta2=.999, weight_decay=.0, finetune=False, **options):
    optimizers = {}
    if finetune:
        lr /= 10
        print("Fine-tuning mode activated, dropping learning rate to {}".format(lr))
    for name in networks:
        net = networks[name]
        optimizers[name] = optim.Adam(net.parameters(), lr=lr, betas=(beta1, beta2), weight_decay=weight_decay)
    return optimizers

def get_optimizers_semisuper(networks, lr=.001, beta1=.5, beta2=.999, weight_decay=.0, finetune=False, **options):
    optimizers = {}
    if finetune:
        lr /= 10
        print("Fine-tuning mode activated, dropping learning rate to {}".format(lr))
    for name in networks:
        net = networks[name]
        optimizers[name] = optim.Adam(net.parameters(), lr=lr, betas=(beta1, beta2), weight_decay=weight_decay)
    return optimizers

def get_optimizers_basic(networks, lr=.001, beta1=.5, beta2=.999, weight_decay=.0, finetune=False, **options):
    optimizers = {}
    if finetune:
        lr /= 10
        print("Fine-tuning mode activated, dropping learning rate to {}".format(lr))
    for name in networks:
        net = networks[name]
        optimizers[name] = optim.Adam(net.parameters(), lr=lr, betas=(beta1, beta2), weight_decay=weight_decay)
    return optimizers

def get_lr_scheduler(networks, optimizers, **options):
    schedulers = {}
    for name in networks:
        schedulers[name] = optim.lr_scheduler.CosineAnnealingLR(optimizers[name], T_max=20, eta_min=0)
    return schedulers

def get_optimizers(networks, lr=.001, beta1=.5, beta2=.999, weight_decay=.0, finetune=False, **options):
    optimizers = {}
    if finetune:
        lr /= 10
        print("Fine-tuning mode activated, dropping learning rate to {}".format(lr))
    for name in networks:
        net = networks[name]
        optimizers[name] = optim.Adam(net.parameters(), lr=lr, betas=(beta1, beta2), weight_decay=weight_decay)
    return optimizers


def get_pth_by_epoch(result_dir, name, epoch=None):
    checkpoint_path = os.path.join(result_dir, 'checkpoints/')
    ensure_directory_exists(checkpoint_path)
    files = os.listdir(checkpoint_path)
    suffix = '.pth'
    if epoch is not None:
        suffix = 'epoch_{:04d}.pth'.format(epoch)
    files = [f for f in files if '{}_epoch'.format(name) in f]
    if not files:
        return None
    files = [os.path.join(checkpoint_path, fn) for fn in files]
    files.sort(key=lambda x: os.stat(x).st_mtime)

    # return files[-1]
    return files[epoch-1]

def get_basic_pth_by_epoch(result_dir, name, epoch=None):
    checkpoint_path = os.path.join(result_dir, 'basic_checkpoints/')
    ensure_directory_exists(checkpoint_path)
    files = os.listdir(checkpoint_path)
    suffix = '.pth'
    if epoch is not None:
        suffix = 'epoch_{:04d}.pth'.format(epoch)
    files = [f for f in files if '{}_epoch'.format(name) in f]
    if not files:
        return None
    files = [os.path.join(checkpoint_path, fn) for fn in files]
    files.sort(key=lambda x: os.stat(x).st_mtime)

    return files[-1]
    # return files[epoch-1]