import json
import os

import libmr
import torch
import numpy as np
from torch.autograd import Variable
import torch.nn.functional as F
from sklearn.metrics import roc_curve, roc_auc_score

from plotting import plot_xy

WEIBULL_TAIL_SIZE = 20

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
    customOneHot = torch.squeeze(customOneHot)
    return customOneHot

def basic_VGG_evaluate_classifier(networks, dataloader, **options):
    for net in networks.values():
        net.eval()

    netC = networks['classifier']    
    fold = options.get('fold', 'evaluation')

    classification_closed_correct = 0
    classification_total = 0

    with torch.no_grad():
        for i, (images, class_labels) in enumerate(dataloader):
            images = images.unsqueeze(1)
            images = Variable(images, requires_grad=False).cuda()
            labels = Variable(class_labels, requires_grad=False).cuda()
            cur_batch_size = images.shape[0]

            net_y = netC(images)
            class_predictoins = F.softmax(net_y, dim=1)

            _, predicted = class_predictoins.max(1)
            classification_closed_correct += sum(predicted.data == labels)
            classification_total += len(labels)
        
    basic_stat = {
        fold: {
            'closed_set_image_class_accuracy': float(classification_closed_correct) / (classification_total),
        }
    }

    return basic_stat

def SS_evaluate_classifier(networks, dataloader, **options):
    for net in networks.values():
        net.eval()

    netC = networks['classifier']
    netG = networks['generator']     

    fold = options.get('fold', 'evaluation')

    classification_closed_correct = 0
    classification_total = 0

    with torch.no_grad():
        for i, (images, class_labels) in enumerate(dataloader):
            images = Variable(images, requires_grad=False).cuda()
            labels = Variable(class_labels, requires_grad=False).cuda()
            cur_batch_size = images.shape[0]

            z = torch.randn((cur_batch_size, 100, 1, 1))
            z = Variable(z).cuda()

            samples = netG(z)
            # samples = samples.mul(0.5).add(0.5)
            concatedImgs = torch.zeros((cur_batch_size, 6, 32, 32)).cuda()
            concatedImgs[:, 0:3:1, :, :] = samples
            concatedImgs[:, 3:6:1, :, :] = images

            net_y, _ = netC(concatedImgs)
            class_predictoins = F.softmax(net_y, dim=1)

            _, predicted = class_predictoins.max(1)
            classification_closed_correct += sum(predicted.data == labels)
            classification_total += len(labels)
        
    basic_stat = {
        fold: {
            'closed_set_image_class_accuracy': float(classification_closed_correct) / (classification_total),
        }
    }

    with torch.no_grad():
        for i, (images, class_labels) in enumerate(dataloader):
            images = Variable(images, requires_grad=False).cuda()
            labels = Variable(class_labels, requires_grad=False).cuda()
            cur_batch_size = images.shape[0]

            z = torch.randn((cur_batch_size, 100, 1, 1))
            z = Variable(z).cuda()

            samples = netG(z)
            # samples = samples.mul(0.5).add(0.5)
            # images = images.mul(0.5).add(0.5)
            concatedImgs = torch.zeros((cur_batch_size, 6, 32, 32)).cuda()
            concatedImgs[:, 0:3:1, :, :] = samples
            concatedImgs[:, 3:6:1, :, :] = images


            # Augment transform images
            transIdx = torch.randint(1, 9, (1, cur_batch_size))
            Rot_Idx = torch.ceil(torch.true_divide(transIdx, 2)) # 0.=0 / 1.=90 / 2.=180/ 3.=270
            flip_Idx = torch.remainder(transIdx, 2)

            transedImgs = torch.zeros((concatedImgs.size(0), concatedImgs.size(1), concatedImgs.size(2), concatedImgs.size(3)))
            initial_img = concatedImgs

            for idx, _ in enumerate(transIdx[0]):
                transedImgs[idx, :, :, :] = torch.rot90(initial_img[idx], int(Rot_Idx[0][idx]), [1, 2])
                if flip_Idx[0][idx] == 1:
                    transedImgs[idx, :, :, :] = torch.flip(transedImgs[idx, :, :, :], [1, ])

            transedImgs = Variable(transedImgs, requires_grad=False).cuda()
            SSlabel = custom_one_hot_embedding(transIdx-1, 8).cuda()

            _, net_y = netC(transedImgs)
            _, SSlabels_idx = SSlabel.max(dim=1)
            class_predictoins = F.softmax(net_y, dim=1)

            _, predicted = class_predictoins.max(1)    
            classification_closed_correct += sum(predicted.data == SSlabels_idx)
            classification_total += len(SSlabels_idx)
        
    SS_stat = {
        fold: {
            'closed_set_self_supervised_accuracy': float(classification_closed_correct) / (classification_total),
        }
    }

    return basic_stat, SS_stat

def self_supervised_evaluate_classifier(networks, dataloader, open_set_dataloader=None, **options):
    for net in networks.values():
        net.eval()
    if options.get('mode') == 'baseline':
        print("Using the basic classifier")
        netC = networks['classifier']
        netG = networks['generator']
    elif options.get('mode') == 'weibull':
        print("Weibull mode: Using the basic classifier")
        netC = networks['classifier']
        netG = networks['generator']
    elif options.get('mode') == 'SS_baseline':
        print("Using the Semi-Supervised classifier(basic)")
        netC = networks['classifier']
        netG = networks['generator']
    else:
        print("no mode")
        print("Using the Semi-Supervised classifier(basic)")
        netC = networks['classifier']
        netG = networks['generator']
    fold = options.get('fold', 'evaluation')

    classification_closed_correct = 0
    classification_total = 0

    with torch.no_grad():
        for i, (images, class_labels) in enumerate(dataloader):
            images = Variable(images, requires_grad=False).cuda()
            labels = Variable(class_labels, requires_grad=False).cuda()
            cur_batch_size = images.shape[0]


            z = torch.randn((cur_batch_size, 100, 1, 1))
            z = Variable(z).cuda()

            samples = netG(z)
            samples = samples.mul(0.5).add(0.5)
            # images = images.mul(0.5).add(0.5)
            concatedImgs = torch.zeros((cur_batch_size, 6, 32, 32)).cuda()
            concatedImgs[:, 0:3:1, :, :] = samples
            concatedImgs[:, 3:6:1, :, :] = images


            # # Augment transform images
            # transIdx = torch.randint(1, 9, (1, cur_batch_size))
            # Rot_Idx = torch.ceil(torch.true_divide(transIdx, 2)) # 0.=0 / 1.=90 / 2.=180/ 3.=270
            # flip_Idx = torch.remainder(transIdx, 2)

            # transedImgs = torch.zeros((concatedImgs.size(0), concatedImgs.size(1), concatedImgs.size(2), concatedImgs.size(3)))
            # initial_img = concatedImgs

            # for idx, _ in enumerate(transIdx[0]):
            #     transedImgs[idx, :, :, :] = torch.rot90(initial_img[idx], int(Rot_Idx[0][idx]), [1, 2])
            #     if flip_Idx[0][idx] == 1:
            #         transedImgs[idx, :, :, :] = torch.flip(transedImgs[idx, :, :, :], [1, ])

            # transedImgs = Variable(transedImgs, requires_grad=False).cuda()
            # SSlabel = custom_one_hot_embedding(transIdx-1, 8).cuda()

            net_y, _ = netC(concatedImgs)
            # _, net_y = netC(transedImgs)
            # _, SSlabels_idx = SSlabel.max(dim=1)
            class_predictoins = F.softmax(net_y, dim=1)

            _, predicted = class_predictoins.max(1)
            classification_closed_correct += sum(predicted.data == labels)
            classification_total += len(labels)
        
            # classification_closed_correct += sum(predicted.data == SSlabels_idx)
            # classification_total += len(SSlabels_idx)
        
    stats = {
        fold: {
            'closed_set_image_class_accuracy': float(classification_closed_correct) / (classification_total),
        }
    }

    return stats

def basic_evaluate_classifier(networks, dataloader, open_set_dataloader=None, **options):
    for net in networks.values():
        net.eval()
    if options.get('mode') == 'baseline':
        print("Using the basic classifier")
        netC = networks['classifier']
        netG = networks['generator']
    elif options.get('mode') == 'weibull':
        print("Weibull mode: Using the basic classifier")
        netC = networks['classifier']
        netG = networks['generator']
    else:
        print("Using the Semi-Supervised classifier(basic)")
        netC = networks['classifier']
        netG = networks['generator']
    fold = options.get('fold', 'evaluation')

    classification_closed_correct = 0
    classification_total = 0

    with torch.no_grad():
        for i, (images, class_labels) in enumerate(dataloader):

            images = Variable(images, requires_grad=False).cuda()
            labels = Variable(class_labels, requires_grad=False).cuda()
            cur_batch_size = images.shape[0]

            z = torch.randn((cur_batch_size, 100, 1, 1))
            z = Variable(z).cuda()

            samples = netG(z)
            # samples = samples.mul(0.5).add(0.5)
            concatedImgs = torch.zeros((cur_batch_size, 6, 32, 32)).cuda()
            concatedImgs[:, 0:3:1, :, :] = samples
            concatedImgs[:, 3:6:1, :, :] = images
            # concatedImgs[:, 3:6:1, :, :] = samples

            # net_y = netC(concatedImgs)
            net_y, _ = netC(concatedImgs)
            class_predictoins = F.softmax(net_y, dim=1)

            _, predicted = class_predictoins.max(1)
            classification_closed_correct += sum(predicted.data == labels)
            classification_total += len(labels)
        
    stats = {
        fold: {
            'closed_set_accuracy': float(classification_closed_correct) / (classification_total),
        }
    }
    return stats
    
def evaluate_classifier(networks, dataloader, open_set_dataloader=None, **options):
    for net in networks.values():
        net.eval()
    if options.get('mode') == 'baseline':
        print("Using the K-class classifier")
        netC = networks['classifier_k']
    elif options.get('mode') == 'weibull':
        print("Weibull mode: Using the K-class classifier")
        netC = networks['classifier_k']
    else:
        print("Using the K open set classifier")
        # netC = networks['classifier_kplusone']
        netC = networks['classifier']
    fold = options.get('fold', 'evaluation')
    
    classification_closed_correct = 0
    classification_total = 0
    with torch.no_grad():
        for images, labels in dataloader:
            #images = Variable(images, volatile=True)
            # Predict a classification among known classes
            net_y = netC(images)
            class_predictions = F.softmax(net_y, dim=1)
            
            _, predicted = class_predictions.max(1)
            classification_closed_correct += sum(predicted.data == labels)
            classification_total += len(labels)

    stats = {
        fold: {
            'closed_set_accuracy': float(classification_closed_correct) / (classification_total),
        }
    }
    return stats


def pca(vectors):
    from sklearn.decomposition import PCA
    pca = PCA(n_components=2)
    pca.fit(vectors)
    return pca.transform(vectors)

def evaluate_SS_openset(networks, dataloader_on, dataloader_off, **options):
    from pprint import pprint
    for net in networks.values():
        net.eval()

    d_scores_on = get_SS_openset_scores(dataloader_on, networks, **options)
    d_scores_off = get_SS_openset_scores(dataloader_off, networks, **options)

    y_true = np.array([0]*len(d_scores_on) + [1]*len(d_scores_off))
    y_discriminator = np.concatenate([d_scores_on, d_scores_off])
    print(d_scores_on[0:50])
    print(d_scores_off[0:50])

    auc_d, plot_d = plot_roc(y_true, y_discriminator, 'Discriminator ROC vs {}'.format(dataloader_off.dsf.name), **options)

    save_plot(plot_d, 'roc_discriminator', **options)

    return {
        'auc_discriminator': auc_d,
    }

# Open Set Classification
# Given two datasets, one on-manifold and another off-manifold, predict
# whether each item is on-manifold or off-manifold using the discriminator
# or the autoencoder loss.
# Plot an ROC curve for each and report AUC
# dataloader_on: Test set of the same items the network was trained on
# dataloader_off: Separate dataset from a different distribution
def evaluate_openset(networks, dataloader_on, dataloader_off, **options):
    for net in networks.values():
        net.eval()
        
    d_scores_on = get_openset_scores(dataloader_on, networks, **options)
    d_scores_off = get_openset_scores(dataloader_off, networks, **options)
    
    y_true = np.array([0] * len(d_scores_on) + [1] * len(d_scores_off)) # open set을 계산해야 하니 중요한 것은 아예 모르는 데이터로 모인 데이터셋을 사용해야 한다.
    y_discriminator = np.concatenate([d_scores_on, d_scores_off])

    auc_d, plot_d = plot_roc(y_true, y_discriminator, 'Discriminator ROC vs {}'.format(dataloader_off.dsf.name), **options)

    save_plot(plot_d, 'roc_discriminator', **options)

    return {
        'auc_discriminator': auc_d,
    }


def combine_scores(score_list):
    example_count = len(score_list[0])
    assert all(len(x) == example_count for x in score_list)

    normalized_scores = np.ones(example_count)
    for score in score_list:
        score -= score.min()
        score /= score.max()
        normalized_scores *= score
        normalized_scores /= normalized_scores.max()
    return normalized_scores


def save_plot(plot, title, **options):
    print
    current_epoch = options.get('epoch', 0)
    comparison_name = options['comparison_dataset'].split('/')[-1].replace('.dataset', '')
    filename = 'plot_{}_vs_{}_epoch_{:04d}.png'.format(title, comparison_name, current_epoch)
    filename = os.path.join(options['result_dir'], filename)
    plot.figure.savefig(filename)
    
def get_SS_openset_scores(dataloader, networks, dataloader_train=None, **options):
    if options.get('mode') == 'weibull':
        print('Using weibull mode')
        openset_scores = SS_openset_weibull(dataloader, dataloader_train, networks)
    elif options.get('mode') == 'fuxin':
        print('Using Fuxin mode')
        openset_scores = SS_openset_fuxin(dataloader, networks)
    elif options.get('mode') == 'baseline':
        openset_scores = SS_openset_softmax_confidence(dataloader, networks)
    elif options.get('mode') == 'SS_baseline':
        openset_scores = SS_paper_openset_softmax_confidence(dataloader, networks)
    else:
        print(f"Using Default({options.get('mode')}) mode")
        openset_scores = SS_openset_softmax_confidence(dataloader, networks)



    # if options.get('mode') == 'weibull':
    #     openset_scores = openset_weibull(dataloader, dataloader_train, networks['classifier_k'])
    # elif options.get('mode') and 'weibull' in options['mode']:
    #     print("get_openset_scores 2번임 weibull kplueone 사용할때")
    #     openset_scores = openset_weibull(dataloader, dataloader_train, networks['classifier_kplusone'])
    # elif options.get('mode') == 'baseline':
    #     openset_scores = openset_softmax_confidence(dataloader, networks['classifier_k'])
    # elif options.get('mode') == 'autoencoder':
    #     openset_scores = openset_autoencoder(dataloader, networks)
    # elif options.get('mode') == 'fuxin':
    #     print('Using FUXIN mode')
    #     print("현재 사용하는 데이터셋은 :{}".format(dataloader.dsf.name))
    #     openset_scores = openset_fuxin(dataloader, networks['classifier_kplusone'])
    # else:
    #     print('Using DEFAULT mode')
    #     print("현재 사용하는 데이터셋은 :{}".format(dataloader.dsf.name))
    #     # openset_scores = openset_kplusone(dataloader, networks['classifier_kplusone'])
    #     openset_scores = openset_kplusone(dataloader, networks['classifier'])
    return openset_scores

def get_openset_scores(dataloader, networks, dataloader_train=None, **options):
    if options.get('mode') == 'weibull':
        openset_scores = openset_weibull(dataloader, dataloader_train, networks['classifier_k'])
    elif options.get('mode') and 'weibull' in options['mode']:
        print("get_openset_scores 2번임 weibull kplueone 사용할때")
        openset_scores = openset_weibull(dataloader, dataloader_train, networks['classifier_kplusone'])
    elif options.get('mode') == 'baseline':
        openset_scores = openset_softmax_confidence(dataloader, networks['classifier_k'])
    elif options.get('mode') == 'autoencoder':
        openset_scores = openset_autoencoder(dataloader, networks)
    elif options.get('mode') == 'fuxin':
        print('Using FUXIN mode')
        print("현재 사용하는 데이터셋은 :{}".format(dataloader.dsf.name))
        openset_scores = openset_fuxin(dataloader, networks['classifier_kplusone'])
    else:
        print('Using DEFAULT mode')
        print("현재 사용하는 데이터셋은 :{}".format(dataloader.dsf.name))
        # openset_scores = openset_kplusone(dataloader, networks['classifier_kplusone'])
        openset_scores = openset_kplusone(dataloader, networks['classifier'])
    return openset_scores


def openset_autoencoder(dataloader, networks, scale=4):
    netE = networks['encoder']
    netG = networks['generator']
    netE.train()
    netG.train()

    openset_scores = []
    for images, labels in dataloader:
        images = Variable(images)
        reconstructions = netG(netE(images, 4), 4)
        mse = ((reconstructions - images) ** 2).sum(dim=-1).sum(dim=-1).sum(dim=-1)
        openset_scores.extend([v for v in mse.data.cpu().numpy()])
    return openset_scores

def SS_openset_weibull(dataloader_test, dataloader_train, networks):
    # First generate pre-softmax 'activation vectors' for all training examples
    print("Weibull: computing features for all correctly-classified training data")
    netC = networks['classifier']
    netG = networks['generator']
    activation_vectors = {}

    for images, labels in dataloader_train:
        cur_batch_size = images.shape[0]

        # Create concated image(gen image + basic image), ch6
        z = torch.randn((cur_batch_size, 100, 1, 1))
        z = Variable(z).cuda()
        
        samples = netG(z)
        # samples = samples.mul(0.5).add(0.5)
        concatedImgs = torch.zeros((cur_batch_size, 6, 32, 32)).cuda()
        # concatedImgs[:, 0:3:1, :, :] = images
        # concatedImgs[:, 3:6:1, :, :] = samples
        concatedImgs[:, 0:3:1, :, :] = samples
        concatedImgs[:, 3:6:1, :, :] = images

        logits,_ = netC(concatedImgs)

        correctly_labeled = (logits.data.max(1)[1] == labels)
        labels_np = labels.cpu().numpy()
        logits_np = logits.data.cpu().numpy()
        for i, label in enumerate(labels_np):
            if not correctly_labeled[i]:
                continue
            # If correctly labeled, add this to the list of activation_vectors for this class
            # 맞았다면, 이 다음 줄로 내려갈 수 있다.
            if label not in activation_vectors: # 클래스만큼 빈 칸을 만들기 위하여
                activation_vectors[label] = []
            activation_vectors[label].append(logits_np[i]) # 맞은 것들의 logits 값들을 activation_vector로 넣어둠
    print("Computed activation_vectors for {} known classes".format(len(activation_vectors)))
    for class_idx in activation_vectors:
        print("분류기가 맞았다고 생각하는 Class {}: {} images".format(class_idx, len(activation_vectors[class_idx]))) # 원래 있던 코드라인임.

    # Compute a mean activation vector for each class
    print("Weibull computing mean activation vectors...")
    mean_activation_vectors = {}
    for class_idx in activation_vectors:
        mean_activation_vectors[class_idx] = np.array(activation_vectors[class_idx]).mean(axis=0)

    # Initialize one libMR Wiebull object for each class
    print("Fitting Weibull to distance distribution of each class")
    weibulls = {}
    for class_idx in activation_vectors:
        distances = []
        mav = mean_activation_vectors[class_idx]
        for v in activation_vectors[class_idx]:
            distances.append(np.linalg.norm(v - mav)) # 차이를 distance에 넣어줌
        mr = libmr.MR()
        tail_size = min(len(distances), WEIBULL_TAIL_SIZE) # WEIBULL_TAIL_SIZE=20
        mr.fit_high(distances, tail_size)
        weibulls[class_idx] = mr
        print("Weibull params for class {}: {}".format(class_idx, mr.get_params()))

    # Apply Weibull score to every logit
    weibull_scores = []
    logits = []
    classes = activation_vectors.keys()
    for images, labels in dataloader_test:        
        cur_batch_size = images.shape[0]

        # Create concated image(gen image + basic image), ch6
        z = torch.randn((cur_batch_size, 100, 1, 1))
        z = Variable(z).cuda()
        
        samples = netG(z)
        samples = samples.mul(0.5).add(0.5)
        concatedImgs = torch.zeros((cur_batch_size, 6, 32, 32)).cuda()
        # concatedImgs[:, 0:3:1, :, :] = images
        # concatedImgs[:, 3:6:1, :, :] = samples
        concatedImgs[:, 0:3:1, :, :] = samples
        concatedImgs[:, 3:6:1, :, :] = images

        # batch_logits = netC(images).data.cpu().numpy()
        b_logits, _ = netC(concatedImgs)
        batch_logits = b_logits.data.cpu().numpy()
        batch_weibull = np.zeros(shape=batch_logits.shape)
        for activation_vector in batch_logits:
            weibull_row = np.ones(len(classes))
            for class_idx in classes:
                mav = mean_activation_vectors[class_idx]
                dist = np.linalg.norm(activation_vector - mav)
                weibull_row[class_idx] = 1 - weibulls[class_idx].w_score(dist)
            weibull_scores.append(weibull_row)
            logits.append(activation_vector)
    weibull_scores = np.array(weibull_scores)
    logits = np.array(logits)

    # The following is as close as possible to the precise formulation in
    #   https://arxiv.org/pdf/1511.06233.pdf
    #N, K = logits.shape
    #alpha = np.ones((N, K))
    #for i in range(N):
    #    alpha[i][logits[i].argsort()] = np.arange(K) / (K - 1)
    #adjusted_scores = alpha * weibull_scores + (1 - alpha)
    #prob_open_set = (logits * (1 - adjusted_scores)).sum(axis=1)
    #return prob_open_set

    # But this is better
    # Logits must be positive (lower w score should mean lower probability)
    #shifted_logits = (logits - np.expand_dims(logits.min(axis=1), -1))
    #adjusted_scores = alpha * weibull_scores + (1 - alpha)
    #openmax_scores = -np.log(np.sum(np.exp(shifted_logits * adjusted_scores), axis=1))
    #return np.array(openmax_scores)

    # Let's just ignore alpha and ignore shifting
    openmax_scores = -np.log(np.sum(np.exp(logits * weibull_scores), axis=1))
    return np.array(openmax_scores)


def openset_weibull(dataloader_test, dataloader_train, netC):
    # First generate pre-softmax 'activation vectors' for all training examples
    print("Weibull: computing features for all correctly-classified training data")
    activation_vectors = {}
    for images, labels in dataloader_train:
        logits = netC(images)
        correctly_labeled = (logits.data.max(1)[1] == labels)
        labels_np = labels.cpu().numpy()
        logits_np = logits.data.cpu().numpy()
        for i, label in enumerate(labels_np):
            if not correctly_labeled[i]:
                continue
            # If correctly labeled, add this to the list of activation_vectors for this class
            # 맞았다면, 이 다음 줄로 내려갈 수 있다.
            if label not in activation_vectors: # 클래스만큼 빈 칸을 만들기 위하여
                activation_vectors[label] = []
            activation_vectors[label].append(logits_np[i]) # 맞은 것들의 logits 값들을 activation_vector로 넣어둠
    print("Computed activation_vectors for {} known classes".format(len(activation_vectors)))
    for class_idx in activation_vectors:
        print("분류기가 맞았다고 생각하는 Class {}: {} images".format(class_idx, len(activation_vectors[class_idx]))) # 원래 있던 코드라인임.

    # Compute a mean activation vector for each class
    print("Weibull computing mean activation vectors...")
    mean_activation_vectors = {}
    for class_idx in activation_vectors:
        mean_activation_vectors[class_idx] = np.array(activation_vectors[class_idx]).mean(axis=0)

    # Initialize one libMR Wiebull object for each class
    print("Fitting Weibull to distance distribution of each class")
    weibulls = {}
    for class_idx in activation_vectors:
        distances = []
        mav = mean_activation_vectors[class_idx]
        for v in activation_vectors[class_idx]:
            distances.append(np.linalg.norm(v - mav)) # 차이를 distance에 넣어줌
        mr = libmr.MR()
        tail_size = min(len(distances), WEIBULL_TAIL_SIZE) # WEIBULL_TAIL_SIZE=20
        mr.fit_high(distances, tail_size)
        weibulls[class_idx] = mr
        print("Weibull params for class {}: {}".format(class_idx, mr.get_params()))

    # Apply Weibull score to every logit
    weibull_scores = []
    logits = []
    classes = activation_vectors.keys()
    for images, labels in dataloader_test:
        batch_logits = netC(images).data.cpu().numpy()
        batch_weibull = np.zeros(shape=batch_logits.shape)
        for activation_vector in batch_logits:
            weibull_row = np.ones(len(classes))
            for class_idx in classes:
                mav = mean_activation_vectors[class_idx]
                dist = np.linalg.norm(activation_vector - mav)
                weibull_row[class_idx] = 1 - weibulls[class_idx].w_score(dist)
            weibull_scores.append(weibull_row)
            logits.append(activation_vector)
    weibull_scores = np.array(weibull_scores)
    logits = np.array(logits)

    # The following is as close as possible to the precise formulation in
    #   https://arxiv.org/pdf/1511.06233.pdf
    #N, K = logits.shape
    #alpha = np.ones((N, K))
    #for i in range(N):
    #    alpha[i][logits[i].argsort()] = np.arange(K) / (K - 1)
    #adjusted_scores = alpha * weibull_scores + (1 - alpha)
    #prob_open_set = (logits * (1 - adjusted_scores)).sum(axis=1)
    #return prob_open_set

    # But this is better
    # Logits must be positive (lower w score should mean lower probability)
    #shifted_logits = (logits - np.expand_dims(logits.min(axis=1), -1))
    #adjusted_scores = alpha * weibull_scores + (1 - alpha)
    #openmax_scores = -np.log(np.sum(np.exp(shifted_logits * adjusted_scores), axis=1))
    #return np.array(openmax_scores)

    # Let's just ignore alpha and ignore shifting
    openmax_scores = -np.log(np.sum(np.exp(logits * weibull_scores), axis=1))
    return np.array(openmax_scores)


def openset_kplusone(dataloader, netC):
    openset_scores = []
    with torch.no_grad():
        for i, (images, labels) in enumerate(dataloader):
            #images = Variable(images, volatile=True)
            preds = netC(images)
            # The implicit K+1th class (the open set class) is computed
            #  by assuming an extra linear output with constant value 0
            z = torch.exp(preds).sum(dim=1)
            prob_known = z / (z + 1)
            prob_unknown = 1 - prob_known
            openset_scores.extend(prob_unknown.data.cpu().numpy())
    return np.array(openset_scores)


def openset_softmax_confidence(dataloader, netC):
    openset_scores = []
    with torch.no_grad():
        for i, (images, labels) in enumerate(dataloader):
            #images = Variable(images, volatile=True)
            preds = F.softmax(netC(images), dim=1)
            openset_scores.extend(preds.max(dim=1)[0].data.cpu().numpy())
            # classifier가 몇 score로 그 클래스로 정답을 내렸는지 모은다.
    return -np.array(openset_scores)

def SS_paper_openset_softmax_confidence(dataloader, networks):
    netC = networks['classifier']
    netG = networks['generator']
    openset_scores = []

    # cnt = 0
    with torch.no_grad():
        for i, (images, labels) in enumerate(dataloader):
            cur_batch_size = images.shape[0]

            # Create concated image(gen image + basic image), ch6
            z = torch.randn((cur_batch_size, 100, 1, 1))
            z = Variable(z).cuda()
            
            samples = netG(z)
            # samples = samples.mul(0.5).add(0.5)
            concatedImgs = torch.zeros((cur_batch_size, 6, 32, 32)).cuda()
            # concatedImgs[:, 0:3:1, :, :] = images
            # concatedImgs[:, 3:6:1, :, :] = samples
            concatedImgs[:, 0:3:1, :, :] = samples
            concatedImgs[:, 3:6:1, :, :] = images

            #images = Variable(images, volatile=True)
            logits, _ = netC(concatedImgs)
            
            #images = Variable(images, volatile=True)
            # preds = F.softmax(logits, dim=1)

            lamda = 0.9
            # conditioned_pred = preds.max(dim=1)[0]
            conditioned_pred = logits.max(dim=1)[0]
            # conditioned_pred[conditioned_pred>1] = 1
            conditioned_pred[conditioned_pred<lamda] = -1
            # conditioned_pred[conditioned_pred>=lamda] = 100
            # if 'cifar100-toCIFAR10-u0' == dataloader.dsf.name:

            # preds = F.softmax(conditioned_pred, dim=1)
            # openset_scores.extend(preds.max(dim=1)[0].data.cpu().numpy())
            openset_scores.extend(conditioned_pred.data.cpu().numpy())
            # classifier가 몇 score로 그 클래스로 정답을 내렸는지 모은다.
    # print("="*60)
    # print(cnt)
    return -np.array(openset_scores)

def SS_openset_softmax_confidence(dataloader, networks):
    netC = networks['classifier']
    netG = networks['generator']
    openset_scores = []
    with torch.no_grad():
        for i, (images, labels) in enumerate(dataloader):
            cur_batch_size = images.shape[0]

            # Create concated image(gen image + basic image), ch6
            z = torch.randn((cur_batch_size, 100, 1, 1))
            z = Variable(z).cuda()
            
            samples = netG(z)
            # samples = samples.mul(0.5).add(0.5)
            concatedImgs = torch.zeros((cur_batch_size, 6, 32, 32)).cuda()
            # concatedImgs[:, 0:3:1, :, :] = images
            # concatedImgs[:, 3:6:1, :, :] = samples
            concatedImgs[:, 0:3:1, :, :] = samples
            concatedImgs[:, 3:6:1, :, :] = images

            #images = Variable(images, volatile=True)
            logits, _ = netC(concatedImgs)
            #images = Variable(images, volatile=True)
            preds = F.softmax(logits, dim=1)
            openset_scores.extend(preds.max(dim=1)[0].data.cpu().numpy())
            # classifier가 몇 score로 그 클래스로 정답을 내렸는지 모은다.
    return -np.array(openset_scores)

def SS_openset_fuxin(dataloader, networks):
    netC = networks['classifier']
    netG = networks['generator']
    openset_scores = []
    with torch.no_grad():
        for i, (images, labels) in enumerate(dataloader):
            cur_batch_size = images.shape[0]

            # Create concated image(gen image + basic image), ch6
            z = torch.randn((cur_batch_size, 100, 1, 1))
            z = Variable(z).cuda()
            
            samples = netG(z)
            # samples = samples.mul(0.5).add(0.5)
            concatedImgs = torch.zeros((cur_batch_size, 6, 32, 32)).cuda()
            # concatedImgs[:, 0:3:1, :, :] = images
            # concatedImgs[:, 3:6:1, :, :] = samples
            concatedImgs[:, 0:3:1, :, :] = samples
            concatedImgs[:, 3:6:1, :, :] = images

            #images = Variable(images, volatile=True)
            logits, _ = netC(concatedImgs)
            augmented_logits = F.pad(logits, pad=(0,1))
            # The implicit K+1th class (the open set class) is computed
            #  by assuming an extra linear output with constant value 0
            preds = F.softmax(augmented_logits)
            #preds = augmented_logits
            prob_unknown = preds[:, -1]
            prob_known = preds[:, :-1].max(dim=1)[0]
            prob_open = prob_unknown - prob_known

            openset_scores.extend(prob_open.data.cpu().numpy())

    return np.array(openset_scores)

def openset_fuxin(dataloader, netC):
    openset_scores = []
    with torch.no_grad():
        for i, (images, labels) in enumerate(dataloader):
            #images = Variable(images, volatile=True)
            logits = netC(images)
            augmented_logits = F.pad(logits, pad=(0,1))
            # The implicit K+1th class (the open set class) is computed
            #  by assuming an extra linear output with constant value 0
            preds = F.softmax(augmented_logits)
            #preds = augmented_logits
            prob_unknown = preds[:, -1]
            prob_known = preds[:, :-1].max(dim=1)[0]
            prob_open = prob_unknown - prob_known

            openset_scores.extend(prob_open.data.cpu().numpy())
    return np.array(openset_scores)


def plot_roc(y_true, y_score, title="Receiver Operating Characteristic", **options):
    fpr, tpr, thresholds = roc_curve(y_true, y_score)
    auc_score = roc_auc_score(y_true, y_score)
    plot = plot_xy(fpr, tpr, x_axis="False Positive Rate", y_axis="True Positive Rate", title=title)
    if options.get('roc_output'):
        print("Saving ROC scores to file {}".format(options['roc_output']))
        np.save(options['roc_output'], (fpr, tpr))
    return auc_score, plot


def save_evaluation(new_results, result_dir, epoch):
    if not os.path.exists('evaluations'):
        os.mkdir('evaluations')
    filename = 'evaluations/eval_epoch_{:06d}.json'.format(epoch)
    filename = os.path.join(result_dir, filename)
    filename = os.path.expanduser(filename)
    if os.path.exists(filename):
        old_results = json.load(open(filename))
    else:
        old_results = {}
    old_results.update(new_results)
    with open(filename, 'w') as fp:
        json.dump(old_results, fp, indent=2, sort_keys=True)

from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt

def plot_confusion_matrix(y_true, y_pred, classes,
                          normalize=False,
                          title=None,
                          cmap=plt.cm.Blues, 
                          pngName="./test.png"):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if not title:
        if normalize:
            title = 'Normalized confusion matrix'
        else:
            title = 'Confusion matrix, without normalization'

    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    # Only use the labels that appear in the data
    #classes = classes[unique_labels(y_true, y_pred)]
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()

    plt.savefig(pngName)
    return ax

def draw_basic_confidence_map(networks, dataloader, **options):
    for net in networks.values():
        net.eval()

    netC = networks['classifier']
    # fold = options.get('fold', 'evaluation')

    # print(dataloader.dsf.count(dataloader.fold))
    predicted_array = torch.zeros(dataloader.dsf.count(dataloader.fold), dtype=int)
    class_labels_array = torch.zeros(dataloader.dsf.count(dataloader.fold))
    cnt = 0

    with torch.no_grad():
        for i, (images, class_labels) in enumerate(dataloader):
            images = images.unsqueeze(1)
            images = Variable(images, requires_grad=False).cuda()
            labels = Variable(class_labels, requires_grad=False).cuda()
            # cur_batch_size = images.shape[0]

            net_y = netC(images)
            class_predictoins = F.softmax(net_y, dim=1)

            _, predicted = class_predictoins.max(1)
            predicted_array[cnt:cnt+len(labels)] = predicted
            class_labels_array[cnt:cnt+len(labels)] = labels
            cnt += len(labels)
        
    plot_confusion_matrix(class_labels_array, predicted_array, pngName="./confusion_matrix.png",classes=dataloader.lab_conv.labels, title='Confusion matrix')
    plot_confusion_matrix(class_labels_array, predicted_array, pngName="./confusion_matrix_Norm.png",classes=dataloader.lab_conv.labels, normalize=True, title='Confusion matrix with normalization')
    return True

def draw_confidence_map(networks, dataloader, **options):
    for net in networks.values():
        net.eval()

    netC = networks['classifier']
    netG = networks['generator']
    fold = options.get('fold', 'evaluation')

    # print(dataloader.dsf.count(dataloader.fold))
    predicted_array = torch.zeros(dataloader.dsf.count(dataloader.fold), dtype=int)
    class_labels_array = torch.zeros(dataloader.dsf.count(dataloader.fold))
    cnt = 0

    with torch.no_grad():
        for i, (images, class_labels) in enumerate(dataloader):

            images = Variable(images, requires_grad=False).cuda()
            labels = Variable(class_labels, requires_grad=False).cuda()
            cur_batch_size = images.shape[0]

            z = torch.randn((cur_batch_size, 100, 1, 1))
            z = Variable(z).cuda()

            samples = netG(z)

            concatedImgs = torch.zeros((cur_batch_size, 6, 32, 32)).cuda()
            concatedImgs[:, 0:3:1, :, :] = samples
            concatedImgs[:, 3:6:1, :, :] = images

            net_y, _ = netC(concatedImgs)
            class_predictoins = F.softmax(net_y, dim=1)

            _, predicted = class_predictoins.max(1)
            predicted_array[cnt:cnt+len(labels)] = predicted
            class_labels_array[cnt:cnt+len(labels)] = labels
            cnt += len(labels)
        
    plot_confusion_matrix(class_labels_array, predicted_array, pngName="./confusion_matrix.png",classes=dataloader.lab_conv.labels, title='Confusion matrix')
    plot_confusion_matrix(class_labels_array, predicted_array, pngName="./confusion_matrix_Norm.png",classes=dataloader.lab_conv.labels, normalize=True, title='Confusion matrix with normalization')
    return True
