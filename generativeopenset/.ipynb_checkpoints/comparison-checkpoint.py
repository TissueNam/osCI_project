import evaluation
from dataloader import CustomDataloader

from pprint import pprint

def evaluate_with_comparison(networks, dataloader, **options):
    comparison_dataloader = get_comparison_dataloader(**options)
    if comparison_dataloader:
        options['fold'] = 'openset_{}'.format(comparison_dataloader.dsf.name)
    if options.get('mode'):
        options['fold'] += '_{}'.format(options['mode'])
        #머임 fold 이름: openset_cifar10-split0b_baseline *baseline 모드에 경우
    if options.get('aux_dataset'):
        aux_dataset = CustomDataloader(options['aux_dataset'])
        options['fold'] = '{}_{}'.format(options.get('fold'), aux_dataset.dsf.count())

    new_results = evaluation.evaluate_classifier(networks, dataloader, comparison_dataloader, **options)

    if comparison_dataloader:
        print("comparision_dataloader 가 있을때 evaludate_openset으로 {}으로 평가.".format(dataloader.dsf.name))
        openset_results = evaluation.evaluate_openset(networks, dataloader, comparison_dataloader, **options)
        pprint(new_results)
        new_results[options['fold']].update(openset_results)
    return new_results


def get_comparison_dataloader(comparison_dataset=None, **options):
    if not comparison_dataset:
        return
    comparison_options = options.copy()
    comparison_options['dataset'] = comparison_dataset
    comparison_options['last_batch'] = True
    comparison_options['shuffle'] = False
    comparison_dataloader = CustomDataloader(**comparison_options)
    return comparison_dataloader
