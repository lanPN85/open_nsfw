from __future__ import print_function
from argparse import ArgumentParser

import os
import caffe
import numpy as np

import classify_nsfw as nsfw

NSFW_CUTOFF = 0.8
SFW_CUTOFF = 0.2


def parse_arguments():
    parser = ArgumentParser()
    parser.add_argument('--model-def', dest='model_def', default='nsfw_model/deploy.prototxt')
    parser.add_argument('--pretrained-model', dest='pretrained_model', default='resnet_50_1by2_nsfw.caffemodel')
    parser.add_argument('--folder', dest='folder', default='data/sfw',
                        help='Path to the folder containing images')
    return parser.parse_args()


def list_full_dir(path):
    files = os.listdir(path)
    res = []
    for f in files:
        full_path = os.path.join(path, f)
        if os.path.isfile(full_path):
            res.append(full_path)
    return res


def score_loop(file_list, transformer, net):
    score_list = []
    length = len(file_list)

    for i, fname in enumerate(file_list):
        f = open(fname)
        img = f.read()
        f.close()
        print('%d / %d ...' % (i + 1, length))
        scores = nsfw.caffe_preprocess_and_compute(img, caffe_transformer=transformer, caffe_net=net,
                                                   output_layers=['prob'])
        if scores[1] is None:
            print('WARNING: Error predicting %s' % fname)
        else:
            score_list.append(scores[1])
    print()
    return score_list


def summarize(scores):
    metrics = []
    npscores = np.asarray(scores, dtype=np.float)

    # Total images
    metrics.append(('# Images', len(scores)))

    # Average NSFW score
    avg = np.mean(npscores)
    metrics.append(('Average NSFW score', avg))

    # Highest score
    top = np.max(npscores)
    metrics.append(('Highest NSFW score', top))

    # Lowest score
    bottom = np.min(npscores)
    metrics.append(('Lowest NSFW score', bottom))

    # NSFW, SFW & Unsure count
    nsfw_count = 0
    sfw_count = 0
    unsure_count = 0
    for s in scores:
        if s >= NSFW_CUTOFF:
            nsfw_count += 1
        elif s > SFW_CUTOFF:
            unsure_count += 1
        else:
            sfw_count += 1
    metrics.append(('NSFW Images', nsfw_count))
    metrics.append(('SFW Images', sfw_count))
    metrics.append(('Unsure Images', unsure_count))

    return metrics


def show_metrics(metrics):
    for met in metrics:
        print('%s: %s' % (met[0], met[1]))


def main(args):
    print('Loading model...')
    nsfw_net = caffe.Net(args.model_def, 1,
                         weights=args.pretrained_model)
    caffe_transformer = nsfw.load_transformer(nsfw_net)
    print('Done.')

    files = list_full_dir(args.folder)

    print('Scoring...')
    scores = score_loop(files, caffe_transformer, nsfw_net)

    print('\nResults:')
    show_metrics(summarize(scores))


if __name__ == '__main__':
    main(parse_arguments())
