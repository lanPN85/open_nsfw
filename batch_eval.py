from argparse import ArgumentParser

import os
import caffe
import numpy as np

import classify_nsfw as nsfw


def parse_arguments():
    parser = ArgumentParser()
    parser.add_argument('--model-def', dest='model_def', default='nsfw_model/deploy.prototxt')
    parser.add_argument('--pretrained-model', dest='pretrained_model', default='resnet_50_1by2_nsfw.caffemodel')
    parser.add_argument('--sfw', dest='sfw', default='data/sfw',
                        help='Path to the folder containing SFW images')
    parser.add_argument('--nsfw', dest='nsfw', default='data/nsfw',
                        help='Path to the folder containing NSFW images')
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

    for fname in file_list:
        f = open(fname)
        img = f.read()
        f.close()
        scores = nsfw.caffe_preprocess_and_compute(img, caffe_transformer=transformer, caffe_net=net,
                                                   output_layers=['prob'])
        if scores[1] is None:
            print 'WARNING: Error predicting %s' % fname
        else:
            score_list.append(scores[1])
    return score_list


def summarize(scores):
    metrics = []
    npscores = np.asarray(scores, dtype=np.float)

    # Average NSFW score
    avg = np.mean(npscores)
    metrics.append(('Average NSFW score', avg))

    # Highest score
    top = np.max(npscores)
    metrics.append(('Highest NSFW score', top))

    return metrics


def show_metrics(metrics):
    for met in metrics:
        print '%s: %s' % (met[0], met[1])


def main(args):
    print 'Loading model...'
    nsfw_net = caffe.Net(args.model_def,  # pylint: disable=invalid-name
                         1, weights=args.pretrained_model)
    caffe_transformer = nsfw.load_transformer(nsfw_net)
    print 'Done.'

    sfw_files = list_full_dir(args.sfw)
    nsfw_files = list_full_dir(args.nsfw)

    print 'Scoring...'
    sfw_scores = score_loop(sfw_files, caffe_transformer, nsfw_net)
    nsfw_scores = score_loop(nsfw_files, caffe_transformer, nsfw_net)

    print '\nResults on SFW images:'
    show_metrics(summarize(sfw_scores))
    print '\nResults on NSFW images:'
    show_metrics(summarize(nsfw_scores))


if __name__ == '__main__':
    main(parse_arguments())
