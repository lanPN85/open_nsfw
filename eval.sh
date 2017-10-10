#!/usr/bin/env bash
python2 batch_eval.py --model-def nsfw_model/deploy.prototxt --pretrained-model nsfw_model/resnet_50_1by2_nsfw.caffemodel\
 --folder ~/Pictures/ 2> data/run.log
