#!/usr/bin/env bash
caffe train -iterations 200\
 -model nsfw_model/train_extra.prototxt\
 -solver nsfw_model/solver_extra.prototxt\
 -weights nsfw_model/resnet_50_nsfw_extra.caffemodel
