#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os, sys
import math
import argparse
import numpy as np
from easydict import EasyDict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

GMFLOW = os.path.join("submodules", "GMFlow")
sys.path.insert(0, GMFLOW)

CLASSIFIER_MODELS = [
  ("alexnet",                 "AlexNet_Weights.IMAGENET1K_V1"),                       # Acc@1: 56.522, Acc@5: 79.066, Params: 61.1M,   GFLOPS: 0.71

  ("convnext_base",           "ConvNeXt_Base_Weights.IMAGENET1K_V1"),                 # Acc@1: 84.062, Acc@5: 96.870, Params: 88.6M,   GFLOPS: 15.36
  ("convnext_large",          "ConvNeXt_Large_Weights.IMAGENET1K_V1"),                # Acc@1: 84.414, Acc@5: 96.976, Params: 197.8M,  GFLOPS: 34.36
  ("convnext_small",          "ConvNeXt_Small_Weights.IMAGENET1K_V1"),                # Acc@1: 83.616, Acc@5: 96.650, Params: 50.2M,   GFLOPS: 8.68
  ("convnext_tiny",           "ConvNeXt_Tiny_Weights.IMAGENET1K_V1"),                 # Acc@1: 82.520, Acc@5: 96.146, Params: 28.6M,   GFLOPS: 4.46

  ("densenet121",             "DenseNet121_Weights.IMAGENET1K_V1"),                   # Acc@1: 74.434, Acc@5: 91.972, Params: 8.0M,    GFLOPS: 2.83
  ("densenet161",             "DenseNet161_Weights.IMAGENET1K_V1"),                   # Acc@1: 77.138, Acc@5: 93.560, Params: 28.7M,   GFLOPS: 7.73
  ("densenet169",             "DenseNet169_Weights.IMAGENET1K_V1"),                   # Acc@1: 75.600, Acc@5: 92.806, Params: 14.1M,   GFLOPS: 3.36
  ("densenet201",             "DenseNet201_Weights.IMAGENET1K_V1"),                   # Acc@1: 76.896, Acc@5: 93.370, Params: 20.0M,   GFLOPS: 4.29

  ("efficientnet_b0",         "EfficientNet_B0_Weights.IMAGENET1K_V1"),               # Acc@1: 77.692, Acc@5: 93.532, Params: 5.3M,    GFLOPS: 0.39
  ("efficientnet_b1",         "EfficientNet_B1_Weights.IMAGENET1K_V1"),               # Acc@1: 78.642, Acc@5: 94.186, Params: 7.8M,    GFLOPS: 0.69
  ("efficientnet_b1",         "EfficientNet_B1_Weights.IMAGENET1K_V2"),               # Acc@1: 79.838, Acc@5: 94.934, Params: 7.8M,    GFLOPS: 0.69
  ("efficientnet_b2",         "EfficientNet_B2_Weights.IMAGENET1K_V1"),               # Acc@1: 80.608, Acc@5: 95.310, Params: 9.1M,    GFLOPS: 1.09
  ("efficientnet_b3",         "EfficientNet_B3_Weights.IMAGENET1K_V1"),               # Acc@1: 82.008, Acc@5: 96.054, Params: 12.2M,   GFLOPS: 1.83
  ("efficientnet_b4",         "EfficientNet_B4_Weights.IMAGENET1K_V1"),               # Acc@1: 83.384, Acc@5: 96.594, Params: 19.3M,   GFLOPS: 4.39
  ("efficientnet_b5",         "EfficientNet_B5_Weights.IMAGENET1K_V1"),               # Acc@1: 83.444, Acc@5: 96.628, Params: 30.4M,   GFLOPS: 10.27
  ("efficientnet_b6",         "EfficientNet_B6_Weights.IMAGENET1K_V1"),               # Acc@1: 84.008, Acc@5: 96.916, Params: 43.0M,   GFLOPS: 19.07
  ("efficientnet_b7",         "EfficientNet_B7_Weights.IMAGENET1K_V1"),               # Acc@1: 84.122, Acc@5: 96.908, Params: 66.3M,   GFLOPS: 37.75
  ("efficientnet_v2_l",       "EfficientNet_V2_L_Weights.IMAGENET1K_V1"),             # Acc@1: 85.808, Acc@5: 97.788, Params: 118.5M,  GFLOPS: 56.08
  ("efficientnet_v2_m",       "EfficientNet_V2_M_Weights.IMAGENET1K_V1"),             # Acc@1: 85.112, Acc@5: 97.156, Params: 54.1M,   GFLOPS: 24.58
  ("efficientnet_v2_s",       "EfficientNet_V2_S_Weights.IMAGENET1K_V1"),             # Acc@1: 84.228, Acc@5: 96.878, Params: 21.5M,   GFLOPS: 8.37

  ("googlenet",               "GoogLeNet_Weights.IMAGENET1K_V1"),                     # Acc@1: 69.778, Acc@5: 89.530, Params: 6.6M,    GFLOPS: 1.50
  ("inception_v3",            "Inception_V3_Weights.IMAGENET1K_V1"),                  # Acc@1: 77.294, Acc@5: 93.450, Params: 27.2M,   GFLOPS: 5.71

  ("mnasnet0_5",              "MNASNet0_5_Weights.IMAGENET1K_V1"),                    # Acc@1: 67.734, Acc@5: 87.490, Params: 2.2M,    GFLOPS: 0.10
  ("mnasnet0_75",             "MNASNet0_75_Weights.IMAGENET1K_V1"),                   # Acc@1: 71.180, Acc@5: 90.496, Params: 3.2M,    GFLOPS: 0.21
  ("mnasnet1_0",              "MNASNet1_0_Weights.IMAGENET1K_V1"),                    # Acc@1: 73.456, Acc@5: 91.510, Params: 4.4M,    GFLOPS: 0.31
  ("mnasnet1_3",              "MNASNet1_3_Weights.IMAGENET1K_V1"),                    # Acc@1: 76.506, Acc@5: 93.522, Params: 6.3M,    GFLOPS: 0.53

  ("maxvit_t",                "MaxVit_T_Weights.IMAGENET1K_V1"),                      # Acc@1: 83.700, Acc@5: 96.722, Params: 30.9M,   GFLOPS: 5.56

  ("mobilenet_v2",            "MobileNet_V2_Weights.IMAGENET1K_V1"),                  # Acc@1: 71.878, Acc@5: 90.286, Params: 3.5M,    GFLOPS: 0.30
  ("mobilenet_v2",            "MobileNet_V2_Weights.IMAGENET1K_V2"),                  # Acc@1: 72.154, Acc@5: 90.822, Params: 3.5M,    GFLOPS: 0.30
  ("mobilenet_v3_large",      "MobileNet_V3_Large_Weights.IMAGENET1K_V1"),            # Acc@1: 74.042, Acc@5: 91.340, Params: 5.5M,    GFLOPS: 0.22
  ("mobilenet_v3_large",      "MobileNet_V3_Large_Weights.IMAGENET1K_V2"),            # Acc@1: 75.274, Acc@5: 92.566, Params: 5.5M,    GFLOPS: 0.22
  ("mobilenet_v3_small",      "MobileNet_V3_Small_Weights.IMAGENET1K_V1"),            # Acc@1: 67.668, Acc@5: 87.402, Params: 2.5M,    GFLOPS: 0.06

  ("regnet_x_16gf",           "RegNet_X_16GF_Weights.IMAGENET1K_V1"),                 # Acc@1: 80.058, Acc@5: 94.944, Params: 54.3M,   GFLOPS: 15.94
  ("regnet_x_16gf",           "RegNet_X_16GF_Weights.IMAGENET1K_V2"),                 # Acc@1: 82.716, Acc@5: 96.196, Params: 54.3M,   GFLOPS: 15.94
  ("regnet_x_1_6gf",          "RegNet_X_1_6GF_Weights.IMAGENET1K_V1"),                # Acc@1: 77.040, Acc@5: 93.440, Params: 9.2M,    GFLOPS: 1.60
  ("regnet_x_1_6gf",          "RegNet_X_1_6GF_Weights.IMAGENET1K_V2"),                # Acc@1: 79.668, Acc@5: 94.922, Params: 9.2M,    GFLOPS: 1.60
  ("regnet_x_32gf",           "RegNet_X_32GF_Weights.IMAGENET1K_V1"),                 # Acc@1: 80.622, Acc@5: 95.248, Params: 107.8M,  GFLOPS: 31.74
  ("regnet_x_32gf",           "RegNet_X_32GF_Weights.IMAGENET1K_V2"),                 # Acc@1: 83.014, Acc@5: 96.288, Params: 107.8M,  GFLOPS: 31.74
  ("regnet_x_3_2gf",          "RegNet_X_3_2GF_Weights.IMAGENET1K_V1"),                # Acc@1: 78.364, Acc@5: 93.992, Params: 15.3M,   GFLOPS: 3.18
  ("regnet_x_3_2gf",          "RegNet_X_3_2GF_Weights.IMAGENET1K_V2"),                # Acc@1: 81.196, Acc@5: 95.430, Params: 15.3M,   GFLOPS: 3.18
  ("regnet_x_400mf",          "RegNet_X_400MF_Weights.IMAGENET1K_V1"),                # Acc@1: 72.834, Acc@5: 90.950, Params: 5.5M,    GFLOPS: 0.41
  ("regnet_x_400mf",          "RegNet_X_400MF_Weights.IMAGENET1K_V2"),                # Acc@1: 74.864, Acc@5: 92.322, Params: 5.5M,    GFLOPS: 0.41
  ("regnet_x_800mf",          "RegNet_X_800MF_Weights.IMAGENET1K_V1"),                # Acc@1: 75.212, Acc@5: 92.348, Params: 7.3M,    GFLOPS: 0.80
  ("regnet_x_800mf",          "RegNet_X_800MF_Weights.IMAGENET1K_V2"),                # Acc@1: 77.522, Acc@5: 93.826, Params: 7.3M,    GFLOPS: 0.80
  ("regnet_x_8gf",            "RegNet_X_8GF_Weights.IMAGENET1K_V1"),                  # Acc@1: 79.344, Acc@5: 94.686, Params: 39.6M,   GFLOPS: 8.00
  ("regnet_x_8gf",            "RegNet_X_8GF_Weights.IMAGENET1K_V2"),                  # Acc@1: 81.682, Acc@5: 95.678, Params: 39.6M,   GFLOPS: 8.00

  ("regnet_y_128gf",          "RegNet_Y_128GF_Weights.IMAGENET1K_SWAG_E2E_V1"),       # Acc@1: 88.228, Acc@5: 98.682, Params: 644.8M,  GFLOPS: 374.57
  ("regnet_y_128gf",          "RegNet_Y_128GF_Weights.IMAGENET1K_SWAG_LINEAR_V1"),    # Acc@1: 86.068, Acc@5: 97.844, Params: 644.8M,  GFLOPS: 127.52
  ("regnet_y_16gf",           "RegNet_Y_16GF_Weights.IMAGENET1K_V1"),                 # Acc@1: 80.424, Acc@5: 95.240, Params: 83.6M,   GFLOPS: 15.91
  ("regnet_y_16gf",           "RegNet_Y_16GF_Weights.IMAGENET1K_V2"),                 # Acc@1: 82.886, Acc@5: 96.328, Params: 83.6M,   GFLOPS: 15.91
  ("regnet_y_16gf",           "RegNet_Y_16GF_Weights.IMAGENET1K_SWAG_E2E_V1"),        # Acc@1: 86.012, Acc@5: 98.054, Params: 83.6M,   GFLOPS: 46.73
  ("regnet_y_16gf",           "RegNet_Y_16GF_Weights.IMAGENET1K_SWAG_LINEAR_V1"),     # Acc@1: 83.976, Acc@5: 97.244, Params: 83.6M,   GFLOPS: 15.91
  ("regnet_y_1_6gf",          "RegNet_Y_1_6GF_Weights.IMAGENET1K_V1"),                # Acc@1: 77.950, Acc@5: 93.966, Params: 11.2M,   GFLOPS: 1.61
  ("regnet_y_1_6gf",          "RegNet_Y_1_6GF_Weights.IMAGENET1K_V2"),                # Acc@1: 80.876, Acc@5: 95.444, Params: 11.2M,   GFLOPS: 1.61
  ("regnet_y_32gf",           "RegNet_Y_32GF_Weights.IMAGENET1K_V1"),                 # Acc@1: 80.878, Acc@5: 95.340, Params: 145.0M,  GFLOPS: 32.28
  ("regnet_y_32gf",           "RegNet_Y_32GF_Weights.IMAGENET1K_V2"),                 # Acc@1: 83.368, Acc@5: 96.498, Params: 145.0M,  GFLOPS: 32.28
  ("regnet_y_32gf",           "RegNet_Y_32GF_Weights.IMAGENET1K_SWAG_E2E_V1"),        # Acc@1: 86.838, Acc@5: 98.362, Params: 145.0M,  GFLOPS: 94.83
  ("regnet_y_32gf",           "RegNet_Y_32GF_Weights.IMAGENET1K_SWAG_LINEAR_V1"),     # Acc@1: 84.622, Acc@5: 97.480, Params: 145.0M,  GFLOPS: 32.28
  ("regnet_y_3_2gf",          "RegNet_Y_3_2GF_Weights.IMAGENET1K_V1"),                # Acc@1: 78.948, Acc@5: 94.576, Params: 19.4M,   GFLOPS: 3.18
  ("regnet_y_3_2gf",          "RegNet_Y_3_2GF_Weights.IMAGENET1K_V2"),                # Acc@1: 81.982, Acc@5: 95.972, Params: 19.4M,   GFLOPS: 3.18
  ("regnet_y_400mf",          "RegNet_Y_400MF_Weights.IMAGENET1K_V1"),                # Acc@1: 74.046, Acc@5: 91.716, Params: 4.3M,    GFLOPS: 0.40
  ("regnet_y_400mf",          "RegNet_Y_400MF_Weights.IMAGENET1K_V2"),                # Acc@1: 75.804, Acc@5: 92.742, Params: 4.3M,    GFLOPS: 0.40
  ("regnet_y_800mf",          "RegNet_Y_800MF_Weights.IMAGENET1K_V1"),                # Acc@1: 76.420, Acc@5: 93.136, Params: 6.4M,    GFLOPS: 0.83
  ("regnet_y_800mf",          "RegNet_Y_800MF_Weights.IMAGENET1K_V2"),                # Acc@1: 78.828, Acc@5: 94.502, Params: 6.4M,    GFLOPS: 0.83
  ("regnet_y_8gf",            "RegNet_Y_8GF_Weights.IMAGENET1K_V1"),                  # Acc@1: 80.032, Acc@5: 95.048, Params: 39.4M,   GFLOPS: 8.47
  ("regnet_y_8gf",            "RegNet_Y_8GF_Weights.IMAGENET1K_V2"),                  # Acc@1: 82.828, Acc@5: 96.330, Params: 39.4M,   GFLOPS: 8.47

  ("resnext101_32x8d",        "ResNeXt101_32X8D_Weights.IMAGENET1K_V1"),              # Acc@1: 79.312, Acc@5: 94.526, Params: 88.8M,   GFLOPS: 16.41
  ("resnext101_32x8d",        "ResNeXt101_32X8D_Weights.IMAGENET1K_V2"),              # Acc@1: 82.834, Acc@5: 96.228, Params: 88.8M,   GFLOPS: 16.41
  ("resnext101_64x4d",        "ResNeXt101_64X4D_Weights.IMAGENET1K_V1"),              # Acc@1: 83.246, Acc@5: 96.454, Params: 83.5M,   GFLOPS: 15.46
  ("resnext50_32x4d",         "ResNeXt50_32X4D_Weights.IMAGENET1K_V1"),               # Acc@1: 77.618, Acc@5: 93.698, Params: 25.0M,   GFLOPS: 4.23
  ("resnext50_32x4d",         "ResNeXt50_32X4D_Weights.IMAGENET1K_V2"),               # Acc@1: 81.198, Acc@5: 95.340, Params: 25.0M,   GFLOPS: 4.23

  ("resnet101",               "ResNet101_Weights.IMAGENET1K_V1"),                     # Acc@1: 77.374, Acc@5: 93.546, Params: 44.5M,   GFLOPS: 7.80
  ("resnet101",               "ResNet101_Weights.IMAGENET1K_V2"),                     # Acc@1: 81.886, Acc@5: 95.780, Params: 44.5M,   GFLOPS: 7.80
  ("resnet152",               "ResNet152_Weights.IMAGENET1K_V1"),                     # Acc@1: 78.312, Acc@5: 94.046, Params: 60.2M,   GFLOPS: 11.51
  ("resnet152",               "ResNet152_Weights.IMAGENET1K_V2"),                     # Acc@1: 82.284, Acc@5: 96.002, Params: 60.2M,   GFLOPS: 11.51
  ("resnet18",                "ResNet18_Weights.IMAGENET1K_V1"),                      # Acc@1: 69.758, Acc@5: 89.078, Params: 11.7M,   GFLOPS: 1.81
  ("resnet34",                "ResNet34_Weights.IMAGENET1K_V1"),                      # Acc@1: 73.314, Acc@5: 91.420, Params: 21.8M,   GFLOPS: 3.66
  ("resnet50",                "ResNet50_Weights.IMAGENET1K_V1"),                      # Acc@1: 76.130, Acc@5: 92.862, Params: 25.6M,   GFLOPS: 4.09
  ("resnet50",                "ResNet50_Weights.IMAGENET1K_V2"),                      # Acc@1: 80.858, Acc@5: 95.434, Params: 25.6M,   GFLOPS: 4.09

  ("shufflenet_v2_x0_5",      "ShuffleNet_V2_X0_5_Weights.IMAGENET1K_V1"),            # Acc@1: 60.552, Acc@5: 81.746, Params: 1.4M,    GFLOPS: 0.04
  ("shufflenet_v2_x1_0",      "ShuffleNet_V2_X1_0_Weights.IMAGENET1K_V1"),            # Acc@1: 69.362, Acc@5: 88.316, Params: 2.3M,    GFLOPS: 0.14
  ("shufflenet_v2_x1_5",      "ShuffleNet_V2_X1_5_Weights.IMAGENET1K_V1"),            # Acc@1: 72.996, Acc@5: 91.086, Params: 3.5M,    GFLOPS: 0.30
  ("shufflenet_v2_x2_0",      "ShuffleNet_V2_X2_0_Weights.IMAGENET1K_V1"),            # Acc@1: 76.230, Acc@5: 93.006, Params: 7.4M,    GFLOPS: 0.58

  ("squeezenet1_0",           "SqueezeNet1_0_Weights.IMAGENET1K_V1"),                 # Acc@1: 58.092, Acc@5: 80.420, Params: 1.2M,    GFLOPS: 0.82
  ("squeezenet1_1",           "SqueezeNet1_1_Weights.IMAGENET1K_V1"),                 # Acc@1: 58.178, Acc@5: 80.624, Params: 1.2M,    GFLOPS: 0.35

  ("swin_b",                  "Swin_B_Weights.IMAGENET1K_V1"),                        # Acc@1: 83.582, Acc@5: 96.640, Params: 87.8M,   GFLOPS: 15.43
  ("swin_s",                  "Swin_S_Weights.IMAGENET1K_V1"),                        # Acc@1: 83.196, Acc@5: 96.360, Params: 49.6M,   GFLOPS: 8.74
  ("swin_t",                  "Swin_T_Weights.IMAGENET1K_V1"),                        # Acc@1: 81.474, Acc@5: 95.776, Params: 28.3M,   GFLOPS: 4.49
  ("swin_v2_b",               "Swin_V2_B_Weights.IMAGENET1K_V1"),                     # Acc@1: 84.112, Acc@5: 96.864, Params: 87.9M,   GFLOPS: 20.32
  ("swin_v2_s",               "Swin_V2_S_Weights.IMAGENET1K_V1"),                     # Acc@1: 83.712, Acc@5: 96.816, Params: 49.7M,   GFLOPS: 11.55
  ("swin_v2_t",               "Swin_V2_T_Weights.IMAGENET1K_V1"),                     # Acc@1: 82.072, Acc@5: 96.132, Params: 28.4M,   GFLOPS: 5.94

  ("vgg11_bn",                "VGG11_BN_Weights.IMAGENET1K_V1"),                      # Acc@1: 70.370, Acc@5: 89.810, Params: 132.9M,  GFLOPS: 7.61
  ("vgg11",                   "VGG11_Weights.IMAGENET1K_V1"),                         # Acc@1: 69.020, Acc@5: 88.628, Params: 132.9M,  GFLOPS: 7.61
  ("vgg13_bn",                "VGG13_BN_Weights.IMAGENET1K_V1"),                      # Acc@1: 71.586, Acc@5: 90.374, Params: 133.1M,  GFLOPS: 11.31
  ("vgg13",                   "VGG13_Weights.IMAGENET1K_V1"),                         # Acc@1: 69.928, Acc@5: 89.246, Params: 133.0M,  GFLOPS: 11.31
  ("vgg16_bn",                "VGG16_BN_Weights.IMAGENET1K_V1"),                      # Acc@1: 73.360, Acc@5: 91.516, Params: 138.4M,  GFLOPS: 15.47
  ("vgg16",                   "VGG16_Weights.IMAGENET1K_V1"),                         # Acc@1: 71.592, Acc@5: 90.382, Params: 138.4M,  GFLOPS: 15.47
  ("vgg16",                   "VGG16_Weights.IMAGENET1K_FEATURES"),                   # Acc@1: nan,    Acc@5: nan,    Params: 138.4M,  GFLOPS: 15.47
  ("vgg19_bn",                "VGG19_BN_Weights.IMAGENET1K_V1"),                      # Acc@1: 74.218, Acc@5: 91.842, Params: 143.7M,  GFLOPS: 19.63
  ("vgg19",                   "VGG19_Weights.IMAGENET1K_V1"),                         # Acc@1: 72.376, Acc@5: 90.876, Params: 143.7M,  GFLOPS: 19.63

  ("vit_b_16",                "ViT_B_16_Weights.IMAGENET1K_V1"),                      # Acc@1: 81.072, Acc@5: 95.318, Params: 86.6M,   GFLOPS: 17.56
  ("vit_b_16",                "ViT_B_16_Weights.IMAGENET1K_SWAG_E2E_V1"),             # Acc@1: 85.304, Acc@5: 97.650, Params: 86.9M,   GFLOPS: 55.48
  ("vit_b_16",                "ViT_B_16_Weights.IMAGENET1K_SWAG_LINEAR_V1"),          # Acc@1: 81.886, Acc@5: 96.180, Params: 86.6M,   GFLOPS: 17.56
  ("vit_b_32",                "ViT_B_32_Weights.IMAGENET1K_V1"),                      # Acc@1: 75.912, Acc@5: 92.466, Params: 88.2M,   GFLOPS: 4.41
  ("vit_h_14",                "ViT_H_14_Weights.IMAGENET1K_SWAG_E2E_V1"),             # Acc@1: 88.552, Acc@5: 98.694, Params: 633.5M,  GFLOPS: 1016.72
  ("vit_h_14",                "ViT_H_14_Weights.IMAGENET1K_SWAG_LINEAR_V1"),          # Acc@1: 85.708, Acc@5: 97.730, Params: 632.0M,  GFLOPS: 167.29
  ("vit_l_16",                "ViT_L_16_Weights.IMAGENET1K_V1"),                      # Acc@1: 79.662, Acc@5: 94.638, Params: 304.3M,  GFLOPS: 61.55
  ("vit_l_16",                "ViT_L_16_Weights.IMAGENET1K_SWAG_E2E_V1"),             # Acc@1: 88.064, Acc@5: 98.512, Params: 305.2M,  GFLOPS: 361.99
  ("vit_l_16",                "ViT_L_16_Weights.IMAGENET1K_SWAG_LINEAR_V1"),          # Acc@1: 85.146, Acc@5: 97.422, Params: 304.3M,  GFLOPS: 61.55
  ("vit_l_32",                "ViT_L_32_Weights.IMAGENET1K_V1"),                      # Acc@1: 76.972, Acc@5: 93.070, Params: 306.5M,  GFLOPS: 15.38

  ("wide_resnet101_2",        "Wide_ResNet101_2_Weights.IMAGENET1K_V1"),              # Acc@1: 78.848, Acc@5: 94.284, Params: 126.9M,  GFLOPS: 22.75
  ("wide_resnet101_2",        "Wide_ResNet101_2_Weights.IMAGENET1K_V2"),              # Acc@1: 82.510, Acc@5: 96.020, Params: 126.9M,  GFLOPS: 22.75
  ("wide_resnet50_2",         "Wide_ResNet50_2_Weights.IMAGENET1K_V1"),               # Acc@1: 78.468, Acc@5: 94.086, Params: 68.9M,   GFLOPS: 11.40
  ("wide_resnet50_2",         "Wide_ResNet50_2_Weights.IMAGENET1K_V2"),               # Acc@1: 81.602, Acc@5: 95.758, Params: 68.9M,   GFLOPS: 11.40
]

# ===============================
# CLI: 요청대로 세 가지만 제어
# ===============================
def get_args():
    p = argparse.ArgumentParser(description="GDual training (only 3 overrides)")
    p.add_argument('--n_steps',    type=int, default=3)
    p.add_argument('--n_classifiers',    type=int, default=0)
    p.add_argument('--log_dir',    type=str, default=None, help="Override TensorBoard/log save dir")
    return p.parse_args()

args = get_args()

# ===============================
# Config (원문 유지 + 3가지만 덮어쓰기)
# ===============================
config = EasyDict()
config.backbone      = 'GMDiT'
config.batch_size    = 10
config.n_valid       = 100
config.CFG           = 1.4
config.latent_size   = (4, 32, 32)

# LR & Scheduler
config.base_lr       = 2e-3
config.end_lr        = 1e-4
config.total_steps   = 10*1000        # 전체 학습 스텝

# ---- 여기만 CLI로 덮어씀 ----
config.n_steps       = args.n_steps
config.log_dir       = args.log_dir or config.log_dir
config.n_classifiers = args.n_classifiers
# -----------------------------

# Loss
config.losses = ['classifier']
config.main_loss = 'classifier'

os.makedirs(config.log_dir, exist_ok=True)

# ===============================
# Model (frozen)
# ===============================
from backbones.gmdit import GMDiT
from utils.general_classifier import Classifier

if config.backbone == 'GMDiT':
    model = GMDiT(trainable=True)  # 내부 구현에 맞춰 유지
    model.set_freeze()
device = model.device
print(model)

if 'classifier' in config.losses:
    arch, weights = CLASSIFIER_MODELS[config.n_classifiers]
    classifier = Classifier(
            arch=arch,
            weights=weights,
        ).to(device)
    
print('done')

# ===============================
# Solver / Optimizer / Scheduler
# ===============================
from solvers.taylor.solver.gdual_solver import GDual_Solver
from solvers.taylor.transform.logaffine_transform import LogAffineTransform
from solvers.taylor.extractor.table_extractor import Extractor

noise_schedule = model.get_noise_schedule()
extractor = Extractor(steps=config.n_steps)
transform = LogAffineTransform(gamma_push=True, gamma_max=2, tau_offset=1, kappa_max=2, eps=1e-2)
solver = GDual_Solver(
    noise_schedule,
    steps=config.n_steps,
    transform=transform,
    param_extractor=extractor,
    skip_type="time_uniform_flow",
    flow_shift=1.0,
    pred_order=1,
    corr_order=2,
    order1_kappa=True,
    order2_kappa=True,
    use_corrector=True,
    time_learning=True,
    train_mode=True,
    checkpoint=False
).to(device)

optimizer = torch.optim.AdamW(solver.parameters(), lr=config.base_lr)

# ---- Scheduler: Pure Cosine ----
from torch.optim.lr_scheduler import CosineAnnealingLR

scheduler = CosineAnnealingLR(
    optimizer,
    T_max=config.total_steps,   # 전체 스텝에 걸쳐 한 번의 코사인
    eta_min=config.end_lr
)

print('solver/optimizer')

# ===============================
# Utils
# ===============================
def abort_if_bad(tag, value, step=None):
    v = float(value.detach().cpu()) if isinstance(value, torch.Tensor) else float(value)
    if (not math.isfinite(v)) or (v >= 100.0):
        msg = f"[EARLY-STOP] {tag} loss={v:.6f}" + (f" @ step {step}" if step is not None else "")
        print(msg, flush=True)
        raise RuntimeError(msg)

def save_checkpoint(global_step, save_dir, solver, optimizer):
    ckpt = {
        "global_step": int(global_step),
        "optim_state_dict": optimizer.state_dict(),
        "solver_state_dict": solver.state_dict(),
        "config": dict(config),
    }
    os.makedirs(save_dir, exist_ok=True)
    step_path = os.path.join(save_dir, f"step_{global_step:08d}.pt")
    torch.save(ckpt, step_path)
    return step_path


def get_classifier_loss(raw_output, targets):
    loss = classifier(raw_output, targets=targets)['loss']
    return loss

@torch.no_grad()
def get_valid_loss(valid_noises, valid_conds, device, solver):
    solver.eval()
    losses = []
    start = 0
    while True:
        if start >= len(valid_noises):
            break
        noises = valid_noises[start:min(start+config.batch_size, len(valid_noises))]
        conds = valid_conds[start:min(start+config.batch_size, len(valid_noises))]
        start += config.batch_size
        model_fn = model.get_model_fn(noise_schedule, pos_conds=conds, guidance_scale=config.CFG)
        with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
            latent_pred = solver.sample(noises, model_fn)['samples']
            if 'classifier' in config.losses:
                outputs = model.decode_vae(latent_pred, raw_output=True)
                loss = get_classifier_loss(outputs['raw_output'], targets=conds)
                losses.append(loss.item())

    return np.mean(losses)


def do_train_loop(device, writer, solver, optimizer, global_step):
    solver.train()
    pbar = tqdm(range(1000))
    
    for _, batch in enumerate(pbar):
        if global_step >= config.total_steps:
            break

        optimizer.zero_grad(set_to_none=True)
        if config.main_loss == 'classifier':
            noises = torch.randn(config.batch_size, *config.latent_size).to(device, non_blocking=True)
            conds = torch.randint(0, 1000, size=(len(noises),)).to(device, non_blocking=True)
        
        model_fn = model.get_model_fn(noise_schedule, pos_conds=conds, guidance_scale=config.CFG)
        with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
            latent_pred = solver.sample(noises, model_fn)['samples']
            if 'classifier' in config.main_loss:
                outputs = model.decode_vae(latent_pred, raw_output=True)
                loss = get_classifier_loss(outputs['raw_output'], targets=conds)
                
        abort_if_bad("train", loss, global_step)  # ← 즉시 중단
        loss.backward()
        torch.nn.utils.clip_grad_norm_(solver.parameters(), 1.0)
        optimizer.step()
        scheduler.step()   # ← lr 업데이트 포인트
        lr_now = optimizer.param_groups[0]["lr"]
        pbar.set_postfix({'loss': loss.item(), 'lr': lr_now})
        global_step += 1
        
    return global_step


# ===============================
# Train
# ===============================

def set_seed(seed=42):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def main():
    set_seed()

    writer = SummaryWriter(config.log_dir)
    print('tensorboard:', config.log_dir)

    global_step = 0
    valid_noises = torch.randn(config.n_valid, *config.latent_size).to(device, non_blocking=True)
    valid_conds = torch.randint(0, 1000, size=(len(valid_noises),)).to(device, non_blocking=True)
    while True:
        loss = get_valid_loss(valid_noises, valid_conds, device, solver)
        writer.add_scalar('valid_loss', loss, global_step)
        save_checkpoint(global_step, config.log_dir, solver, optimizer) 

        if global_step >= config.total_steps:
            break
        
        global_step = do_train_loop(device, writer, solver, optimizer, global_step)

    print('E-N-D')
    
if __name__ == "__main__":
    main()
