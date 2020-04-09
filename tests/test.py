import pathlib
import os
import datetime

import torch
import numpy
import cv2

from compression import Model, ModelConfig

class Main:
    def __init__(self, is_cuda):
        self.is_cuda = is_cuda

        self.model_config = ModelConfig(
            ### Encoder ###
            num_columns_encoder=5,
            num_rows_encoder=5,
            num_input_image_channels=3,
            num_filters_encoder=32,
            kernel_size_encoder=3,
            latent_space_num_channels=3,
            ### Decoder ###
            num_columns_decoder=5,
            num_rows_decoder=5,
            num_filters_decoder=32,
            kernel_size_decoder=3,
            num_final_feature_maps=256,
            reconstructed_image_num_channels=3,
        )

        self.model = Model(self.model_config)

        if self.is_cuda:
            self.model.cuda()

    def summary(self,):
        print("Model Summary")
        num_total_params = sum(p.numel() for p in self.model.parameters())
        num_trainable_params = num_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        num_non_trainable_params = num_total_params - num_trainable_params
        print("Total Params:", num_total_params)
        print("Trainable Params:", num_trainable_params)
        print("Non Trainable Params:", num_non_trainable_params)

    def model_test(self):
        test_input = torch.zeros(
            (1, 3, 512, 512)
        )
        test_output = self.model(test_input)
        print("Output Shape:", test_output.shape)


main = Main(
    is_cuda=False
)

main.summary()
main.model_test()
