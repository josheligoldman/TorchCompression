import torch

from compression.codecs import Encoder, Decoder


class Model(torch.nn.Module):
    def __init__(self, config):
        super(Model, self).__init__()

        self.config = config

        self.encoder = Encoder(
            num_columns=self.config.num_columns_encoder,
            num_rows=self.config.num_rows_encoder,
            in_channels=self.config.num_input_image_channels,
            num_filters=self.config.num_filters_encoder,
            kernel_size=self.config.kernel_size_encoder,
            latent_space_num_channels=self.config.latent_space_num_channels
        )

        self.decoder = Decoder(
            num_columns=self.config.num_columns_decoder,
            num_rows=self.config.num_rows_decoder,
            in_channels=self.config.latent_space_num_channels,
            num_filters=self.config.num_filters_decoder,
            kernel_size=self.config.kernel_size_decoder,
            num_final_feature_maps=self.config.num_final_feature_maps,
            reconstructed_image_num_channels=self.config.reconstructed_image_num_channels,
        )

    def forward(self, input_image):
        encoded = self.encoder(input_image)
        decoded = self.decoder(encoded)

        return decoded



