import torch

from compression.codecs.blocks import EncoderPreProcessingBlock, EncoderBlock, EncoderPostProcessingBlock


class Encoder(torch.nn.Module):
    def __init__(self, num_columns, num_rows, in_channels, num_filters, kernel_size, latent_space_num_channels,):
        super(Encoder, self).__init__()

        self.pre_processing_layer = EncoderPreProcessingBlock(
            in_channels=in_channels,
            num_filters=num_filters,
            kernel_size=kernel_size,
        )

        self.block = EncoderBlock(
            num_columns=num_columns,
            num_rows=num_rows,
            in_channels=num_filters,
            kernel_size=kernel_size
        )

        self.post_processing_layer = EncoderPostProcessingBlock(
            in_channels=num_filters,
            num_filters=latent_space_num_channels,
            kernel_size=kernel_size,
        )

    def forward(self, input_tensor):
        pre_processed_output = self.pre_processing_layer(input_tensor)

        block_output = self.block(pre_processed_output)

        post_processed_output = self.post_processing_layer(block_output)

        return post_processed_output





