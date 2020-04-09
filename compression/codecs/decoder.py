import torch

from compression.codecs.blocks import DecoderPreProcessingBlock, DecoderBlock, DecoderPostProcessingBlock


class Decoder(torch.nn.Module):
    def __init__(self,
        num_columns,
        num_rows,
        in_channels,
        num_filters,
        kernel_size,
        num_final_feature_maps,
        reconstructed_image_num_channels
    ):
        super(Decoder, self).__init__()

        self.pre_processing_layer = DecoderPreProcessingBlock(
            in_channels=in_channels,
            num_filters=num_filters,
            kernel_size=kernel_size,
        )

        self.block = DecoderBlock(
            in_channels=num_filters,
            num_columns=num_columns,
            num_rows=num_rows,
            kernel_size=kernel_size
        )

        self.post_processing_layer = DecoderPostProcessingBlock(
            in_channels=num_filters,
            num_filters=num_final_feature_maps,
            reconstructed_image_num_channels=reconstructed_image_num_channels,
            kernel_size=kernel_size,
        )

    def forward(self, input_tensor):
        pre_processed_output = self.pre_processing_layer(input_tensor)

        block_output = self.block(pre_processed_output)

        post_processed_output = self.post_processing_layer(block_output)

        return post_processed_output
