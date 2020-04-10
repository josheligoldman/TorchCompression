import torch

from compression.codecs.blocks.layers.base import ProcessingLayer, UpSampleLayer, DownSampleLayer, SummedInputLayer


class DecoderBottomLeftCornerLayer(torch.nn.Module):
    def __init__(self, in_channels, kernel_size,):
        super(DecoderBottomLeftCornerLayer, self).__init__()

        self.processing_layer = ProcessingLayer(
            in_channels=in_channels,
            kernel_size=kernel_size,
        )
        self.up_sample_layer = UpSampleLayer(
            in_channels=in_channels,
            kernel_size=kernel_size,
        )

    def forward(self, input_tensor):
        processed_output = self.processing_layer(input_tensor)
        up_sampled_output = self.up_sample_layer(processed_output)

        return tuple((processed_output, up_sampled_output))


class DecoderTopLeftCornerLayer(torch.nn.Module):
    def __init__(self, in_channels, kernel_size,):
        super(DecoderTopLeftCornerLayer, self).__init__()

        self.processing_layer = ProcessingLayer(
            in_channels=in_channels,
            kernel_size=kernel_size,
        )
        self.down_sample_layer = DownSampleLayer(
            in_channels=in_channels,
            kernel_size=kernel_size,
        )

    def forward(self, previous_up_sampled_output):
        processed_output = self.processing_layer(previous_up_sampled_output)
        down_sampled_output = self.down_sample_layer(processed_output)

        return tuple((processed_output, down_sampled_output))


class DecoderBottomRightCornerLayer(torch.nn.Module):
    def __init__(self, in_channels, kernel_size,):
        super(DecoderBottomRightCornerLayer, self).__init__()

        self.input_layer = SummedInputLayer()
        self.processing_layer = ProcessingLayer(
            in_channels=in_channels,
            kernel_size=kernel_size,
        )
        self.up_sample_layer = UpSampleLayer(
            in_channels=in_channels,
            kernel_size=kernel_size,
        )

    def forward(self, previous_processed_output, previous_down_sampled_output):
        summed_input = self.input_layer(
            [previous_processed_output, previous_down_sampled_output]
        )
        processed_output = self.processing_layer(summed_input)
        up_sampled_output = self.up_sample_layer(processed_output)

        return tuple((processed_output, up_sampled_output))


class DecoderTopRightCornerLayer(torch.nn.Module):
    def __init__(self, in_channels, kernel_size,):
        super(DecoderTopRightCornerLayer, self).__init__()

        self.input_layer = SummedInputLayer()
        self.processing_layer = ProcessingLayer(
            in_channels=in_channels,
            kernel_size=kernel_size,
        )

    def forward(self, previous_processed_output, previous_up_sampled_output):
        summed_input = self.input_layer([previous_processed_output, previous_up_sampled_output])
        processed_output = self.processing_layer(summed_input)

        return tuple((processed_output, ))


class DecoderLeftEdgeLayer(torch.nn.Module):
    def __init__(self, in_channels, kernel_size,):
        super(DecoderLeftEdgeLayer, self).__init__()

        self.processing_layer = ProcessingLayer(
            in_channels=in_channels,
            kernel_size=kernel_size,
        )
        self.up_sample_layer = UpSampleLayer(
            in_channels=in_channels,
            kernel_size=kernel_size,
        )
        self.down_sample_layer = UpSampleLayer(
            in_channels=in_channels,
            kernel_size=kernel_size,
        )

    def forward(self, previous_up_sampled_output):
        processed_output = self.processing_layer(previous_up_sampled_output)
        up_sampled_output = self.up_sample_layer(processed_output)
        down_sampled_output = self.down_sample_layer(processed_output)

        return tuple((processed_output, up_sampled_output, down_sampled_output))


class DecoderTopEdgeLayer(torch.nn.Module):
    def __init__(self, in_channels, kernel_size,):
        super(DecoderTopEdgeLayer, self).__init__()

        self.input_layer = SummedInputLayer()
        self.processing_layer = ProcessingLayer(
            in_channels=in_channels,
            kernel_size=kernel_size,
        )
        self.down_sample_layer = DownSampleLayer(
            in_channels=in_channels,
            kernel_size=kernel_size,
        )

    def forward(self, previous_processed_output, previous_up_sampled_output):
        summed_input = self.input_layer(
            [previous_processed_output, previous_up_sampled_output]
        )
        processed_output = self.processing_layer(summed_input)
        down_sampled_output = self.down_sample_layer(processed_output)

        return tuple((processed_output, down_sampled_output))


class DecoderRightEdgeLayer(torch.nn.Module):
    def __init__(self, in_channels, kernel_size,):
        super(DecoderRightEdgeLayer, self).__init__()

        self.input_layer = SummedInputLayer()
        self.processing_layer = ProcessingLayer(
            in_channels=in_channels,
            kernel_size=kernel_size,
        )
        self.up_sample_layer = UpSampleLayer(
            in_channels=in_channels,
            kernel_size=kernel_size,
        )

    def forward(self, previous_processed_output, previous_up_sampled_output, previous_down_sampled_output):
        summed_input = self.input_layer(
            [previous_processed_output, previous_up_sampled_output, previous_down_sampled_output]
        )
        processed_output = self.processing_layer(summed_input)
        up_sampled_output = self.up_sample_layer(processed_output)

        print("Dec Right Shape", processed_output.shape)

        return tuple((processed_output, up_sampled_output))


class DecoderBottomEdgeLayer(torch.nn.Module):
    def __init__(self, in_channels, kernel_size,):
        super(DecoderBottomEdgeLayer, self).__init__()

        self.input_layer = SummedInputLayer()
        self.processing_layer = ProcessingLayer(
            in_channels=in_channels,
            kernel_size=kernel_size,
        )
        self.up_sample_layer = UpSampleLayer(
            in_channels=in_channels,
            kernel_size=kernel_size,
        )

    def forward(self, previous_processed_output, previous_down_sampled_output):
        summed_input = self.input_layer(
            [previous_processed_output, previous_down_sampled_output]
        )
        processed_output = self.processing_layer(summed_input)
        up_sampled_output = self.up_sample_layer(processed_output)

        return tuple((processed_output, up_sampled_output))


class DecoderMiddleLayer(torch.nn.Module):
    def __init__(self, in_channels, kernel_size,):
        super(DecoderMiddleLayer, self).__init__()

        self.in_channels = in_channels
        self.kernel_size = kernel_size

        self.input_layer = SummedInputLayer()
        self.processing_layer = ProcessingLayer(
            in_channels=in_channels,
            kernel_size=kernel_size,
        )
        self.up_sample_layer = UpSampleLayer(
            in_channels=in_channels,
            kernel_size=kernel_size,
        )
        self.down_sample_layer = DownSampleLayer(
            in_channels=in_channels,
            kernel_size=kernel_size,
        )

    def forward(self, previous_processed_output, previous_up_sampled_output, previous_down_sampled_output):
        summed_input = self.input_layer(
            [previous_processed_output, previous_up_sampled_output, previous_down_sampled_output]
        )

        processed_output = self.processing_layer(summed_input)
        up_sampled_output = self.up_sample_layer(processed_output)
        down_sampled_output = self.down_sample_layer(processed_output)

        return tuple((processed_output, up_sampled_output, down_sampled_output))