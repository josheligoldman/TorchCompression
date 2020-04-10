import torch

from compression.codecs.blocks.layers.base import ProcessingLayer, UpSampleLayer, DownSampleLayer, SummedInputLayer


class EncoderTopLeftCornerLayer(torch.nn.Module):
    def __init__(self, in_channels, kernel_size,):
        super(EncoderTopLeftCornerLayer, self).__init__()

        self.processing_layer = ProcessingLayer(
            in_channels=in_channels,
            kernel_size=kernel_size,
        )
        self.down_sample_layer = DownSampleLayer(
            in_channels=in_channels,
            kernel_size=kernel_size,
        )

    def forward(self, input_tensor):
        processed_output = self.processing_layer(input_tensor)
        down_sampled_output = self.down_sample_layer(processed_output)

        return tuple((processed_output, down_sampled_output))


class EncoderBottomLeftCornerLayer(torch.nn.Module):
    def __init__(self, in_channels, kernel_size,):
        super(EncoderBottomLeftCornerLayer, self).__init__()

        self.processing_layer = ProcessingLayer(
            in_channels=in_channels,
            kernel_size=kernel_size,
        )
        self.up_sample_layer = UpSampleLayer(
            in_channels=in_channels,
            kernel_size=kernel_size,
        )

    def forward(self, previous_down_sampled_output):
        processed_output = self.processing_layer(previous_down_sampled_output)
        up_sampled_output = self.up_sample_layer(processed_output)

        return tuple((processed_output, up_sampled_output))


class EncoderTopRightCornerLayer(torch.nn.Module):
    def __init__(self, in_channels, kernel_size,):
        super(EncoderTopRightCornerLayer, self).__init__()

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
        summed_input = self.input_layer([previous_processed_output, previous_up_sampled_output])
        processed_output = self.processing_layer(summed_input)
        down_sampled_output = self.down_sample_layer(processed_output)

        return tuple((processed_output, down_sampled_output))


class EncoderBottomRightCornerLayer(torch.nn.Module):
    def __init__(self, in_channels, kernel_size,):
        super(EncoderBottomRightCornerLayer, self).__init__()

        self.input_layer = SummedInputLayer()
        self.processing_layer = ProcessingLayer(
            in_channels=in_channels,
            kernel_size=kernel_size,
        )

    def forward(self, previous_processed_output, previous_down_sampled_output):
        summed_input = self.input_layer(
            [previous_processed_output, previous_down_sampled_output]
        )
        processed_output = self.processing_layer(summed_input)

        return tuple((processed_output, ))


class EncoderLeftEdgeLayer(torch.nn.Module):
    def __init__(self, in_channels, kernel_size,):
        super(EncoderLeftEdgeLayer, self).__init__()

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

    def forward(self, previous_down_sampled_output):
        processed_output = self.processing_layer(previous_down_sampled_output)
        up_sampled_output = self.up_sample_layer(processed_output)
        down_sampled_output = self.down_sample_layer(processed_output)

        return tuple((processed_output, up_sampled_output, down_sampled_output))


class EncoderTopEdgeLayer(torch.nn.Module):
    def __init__(self, in_channels, kernel_size,):
        super(EncoderTopEdgeLayer, self).__init__()

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


class EncoderRightEdgeLayer(torch.nn.Module):
    def __init__(self, in_channels, kernel_size,):
        super(EncoderRightEdgeLayer, self).__init__()

        self.input_layer = SummedInputLayer()
        self.processing_layer = ProcessingLayer(
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
        down_sampled_output = self.down_sample_layer(processed_output)

        print("Enc Right Shape", processed_output.shape)

        return tuple((processed_output, down_sampled_output))


class EncoderBottomEdgeLayer(torch.nn.Module):
    def __init__(self, in_channels, kernel_size,):
        super(EncoderBottomEdgeLayer, self).__init__()

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


class EncoderMiddleLayer(torch.nn.Module):
    def __init__(self, in_channels, kernel_size,):
        super(EncoderMiddleLayer, self).__init__()

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