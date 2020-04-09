import torch

from compression.codecs.blocks.layers.base.layers import ProcessingLayer, UpSampleLayer, DownSampleLayer, SummedInputLayer


class BottomLeftCornerLayer(torch.nn.Module):
    def __init__(self, in_channels, kernel_size,):
        super(BottomLeftCornerLayer, self).__init__()

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


class TopLeftCornerLayer(torch.nn.Module):
    def __init__(self, in_channels, kernel_size,):
        super(TopLeftCornerLayer, self).__init__()

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


class BottomRightCornerLayer(torch.nn.Module):
    def __init__(self, in_channels, kernel_size,):
        super(BottomRightCornerLayer, self).__init__()

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


class TopRightCornerLayer(torch.nn.Module):
    def __init__(self, in_channels, kernel_size,):
        super(TopRightCornerLayer, self).__init__()

        self.input_layer = SummedInputLayer()
        self.processing_layer = ProcessingLayer(
            in_channels=in_channels,
            kernel_size=kernel_size,
        )

    def forward(self, previous_processed_output, previous_up_sampled_output):
        summed_input = self.input_layer([previous_processed_output, previous_up_sampled_output])
        processed_output = self.processing_layer(summed_input)

        return tuple((processed_output, ))


class LeftEdgeLayer(torch.nn.Module):
    def __init__(self, in_channels, kernel_size,):
        super(LeftEdgeLayer, self).__init__()

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


class TopEdgeLayer(torch.nn.Module):
    def __init__(self, in_channels, kernel_size,):
        super(TopEdgeLayer, self).__init__()

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


class RightEdgeLayer(torch.nn.Module):
    def __init__(self, in_channels, kernel_size,):
        super(RightEdgeLayer, self).__init__()

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

        return tuple((processed_output, up_sampled_output))


class BottomEdgeLayer(torch.nn.Module):
    def __init__(self, in_channels, kernel_size,):
        super(BottomEdgeLayer, self).__init__()

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


class MiddleLayer(torch.nn.Module):
    def __init__(self, in_channels, kernel_size,):
        super(MiddleLayer, self).__init__()

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