import torch

from compression.codecs.blocks.layers import DecoderTopLeftCornerLayer, DecoderTopRightCornerLayer
from compression.codecs.blocks.layers import DecoderBottomLeftCornerLayer, DecoderBottomRightCornerLayer
from compression.codecs.blocks.layers import DecoderTopEdgeLayer, DecoderBottomEdgeLayer
from compression.codecs.blocks.layers import DecoderLeftEdgeLayer, DecoderRightEdgeLayer
from compression.codecs.blocks.layers import DecoderMiddleLayer

from compression.codecs.blocks.layers.base import ConvolutionLayer


class DecoderPreProcessingBlock(torch.nn.Module):
    def __init__(self, in_channels, num_filters, kernel_size,):
        super(DecoderPreProcessingBlock, self).__init__()

        self.convolution_layer = ConvolutionLayer(
            in_channels=in_channels,
            num_filters=num_filters,
            kernel_size=kernel_size,
            stride=(1, 1)
        )

    def forward(self, input_tensor):
        convolution_output = self.convolution_layer(input_tensor)

        return convolution_output


class DecoderBlock(torch.nn.Module):
    def __init__(self, num_columns, num_rows, in_channels, kernel_size,):
        super(DecoderBlock, self).__init__()

        self.num_columns = num_columns
        self.num_rows = num_rows

        self.in_channels = in_channels
        self.kernel_size = kernel_size

        self.list_layers = torch.nn.ModuleList()

        for column in range(self.num_columns):
            column_list = torch.nn.ModuleList()
            if column == 0:
                # if its the first column
                column_list.append(
                        DecoderBottomLeftCornerLayer(
                            in_channels=in_channels,
                            kernel_size=kernel_size,
                        )
                )
                for row in range(int(self.num_rows - 2)):
                    column_list.append(
                        DecoderLeftEdgeLayer(
                            in_channels=in_channels,
                            kernel_size=kernel_size,
                        )
                    )
                column_list.append(
                    DecoderTopLeftCornerLayer(
                        in_channels=in_channels,
                        kernel_size=kernel_size,
                    )
                )
            elif column == (self.num_columns - 1):
                # if its the last column
                column_list.append(
                    DecoderBottomRightCornerLayer(
                        in_channels=in_channels,
                        kernel_size=kernel_size,
                    )
                )
                for row in range(int(self.num_rows - 2)):
                    column_list.append(
                        DecoderRightEdgeLayer(
                            in_channels=in_channels,
                            kernel_size=kernel_size,
                        )
                    )
                column_list.append(
                    DecoderTopRightCornerLayer(
                        in_channels=in_channels,
                        kernel_size=kernel_size,
                    )
                )
            else:
                # if its one of the middle columns
                column_list.append(
                    DecoderBottomEdgeLayer(
                        in_channels=in_channels,
                        kernel_size=kernel_size,
                    )
                )
                for row in range(int(self.num_rows - 2)):
                    column_list.append(
                        DecoderMiddleLayer(
                            in_channels=in_channels,
                            kernel_size=kernel_size,
                        )
                    )
                column_list.append(
                    DecoderTopEdgeLayer(
                        in_channels=in_channels,
                        kernel_size=kernel_size,
                    )
                )

            self.list_layers.append(column_list)

    def forward(self, input_tensor):
        # layer outputs are always in the following order: process, up, down
        list_column_outputs = [[]] * self.num_rows
        for column_index, column in enumerate(self.list_layers):
            # First Column
            if column_index == 0:
                for layer_index, layer in enumerate(column):
                    # bottom left corner
                    if layer_index == 0:
                        layer_output = layer(input_tensor)
                    # the rest of the left side including bottom left corner for they all take previous down as input
                    else:
                        layer_output = layer(
                            list_column_outputs[layer_index-1][1]
                        )
                    list_column_outputs[layer_index] = layer_output
            # Last Column
            elif column_index == (len(self.list_layers) - 1):
                for layer_index, layer in enumerate(column):
                    # bottom right Corner
                    if layer_index == 0:
                        # inputs: processed, up
                        layer_output = layer(
                            list_column_outputs[layer_index][0],
                            list_column_outputs[layer_index+1][-1]
                        )
                    # top right corner
                    elif layer_index == (len(column) - 1):
                        layer_output = layer(
                            list_column_outputs[layer_index][0],
                            list_column_outputs[layer_index-1][1]
                        )
                    # right edge
                    else:
                        layer_output = layer(
                            list_column_outputs[layer_index][0],
                            list_column_outputs[layer_index - 1][1],
                            list_column_outputs[layer_index + 1][-1],
                        )
                    list_column_outputs[layer_index] = layer_output
            else:
                for layer_index, layer in enumerate(column):
                    # bottom edge
                    if layer_index == 0:
                        layer_output = layer(
                            list_column_outputs[layer_index][0],
                            list_column_outputs[layer_index+1][-1],
                        )
                    # top edge
                    elif layer_index == (len(column) - 1):
                        layer_output = layer(
                            list_column_outputs[layer_index][0],
                            list_column_outputs[layer_index-1][1],
                        )
                    # middle edge
                    else:
                        layer_output = layer(
                            list_column_outputs[layer_index][0],
                            list_column_outputs[layer_index - 1][1],
                            list_column_outputs[layer_index + 1][-1],
                        )
                    list_column_outputs[layer_index] = layer_output

        return list_column_outputs[-1][0]


class DecoderPostProcessingBlock(torch.nn.Module):
    def __init__(self, in_channels, num_filters, reconstructed_image_num_channels, kernel_size,):
        super(DecoderPostProcessingBlock, self).__init__()

        self.feature_extraction_layer = ConvolutionLayer(
            in_channels=in_channels,
            num_filters=num_filters,
            kernel_size=kernel_size,
            stride=(1, 1)
        )

        self.convolution_layer = ConvolutionLayer(
            in_channels=num_filters,
            num_filters=reconstructed_image_num_channels,
            kernel_size=kernel_size,
            stride=(1, 1)
        )

    def forward(self, input_tensor):
        feature_extraction_output = self.feature_extraction_layer(
            input_tensor
        )

        restored_image = self.convolution_layer(
            feature_extraction_output
        )

        return restored_image


