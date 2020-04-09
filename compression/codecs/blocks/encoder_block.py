import torch


from compression.codecs.blocks.layers import EncoderTopLeftCornerLayer, EncoderTopRightCornerLayer
from compression.codecs.blocks.layers import EncoderBottomLeftCornerLayer, EncoderBottomRightCornerLayer
from compression.codecs.blocks.layers import EncoderTopEdgeLayer, EncoderBottomEdgeLayer
from compression.codecs.blocks.layers import EncoderLeftEdgeLayer, EncoderRightEdgeLayer
from compression.codecs.blocks.layers import EncoderMiddleLayer

from compression.codecs.blocks.layers.base import ConvolutionLayer


class EncoderPreProcessingBlock(torch.nn.Module):
    def __init__(self, in_channels, num_filters, kernel_size,):
        super(EncoderPreProcessingBlock, self).__init__()

        self.convolution_layer = ConvolutionLayer(
            in_channels=in_channels,
            num_filters=num_filters,
            kernel_size=kernel_size,
            stride=(1, 1)
        )

    def forward(self, input_tensor):
        convolution_output = self.convolution_layer(input_tensor)

        return convolution_output


class EncoderBlock(torch.nn.Module):
    def __init__(self, num_columns, num_rows, in_channels, kernel_size,):
        super(EncoderBlock, self).__init__()

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
                    EncoderTopLeftCornerLayer(
                        in_channels=in_channels,
                        kernel_size=kernel_size,
                    )
                )
                for row in range(int(self.num_rows - 2)):
                    column_list.append(
                        EncoderLeftEdgeLayer(
                            in_channels=in_channels,
                            kernel_size=kernel_size,
                        )
                    )
                column_list.append(
                    EncoderBottomLeftCornerLayer(
                        in_channels=in_channels,
                        kernel_size=kernel_size,
                    )
                )
            elif column == (self.num_columns - 1):
                # if its the last column
                column_list.append(
                    EncoderTopRightCornerLayer(
                        in_channels=in_channels,
                        kernel_size=kernel_size,
                    )
                )
                for row in range(int(self.num_rows - 2)):
                    column_list.append(
                        EncoderRightEdgeLayer(
                            in_channels=in_channels,
                            kernel_size=kernel_size,
                        )
                    )
                column_list.append(
                    EncoderBottomRightCornerLayer(
                        in_channels=in_channels,
                        kernel_size=kernel_size,
                    )
                )
            else:
                # if its one of the middle columns
                column_list.append(
                    EncoderTopEdgeLayer(
                        in_channels=in_channels,
                        kernel_size=kernel_size,
                    )
                )
                for row in range(int(self.num_rows - 2)):
                    column_list.append(
                        EncoderMiddleLayer(
                            in_channels=in_channels,
                            kernel_size=kernel_size,
                        )
                    )
                column_list.append(
                    EncoderBottomEdgeLayer(
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
                    # top left corner
                    if layer_index == 0:
                        layer_output = layer(input_tensor)
                    # the rest of the left side including bottom left corner for they all take previous down as input
                    else:
                        layer_output = layer(
                            list_column_outputs[layer_index-1][-1]
                        )
                    list_column_outputs[layer_index] = layer_output
            # Last Column
            elif column_index == (len(self.list_layers) - 1):
                for layer_index, layer in enumerate(column):
                    # top right Corner
                    if layer_index == 0:
                        # inputs: processed, up
                        layer_output = layer(
                            list_column_outputs[layer_index][0],
                            list_column_outputs[layer_index+1][1]
                        )
                    # bottom right corner
                    elif layer_index == (len(column) - 1):
                        layer_output = layer(
                            list_column_outputs[layer_index][0],
                            list_column_outputs[layer_index-1][-1]
                        )
                    else:
                        layer_output = layer(
                            list_column_outputs[layer_index][0],
                            list_column_outputs[layer_index + 1][1],
                            list_column_outputs[layer_index - 1][-1],
                        )
                    list_column_outputs[layer_index] = layer_output
            else:
                for layer_index, layer in enumerate(column):
                    # top corner
                    if layer_index == 0:
                        layer_output = layer(
                            list_column_outputs[layer_index][0],
                            list_column_outputs[layer_index+1][1],
                        )
                    elif layer_index == (len(column) - 1):
                        layer_output = layer(
                            list_column_outputs[layer_index][0],
                            list_column_outputs[layer_index-1][-1],
                        )
                    else:
                        layer_output = layer(
                            list_column_outputs[layer_index][0],
                            list_column_outputs[layer_index + 1][1],
                            list_column_outputs[layer_index - 1][-1],
                        )
                    list_column_outputs[layer_index] = layer_output

        return list_column_outputs[-1][0]


class EncoderPostProcessingBlock(torch.nn.Module):
    def __init__(self, in_channels, num_filters, kernel_size,):
        super(EncoderPostProcessingBlock, self).__init__()

        self.convolution_layer = ConvolutionLayer(
            in_channels=in_channels,
            num_filters=num_filters,
            kernel_size=kernel_size,
            stride=(1, 1)
        )

    def forward(self, input_tensor):
        convolution_output = self.convolution_layer(input_tensor)

        return convolution_output

