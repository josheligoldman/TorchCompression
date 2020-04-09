import torch


class TanhActivation(torch.nn.Module):
    def __init__(self):
        super(TanhActivation, self).__init__()

        self.divisor = torch.tensor(3.0, dtype=torch.float32)
        self.coefficient = torch.tensor(3.0, dtype=torch.float32)

    def forward(self, input_tensor):
        activated = torch.tanh(
            input_tensor / self.divisor
        ) * self.coefficient

        return activated


class ConvolutionPreProcessing(torch.nn.Module):
    def __init__(self, in_channels):
        super(ConvolutionPreProcessing, self).__init__()

        self.pre_processing = torch.nn.Sequential(
            torch.nn.BatchNorm2d(in_channels),
            TanhActivation()
        )

    def forward(self, input_tensor):
        pre_processed = self.pre_processing(input_tensor)

        return pre_processed


class ConvolutionLayer(torch.nn.Module):
    def __init__(self, in_channels, num_filters, kernel_size, stride):
        super(ConvolutionLayer, self).__init__()

        self.convolution_layer = torch.nn.Sequential(
            ConvolutionPreProcessing(in_channels),
            torch.nn.Conv2d(
                in_channels=in_channels,
                out_channels=num_filters,
                kernel_size=kernel_size,
                stride=stride,
                padding=1,
            )
        )

    def forward(self, input_tensor):
        convolution_output = self.convolution_layer(
            input_tensor
        )

        return convolution_output


class TransposedConvolutionLayer(torch.nn.Module):
    def __init__(self, in_channels, num_filters, kernel_size, stride=(2, 2), **kwargs):
        super(TransposedConvolutionLayer, self).__init__()
        self.transposed_convolution_layer = torch.nn.Sequential(
            ConvolutionPreProcessing(in_channels),
            torch.nn.ConvTranspose2d(
                in_channels=in_channels,
                out_channels=num_filters,
                kernel_size=kernel_size,
                stride=stride,
                padding=1,
            )
        )

    def forward(self, input_tensor):
        transposed_convolution_output = self.transposed_convolution_layer(
            input_tensor
        )

        return transposed_convolution_output


class SummationLayer(torch.nn.Module):
    def __init__(self):
        super(SummationLayer, self).__init__()

    def forward(self, list_input_tensors):
        total_sum = list_input_tensors[0]
        for tensor_index in range(1, len(list_input_tensors)):
            total_sum += list_input_tensors[tensor_index]

        return total_sum


class ConcatenationLayer(torch.nn.Module):
    def __init__(self, axis=-1):
        super(ConcatenationLayer, self).__init__()

        self.axis = axis

    def forward(self, list_input_tensors):
        concatenated = torch.cat(
            list_input_tensors,
            dim=self.axis
        )

        return concatenated


class SliceLayer(torch.nn.Module):
    def __init__(self):
        super(SliceLayer, self).__init__()

    def forward(self, input_tensor, height, width):
        sliced = input_tensor[
            :,
            :
            : height,
            : width,
        ]

        return sliced


class GroupSliceLayer(torch.nn.Module):
    def __init__(self):
        super(GroupSliceLayer, self).__init__()

        self.slice_layer = SliceLayer()

    def forward(self, list_input_tensors):
        list_heights = [tensor.shape[2] for tensor in list_input_tensors]
        list_widths = [tensor.shape[3] for tensor in list_input_tensors]

        slice_height = min(list_heights)
        slice_width = min(list_widths)

        list_sliced_tensors = []
        for tensor in list_input_tensors:
            sliced_tensor = self.slice_layer(tensor, slice_height, slice_width)
            list_sliced_tensors.append(sliced_tensor)

        return list_sliced_tensors


class ConcatenatedConvolutionLayer(torch.nn.Module):
    def __init__(self, in_channels, num_filters, kernel_size):
        super(ConcatenatedConvolutionLayer, self).__init__()

        self.convolution_layer = ConvolutionLayer(
            in_channels=in_channels,
            num_filters=num_filters,
            stride=(1, 1),
            kernel_size=kernel_size,
        )

        self.concatenation_layer = ConcatenationLayer()

    def forward(self, input_tensor):
        convolution_layer_output = self.convolution_layer(input_tensor)

        concatenated = self.concatenation_layer(
            [input_tensor, convolution_layer_output]
        )

        return concatenated


class SummedConvolutionLayer(torch.nn.Module):
    def __init__(self, in_channels, num_filters, kernel_size,):
        super(SummedConvolutionLayer, self).__init__()

        self.convolution_layer = ConvolutionLayer(
            in_channels=in_channels,
            num_filters=num_filters,
            kernel_size=kernel_size,
            stride=(1, 1),
        )

        self.summation_layer = SummationLayer()

    def forward(self, input_tensor):
        convolution_layer_output = self.convolution_layer(input_tensor)

        summed = self.summation_layer(
            [input_tensor, convolution_layer_output]
        )

        return summed


class SplitLayer(torch.nn.Module):
    def __init__(self, num_splits, axis):
        super(SplitLayer, self).__init__()

        self.num_splits = num_splits
        self.axis = axis

    def forward(self, input_tensor):
        splits = torch.split(
            input_tensor,
            self.num_splits,
            dim=self.axis
        )

        return splits


class ResidualLayer(torch.nn.Module):
    def __init__(self, in_channels, kernel_size,):
        super(ResidualLayer, self).__init__()

        self.convolutions = torch.nn.Sequential(
            ConvolutionLayer(
                in_channels=in_channels,
                num_filters=in_channels,
                kernel_size=kernel_size,
                stride=(1, 1)
            ),
            ConvolutionLayer(
                in_channels=in_channels,
                num_filters=in_channels,
                kernel_size=kernel_size,
                stride=(1, 1)
            )
        )

        self.summation_layer = SummationLayer()

    def forward(self, input_tensor):
        convolution_output = self.convolutions(input_tensor)

        summed = self.summation_layer(
            [input_tensor, convolution_output]
        )

        return summed


class DenselyConnectedConvolutionLayer(torch.nn.Module):
    def __init__(self, in_channels, kernel_size,):
        super(DenselyConnectedConvolutionLayer, self).__init__()

        self.convolution_layer_1 = ConvolutionLayer(
            in_channels=in_channels,
            num_filters=in_channels,
            kernel_size=kernel_size,
            stride=(1, 1)
        )
        self.concatenation_layer_1 = ConcatenationLayer()

        self.convolution_layer_1 = ConvolutionLayer(
            in_channels=in_channels,
            num_filters=in_channels,
            kernel_size=kernel_size,
            stride=(1, 1)
        )
        self.concatenation_layer_2 = ConcatenationLayer()

        self.convolution_layer_1 = ConvolutionLayer(
            in_channels=in_channels,
            num_filters=in_channels,
            kernel_size=kernel_size,
            stride=(1, 1)
        )
        self.concatenation_layer_3 = ConcatenationLayer()

        self.convolution_layer_1 = ConvolutionLayer(
            in_channels=in_channels,
            num_filters=in_channels,
            kernel_size=kernel_size,
            stride=(1, 1)
        )
        self.concatenation_layer_4 = ConcatenationLayer()

    def forward(self, input_tensor):
        convolution_output_1 = self.convolution_layer_1(input_tensor)
        concat_1 = self.concatenation_layer_1(
            [input_tensor, convolution_output_1]
        )

        convolution_output_2 = self.convolution_layer_2(concat_1)
        concat_2 = self.concatenation_layer_2(
            [concat_1, convolution_output_2]
        )

        convolution_output_3 = self.convolution_layer_3(concat_2)
        concat_3 = self.concatenation_layer_3(
            [concat_2, convolution_output_3]
        )

        convolution_output_4 = self.convolution_layer_4(concat_3)
        concat_4 = self.concatenation_layer_4(
            [concat_3, convolution_output_4]
        )

        return concat_4


class InceptionLayer(torch.nn.Module):
    def __init__(self, in_channels, num_intermediate_filters):
        super(InceptionLayer, self).__init__()

        # 1 x 1
        self.convolution_layer_a = ConvolutionLayer(
            in_channels=in_channels,
            kernel_size=1,
            num_filters=in_channels,
            stride=(1, 1)
        )

        # 3 x 3
        self.convolution_layer_b_1 = ConvolutionLayer(
            in_channels=in_channels,
            kernel_size=1,
            num_filters=num_intermediate_filters,
            stride = (1, 1)
        )
        self.convolution_layer_b_2 = ConvolutionLayer(
            in_channels=num_intermediate_filters,
            kernel_size=3,
            num_filters=num_intermediate_filters,
            stride=(1, 1)
        )
        self.convolution_layer_b_3 = ConvolutionLayer(
            in_channels=num_intermediate_filters,
            kernel_size=1,
            num_filters=in_channels,
            stride=(1, 1)
        )

        # 3 x 3 -> 3 x 3
        self.convolution_layer_c_1 = ConvolutionLayer(
            in_channels=in_channels,
            kernel_size=1,
            num_filters=num_intermediate_filters,
            stride=(1, 1)
        )
        self.convolution_layer_c_2 = ConvolutionLayer(
            in_channels=num_intermediate_filters,
            kernel_size=3,
            num_filters=num_intermediate_filters,
            stride=(1, 1)
        )
        self.convolution_layer_c_3 = ConvolutionLayer(
            in_channels=num_intermediate_filters,
            kernel_size=3,
            num_filters=num_intermediate_filters,
            stride=(1, 1)
        )
        self.convolution_layer_c_4 = ConvolutionLayer(
            in_channels=num_intermediate_filters,
            kernel_size=1,
            num_filters=in_channels,
            stride=(1, 1)
        )

        self.adding_layer = SummationLayer()

    def forward(self, input_tensor):
        # 1 x 1
        convolution_output_a = self.convolution_layer_a(input_tensor)

        # 3 x 3
        convolution_output_b_1 = self.convolution_layer_b_1(input_tensor)
        convolution_output_b_2 = self.convolution_layer_b_2(convolution_output_b_1)
        convolution_output_b_3 = self.convolution_layer_b_3(convolution_output_b_2)

        # 3 x 3 -> 3 x 3
        convolution_output_c_1 = self.convolution_layer_c_1(input_tensor)
        convolution_output_c_2 = self.convolution_layer_c_2(convolution_output_c_1)
        convolution_output_c_3 = self.convolution_layer_c_3(convolution_output_c_2)
        convolution_output_c_4 = self.convolution_layer_c_4(convolution_output_c_3)

        summed = self.adding_layer(
            [
                input_tensor,
                convolution_output_a,
                convolution_output_b_3,
                convolution_output_c_4,
            ]
        )

        return summed


class SummedInputLayer(torch.nn.Module):
    def __init__(self):
        super(SummedInputLayer, self).__init__()

        self.group_slice_layer = GroupSliceLayer()

        self.summation_layer = SummationLayer()

    def forward(self, list_input_tensors):
        sliced_outputs = self.group_slice_layer(list_input_tensors)
        summed_outputs = self.summation_layer(sliced_outputs)

        return summed_outputs


class ProcessingLayer(torch.nn.Module):
    def __init__(self, in_channels, kernel_size,):
        super(ProcessingLayer, self).__init__()

        self.convolution_layer = SummedConvolutionLayer(
            in_channels=in_channels,
            num_filters=in_channels,
            kernel_size=kernel_size,
        )

    def forward(self, input_tensor):
        processed_output = self.convolution_layer(input_tensor)

        return processed_output


class UpSampleLayer(torch.nn.Module):
    def __init__(self, in_channels, kernel_size,):
        super(UpSampleLayer, self).__init__()

        self.up_sample_layer = TransposedConvolutionLayer(
            in_channels=in_channels,
            num_filters=in_channels,
            kernel_size=kernel_size,
            stride=(2, 2),
        )

    def forward(self, input_tensor):
        up_sample_output = self.up_sample_layer(input_tensor)

        return up_sample_output


class DownSampleLayer(torch.nn.Module):
    def __init__(self, in_channels, kernel_size,):
        super(DownSampleLayer, self).__init__()

        self.down_sample_layer = ConvolutionLayer(
            in_channels=in_channels,
            num_filters=in_channels,
            kernel_size=kernel_size,
            stride=(2, 2),
        )

    def forward(self, input_tensor):
        down_sampled_output = self.down_sample_layer(input_tensor)

        return down_sampled_output


