import torch
import tensorflow


class TransposedConvolutionLayer(torch.nn.Module):
    def __init__(self):
        super(TransposedConvolutionLayer, self).__init__()

        self.layer = torch.nn.ConvTranspose2d(
            in_channels=3,
            out_channels=3,
            kernel_size=3,
            stride=2,
        )

    def forward(self, input_tensor):
        return self.layer(input_tensor)

    def test(self):
        test_input = torch.rand(
            (1, 3, 33, 33)
        )
        test_output = self.forward(test_input)
        print("Torch Out Shape:", test_output.shape)


class Conv2DTransposed(tensorflow.keras.layers.Layer):
    def __init__(self):
        super(Conv2DTransposed, self).__init__()

        self.padder = tensorflow.keras.layers.ZeroPadding2D(
            padding=(1, 1),
            data_format='channels_first'
        )
        self.layer = tensorflow.keras.layers.Conv2DTranspose(
            filters=3,
            kernel_size=3,
            strides=2,
            padding='same',
            data_format='channels_first'
        )

    def call(self, input_tensor):
        # padded = self.padder(input_tensor)
        return self.layer(input_tensor)

    def test(self):
        test_input = tensorflow.random.uniform(
            (1, 3, 33, 33)
        )
        test_output = self.call(test_input)
        print("TF Out Shape:", test_output.shape)


tf_layer = TransposedConvolutionLayer()
torch_layer = Conv2DTransposed()

tf_layer.test()
torch_layer.test()

