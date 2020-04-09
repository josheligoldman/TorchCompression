class ModelConfig:
    def __init__(
        self,
        num_columns_encoder,
        num_rows_encoder,
        num_input_image_channels,
        num_filters_encoder,
        kernel_size_encoder,
        latent_space_num_channels,
        num_columns_decoder,
        num_rows_decoder,
        num_filters_decoder,
        kernel_size_decoder,
        num_final_feature_maps,
        reconstructed_image_num_channels,
    ):
        self.num_columns_encoder = num_columns_encoder
        self.num_rows_encoder = num_rows_encoder
        self.num_input_image_channels = num_input_image_channels
        self.num_filters_encoder = num_filters_encoder
        self.kernel_size_encoder = kernel_size_encoder
        self.latent_space_num_channels = latent_space_num_channels

        self.num_columns_decoder = num_columns_decoder
        self.num_rows_decoder = num_rows_decoder
        self.num_filters_decoder = num_filters_decoder
        self.kernel_size_decoder = kernel_size_decoder
        self.num_final_feature_maps = num_final_feature_maps
        self.reconstructed_image_num_channels = reconstructed_image_num_channels


