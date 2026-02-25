import tensorflow as tf
from tensorflow.keras.layers import Input, Conv1D, Conv1DTranspose, Dense, Reshape, GlobalAveragePooling1D
from tensorflow.keras.models import Model


def neural_configuration_1d(encoder_filters, kernel_sizes, strides):
    """
    Returns:
      encoder_layers: list of (filters, kernel, stride)
      decoder_layers: reversed list for Conv1DTranspose
    """
    if not (len(encoder_filters) == len(kernel_sizes) == len(strides)):
        raise ValueError("encoder_filters, kernel_sizes, strides must have the same length")

    encoder = list(zip(encoder_filters, kernel_sizes, strides))
    decoder = list(reversed(encoder))
    return encoder, decoder


def build_model_1d(
    Nx: int,
    encoder_layers,
    decoder_layers,
    dh: int,
    n_channels: int = 1,
):
    """
    1D convolutional autoencoder for KS signals.

    Input:  (Nx, 1)
    Latent: (dh,)
    Output: (Nx, 1)
    """
    inputs = Input(shape=(Nx, n_channels))
    x = inputs

    # Encoder
    for filters, k, s in encoder_layers:
        x = Conv1D(filters, k, strides=s, padding="same", activation="gelu")(x)

    length_enc = int(x.shape[1])
    channels_enc = int(x.shape[2])

    x = Flatten()(x)
    latent = Dense(dh, name="latent")(x)

    # Decoder
    x = Dense(length_enc * channels_enc, activation="gelu")(latent)
    x = Reshape((length_enc, channels_enc))(x)

    for filters, k, s in decoder_layers:
        x = Conv1DTranspose(filters, k, strides=s, padding="same", activation="gelu")(x)

    outputs = Conv1D(n_channels, kernel_size=3, padding="same", activation=None, name="decoded")(x)

    return Model(inputs, outputs)
