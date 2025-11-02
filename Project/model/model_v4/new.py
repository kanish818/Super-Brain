import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.losses import MeanSquaredError

# Define the encoder model
def build_encoder(input_shape):
    inputs = layers.Input(shape=input_shape)
    x = layers.Conv1D(32, 3, activation='relu', strides=2, padding='same')(inputs)
    x = layers.MaxPooling1D(2)(x)
    x = layers.Flatten()(x)
    x = layers.Dense(64, activation='relu')(x)
    
    z_mean = layers.Dense(16, name='z_mean')(x)
    z_log_var = layers.Dense(16, name='z_log_var')(x)

    # Sampling layer (reparameterization trick)
    def sampling(args):
        z_mean, z_log_var = args
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon

    z = layers.Lambda(sampling, output_shape=(16,))([z_mean, z_log_var])
    
    encoder = models.Model(inputs, [z_mean, z_log_var, z], name="encoder")
    return encoder

# Define the decoder model
def build_decoder(latent_dim):
    latent_inputs = layers.Input(shape=(latent_dim,))
    x = layers.Dense(64, activation='relu')(latent_inputs)
    x = layers.Dense(10, activation='relu')(x)
    x = layers.Reshape((10, 1))(x)
    x = layers.LSTM(64, return_sequences=True)(x)
    x = layers.Conv1D(32, 3, activation='relu', padding='same')(x)
    outputs = layers.Dense(1)(x)
    
    decoder = models.Model(latent_inputs, outputs, name="decoder")
    return decoder

# Define a custom loss layer for VAE
class VAELoss(layers.Layer):
    def __init__(self, z_mean, z_log_var, **kwargs):
        super(VAELoss, self).__init__(**kwargs)
        self.z_mean = z_mean
        self.z_log_var = z_log_var

    def call(self, y_true, y_pred):
        # Reconstruction loss
        reconstruction_loss = tf.reduce_mean(MeanSquaredError()(y_true, y_pred))

        # KL divergence loss
        kl_loss = -0.5 * tf.reduce_mean(
            tf.reduce_sum(1 + self.z_log_var - tf.square(self.z_mean) - tf.exp(self.z_log_var), axis=-1)
        )

        return reconstruction_loss + kl_loss

# Define the VAE model
def build_vae(encoder, decoder, input_shape):
    vae_input = layers.Input(shape=input_shape)
    z_mean, z_log_var, z = encoder(vae_input)
    vae_output = decoder(z)
    
    # Create the VAE model
    vae = models.Model(vae_input, vae_output)

    # Define the loss function directly in the model's call method
    def vae_loss(y_true, y_pred):
        reconstruction_loss = tf.reduce_mean(MeanSquaredError()(y_true, y_pred))
        kl_loss = -0.5 * tf.reduce_mean(
            tf.reduce_sum(1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var), axis=-1)
        )
        return reconstruction_loss + kl_loss
    
    vae.add_loss(vae_loss(vae_input, vae_output))

    vae.compile(optimizer='adam')
    return vae

# Load or generate your data (e.g., using random data here)
x_train = np.random.rand(1000, 10, 1)  # Example input data (1000 samples of shape 10x1)
x_test = np.random.rand(200, 10, 1)    # Example test data (200 samples of shape 10x1)

# Build the encoder and decoder models
encoder = build_encoder(input_shape=(10, 1))
decoder = build_decoder(latent_dim=16)

# Build and compile the VAE model
vae = build_vae(encoder, decoder, input_shape=(10, 1))

# Train the VAE
vae.fit(x_train, x_train, epochs=20, batch_size=32)

# Evaluate the VAE on the test data (reconstruction loss)
reconstruction_loss = vae.evaluate(x_test, x_test)
print("Reconstruction Loss on Test Data: ", reconstruction_loss)
