import numpy as np
import matplotlib.pyplot as plt

from tensorflow.keras.layers import Input, Conv2D, Flatten, Dense
from tensorflow.keras.layers import Reshape, Conv2DTranspose
from tensorflow.keras.models import Model
from tensorflow.keras        import backend as K
from tensorflow.keras.utils  import plot_model 

def get_mnist_dataset(file_name="mnist.npz"):
    # returns normalized images
    mnist = np.load(file_name, allow_pickle=True)
    # print(mnist.files)
    X_train = mnist['x_train']
    X_test  = mnist['x_test']
    y_train = mnist['y_train']
    y_test  = mnist['y_test']
    # image reshape to (28, 28, 1)
    image_size = X_train.shape[1]
    X_train = np.reshape(X_train, [-1, image_size, image_size, 1])
    X_test  = np.reshape(X_test,  [-1, image_size, image_size, 1])
    # pixel rescaling
    X_train = X_train.astype('float')/255
    X_test  = X_test.astype('float')/255
    return X_train, X_test, y_train, y_test

def view_first_mnist_images_of_digits(X, y):
    # view first images of digits
    fig, ax = plt.subplots(nrows=2, ncols=5, sharex=True, sharey=True)
    ax = ax.flatten()
    for i in range(10):
        img = X[y==i][0].reshape(28,28)
        ax[i].imshow(img, cmap='Greys')
    ax[0].set_xticks([])
    ax[0].set_yticks([])
    plt.tight_layout()
    plt.show()

def view_first_10_mnist_image_of_digit(X, y, digit):
    # view first images for digits
    fig, ax = plt.subplots(nrows=2, ncols=5, sharex=True, sharey=True)
    ax = ax.flatten()
    for i in range(10):
        img = X[y==digit][i].reshape(28,28)
        ax[i].imshow(img, cmap='Greys')
    ax[0].set_xticks([])
    ax[0].set_yticks([])
    plt.tight_layout()
    plt.show()

def view(X_in, X_out):
    fig, ax = plt.subplots(nrows=1, ncols=2, sharex=True, sharey=True)
    ax = ax.flatten()
    ax[0].imshow(X_in[0].reshape(28,28), cmap='Greys')
    ax[1].imshow(X_out[0].reshape(28,28), cmap='Greys')
    ax[0].set_xticks([])
    ax[0].set_yticks([])
    plt.tight_layout()
    plt.show()

# === Network parameters ======
batch_size   = 128
epochs       = 100
entry_shape  = (28, 28, 1)
# shape of data point (e.g., shape of each image)
# -----------------------------
n_channels   = entry_shape[-1]       # : 1
original_dim = np.prod(entry_shape)  # : 28 * 28 * 1 = 784
# -----------------------------
intermed_dim = 64
latent_dim   = 16
conv_filters = [32, 64]
kernel_size  = 3
strides      = 2
# -----------------------------

# === Encoder part ============
x_in  = Input(shape=entry_shape)
x_enc = x_in
x_enc = Conv2D(filters=conv_filters[0], kernel_size=kernel_size,
               strides=strides, padding='same',
               activation='relu')(x_enc)
x_enc = Conv2D(filters=conv_filters[1], kernel_size=kernel_size,
               strides=strides, padding='same',
               activation='relu')(x_enc)
conv_output_shape = K.int_shape(x_enc)
# (None,7, 7, 64): Decoder needs this info
# --- latent space
x_enc = Flatten()(x_enc)
flat_output_shape = K.int_shape(x_enc)
# (None, 3136) = (None, 7*7*64): Decoder needs this info
z = Dense(latent_dim, activation=None)(x_enc)
# --- build model
Encoder = Model(x_in, z, name='encoder')
Encoder.summary()

# === Decoder part ============
z_in  = Input(shape=latent_dim)
x_dec = z_in
x_dec = Dense(flat_output_shape[1], activation=None)(x_dec) # (7*7*64=3136)
x_dec = Reshape((conv_output_shape[1:]))(x_dec)
x_dec = Conv2DTranspose(filters=conv_filters[1], kernel_size=kernel_size,
                        strides=strides,padding='same',
                        activation='relu')(x_dec)
x_dec = Conv2DTranspose(filters=conv_filters[0], kernel_size=kernel_size,
                        strides=strides, padding='same',
                        activation='relu')(x_dec)
# --- to original shape-------
x_out = Conv2DTranspose(filters=n_channels, kernel_size=kernel_size,
                        strides=1, padding='same',
                        activation='sigmoid')(x_dec)
# --- build model
Decoder = Model(z_in, x_out, name='decoder')
Decoder.summary()

# === Autoencoder ============
z     = Encoder(x_in)
x_out = Decoder(z)
AE    = Model(x_in, x_out, name='autoencoder')
AE.summary()
AE.compile(loss='binary_crossentropy', optimizer='adam')
#AE.compile(loss='mse', optimizer='adam')
 
X_train, X_test, y_train, y_test = get_mnist_dataset(file_name="mnist.npz") 
X_train = X_train[:5]

AE.fit(X_train, X_train, epochs=epochs, batch_size=batch_size,
       shuffle=True, validation_split=0.0)

# === Reconstruction Example
for i in range(5):
    x_ori = X_train[i:i+1]
    x_rec = AE.predict(x_ori)
    view(x_ori, x_rec)
