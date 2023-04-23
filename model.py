import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Input, Dense, Reshape, Flatten, Dropout, Embedding, multiply
from tensorflow.keras.layers import BatchNormalization, Activation, ZeroPadding2D
from tensorflow.keras.layers import LeakyReLU
from tensorflow.keras.layers import UpSampling2D, Conv2DTranspose, Conv2D
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.optimizers import Adam

# Load the sonnets from the Sonnet.txt file
sonnet_file = open('./data/Sonnet.txt', 'r')
sonnets = sonnet_file.read().split('\n\n')
sonnet_file.close()

# Clean the sonnets by removing extra whitespaces and converting to lowercase
cleaned_sonnets = []
for sonnet in sonnets:
    sonnet = sonnet.strip().lower()
    sonnet = ' '.join(sonnet.split())
    cleaned_sonnets.append(sonnet)

# Preprocess the sonnets using the Tokenizer
tokenizer = Tokenizer()
tokenizer.fit_on_texts(cleaned_sonnets)
sequences = tokenizer.texts_to_sequences(cleaned_sonnets)
max_sequence_length = max([len(seq) for seq in sequences])
padded_sequences = pad_sequences(sequences, maxlen=max_sequence_length, padding='post')
vocabulary_size = len(tokenizer.word_index) + 1

# Define the generator model
def build_generator():
    model = Sequential()
    model.add(Dense(256, input_dim=100))
    model.add(LeakyReLU(alpha=0.2))
    model.add(BatchNormalization(momentum=0.8))
    model.add(Dense(512))
    model.add(LeakyReLU(alpha=0.2))
    model.add(BatchNormalization(momentum=0.8))
    model.add(Dense(1024))
    model.add(LeakyReLU(alpha=0.2))
    model.add(BatchNormalization(momentum=0.8))
    model.add(Dense(max_sequence_length * vocabulary_size, activation='softmax'))
    model.add(Reshape((max_sequence_length, vocabulary_size)))
    return model

# Define the discriminator model
def build_discriminator():
    model = Sequential()
    model.add(Flatten(input_shape=(max_sequence_length, vocabulary_size)))
    model.add(Dense((max_sequence_length * vocabulary_size) // 2))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dense((max_sequence_length * vocabulary_size) // 4))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dense(1, activation='sigmoid'))
    return model

# Define the combined generator/discriminator model
def build_combined(generator, discriminator):
    discriminator.trainable = False
    sequence_input = Input(shape=(100,))
    generated_sequence = generator(sequence_input)
    validity = discriminator(generated_sequence)
    combined_model = Model(sequence_input, validity)
    combined_model.compile(loss='binary_crossentropy', optimizer=Adam(0.0002, 0.5))
    return combined_model

# Build the generator, discriminator, and combined models
generator = build_generator()
discriminator = build_discriminator()
combined = build_combined(generator, discriminator)

# Train the GAN model
batch_size = 32
epochs = 10000

valid = np.ones((batch_size, 1))
fake = np.zeros((batch_size, 1))

for epoch in range(epochs):
    # Train discriminator on real sequences
    sequence_indices = np.random.randint(0, padded_sequences.shape[0], batch_size)
    real_sequences = padded_sequences[sequence_indices]
    discriminator_loss_real = discriminator.train_on_batch(real_sequences, valid)

    # Train discriminator on generated sequences
    noise = np.random.normal(0, 1, (batch_size, 100))
    generated_sequences = generator.predict(noise)
    discriminator_loss_fake = discriminator.train_on_batch(generated_sequences, fake)

    # Train generator
    noise = np.random.normal(0, 1, (batch_size, 100))
    generator_loss = combined.train_on_batch(noise, valid)

    # Print progress every 100 epochs
    if epoch % 100 == 0:
        print(f"Epoch {epoch} - Discriminator loss on real sequences: {discriminator_loss_real} - Discriminator loss on generated sequences: {discriminator_loss_fake} - Generator loss: {generator_loss}")

# Generate a sonnet using the trained generator
noise = np.random.normal(0, 1, (1, 100))
generated_sequence = generator.predict(noise)
generated_sonnet = []
for sequence in generated_sequence[0]:
    index = np.argmax(sequence)
    word = tokenizer.index_word[index]
    if word is None:
        break
    generated_sonnet.append(word)
generated_sonnet = ' '.join(generated_sonnet)
print(generated_sonnet)
