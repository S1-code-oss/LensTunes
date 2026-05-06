import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers, Model, Sequential
import matplotlib.pyplot as plt

TRAIN_DIR = 'train'
TEST_DIR  = 'test'


train_gen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True
)
test_gen = ImageDataGenerator(rescale=1./255)

train_data = train_gen.flow_from_directory(TRAIN_DIR, target_size=(48,48),
                                            batch_size=64, color_mode='grayscale')
test_data  = test_gen.flow_from_directory(TEST_DIR,  target_size=(48,48),
                                            batch_size=64, color_mode='grayscale')


model = Sequential([
    layers.Input(shape=(48,48,1)),

    layers.Conv2D(32, (3,3), activation='relu', padding='same'),
    layers.BatchNormalization(),
    layers.MaxPooling2D(2,2),
    layers.Dropout(0.25),

    layers.Conv2D(64, (3,3), activation='relu', padding='same'),
    layers.BatchNormalization(),
    layers.MaxPooling2D(2,2),
    layers.Dropout(0.25),

    layers.Conv2D(128, (3,3), activation='relu', padding='same'),
    layers.BatchNormalization(),
    layers.MaxPooling2D(2,2),
    layers.Dropout(0.25),

    layers.Flatten(),
    layers.Dense(256, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(7, activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()

history = model.fit(train_data, epochs=30, validation_data=test_data)

model.save('emotion_model.keras')
print("Model saved!")

plt.plot(history.history['accuracy'], label='Train')
plt.plot(history.history['val_accuracy'], label='Validation')
plt.title('Training Accuracy')
plt.legend()
plt.savefig('training_plot.png')
print("Plot saved!")