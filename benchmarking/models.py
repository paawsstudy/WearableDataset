# This file contains the single- and multi-sensor-based CNNs used in the accompanying manuscript.
import tensorflow as tf 

# Single-sensor-based CNN.
# Params: input_shape (tuple) - the shape of the passed in input, output_shape (tuple) - the shape 
# of the labels.
# returns: cnn (tf model) - the single-sensor model.
def single_sensor_CNN(input_shape, output_shape):
    cnn = tf.keras.models.Sequential()
    cnn.add(tf.keras.layers.Input(shape=input_shape))
    cnn.add(tf.keras.layers.Conv1D(9, (5), strides=(1), padding="same", activation='relu'))
    cnn.add(tf.keras.layers.MaxPooling1D(pool_size=2, strides=1, padding='valid'))
    cnn.add(tf.keras.layers.Conv1D(27, (5), strides=(1), padding='same', activation='relu'))
    cnn.add(tf.keras.layers.MaxPooling1D(pool_size=2, strides=1, padding='valid'))
    cnn.add(tf.keras.layers.Conv1D(54, (5), strides=(1), padding='same', activation='relu'))
    cnn.add(tf.keras.layers.MaxPooling1D(pool_size=2, strides=1, padding='valid'))
    cnn.add(tf.keras.layers.Dropout(.25))
    cnn.add(tf.keras.layers.Flatten())
    cnn.add(tf.keras.layers.Dense(output_shape, activation=tf.keras.activations.softmax))

    return cnn

# Muli-sensor-based CNN.
# Params: input_shape (tuple) - the shape of the passed in input, output_shape (tuple) - the shape 
# of the labels.
# returns: cnn (tf model) - the multi-sensor model.
class multi_sensor_CNN(tf.keras.models.Model):
  def __init__(self, input_shape, output_shape):
    super(multi_sensor_CNN, self).__init__() 

    # feature extraction of sensor 1
    self.sens_1 = tf.keras.Sequential([
      tf.keras.layers.Input(shape=(400, 3)),
      tf.keras.layers.Conv1D(9, (5), strides=(1), padding="same", activation='relu'),
      tf.keras.layers.MaxPooling1D(pool_size=2, strides=1, padding='valid'),
      tf.keras.layers.Conv1D(27, (5), strides=(1), padding='same', activation='relu'),
      tf.keras.layers.MaxPooling1D(pool_size=2, strides=1, padding='valid'),
    ])

    # feature extraction of sensor 2
    self.sens_2 = tf.keras.Sequential([
      tf.keras.layers.Input(shape=(400, 3)),
      tf.keras.layers.Conv1D(9, (5), strides=(1), padding="same", activation='relu'),
      tf.keras.layers.MaxPooling1D(pool_size=2, strides=1, padding='valid'),
      tf.keras.layers.Conv1D(27, (5), strides=(1), padding='same', activation='relu'),
      tf.keras.layers.MaxPooling1D(pool_size=2, strides=1, padding='valid'),
    ])

    # feature concatenation and additional extraction
    self.post_conv = tf.keras.Sequential([
      tf.keras.layers.Input(shape=(398, 27*2)), # 2 for number of sensors 
      tf.keras.layers.Conv1D(54, (5), strides=(1), padding="same", activation='relu'),
      tf.keras.layers.MaxPooling1D(pool_size=2, strides=1, padding='valid'),
    ])

    # block for classifcation
    self.classifier = tf.keras.Sequential([
      tf.keras.layers.Input(shape=(397, 54)),
      tf.keras.layers.Dropout(.25),
      tf.keras.layers.Flatten(),
      tf.keras.layers.Dense(output_shape, activation=tf.keras.activations.softmax),
    ])

  def call(self, x):
    sens1 = self.sens_1(x[:, :, 0:3])
    sens2 = self.sens_2(x[:, :, 3:])
    post = self.post_conv(tf.concat([sens1, sens2], 2)) # 2 for number of sensors 
    classified = self.classifier(post)
    return classified