import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Bidirectional, Dense, Dropout, Conv1D, MaxPooling1D, Flatten, Conv2D, MaxPooling2D
from tensorflow.keras.optimizers import Adam

from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model

from kapre import STFT, Magnitude, ApplyFilterbank, MagnitudeToDecibel


def get_tiny_cnn_model_accurate(nb_classes=4) -> (Sequential, tf.keras.callbacks.LearningRateScheduler):
   pass

def get_tiny_cnn_model_fastest(nb_classes=4) -> (Sequential, tf.keras.callbacks.LearningRateScheduler):
    pass

def get_tiny_cnn_model_fittest(nb_classes=4) -> (Sequential, tf.keras.callbacks.LearningRateScheduler):
    # /data/du92wufe/Documents/EvoNAS/Results/ga_20240111-100825_spoken_languages/Generation_1/versatile_marmot_0/models/model_untrained.h5
    # /data/du92wufe/Documents/EvoNAS/Results/ga_20240111-100825_spoken_languages/Generation_9/humongous_penguin_8/models/model_untrained.h5
    model = tf.keras.models.load_model("/data/du92wufe/Documents/EvoNAS/Results/ga_20240111-100825_spoken_languages/Generation_9/humongous_penguin_8/models/model_untrained.h5", 
                                        custom_objects={'STFT': STFT,
                                                          'Magnitude': Magnitude,
                                                          'ApplyFilterbank': ApplyFilterbank,
                                                          'MagnitudeToDecibel': MagnitudeToDecibel})
    
    new_model = tf.keras.Sequential()
    new_model.add(tf.keras.Input(shape=(model.input_shape[1], model.input_shape[2])))
    for layer in model.layers[:-1]:
        if type(layer) == ApplyFilterbank:
            kwargs = {
                 'sample_rate': 8000,
                 'n_freq': 112 // 2 + 1,
                 'n_mels': 44,
                 'f_min': 0,
                 'f_max': 8000,
                 'htk': False,
                 'norm': 'slaney',
            }
            #kwargs = {
            #     'sample_rate': 8000,
            #     'n_freq': 96 // 2 + 1,
            #     'n_mels': 52,
            #     'f_min': 0,
            #     'f_max': 8000,
            #     'htk': False,
            #     'norm': 'slaney',
            #}
            new_model.add(ApplyFilterbank(type="mel", filterbank_kwargs=kwargs, data_format='channels_last'))
        else:
            new_model.add(layer)
    
    new_model.add(Dense(nb_classes, activation='softmax'))
    
    new_model.compile(loss='categorical_crossentropy', optimizer=Adam(learning_rate=0.0001), metrics=['accuracy'])

    def lr_decay(epoch):
        if epoch < 25:
            return 0.0005
        elif epoch < 50:
            return 0.00001
        else:
            return 0.000001
    
    return new_model, tf.keras.callbacks.LearningRateScheduler(lr_decay)

#def get_tiny_cnn_model(nb_classes=4) -> (Sequential, tf.keras.callbacks.LearningRateScheduler):
#    model = Sequential()
#    model.add(STFT(input_shape=(4000, 1), n_fft=512, hop_length=256, input_data_format='channels_last', output_data_format='channels_last', name='stft'))
#    model.add(Magnitude(name='magnitude'))
#    kwargs = {'sample_rate': 2048,
#              'n_freq': 512 // 2 + 1,
#              'n_mels': 128
#    }
#    #model.add(ApplyFilterbank(type='mel', filterbank_kwargs=kwargs))
#    #model.add(MagnitudeToDecibel(name='magnitude_to_decibel'))
#
#    model.add(Conv2D(8, kernel_size=(3, 3), activation='relu'))
#    #model.add(MaxPooling2D(pool_size=(2, 2)))
#    #model.add(Conv2D(16, kernel_size=(3, 3), activation='relu'))
#    model.add(GlobalAveragePooling2D())
#    model.add(Dense(16, activation='relu'))
#    #model.add(Dense(64, activation='relu'))
#    model.add(Dense(nb_classes, activation='softmax'))
#
#  
#    model.compile(loss='categorical_crossentropy', optimizer=Adam(learning_rate=0.0001), metrics=['accuracy'])
#
#    def lr_decay(epoch):
#        if epoch < 25:
#            return 0.001
#        elif epoch < 50:
#            return 0.0005
#        else:
#            return 0.00001
#    
#    return model, tf.keras.callbacks.LearningRateScheduler(lr_decay)

def get_rnn_model(nb_classes=4) -> (Sequential, tf.keras.callbacks.LearningRateScheduler):
    # Define the model
    model = Sequential()
    model.add(Bidirectional(LSTM(128, return_sequences=True), input_shape=(128, 32)))
    model.add(Bidirectional(LSTM(256, return_sequences=True)))
    model.add(Bidirectional(LSTM(512)))
    model.add(Dropout(0.5))
    model.add(Dense(256, activation='relu'))
    model.add(Dense(128, activation='relu'))
    model.add(Dense(nb_classes, activation='softmax'))

    model.compile(loss='categorical_crossentropy', optimizer=Adam(learning_rate=0.0001), metrics=['accuracy'])

    def lr_decay(epoch):
        if epoch < 25:
            return 0.0001
        elif epoch < 50:
            return 0.00005
        else:
            return 0.00001
    
    return model, tf.keras.callbacks.LearningRateScheduler(lr_decay)

def get_cnn_effnet_model(nb_classes=4) -> (Sequential, tf.keras.callbacks.LearningRateScheduler):
    base_model = EfficientNetB0(weights='imagenet', include_top=False, input_shape=(128, 32, 3))

    # Add a global average pooling layer
    x = base_model.output
    x = GlobalAveragePooling2D()(x)

    # Add a fully connected layer
    x = Dense(1024, activation='relu')(x)

    # Add the output layer
    predictions = Dense(nb_classes, activation='softmax')(x)

    # Create the model
    model = Model(inputs=base_model.input, outputs=predictions)

    # Compile the model
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    def lr_decay(epoch):
        if epoch < 25:
            return 0.001
        elif epoch < 50:
            return 0.0005
        else:
            return 0.0001
        
    return model, tf.keras.callbacks.LearningRateScheduler(lr_decay)

def get_cnn_2d_model(nb_classes=4) -> (Sequential, tf.keras.callbacks.LearningRateScheduler):
    model = Sequential()
    model.add(Conv2D(16, kernel_size=(3, 3), activation='relu', input_shape=(128, 32, 1)))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(32, kernel_size=(3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
    model.add(Flatten())
    model.add(Dense(256, activation='relu'))
    model.add(Dense(128, activation='relu'))
    model.add(Dense(nb_classes, activation='softmax'))

    model.compile(loss='categorical_crossentropy', optimizer=Adam(learning_rate=0.0001), metrics=['accuracy'])

    def lr_decay(epoch):
        if epoch < 25:
            return 0.0001
        elif epoch < 50:
            return 0.00005
        else:
            return 0.00001
    
    return model, tf.keras.callbacks.LearningRateScheduler(lr_decay)

def get_cnn_1d_model(nb_classes=4) -> (Sequential, tf.keras.callbacks.LearningRateScheduler):
    model = Sequential()
    model.add(Conv1D(8, kernel_size=3, activation='relu', input_shape=(8000, 1)))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Conv1D(16, kernel_size=3, activation='relu'))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Conv1D(32, kernel_size=3, activation='relu'))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Conv1D(64, kernel_size=3, activation='relu'))
    model.add(Flatten())
    model.add(Dense(256, activation='relu'))
    model.add(Dense(128, activation='relu'))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(nb_classes, activation='softmax'))

    model.compile(loss='categorical_crossentropy', optimizer=Adam(learning_rate=0.0001), metrics=['accuracy'])

    def lr_decay(epoch):
        if epoch < 25:
            return 0.0001
        elif epoch < 50:
            return 0.00005
        else:
            return 0.00001
    
    return model, tf.keras.callbacks.LearningRateScheduler(lr_decay)

def get_transformer_model(nb_classes=4) -> (Sequential, tf.keras.callbacks.LearningRateScheduler):
    model = transformer(time_steps=32,
      num_layers=4,
      units=1024,
      d_model=128,
      num_heads=16,
      dropout=0.1,
      output_size=nb_classes,  
      projection="linear")
    
    model.compile(optimizer=tf.keras.optimizers.Adam(0.00001), loss='categorical_crossentropy', metrics=['accuracy'])

    def lr_decay(epoch):
      if epoch < 30:
          return 0.0001
      elif epoch < 50:
          return 0.00001
      elif epoch < 70:
          return 0.000001
      else:
          return 0.0000001

    return model, tf.keras.callbacks.LearningRateScheduler(lr_decay)


def transformer(time_steps,
                num_layers,
                units,
                d_model,
                num_heads,
                dropout,
                output_size,
                projection,
                name="transformer"):
  inputs = tf.keras.Input(shape=(None, d_model), name="inputs")
  
  
  enc_padding_mask = tf.keras.layers.Lambda(
      create_padding_mask, output_shape=(1, 1, None),
      name='enc_padding_mask')(tf.dtypes.cast(
          
    #Like our input has a dimension of length X d_model but the masking is applied to a vector
    # We get the sum for each row and result is a vector. So, if result is 0 it is because in that position was masked      
    tf.math.reduce_sum(
    inputs,
    axis=2,
    keepdims=False,
    name=None
), tf.int32))
  

  enc_outputs = encoder(
      time_steps=time_steps,
      num_layers=num_layers,
      units=units,
      d_model=d_model,
      num_heads=num_heads,
      dropout=dropout,
      projection=projection,
      name='encoder'
  )(inputs=[inputs, enc_padding_mask])

  #We reshape for feeding our FC in the next step
  outputs=tf.reshape(enc_outputs,(-1,time_steps*d_model))
  
  #We predict our class
  outputs = tf.keras.layers.Dense(units=output_size,use_bias=True,activation='softmax', name="outputs")(outputs)

  return tf.keras.Model(inputs=[inputs], outputs=outputs, name='audio_class')


class MultiHeadAttention(tf.keras.layers.Layer):

  def __init__(self, d_model, num_heads, name="multi_head_attention"):
    super(MultiHeadAttention, self).__init__(name=name)
    self.num_heads = num_heads
    self.d_model = d_model

    assert d_model % self.num_heads == 0

    self.depth = d_model // self.num_heads

    self.query_dense = tf.keras.layers.Dense(units=d_model)
    self.key_dense = tf.keras.layers.Dense(units=d_model)
    self.value_dense = tf.keras.layers.Dense(units=d_model)

    self.dense = tf.keras.layers.Dense(units=d_model)

  def split_heads(self, inputs, batch_size):
    inputs = tf.reshape(
        inputs, shape=(batch_size, -1, self.num_heads, self.depth))
    return tf.transpose(inputs, perm=[0, 2, 1, 3])

  def call(self, inputs):
    query, key, value, mask = inputs['query'], inputs['key'], inputs[
        'value'], inputs['mask']
    batch_size = tf.shape(query)[0]

    # linear layers
    query = self.query_dense(query)
    key = self.key_dense(key)
    value = self.value_dense(value)

    # split heads
    query = self.split_heads(query, batch_size)
    key = self.split_heads(key, batch_size)
    value = self.split_heads(value, batch_size)

    scaled_attention = scaled_dot_product_attention(query, key, value, mask)

    scaled_attention = tf.transpose(scaled_attention, perm=[0, 2, 1, 3])

    concat_attention = tf.reshape(scaled_attention,
                                  (batch_size, -1, self.d_model))

    outputs = self.dense(concat_attention)

    return outputs
  
class PositionalEncoding(tf.keras.layers.Layer):

  def __init__(self, position, d_model):
    super(PositionalEncoding, self).__init__()
    self.pos_encoding = self.positional_encoding(position, d_model)

  def get_angles(self, position, i, d_model):
    angles = 1 / tf.pow(10000, (2 * (i // 2)) / tf.cast(d_model, tf.float32))
    return position * angles

  def positional_encoding(self, position, d_model):
    angle_rads = self.get_angles(
        position=tf.range(position, dtype=tf.float32)[:, tf.newaxis],
        i=tf.range(d_model, dtype=tf.float32)[tf.newaxis, :],
        d_model=d_model)
    # apply sin to even index in the array
    sines = tf.math.sin(angle_rads[:, 0::2])
    # apply cos to odd index in the array
    cosines = tf.math.cos(angle_rads[:, 1::2])

    pos_encoding = tf.concat([sines, cosines], axis=-1)
    pos_encoding = pos_encoding[tf.newaxis, ...]
    return tf.cast(pos_encoding, tf.float32)

  def call(self, inputs):
    return inputs + self.pos_encoding[:, :tf.shape(inputs)[1], :]
  
# This allows to the transformer to know where there is real data and where it is padded
def create_padding_mask(seq):
  seq = tf.cast(tf.math.equal(seq, 0), tf.float32)
  
  # add extra dimensions to add the padding
  # to the attention logits.
  return seq[:, tf.newaxis, tf.newaxis, :]  # (batch_size, 1, 1, seq_len)

def encoder_layer(units, d_model, num_heads, dropout,name="encoder_layer"):
  inputs = tf.keras.Input(shape=(None,d_model ), name="inputs")
  padding_mask = tf.keras.Input(shape=(1, 1, None), name="padding_mask")

  attention = MultiHeadAttention(
      d_model, num_heads, name="attention")({
          'query': inputs,
          'key': inputs,
          'value': inputs,
          'mask': padding_mask
      })
  attention = tf.keras.layers.Dropout(rate=dropout)(attention)
  attention = tf.keras.layers.LayerNormalization(
      epsilon=1e-6)(inputs + attention)

  outputs = tf.keras.layers.Dense(units=units, activation='relu')(attention)
  outputs = tf.keras.layers.Dense(units=d_model)(outputs)
  outputs = tf.keras.layers.Dropout(rate=dropout)(outputs)
  outputs = tf.keras.layers.LayerNormalization(
      epsilon=1e-6)(attention + outputs)

  return tf.keras.Model(
      inputs=[inputs, padding_mask], outputs=outputs, name=name)

def scaled_dot_product_attention(query, key, value, mask):
  matmul_qk = tf.matmul(query, key, transpose_b=True)

  depth = tf.cast(tf.shape(key)[-1], tf.float32)
  logits = matmul_qk / tf.math.sqrt(depth)

  # add the mask zero out padding tokens.
  if mask is not None:
    logits += (mask * -1e9)

  attention_weights = tf.nn.softmax(logits, axis=-1)

  return tf.matmul(attention_weights, value)

def encoder(time_steps,
            num_layers,
            units,
            d_model,
            num_heads,
            dropout,
            projection,
            name="encoder"):
  
  inputs = tf.keras.Input(shape=(None, d_model), name="inputs")
  padding_mask = tf.keras.Input(shape=(1, 1, None), name="padding_mask")
  
  if projection=='linear':
    ## We implement a linear projection based on Very Deep Self-Attention Networks for End-to-End Speech Recognition. Retrieved from https://arxiv.org/abs/1904.13377
    projection=tf.keras.layers.Dense( d_model,use_bias=True, activation='linear')(inputs)
    print('linear')
  
  else:
    projection=tf.identity(inputs)
    print('none')
   
  projection *= tf.math.sqrt(tf.cast(d_model, tf.float32))
  projection = PositionalEncoding(time_steps, d_model)(projection)

  outputs = tf.keras.layers.Dropout(rate=dropout)(projection)

  for i in range(num_layers):
    outputs = encoder_layer(
        units=units,
        d_model=d_model,
        num_heads=num_heads,
        dropout=dropout,
        name="encoder_layer_{}".format(i),
    )([outputs, padding_mask])
 
  return tf.keras.Model(
      inputs=[inputs, padding_mask], outputs=outputs, name=name)