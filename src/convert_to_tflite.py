import tensorflow as tf
from kapre import STFT, Magnitude, STFTTflite, MagnitudeTflite, ApplyFilterbank, MagnitudeToDecibel
import numpy as np
import argparse

from baseline_models import get_tiny_cnn_model

gpus = tf.config.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)


def substitute_tflite_layer(model, input_shape):
    """ Preprocessing layers are critical on MCUs since some TF ops (e.g. RANGE) are not implemented for
    TFLite-micro. Therefore, these layers need to be replaced with special implementations to be able to deploy the
    model. """
    model_with_tflite_layers = tf.keras.Sequential()
    model_with_tflite_layers.add(tf.keras.Input(shape=input_shape))

    for layer in model.layers:
        if type(layer) == STFT:
            model_with_tflite_layers.add(
                STFTTflite(input_shape=layer.input_shape[1::], n_fft=layer.n_fft, hop_length=layer.hop_length,
                           input_data_format='channels_last', output_data_format='channels_last'))
        elif type(layer) == Magnitude:
            model_with_tflite_layers.add(MagnitudeTflite())
        else:
            model_with_tflite_layers.add(layer)

    return model_with_tflite_layers


def convert_to_tflite(model, representative_data=None):
    # create TFLite converter object
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS,
                                           tf.lite.OpsSet.SELECT_TF_OPS]

    if representative_data is not None:
        def representative_dataset():
            for _ in range(len(representative_data)):
                yield [representative_data.astype(np.float32)]

        converter.representative_dataset = representative_dataset
        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS,
                                               tf.lite.OpsSet.TFLITE_BUILTINS_INT8,
                                               tf.lite.OpsSet.SELECT_TF_OPS]
        converter.inference_input_type = tf.int8
        converter.inference_output_type = tf.int8

    # converter specifications
    converter.optimizations = [tf.lite.Optimize.DEFAULT]

    tflite_model = converter.convert()
    return tflite_model

if __name__ == "__main__":
    # args
    parser = argparse.ArgumentParser()
    parser.add_argument("model_weights_path", type=str, default="", help="The model to convert to TFLite")
    args = parser.parse_args()

    #model, _ = get_tiny_cnn_model(4)
    model = tf.keras.models.load_model("/data/du92wufe/Documents/EvoNAS/Results/ga_20240111-100825_spoken_languages/Generation_10/radical_nautilus_9/models/model_untrained.h5", 
                                       custom_objects={'STFT': STFT,
                                                         'Magnitude': Magnitude,
                                                         'ApplyFilterbank': ApplyFilterbank,
                                                         'MagnitudeToDecibel': MagnitudeToDecibel})
    new_model = tf.keras.Sequential()
    for layer in model.layers:
        if type(layer) == ApplyFilterbank:
            kwargs = {
                    'sample_rate': 8000,
                    'n_freq': 96 // 2 + 1,
                    'n_mels': 52,
                    'f_min': 0,
                    'f_max': 8000,
                    'htk': False,
                    'norm': 'slaney',
                }
            new_model.add(ApplyFilterbank(type="mel", filterbank_kwargs=kwargs, data_format='channels_last'))
        else:
            new_model.add(layer)

    input_shape = model.input_shape[1::]

    # substitute preprocessing layers
    model = substitute_tflite_layer(model, input_shape=input_shape)

    model.summary()

    # convert to TFLite
    tflite_model = convert_to_tflite(model, np.random.uniform(size=(200, input_shape[0], input_shape[1])))

    # save the model
    tflite_path = args.model_weights_path.replace("model.h5", "model.tflite")
    with open(tflite_path, "wb") as f:
        f.write(tflite_model)