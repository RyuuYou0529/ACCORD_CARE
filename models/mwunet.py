from tensorflow.keras.layers import *
from tensorflow.keras.models import *
from tensorflow.keras.optimizers import *
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard, ReduceLROnPlateau
import tensorflow as tf

import os
import warnings

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
warnings.filterwarnings("ignore")

class mwunet():
    def __init__(self, 
                input_shape=[128,128,1],
                output_shape=[128,128,1],
                n_conv_per_scale=3,
                conv_kernel_size=3,
                n_filters_per_scale=[16, 32, 64],
                if_droupout = True,
                droupout_rate=0.5,
                pretrained_weights=None) -> None:
        with tf.name_scope('input'):
            inputs = Input(input_shape, name='Inputs')

        with tf.name_scope('down_1'):
            down1 = self.DWT(inputs)
            for _ in range(n_conv_per_scale):
                down1 = Conv2D(n_filters_per_scale[0], conv_kernel_size, padding='same')(down1)
                down1 = Activation(activation='relu')(down1)
            if if_droupout:
                down1 = Dropout(droupout_rate)(down1)

        with tf.name_scope('down_2'):
            down2 = self.DWT(down1)
            for _ in range(n_conv_per_scale):
                down2 = Conv2D(n_filters_per_scale[1], conv_kernel_size, padding='same')(down2)
                down2 = Activation(activation='relu')(down2)
            if if_droupout:
                down2 = Dropout(droupout_rate)(down2)

        with tf.name_scope('middle'):
            middle = self.DWT(down2)
            for _ in range(n_conv_per_scale - 1):
                middle = Conv2D(n_filters_per_scale[2], conv_kernel_size, padding='same')(middle)
                middle = Activation(activation='relu')(middle)
            middle = Conv2D(n_filters_per_scale[1] * 4, conv_kernel_size, padding='same')(middle)
            middle = Activation(activation='relu')(middle)
            if if_droupout:
                middle = Dropout(droupout_rate)(middle)

        with tf.name_scope('up_1'):
            up1 = self.IWT(middle, output_shape[0] // 8)
            up1 = up1 + down2
            for _ in range(n_conv_per_scale - 1):
                up1 = Conv2D(n_filters_per_scale[1], conv_kernel_size, padding='same')(up1)
                up1 = Activation(activation='relu')(up1)
            up1 = Conv2D(n_filters_per_scale[0] * 4, conv_kernel_size, padding='same')(up1)
            up1 = Activation(activation='relu')(up1)
            # if if_droupout:
            #     up1 = Dropout(droupout_rate)(up1)

        with tf.name_scope('up_2'):
            up2 = self.IWT(up1, output_shape[0] // 4)
            up2 = up2 + down1
            for _ in range(n_conv_per_scale):
                up2 = Conv2D(n_filters_per_scale[0], conv_kernel_size, padding='same')(up2)
                up2 = Activation(activation='relu')(up2)
            # if if_droupout:
            #     up2 = Dropout(droupout_rate)(up2)

        conv = Conv2D(4, 1, padding='same')(up2)
        outputs = self.IWT(conv, output_shape[0] // 2)
        outputs = inputs + outputs

        self.model = Model(inputs, outputs)

        if pretrained_weights:
            self.model.load_weights(pretrained_weights)

    def SSIM(self, y_true, y_pred):
        return tf.reduce_mean(tf.image.ssim(y_true, y_pred, 1.0))


    def PSNR(self, y_true, y_pred):
        return tf.reduce_mean(tf.image.psnr(y_true, y_pred, 1.0))


    def DWT(self, inputs):
        x01 = inputs[:, 0::2] / 2
        x02 = inputs[:, 1::2] / 2
        x1 = x01[:, :, 0::2]
        x2 = x02[:, :, 0::2]
        x3 = x01[:, :, 1::2]
        x4 = x02[:, :, 1::2]
        x_LL = x1 + x2 + x3 + x4
        x_HL = -x1 - x2 + x3 + x4
        x_LH = -x1 + x2 - x3 + x4
        x_HH = x1 - x2 - x3 + x4
        return tf.concat((x_LL, x_HL, x_LH, x_HH), axis=-1)


    def IWT(self, inputs, size):
        in_shape = tf.shape(inputs)

        batch_size = in_shape[0]
        height = in_shape[1]
        width = in_shape[2]
        n_channels = inputs.shape[3] // 4

        outputs = tf.zeros([batch_size, 2 * height, 2 * width, n_channels])

        x1 = inputs[..., 0:n_channels] / 2
        x2 = inputs[..., n_channels:2 * n_channels] / 2
        x3 = inputs[..., 2 * n_channels:3 * n_channels] / 2
        x4 = inputs[..., 3 * n_channels:4 * n_channels] / 2

        x_EE = x1 - x2 - x3 + x4
        x_OE = x1 - x2 + x3 - x4
        x_EO = x1 + x2 - x3 - x4
        x_OO = x1 + x2 + x3 + x4

        # 这里不能使用KerasTensor，会有Shape报错，所以传一个size进来使用
        height_range_E = 2 * tf.range(size)
        height_range_O = height_range_E + 1
        width_range_E = 2 * tf.range(size)
        width_range_O = width_range_E + 1

        scatter_nd_perm = [2, 1, 3, 0]
        outputs_reshaped = tf.transpose(outputs, perm=scatter_nd_perm)

        combos_list = [
            ((height_range_E, width_range_E), x_EE),
            ((height_range_O, width_range_E), x_OE),
            ((height_range_E, width_range_O), x_EO),
            ((height_range_O, width_range_O), x_OO),
        ]
        for (height_range, width_range), x_comb in combos_list:
            h_range, w_range = tf.meshgrid(height_range, width_range)
            h_range = tf.reshape(h_range, (-1,))
            w_range = tf.reshape(w_range, (-1,))
            combo_indices = tf.stack([w_range, h_range], axis=-1)
            combo_reshaped = tf.transpose(x_comb, perm=scatter_nd_perm)
            outputs_reshaped = tf.tensor_scatter_nd_add(
                outputs_reshaped,
                indices=combo_indices,
                updates=tf.reshape(combo_reshaped, (-1, n_channels, batch_size)),
            )

        inverse_scatter_nd_perm = [3, 1, 0, 2]
        outputs = tf.transpose(outputs_reshaped, perm=inverse_scatter_nd_perm)
        return outputs


    def train(self,
            X,Y,
            validation_data,
            train_epochs,
            train_batch_size,
            base_dir,
            train_learning_rate=0.001,
            train_reduce_lr=ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_delta=0),
            name='my_models',
            train_checkpoint='best_weights.hdf5',
            log_dir='logs',
            ):

        save_path = os.path.join(base_dir, name)

        self.model.compile(optimizer=Adam(learning_rate=train_learning_rate, epsilon=1e-6), loss='mae',
                    metrics=['mse'])

        model_checkpoint = ModelCheckpoint(os.path.join(save_path, train_checkpoint), monitor='val_loss', verbose=1, save_best_only=True,
                                        save_weights_only=True, save_freq='epoch')
        tensor_board = TensorBoard(os.path.join(save_path, log_dir), write_graph=True, profile_batch=0)
        
        if train_reduce_lr is not None:
            callback_funcs = [model_checkpoint, tensor_board, train_reduce_lr] 
        else:
            callback_funcs = [model_checkpoint, tensor_board]

        history = self.model.fit(x=X, y=Y, 
                                validation_data=validation_data,
                                epochs=train_epochs,
                                batch_size=train_batch_size,
                                callbacks=callback_funcs)
        return history
        
    def predict(self, input):
        res = self.model.predict(input)
        return res
