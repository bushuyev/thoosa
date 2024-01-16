import tensorflow as tf
from tensorflow.keras.layers import Dense, Flatten, Conv2D
from tensorflow.keras import Model


class SimpModel(Model):
    def __init__(self, *args, **kwargs):
        super(SimpModel, self).__init__(*args, **kwargs)
        self.dense_1 = Dense(2, name="test_in", input_dim=2)
        self.dense_2 = Dense(1, name="test_out")

    #Call function used to make predictions from Rust
    @tf.function
    def call(self, inputs):
        x = self.dense_1(inputs)
        return self.dense_2(x)

    #Train function called from Rust which uses the keras model innate train_step function
    @tf.function
    def training(self, train_data):
        loss = self.train_step(train_data)['loss']
        return loss


if __name__ == '__main__':

    model = SimpModel()

    model.compile(optimizer ='sgd', loss='mse', metrics = ['mae'])

    pred_output = model.call.get_concrete_function(
        tf.TensorSpec(shape=[1, 2],
                      dtype=tf.float32,
                      name='inputs')
    )

    #Get concrete function for the training method
    train_output = model.training.get_concrete_function(
        (
            tf.TensorSpec(shape=[1,2], dtype=tf.float32, name="training_input"),
            tf.TensorSpec(shape=[1,1], dtype=tf.float32, name="training_target")
        )
    )
    model.save('../data_local/simp_model', save_format='tf', signatures={'train': train_output, 'pred': pred_output})


    print("Ok")