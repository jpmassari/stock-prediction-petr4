import tensorflow as tf
from keras.callbacks import Callback
from keras.models import Sequential

class MyCallback(Callback):

	def on_epoch_end(self, epoch, model: Sequential, logs=None):
		# Your custom code to be executed at the end of each epoch
		print("Epoch {} completed".format(epoch))
		model.save("v2")
		tf.keras.callbacks.TensorBoard(log_dir='logs/', histogram_freq=1)