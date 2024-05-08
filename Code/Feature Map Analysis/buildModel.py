#############################
#							#
#       BuildModel.py		#
#							#
#############################

# Base
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress warnings

# Model builder function (VGG16 Transfer Learning)
def buildModelVGG16ExplicitTL(input_shape, classes, tfk, tfkl, seed):
    # VGG16

    # Supernet
    supernet = tfk.applications.VGG16(
        include_top=False,
        weights="imagenet",
		input_shape=input_shape
	)

    # Use the supernet as feature extractor
    supernet.trainable = False

    tl_model = tfk.Sequential()
    for layer in supernet.layers: tl_model.add(layer)
    tl_model.add(tfkl.Flatten(name='Flattening'))
    tl_model.add(tfkl.Dropout(0.3, seed=seed))
    tl_model.add(tfkl.Dense(
        256,
        activation='relu',
        kernel_initializer=tfk.initializers.GlorotUniform(seed)))
    tl_model.add(tfkl.Dropout(0.3, seed=seed))
    tl_model.add(tfkl.Dense(
        classes,
        kernel_initializer=tfk.initializers.GlorotUniform(seed)))

    # Compile the model
    tl_model.compile(loss=tfk.losses.CategoricalCrossentropy(),
                     optimizer=tfk.optimizers.Adam(), metrics=['accuracy'])
    print()
    print('VGG16 Transfer Learning:')
    tl_model.summary()

    # Return the model
    return tl_model

# Model builder function (VGG16 Fine Tuning)
def buildModelVGG16ExplicitFT(model, tfk):
    # VGG16 Fine Tuning

	# Freeze first N layers, e.g., until 14th
	for i, layer in enumerate(model.layers[14:17]):
		layer.trainable = True

	# Compile the model
	model.compile(loss=tfk.losses.CategoricalCrossentropy(),
	              optimizer=tfk.optimizers.Adam(1e-5), metrics=['accuracy'])
	print()
	print('VGG16 Fine Tuning:')
	model.summary()

    # Return the model
	return model