#############################
#							#
#       DataLoader.py		#
#							#
#############################

# Import needed libraries
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt

#Function for loading input data using a preprocessing function
def loadPreprocessedData(training_dir, validation_dir, test_dir, seed, batch_size, preprocess_input):
    train_data_gen = ImageDataGenerator(rotation_range=30,
                                            height_shift_range=50,
                                            width_shift_range=50,
                                            zoom_range=0.3,
                                            horizontal_flip=True,
                                            vertical_flip=True,
                                            preprocessing_function=preprocess_input) 
    # Obtain a data generator with the 'ImageDataGenerator.flow_from_directory' method
    train_gen = train_data_gen.flow_from_directory(directory=training_dir,
                                                        target_size=(256,256),
                                                        color_mode='rgb',
                                                        classes=None, # can be set to labels
                                                        class_mode='categorical',
                                                        batch_size=batch_size,
                                                        shuffle=True,
                                                        seed=seed)


    valid_data_gen = ImageDataGenerator(preprocessing_function=preprocess_input)

    valid_gen = valid_data_gen.flow_from_directory(directory=validation_dir,
                                                target_size=(256,256),
                                                color_mode='rgb',
                                                classes=None, # can be set to labels
                                                class_mode='categorical',
                                                batch_size=batch_size,
                                                shuffle=False,
                                                seed=seed)
    
    test_data_gen_np = ImageDataGenerator(rescale=1/255.)

    test_gen_np = test_data_gen_np.flow_from_directory(directory=test_dir,
                                                    target_size=(256,256),
                                                    color_mode='rgb',
                                                    classes=None, # can be set to labels
                                                    class_mode='categorical',
                                                    batch_size=1,
                                                    shuffle=True,
                                                    seed=seed)

    test_data_gen_p = ImageDataGenerator(preprocessing_function=preprocess_input)

    test_gen_p = test_data_gen_p.flow_from_directory(directory=test_dir,
                                                    target_size=(256,256),
                                                    color_mode='rgb',
                                                    classes=None, # can be set to labels
                                                    class_mode='categorical',
                                                    batch_size=1,
                                                    shuffle=True,
                                                    seed=seed)

    # Return dataset
    return {"train": train_gen, "validation": valid_gen, "test_np": test_gen_np, "test": test_gen_p}