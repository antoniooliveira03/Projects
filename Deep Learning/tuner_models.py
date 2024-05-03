from keras import layers, models, regularizers
from keras.optimizers import Adam 

def build_model(hp):
    
    """
    Build Model is a modification of model_example to work with keras tuner, 
    were the values for units, dropout chance and optimizer learning_rate are 
    optimized
    """

    # hardcoded img_size and number of labels because of hyperparmeters errors
    model = models.Sequential()

    model.add(layers.Conv2D(hp.Int('Conv2D_1', min_value=128, max_value=256, step=32), 
                            (3, 3), activation="relu"))
    
    model.add(layers.MaxPooling2D((2, 2))) 
    
    model.add(layers.Dropout(hp.Float('Dropout_1', min_value=0.1, max_value=0.3, step=0.1)))

    model.add(layers.Conv2D(hp.Int('Conv2D_2', min_value=32, max_value=128, step=32), 
                            (3, 3), activation='relu'))
   
    model.add(layers.MaxPooling2D((2, 2))) 
    
    model.add(layers.Dropout(hp.Float('Dropout_2', min_value=0.1, max_value=0.3, step=0.1)))

    model.add(layers.Conv2D(hp.Int('Conv2D_3', min_value=64, max_value=256, step=64), 
                            (3, 3), activation='relu'))
    
    model.add(layers.MaxPooling2D((2, 2)))
    
    model.add(layers.Dropout(hp.Float('Dropout_3', min_value=0.1, max_value=0.3, step=0.1)))

    model.add(layers.Conv2D(hp.Int('Conv2D_4', min_value=128, max_value=512, step=64), 
                            (3, 3), activation='relu'))
    
    model.add(layers.MaxPooling2D((2, 2))) 

    model.add(layers.Flatten())
    model.add(layers.Dropout(hp.Float('Dropout_4', min_value=0.2, max_value=0.5, step=0.1)))

    model.add(layers.Dense(hp.Int('Dense', min_value=32, max_value=256, step=32), 
                           activation='relu'))

    model.add(layers.Dense(7, activation='softmax'))
    
    # Compile the model
    hp_learning_rate = hp.Choice('learning_rate', values=[1e-2, 1e-3])
    optimizer = Adam(learning_rate=hp_learning_rate)
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

    return model



def model_tuned():

    """
    Model Tuned is the application of the best hyperparameter from build_model
    """

    model = models.Sequential()
    #Conv2D_1
    model.add(layers.Conv2D(128, (3, 3), activation="relu", input_shape=(112, 150, 3)))
    model.add(layers.MaxPooling2D((2, 2))) 
    #Dropout_1
    model.add(layers.Dropout(0.1))
    #Conv2D_2
    model.add(layers.Conv2D(128, (3, 3),activation='relu'))
    model.add(layers.MaxPooling2D((2, 2))) 
    #Dropout_2
    model.add(layers.Dropout(0.2))
    #Conv2D_3
    model.add(layers.Conv2D(192, (3, 3),activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    #Dropout_3
    model.add(layers.Dropout(0.1))
    #Conv2D_4
    model.add(layers.Conv2D(192, (3, 3),activation='relu'))
    model.add(layers.MaxPooling2D((2, 2))) 
    model.add(layers.Flatten())
    #Dropout_4
    model.add(layers.Dropout(0.4))
    #Dense
    model.add(layers.Dense(192))
    model.add(layers.Dense(7, activation='softmax'))
    
    # Compile the model
    optimizer = Adam(learning_rate=0.001)
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

    return model

# Best hyperparameters from Tuner Search
best_hps = {'Conv2D_1': 128,
    'Dropout_1': 0.1,
    'Conv2D_2': 128,
    'Dropout_2': 0.2,
    'Conv2D_3': 192,
    'Dropout_3': 0.1,
    'Conv2D_4': 192,
    'Dropout_4': 0.4,
    'Dense': 192,
    'learning_rate': 0.001,
    'tuner/epochs': 50,
    'tuner/initial_epoch': 17,
    'tuner/bracket': 1,
    'tuner/round': 1,
    'tuner/trial_id': '0077'}