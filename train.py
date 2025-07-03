from model_building import build_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator

train_path = 'dataset/'
batch_size = 32
img_size = (224, 224)

datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)
train_gen = datagen.flow_from_directory(train_path, target_size=img_size, batch_size=batch_size, class_mode='categorical', subset='training')
val_gen = datagen.flow_from_directory(train_path, target_size=img_size, batch_size=batch_size, class_mode='categorical', subset='validation')

model = build_model(num_classes=train_gen.num_classes)
model.fit(train_gen, validation_data=val_gen, epochs=10)

model.save("model/poultry_model.h5")
