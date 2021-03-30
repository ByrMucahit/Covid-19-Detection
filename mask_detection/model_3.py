# Gerekli kütüphaneleri dahil edelim.
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Dense,Dropout,AveragePooling2D,Flatten,Input
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
import os
from tensorflow.keras.models import Sequential
from tensorflow.keras import regularizers
# Eğitim için kullanılacak GPU'nun kontrol edilmesi ve ayarlanması.
physical_devices=tf.config.experimental.list_physical_devices('GPU')
print('Available GPU device:',physical_devices)
tf.config.experimental.set_memory_growth(physical_devices[0],True)

# Bazı eğitim parametrelerinin tanıtılması.
batch_size=32
lr=1e-4

os.chdir('C:/Users/Shazzer/Desktop/mask_detection')

# Veri'nin konumunun eklenmesi
base_dir='C:/Users/Shazzer/Desktop/mask_detection/dataset'
base_dir = r'C:\Users\Shazzer\Desktop\mask_detection\dataset'
train_dir = os.path.join(base_dir,'train')
validation_dir = os.path.join(base_dir,'validation')
test_dir = os.path.join(base_dir,'test')

# Veri'nin etiketleri
categories=['incorrect_mask','with_mask','without_mask']

data_gen=ImageDataGenerator(rescale=1./255,
                            rotation_range=20,
                            width_shift_range=0.2,
                            height_shift_range=0.2,
                            zoom_range=0.15,
                            shear_range=0.15,
                            horizontal_flip=True,
                            fill_mode='nearest')

train_data=data_gen.flow_from_directory(
    train_dir,
    target_size=(224,224),
    color_mode='rgb',
    batch_size=batch_size,
    classes=categories)


val_gen=ImageDataGenerator(rescale=1./255)
val_data=val_gen.flow_from_directory(
    validation_dir,
    target_size=(224,224),
    color_mode='rgb',
    classes=categories,
    batch_size=batch_size)


test_gen=ImageDataGenerator(rescale=1./255)
test_data=test_gen.flow_from_directory(
    test_dir,
    target_size=(224,224),
    color_mode='rgb',
    classes=categories,
    batch_size=batch_size,
    shuffle=True)

# Ön eğitimli modeli yükleme ve katmanlarını dondurma.
conv_base = MobileNetV2(weights="imagenet", include_top=False,input_tensor=Input(shape=(224, 224, 3)))

for layer in conv_base.layers:
    layer.trainable=False

# Modelin özetini yazdırma.    
print(conv_base.summary())

# Kendi modelimizi tanımlayalım.
mask_model=Sequential()
mask_model.add(conv_base)
mask_model.add(AveragePooling2D(pool_size=(7,7)))
mask_model.add(Flatten())
mask_model.add(Dense(128,activation='relu'))
mask_model.add(Dropout(0.3))
mask_model.add(Dense(64,activation='relu',kernel_regularizer=regularizers.l2(0.001)))
mask_model.add(Dropout(0.2))
mask_model.add(Dense(3,activation='softmax'))

# Modelin eğitim parametrelerinin ayarlanması.
mask_model.compile(optimizer=Adam(learning_rate =lr , decay = lr/20),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Eğitim 
hst=mask_model.fit(train_data,batch_size=batch_size,epochs=17,steps_per_epoch=100,validation_data=val_data)

# Model eğitim verilerinin grafikleştirilmesi.
dict=hst.history
dict.keys()
acc=dict['accuracy']
loss=dict['loss']
val_acc=dict['val_accuracy']
val_loss=dict['val_loss']

epok=range(1,len(acc)+1)
plt.clf()
plt.plot(epok,acc,'bo',label='Eğitim Başarımı')
plt.plot(epok,val_acc,'b',label='Doğrulama Başarımı')
plt.title('Eğitim ve Doğrulama Başarımı')
plt.xlabel('Epok')
plt.ylabel('Başarım')
plt.legend()
plt.savefig('mask_model training statistic 1 when epok value is 17.png', dpi=500)

plt.figure()
plt.plot(epok,loss,'bo',label='Eğitim Kaybı')
plt.plot(epok,val_loss,'b',label='Doğrulama Kaybı')
plt.title('Eğitim ve Doğrulama Kaybı')
plt.xlabel('Epok')
plt.ylabel('Kayıp')
plt.legend()
plt.savefig('mask_model training statistic 2 when epok value is 17.png', dpi=500)
plt.show()



# Modelin test edilmesi.
prediction=mask_model.predict(test_data,verbose=2)
print(np.argmax(prediction,axis=1))

# Eğitilen modelin kaydedilmesi.
os.chdir(r'C:\Users\Shazzer\Desktop\mask_detection')
if os.path.isfile(r'C:\Users\Shazzer\Desktop\mask_detection\mask_detection_model_3.h5') is False:
    mask_model.save('mask_detection_model_3.h5')
    

from tensorflow.keras.models import load_model
new_model=load_model(r'C:\Users\Shazzer\Desktop\mask_detection/mask_detection_model_3.h5')
prediction=new_model.predict(test_data)
print(np.argmax(prediction,axis=1))



    
