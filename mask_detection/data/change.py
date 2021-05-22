
from glob import glob
import os
import shutil

base_dir=r"C:\Users\Shazzer\Desktop\Detection-Of-COVID-19-Mask\mask_detection\labels"
mask_dir=os.path.join(base_dir, 'mask')
no_mask_dir=os.path.join(base_dir,'no_mask')
labels_dir=os.path.join(base_dir,'labels')

os.listdir(mask_dir)
print(len(os.listdir(mask_dir)))

os.chdir(mask_dir)

print(glob('*.txt'))


for i in glob('*.txt'):
    src=os.path.join(mask_dir,i)
    dst=os.path.join(labels_dir,i)
    shutil.move(src, dst)
    
    
os.chdir(no_mask_dir)

for i in glob('*.txt'):
    src=os.path.join(no_mask_dir,i)
    dst=os.path.join(labels_dir,i)
    shutil.move(src, dst)


for i in os.listdir(no_mask_dir):
    src=os.path.join(no_mask_dir,i)
    dst=os.path.join(mask_dir,i)
    shutil.move(src,dst)
    
    
    
train_dir=r'C:\Users\Shazzer\Desktop\train'
labels_dir=r'C:\Users\Shazzer\Desktop\labels'
os.mkdir(labels_dir)
os.chdir(train_dir)
for i in glob('*.txt'):
    src=os.path.join(train_dir,i)
    dst=os.path.join(labels_dir,i)
    shutil.move(src, dst)
    
    
base_dir=r'C:\Users\Shazzer\Desktop\train'
valid_dir=r'C:\Users\Shazzer\Desktop\Detection-Of-COVID-19-Mask\mask_detection\data\images\valid'
valid_labels_dir=r'C:\Users\Shazzer\Desktop\Detection-Of-COVID-19-Mask\mask_detection\data\labels\valid'

os.chdir(base_dir)

for i in glob('*.txt'):
    src=os.path.join(base_dir,i)
    dst=os.path.join(valid_labels_dir,i)
    shutil.move(src, dst)

for i in os.listdir(base_dir):
    src=os.path.join(base_dir,i)
    dst=os.path.join(valid_dir,i)
    shutil.move(src, dst)



