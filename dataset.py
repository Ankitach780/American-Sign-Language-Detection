import cv2
import os 
import time
import uuid
from tensorflow.keras.preprocessing.image import ImageDataGenerator

IMAGES_PATH='Images'
labels=['hello','food','say','know','forget','sky','thanks','yes','no']

number_img=15
for label in labels:
    label_path=os.path.join(IMAGES_PATH,label)
    os.makedirs(label_path,exist_ok=True)
    cap=cv2.VideoCapture(0)
    print('collecting images for {}'.format(label))
    time.sleep(5)
    for imgnum in range(number_img):
        ret,frame=cap.read()
        imagename=os.path.join(label_path,'{}.jpg'.format(str(uuid.uuid1())))
        cv2.imwrite(imagename,frame)
        cv2.imshow('frame',frame)
        time.sleep(2)

        if cv2.waitKey(1) & 0xFF ==ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()
print('Images collection completed!')
datagen=ImageDataGenerator(
    rotation_range=10,
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.1,
    horizontal_flip=True,
    fill_mode='nearest'
)
for label in labels:
    label_path = os.path.join(IMAGES_PATH, label)
    print(f'Generating augmented images for {label}...')

    for filename in os.listdir(label_path):
        if filename.endswith('.jpg'):
            img_path = os.path.join(label_path, filename)
            img = cv2.imread(img_path)  # Read the image
            img = img.reshape((1,) + img.shape)  # Reshape for augmentation

            # Generate images
            aug_iter = datagen.flow(img, batch_size=1, save_to_dir=label_path,
                                    save_prefix='aug', save_format='jpg')
            for _ in range(10):  # Generate 5 images
                next(aug_iter)

print('Image generation completed!')