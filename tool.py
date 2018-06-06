import numpy as np
import matplotlib.pyplot as plt
import cv2
        

def display(image_, size, epoch=None, scale=None, array=False, save=None):
    if image_.shape[-1] == 3:
        return display_color(image_, size, epoch=epoch, scale=scale, array=array, save=save)
    else:
        return display_gray(image_, size, epoch=epoch, scale=scale, array=array, save=save)
    
def display_color(image_, size, epoch=None, scale=None, array=False, save=None):
    image = np.copy(image_).astype(np.float32)
    if scale:
        for i in range(image.shape[0]):
            image[i] = image[i] - image[i].min()
            image[i] = image[i]*255/image[i].max()
        #image = (image+1)*255/2
        image = image.round()
    image_width = image.shape[1]
    interval = 2
    canvas = np.zeros((((image_width+interval)*size)+interval,((image_width+interval)*size)+interval, 3))
    for i in range(size):
        for j in range(size):
            temp = image[i*size+j]
            canvas[interval+i*(image_width+interval):interval+i*(image_width+interval)+image_width,
                   interval+j*(image_width+interval):interval+j*(image_width+interval)+image_width,
                   :]=temp
    if epoch is not None:
        plt.title(str(epoch))
    if array:
        return canvas
    elif save is not None:
        plt.imsave(save, canvas.astype(np.uint8)) 
        return None
    plt.imshow(canvas.astype(np.uint8))
    
    plt.show()
    
def display_gray(image_, size, epoch=None, scale=None, array=False, save=None):
    image = np.copy(image_).astype(np.float32)
    if scale:
        for i in range(image.shape[0]):
            image[i] = image[i] - image[i].min()
            image[i] = image[i]*255/image[i].max()
        #image = (image+1)*255/2
        image = image.round()
    image_width = image.shape[1]
    interval = 2
    canvas = np.zeros((((image_width+interval)*size)+interval,((image_width+interval)*size)+interval))
    for i in range(size):
        for j in range(size):
            temp = image[i*size+j]
            canvas[interval+i*(image_width+interval):interval+i*(image_width+interval)+image_width,
                       interval+j*(image_width+interval):interval+j*(image_width+interval)+image_width]=temp
    
    if array:
        return canvas
    
    elif save is not None:
        cv2.imwrite(save, canvas.astype(np.uint8))
        return None
    if epoch is not None:
        plt.title(str(epoch))
    plt.imshow(canvas.astype(np.uint8), cmap='gray')
    plt.show()
    
    
def image_augmentation(image, image2, flip=True):
    image_ = np.copy(image)
    image2_ = np.copy(image2)
    if np.random.choice(2, 1)==1 and flip==True:
        image2_ = np.flip(image2_, 2)
        image_ = np.flip(image_, 2)

    return image_, image2_


