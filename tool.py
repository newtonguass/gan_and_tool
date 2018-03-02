import numpy as np
import matplotlib.pyplot as plt
import dlib
import face_recognition
import cv2
        
pretrained = '/usr/local/lib/python3.5/dist-packages/dlib-19.9.99-py3.5-linux-x86_64.egg/mmod_human_face_detector.dat'
cnn_face_detector = dlib.cnn_face_detection_model_v1(pretrained)



def display(image_, size, epoch=None, scale=None):
    if image_.shape[-1] == 3:
        display_color(image_, size, epoch=epoch, scale=scale)
    else:
        display_gray(image_, size, epoch=epoch, scale=scale)
    
def display_color(image_, size, epoch=None, scale=None):
    image = np.copy(image_).astype(np.float32)
    if scale:
        image = (image+1)*255/2
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
    plt.imshow(canvas.astype(np.uint8))
    if epoch is not None:
        plt.title(str(epoch))
    plt.show()
    
def display_gray(image_, size, epoch=None, scale=None):
    image = np.copy(image_).astype(np.float32)
    if scale:
        image = (image+1)*255/2
        image = image.round()
    image_width = image.shape[1]
    interval = 2
    canvas = np.zeros((((image_width+interval)*size)+interval,((image_width+interval)*size)+interval))
    for i in range(size):
        for j in range(size):
            temp = image[i*size+j]
            canvas[interval+i*(image_width+interval):interval+i*(image_width+interval)+image_width,
                       interval+j*(image_width+interval):interval+j*(image_width+interval)+image_width]=temp
    plt.imshow(canvas.astype(np.uint8), cmap='gray')
    if epoch is not None:
        plt.title(str(epoch))
    plt.show()
    
    
def image_augmentation(image, rate, flip=True):
    image2 = np.copy(image)
    if np.random.choice(2, 1)==1 and flip==True:
        image2 = np.flip(image2, 2)
    if rate>0:
        sun_glasses = np.random.uniform(0.99, -0.99, 1)[0]
        random = np.random.choice(100, len(image), 1)
        index = np.array([i for i in range(len(image))])
        aug = np.random.normal(0, 8, 4).astype(np.int)
        image2[index[random<int(rate*100)], 
               40-aug[0]:63+aug[1], 
               25-2*aug[2]:102+2*aug[3], :] = sun_glasses

    return image2

def crop_image(img):
    dets = cnn_face_detector(img, 1)
    if len(dets)>0:
        top, bottom, left, right = dets[0].rect.top(),dets[0].rect.bottom(),dets[0].rect.left(),dets[0].rect.right()
        y = (top+bottom)//2
        x = (right+left)//2
        length = (bottom-top)//2
        extended = (bottom-top)//3
        while 2*(length+extended)>img.shape[0] or 2*(length+extended)>img.shape[1]:
            extended //= 2
        top = y-length-extended
        bottom = y+length+extended
        left = x-length-extended
        right = x+length+extended
        if top< 0:
            bottom -= top
            top=0
        if bottom>img.shape[0]:
            top -= (bottom -img.shape[0])
            bottom = img.shape[0]
        if left<0:
            right -= left
            left=0
        if right>img.shape[1]:
            left -= (right-img.shape[1])
            right = img.shape[1]
        if len(img.shape)==3:
            return img[top:bottom, left:right, :]
        else:
            return img[top:bottom, left:right]
    else:
        return "not_find"
        
def normalize_face(im): 
    face_landmarks_list = face_recognition.face_landmarks(im)
    rows,cols = im.shape
    left = np.array(face_landmarks_list[0]['left_eye']).mean(0).astype(np.int)
    right = np.array(face_landmarks_list[0]['right_eye']).mean(0).astype(np.int)
    vector = right - left
    angle = -360 * np.arctan(-vector[1]/vector[0])/(np.pi*2)
    M = cv2.getRotationMatrix2D((cols/2,rows/2),angle,1)
    dst = cv2.warpAffine(im,M,(cols,rows))
    return dst
