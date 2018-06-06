import numpy as np
import cv2
import face_recognition            

def normalize_face(im): 
    face_landmarks_list = face_recognition.face_landmarks(im)
    if len(face_landmarks_list )==0:
        return "not_find"
    rows,cols = im.shape
    left = np.array(face_landmarks_list[0]['left_eye']).mean(0).astype(np.int)
    right = np.array(face_landmarks_list[0]['right_eye']).mean(0).astype(np.int)
    lip = np.array(face_landmarks_list[0]['bottom_lip']).mean(0).astype(np.int)
    chin = np.array(face_landmarks_list[0]['chin']).astype(np.int)
    vector = right - left
    middle = (left+right)//2
    angle = -360 * np.arctan(-vector[1]/vector[0])/(np.pi*2)
    M = cv2.getRotationMatrix2D((middle[0], middle[1]),angle,1)
    im = cv2.warpAffine(im,M,(cols,rows))
    face_landmarks_list = face_recognition.face_landmarks(im)
    if len(face_landmarks_list )==0:
        return "not_find"
    middle = np.array(face_landmarks_list[0]['nose_tip']).mean(0).astype(np.int)
    length = int(1.3*np.sqrt(((chin[0]-chin[-1])**2).sum()))
    h_l = length//16
    while middle[1]-h_l*9<0 or middle[1]+h_l*7>rows or middle[0]-h_l*8<0 or middle[0]+h_l*8>cols:
        h_l -= 1
    return im[middle[1]-h_l*9:middle[1]+h_l*7, middle[0]-h_l*8:middle[0]+h_l*8]



import dlib
pretrained = '/home/lianism/jupyter_notebook/master_thesis/session_save/mmod_human_face_detector.dat'
cnn_face_detector = dlib.cnn_face_detection_model_v1(pretrained)
def crop_image(img):
    dets =cnn_face_detector(img, 1)
    if len(dets)>0:
        top, bottom, left, right = dets[0].rect.top(),dets[0].rect.bottom(),dets[0].rect.left(),dets[0].rect.right()
        y = (top+bottom)//2
        x = (right+left)//2
        length = (bottom-top)//2
        extended = (bottom-top)//6
        top = y-length-extended
        bottom = y+length+extended
        left = x-length-extended
        right = x+length+extended
        while top<0 or left<0 or right>img.shape[1] or bottom>img.shape[0]:
            extended -= 1
            top = y-length-extended
            bottom = y+length+extended
            left = x-length-extended
            right = x+length+extended
        if len(img.shape)==3:
            return img[top:bottom, left:right, :]
        else:
            return img[top:bottom, left:right]
        
    else:
        return "not_find"
