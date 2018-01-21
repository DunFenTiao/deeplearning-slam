import os
from PIL import Image
import pylab as pl
from keras.models import model_from_json
from custom_layers_v2 import Cosine,exponential_loss as EL,identity_loss,Aver_4,Aver_6,WeightedLayer
import numpy as np
from data_augmentation import *

Thre = 0.7

def reshape(arr):
    """
    reshape  array from (240,320,3) to (3,320,240)
    """
    reshape_arr = np.empty((3,240,320),dtype='float32')
    reshape_arr[0,:,:] = arr[:,:,0]
    reshape_arr[1,:,:] = arr[:,:,1]
    reshape_arr[2,:,:] = arr[:,:,2]
    return reshape_arr
    
#load model 
model_for_predict = model_from_json(open('model_for_predict_4parts.json','r').read(), custom_objects={"Cosine": Cosine,"Aver_4": Aver_4,'exponential_loss':EL,'Aver_6':Aver_6,'WeightedLayer':WeightedLayer})
model_for_predict.load_weights('model_for_predict_4parts.h5')


import time


#load data
samples = []
testset = []
pairs = []
preds_rotlist = []
rotx = []
y1 = []
y2 = []

for rot in range(0,181,10):
    pairs = []

    img2 = Image.open('../place_dataset/cc_left/0079.jpg')
    img2 = reshape(np.array(img2.resize((320,240),Image.ANTIALIAS),dtype='float32')) / 255.0

    img4 = Image.open('../place_dataset/cc_left/0081.jpg')
    img4 = reshape(np.array(img4.resize((320,240),Image.ANTIALIAS),dtype='float32')) / 255.0
    
    
    img1 = Image.open('../place_dataset/cc_left/1579.jpg')
    img1 = reshape(np.array(img1.resize((320,240),Image.ANTIALIAS),dtype='float32')) / 255.0
    img1 = random_rotation_((img1) , rot)	    
	
    pairs.append([img1,img2])

    img3 = Image.open('../place_dataset/cc_left/1581.jpg')
    img3 = reshape(np.array(img3.resize((320,240),Image.ANTIALIAS),dtype='float32')) / 255.0
    img3 = random_rotation_((img3) , rot)

	
    pairs.append([img3,img4])
    pairs = np.asarray(pairs)

    start = time.clock()
    an=pairs[:,0]
    po=pairs[:,1]
    an_1 = an[:,:,0:120,0:160] 
    an_2 = an[:,:,0:120,160:320]
    an_3 = an[:,:,120:240,0:160]
    an_4 = an[:,:,120:240,160:320]

    pos_1 = po[:,:,0:120,0:160]  
    pos_2 = po[:,:,0:120,160:320]
    pos_3 = po[:,:,120:240,0:160]
    pos_4 = po[:,:,120:240,160:320]  
    input_ap = [an_1,an_2,an_3,an_4,pos_1,pos_2,pos_3,pos_4]    
    test_preds = model_for_predict.predict(input_ap)
    #test_preds = model_for_predict.predict([pairs[:,0],pairs[:,1]])
    end = time.clock()
    print test_preds
    #preds_rotlist.append(rot,test_preds[0],test_preds[1])
    rotx.append(rot)
    y1.append(test_preds[0])
    y2.append(test_preds[1])
    #print "time  %s" %(end-start)
    

#print preds_rotlist
print rotx
print y1
pl.plot(rotx,y1)
pl.plot(rotx,y2)
pl.title('img pair')
pl.xlabel('rot angle')
pl.ylabel('acc')

pl.show()


	