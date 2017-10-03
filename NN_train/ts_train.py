import os
import sys

import h5py 
import keras
import random
import time

from keras.optimizers import SGD
from keras.callbacks import ModelCheckpoint, LearningRateScheduler 
import keras.backend as K
from keras.utils import np_utils

import numpy as np
import cv2

import keras_utils 

from alexnet import ts_alx_net
from resnet.resnet import ResnetBuilder

#import cPickle
import _pickle as cPickle

USE_PARTIAL_LOADING = False
NUM_CLASSES = 5
network_type = "resnet"
imW=227

def preprocess_image(img, resize=False):
    """
    central crop image, and subtract mean
    """
    if resize:
        img = cv2.resize(img, (256,256))

    img_mean = np.zeros((3,256,256))
    img_mean[0,:,:] = 104
    img_mean[1,:,:] = 117
    img_mean[2,:,:] = 123

    imW=227
    DX=int((256-imW)/2)

    # convert BGR to RGB (to match Xiaodong)
    #img = img[:,:,::-1]

    H = img.shape[0]
    W = img.shape[1]
            
    # need to subtract mean, and central crop
    img = np.transpose(img, (2,0,1))
    img = img.astype(np.float32)
    img = img-img_mean

    img = img[:, DX:DX+imW, DX:DX+imW]
    img = img.reshape(1, 3, imW, imW).astype('float32') / 255

    return img#[np.newaxis, :]


def count_train_val_frames():
    """
    """
    import glob
    data_loc = cfg['DATA_LOC']
    '''
    r_imgs = glob.glob(data_loc+"/Red/"+'*.png') #0
    y_imgs = glob.glob(data_loc+"/Yellow/"+'*.png') #1
    g_imgs = glob.glob(data_loc+"/Green/"+'*.png') #2
    '''
    r_imgs = glob.glob(data_loc+"/0/"+'*.jpg') #0
    y_imgs = glob.glob(data_loc+"/1/"+'*.jpg') #1
    g_imgs = glob.glob(data_loc+"/2/"+'*.jpg') #2
    u_imgs = glob.glob(data_loc+"/4/"+'*.jpg') #4

    all_imgs = r_imgs + y_imgs + g_imgs + u_imgs
    N_train = len(all_imgs) #len(r_imgs) + len(y_imgs) + len(g_imgs)
    # TODO: -----------------------------
    N_val = N_train

    print(len(all_imgs), len(r_imgs), len(y_imgs), len(g_imgs))
    # UNKNOWN=4 GREEN=2 YELLOW=1 RED=0
    with open("temp.txt", 'w') as fp:
        for img in r_imgs:
            fp.write(img+" 0\n")
        for img in y_imgs:
            fp.write(img+" 1\n")
        for img in g_imgs:
            fp.write(img+" 2\n")
        for img in u_imgs:
            fp.write(img+" 4\n")

    return N_train, N_val


def generate_batch(batch_size, mode):
    fname = "temp.txt"
    vlist = [x.split() for x in open(fname,'r').read().splitlines()]

    X = np.zeros((batch_size, 3, 227, 227))
    y = np.zeros((batch_size, NUM_CLASSES))

    while 1:
        i = 0
        while i<batch_size:
            k = np.random.randint(len(vlist))    
            filename = vlist[k][0] #os.path.join(BIWI_DBPATH, 'biwi_img', vlist[k][0])

            if os.stat(filename).st_size == 0:
                continue

            img = cv2.imread(filename)

            if img is None:
                print ('Error reading file %s'%filename)
                continue

            class_gt = int(vlist[k][1]) #poses[vlist[k][0]]
            #do_mirror = int(vlist[k][1])
            #crop_scale = float(vlist[k][2])

            crop_scale = 1.0
            do_mirror = 0
            # central crop image to augment data
            if crop_scale<1.0:
                H = img.shape[0]
                W = img.shape[1]
                dy = int(H*(1-crop_scale)*0.5)
                dx = int(W*(1-crop_scale)*0.5)
                img = img[dy:-dy, dx:-dx, :]
                img = cv2.resize(img, (W,H))

            if do_mirror:
                img = np.fliplr(img)

            img = preprocess_image(img, True)

            X[i,:,:,:] = img
            y[i,:] = np_utils.to_categorical(class_gt, NUM_CLASSES)
            i += 1

        yield X, y

 
def train_net(cfg):
    """
    train net
    """

    if network_type == "alexnet":
        K.set_image_dim_ordering('th')
        net = ts_alx_net.build_alexnet(num_out = 5)
        net1 = ts_alx_net.build_alexnet(num_out = 3)
    
        # init
        if USE_PARTIAL_LOADING:
            init_weightfile = 'models/ts_alx.h5' 
            net1.load_weights(init_weightfile)
        
            for i, layer in enumerate(net.layers):
                if net1.layers[i].name == net.layers[i].name:
                    print("set weights for net layer: ", net.layers[i].name, " with values in net1["+str(net1.layers[i].name)+"] ", len(layer.get_weights()))
                    net.layers[i].set_weights(net1.layers[i].get_weights())
                else:
                    print("Not setting weights for net layer: ", net.layers[i].name, " with values in net1["+str(net1.layers[i].name)+"] ", len(layer.get_weights()))
    
            #net.layers[i].set_weights(weights)
        else:
            init_weightfile = 'models/ts_alx.h5' 
            #net.load_weights(init_weightfile)

    elif network_type == "resnet":
        K.set_image_dim_ordering('th')
        net = ResnetBuilder.build_resnet_18((3, imW, imW), NUM_CLASSES)

    sgd = SGD(lr=cfg['LEARNING_RATE'], decay=cfg['DECAY'], momentum=cfg['MOMENTUM'], nesterov=cfg['NESTEROV'])
    #net.compile(optimizer=sgd, loss='mse')
    sgd = SGD(lr=0.02, momentum=0.8)
    net.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=['accuracy'])
    '''
    # also freeze the early layers
    for layer in net.layers:
        if layer.name in ['conv1_1','conv1_2','conv2_1','conv2_2']:
            layer.trainable=False
    '''

    model_filename = 'ts_'+network_type+'_model.h5'
    logfilename = keras_utils.get_logger_filename('log_net')
    logger = keras_utils.init_logger(logfilename)
    history = keras_utils.MyLogger(logfilename, display=20, logger=logger)
    checkpoint = ModelCheckpoint(filepath=model_filename, monitor='val_acc', save_best_only=True)
    lr_schedule = keras_utils.StepLearningRateScheduler(cfg['LEARNING_RATE'], 
        cfg['LEARNING_RATE_DECAY_RATIO'], cfg['LEARNING_RATE_DECAY_STEP'])
    #tbCallBack = keras.callbacks.TensorBoard(log_dir='./Graph', histogram_freq=0, write_graph=True, write_images=True)


    N_train, N_val = count_train_val_frames()
    N_train = N_train/cfg['BATCH_SIZE']*cfg['BATCH_SIZE']
    N_val = N_val/cfg['BATCH_SIZE']*cfg['BATCH_SIZE']

    logger.info('training: N_train %d N_val %d'%(N_train, N_val))

    # print cfg info in logger file
    for key in cfg.keys():
        logger.info('%s: %s'%(key, cfg[key]))

    #loss = net.evaluate_generator(generate_biwi_batch(cfg['BATCH_SIZE'], split, 'test'), N_val)
    #logger.info('Before training: val_loss %f'%loss)
    net.fit_generator(generate_batch(cfg['BATCH_SIZE'], 'train'),
                        samples_per_epoch=N_train/cfg['BATCH_SIZE'], 
                        nb_epoch=cfg['NUM_EPOCHS'],
                        validation_data=generate_batch(cfg['BATCH_SIZE'], 'test'),
                        nb_val_samples=N_val/cfg['BATCH_SIZE'],
                        verbose=0,
                        callbacks=[checkpoint,history,lr_schedule])


def test_net():

    if network_type == 'alexnet':
        net = ts_alx_net.build_alexnet()
    elif network_type == "resnet":
        K.set_image_dim_ordering('th')
        net = ResnetBuilder.build_resnet_18((3, imW, imW), NUM_CLASSES)

    #net.compile(optimizer='sgd',loss='mse')
    model_filename = 'ts_'+network_type+'_model.h5'
    net.load_weights(model_filename)

    fname = "temp.txt"
    vlist = [x.split() for x in open(fname,'r').read().splitlines()]
    N = len(vlist)

    fp = open("eval.txt", 'w')
    incorrect_count = 0
    for i in range(N):

        fname=vlist[i][0]

        if os.stat(fname).st_size == 0:
            continue

        img = cv2.imread(fname)

        img = preprocess_image(img, True)

        pp = net.predict_on_batch(img)

        pp = list(np.squeeze(pp))

        pp = pp.index(max(pp))
        pg = int(vlist[i][1])

        if pp != pg:
            incorrect_count += 1
            string_eval = ('file: %s GT: %d Eval: %d Incorrect: %d\n'%(fname,pg,pp,incorrect_count))
            print(string_eval)
            fp.write(string_eval)
            #cv2.imshow("Incorrect", cv2.imread(fname))
            #cv2.waitKey(0)
    fp.close()
 
if __name__=='__main__':

    cfg = keras_utils.load_config_file('ts_config.yaml')

    if len(sys.argv) > 1 and sys.argv[1] == 'train':
        train_net(cfg)

    test_net()
