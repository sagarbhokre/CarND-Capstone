# Jinwei Gu
# 2016/7/18

from __future__ import print_function

import numpy as np
import math
import time

import keras 

from keras.callbacks import Callback 
from keras import backend as K

#--------------------------------------------------------------------------------
def load_config_file(config_file):
    import yaml
    with open(config_file,'r') as f:
        cfg = yaml.load(f)
    return cfg

def get_logger_filename(prefix):
    a = time.localtime()
    filename = '%d-%02d-%02d-%02d:%02d:%02d.log'%(a.tm_year, a.tm_mon,
            a.tm_mday, a.tm_hour, a.tm_min, a.tm_sec)
    return prefix+'-'+filename

def init_logger(logfilename):
    import logging
    logger = logging.getLogger()

    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(message)s')

    fh = logging.FileHandler(logfilename)
    fh.setLevel(logging.INFO)
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    return logger

def print_cfg(cfg, logger=None):
    if logger:
        print = logger.info
    print('--------------- config ---------------')
    for x in cfg:
        print('%s : %s'%(x, cfg[x]))
    print('--------------- config ---------------')


# convert the convolution weights from theano to tensorflow
# (correlation, similar to caffe) for convolution layers
#
# copied code from 
#
# https://github.com/fchollet/keras/wiki/Converting-convolution-kernels-from-Theano-to-TensorFlow-and-vice-versa
#
# Jinwei Gu. 2016/8/7
def theano2tensorflow(model, weightfile=None):
    from keras.utils.np_utils import convert_kernel
    import tensorflow as tf

    if weightfile:
        model.load_weights(weightfile)
    ops = []
    for layer in model.layers:
       if layer.__class__.__name__ in ['Convolution1D', 'Convolution2D']:
          original_w = K.get_value(layer.W)
          converted_w = convert_kernel(original_w)
          ops.append(tf.assign(layer.W, converted_w).op) 
    K.get_session().run(ops)
    return model

def tensorflow2theano(model, weightfile=None):
    from keras.utils.np_utils import convert_kernel

    if weightfile:
        model.load_weights(weightfile)
    for layer in model.layers:
       if layer.__class__.__name__ in ['Convolution1D', 'Convolution2D']:
          original_w = K.get_value(layer.W)
          converted_w = convert_kernel(original_w)
          K.set_value(layer.W, converted_w)
    return model

def save_weights_to_pkl(net, pkl_filename):
    """
    save net weight to pkl file. Currently only save weights for conv
    layer and dense layer.

    NOTE: for most applications, we should call net.save_weights() and
    save to *.h5 file directly. Often the file is much smaller
    """
    dat={}
    for layer in net.layers:
        if layer.__class__.__name__ in ['Convolution1D', 'Convolution2D', 'Dense']:
            param = [layer.W.get_value(), layer.b.get_value()]
            dat[layer.name] = param

    import cPickle
    with open(pkl_filename,'w') as f:
        cPickle.dump(dat, f)

def load_weights_from_pkl(net, pkl_filename):
    """
    load weights from pkl file. Currently only support conv layer and
    dense layer.

    NOTE: for most applications, we should call net.load_weights() and
    load from *.h5 file directly.
    """
    import cPickle
    with open(pkl_filename,'r') as f:
        dat = cPickle.load(f)

    for layer in net.layers:
        if layer.__class__.__name__ in ['Convolution1D', 'Convolution2D', 'Dense']:
            if layer.name in dat.keys():
                layer.W.set_value(dat[layer.name][0])
                layer.b.set_value(dat[layer.name][1])
    return net



def weightfile_to_modelfile(net, weightfile, modelfile):
    """ convert weight file to a model file """
    net.load_weights(weightfile)
    net.save(modelfile)

def modelfile_to_weightfile(modelfile, weightfile):
    """ convert a full model file to a weight only file """
    net = keras.models.load_model(modelfile)
    net.save_weights(weightfile)


class MyLogger(keras.callbacks.Callback):
    def __init__(self, logfilename, display=1, logger=None):
        self.logfilename = logfilename
        self.display = display 
        if logger:
            self.logger = logger
        else:
            self.logger = init_logger(self.logfilename)
        print ("Logger========:", str(self.logger))

    def on_train_begin(self, logs={}):
        #self.model.summary(logger=self.logger)
        self.best_val_acc = 1e-10
        #self.model.summary()
    
    def on_batch_end(self, batch, logs={}):
        if batch%self.display==0:
            lr = K.eval(self.model.optimizer.lr)#.get_value()
            self.logger.info('epoch %d batch %d lr %f acc %f'%(self.epoch, batch, lr, logs.get('acc')))

    def on_epoch_end(self, epoch, logs={}):
        lr = K.eval(self.model.optimizer.lr)#.get_value()
        val_acc = logs.get('val_acc')
        if val_acc > self.best_val_acc:
            self.best_val_acc = val_acc
        #self.logger.info('epoch %d lr %f val_loss %f best_val_loss %f'%(epoch, lr, val_loss, self.best_val_loss))
        self.logger.info('epoch %d lr %f val_acc %f best_val_acc %f'%(epoch, lr, val_acc, self.best_val_acc))

    def on_epoch_begin(self, epoch, logs={}):
        self.seen = 0
        self.epoch = epoch

class StepLearningRateScheduler(Callback):
    """
    learning rate is multiplied with lr_decay (e.g., 0.1) every lr_epoch
    (e.g., 10). 
    """
    def __init__(self, lr, lr_decay, lr_epoch):
        super(StepLearningRateScheduler, self).__init__()
        self.lr = lr
        self.lr_decay = lr_decay
        self.lr_epoch = lr_epoch

    def get_lrate(self, epoch):
        lrate = self.lr * math.pow(self.lr_decay, math.floor((1+epoch)/self.lr_epoch))
        return lrate

    def on_epoch_begin(self, epoch, logs={}):
        assert hasattr(self.model.optimizer, 'lr'), \
            'Optimizer must have a "lr" attribute.'
        lrate = self.get_lrate(epoch)
        K.set_value(self.model.optimizer.lr, lrate)
        

class VectorLearningRateScheduler(Callback):
    """
    specify a vector for decay ratio and epoch. This is more flexible
    E.g., lr_decay = [0.1, 0.1, 0.5], lr_epoch=[5, 10, 20]
    means the lrate will multiple with 0.1 at the 5th epoch, and then
    multiple with 0.1 at the 10-th epoch, and then multiple with 0.5 at
    the 20th epoch.
    """
    def __init__(self, lr, lr_decay, lr_epoch):
        super(VectorLearningRateScheduler, self).__init__()
        self.lr = lr
        self.lr_decay = lr_decay
        self.lr_epoch = lr_epoch

        self.lr_decay_cumprod = np.cumprod(np.array(lr_decay))

    def on_epoch_begin(self, epoch, logs={}):
        assert hasattr(self.model.optimizer, 'lr'), \
            'Optimizer must have a "lr" attribute.'

        k=[n for n,i in enumerate(self.lr_epoch) if i > epoch]
        if k==[]:
            lrate = self.lr * self.lr_decay_cumprod[-1]
        else:
            k=k[0]
            if k==0:
                lrate = self.lr
            else:
                lrate = self.lr * self.lr_decay_cumprod[k-1]

        K.set_value(self.model.optimizer.lr, lrate)


def my_fit_generator(net, train_data_generator, samples_per_epoch, nb_epoch,
        val_data_generator, nb_val_samples, model_filename, logger, lr_scheduler=None):
    '''
    Simple, single thread routine to do training. The keras
    fit_generator sometimes will crash, due to multi-threading. This is
    a simple alternative.

    Input:
        net -- net model
        train_data_generator -- generator to yield X,y minibatch for training
        nb_epoch -- how many epochs
        samples_per_epoch -- total number of training samples for each epoch
        val_data_generator -- generator to yield X,y minibatch for validation
        nb_val_samples -- number of validation samples
        model_filename -- filename to save the best model (the full model, including the weights, the model, and the optimizer states)
        logger -- logger
        lr_scheduler -- learning rate schedule function, default None
    '''

    best_val_acc = 1e-10
    epoch = 0 

    net.summary(logger=logger)

    while epoch < nb_epoch:
        # set learning rate if needed
        if lr_scheduler:
            lr = lr_scheduler.get_lrate(epoch)
            net.optimizer.lr.set_value(lr)
        else:
            lr = net.optimizer.lr.get_value()

        idx = 0
        i = 0
        while 1:
            X,y = next(train_data_generator)
            idx += y.shape[0] # batch_size
            i += 1
            loss = net.train_on_batch(X,y)
            if i%20 == 0:
                logger.info('epoch %d batch %d lr %f loss %f'%(epoch, i, lr, loss))

            if idx>=samples_per_epoch:
                break

        idx=0
        i=0
        val_loss=0
        while 1:
            X,y = next(val_data_generator)
            idx += y.shape[0] # batch_size
            i += 1
            val_loss += net.test_on_batch(X,y)
            if idx>=nb_val_samples:
                break
        val_loss/=i

        if val_loss<=best_val_loss:
            best_val_loss = val_loss
            net.save(model_filename)

        logger.info('epoch %d lr %f val_loss %f best_val_loss %f'%(epoch, lr, val_loss, best_val_loss))
        epoch += 1


def my_evaluate_generator(net, val_data_generator, val_samples):
    '''
    Simple, single thread routine to do evaluation. The keras
    evaluate_generator sometimes will crash, due to multi-threading. This is
    a simple alternative.

    Input:
        net -- net model
        val_data_generator -- generator to yield X,y minibatch for validation
        nb_val_samples -- number of validation samples
    '''
    
    idx=0
    i=0
    val_loss=0
    while 1:
        X,y = next(val_data_generator)
        idx += y.shape[0]
        i += 1
        val_loss += net.test_on_batch(X,y)
        if idx>=val_samples:
            break
        #print('%d %d'%(idx,val_samples))

    val_loss/=i
    return val_loss


from multiprocessing import Process, Queue

def my_fit_generator_with_prefetch(net, train_data_generator, samples_per_epoch, nb_epoch, 
        val_data_generator, nb_val_samples, model_filename, logger, lr_scheduler=None):
    '''
    Two threads routine to do training. The keras
    fit_generator sometimes will crash, due to multi-threading. This is
    a simple alternative. One for prefetching data

    Input:
        net -- net model
        train_data_generator -- generator to yield X,y minibatch for training
        nb_epoch -- how many epochs
        samples_per_epoch -- total number of training samples for each epoch
        val_data_generator -- generator to yield X,y minibatch for validation
        nb_val_samples -- number of validation samples
        model_filename -- filename to save the best model (the full model, including the weights, the model, and the optimizer states)
        logger -- logger
        lr_scheduler -- learning rate schedule function, default None
    '''

    train_blob_queue = Queue(10)
    train_prefetch_process = BlobFetcher(train_blob_queue, train_data_generator, logger)
    train_prefetch_process.start()

    val_blob_queue = Queue(10)
    val_prefetch_process = BlobFetcher(val_blob_queue, val_data_generator, logger)
    val_prefetch_process.start()

    # Terminate the child process when the parent exists
    def cleanup():
        logger.info('Terminating BlobFetcher')
        train_prefetch_process.terminate()
        val_prefetch_process.terminate()
        train_prefetch_process.join()
        val_prefetch_process.join()
    import atexit
    atexit.register(cleanup)

    best_val_loss = 1e+10
    epoch = 0 

    net.summary(logger=logger)

    while epoch < nb_epoch:
        # set learning rate if needed
        if lr_scheduler:
            lr = lr_scheduler.get_lrate(epoch)
            net.optimizer.lr.set_value(lr)
        else:
            lr = net.optimizer.lr.get_value()

        idx = 0
        i = 0
        while 1:
            X,y = train_blob_queue.get() 
            idx += y.shape[0] # batch_size
            i += 1
            loss = net.train_on_batch(X,y)
            if i%20 == 0:
                logger.info('epoch %d batch %d lr %f loss %f'%(epoch, i, lr, loss))

            if idx>=samples_per_epoch:
                break

        idx=0
        i=0
        val_loss=0
        while 1:
            X,y = val_blob_queue.get()
            idx += y.shape[0] # batch_size
            i += 1
            val_loss += net.test_on_batch(X,y)
            if idx>=nb_val_samples:
                break
        val_loss/=i

        if val_loss<=best_val_loss:
            best_val_loss = val_loss
            net.save(model_filename)

        logger.info('epoch %d lr %f val_loss %f best_val_loss %f'%(epoch, lr, val_loss, best_val_loss))
        epoch += 1


def my_evaluate_generator_with_prefetch(net, val_data_generator, val_samples, logger):
    '''
    Two threads routine to do evaluation. The keras
    evaluate_generator sometimes will crash, due to multi-threading. This is
    a simple alternative.

    Input:
        net -- net model
        val_data_generator -- generator to yield X,y minibatch for validation
        val_samples -- number of validation samples
    '''
 
    val_blob_queue = Queue(10)
    val_prefetch_process = BlobFetcher(val_blob_queue, val_data_generator, logger)
    val_prefetch_process.start()

    # Terminate the child process when the parent exists
    def cleanup():
        logger.info('Terminating BlobFetcher')
        val_prefetch_process.terminate()
        val_prefetch_process.join()
    import atexit
    atexit.register(cleanup)

   
    idx=0
    i=0
    val_loss=0
    while 1:
        X,y = val_blob_queue.get()
        idx += y.shape[0]
        i += 1
        val_loss += net.test_on_batch(X,y)
        if idx>=val_samples:
            break
    val_loss/=i
    return val_loss


   
class BlobFetcher(Process):
    def __init__(self,queue,generator,logger):
        super(BlobFetcher, self).__init__()
        self._queue = queue
        self._generator = generator
        self._logger=logger
    
    def run(self):
        self._logger.info('BlobFetcher started')
        while True:
            X,y=next(self._generator)
            self._queue.put((X,y))
