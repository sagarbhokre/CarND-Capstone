from styx_msgs.msg import TrafficLight
import ts_alx_net, os, cv2
import numpy as np

class TLClassifier(object):
    def __init__(self):
        #load classifier
        #build NN
        self.net = ts_alx_net.build_alexnet()
        self.net.load_weights('light_classification/ts_alxnet_z1.h5')
        pass


    def preprocess_image(self, img, resize=False):
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
        DX=(256-imW)/2
    
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

    def get_classification(self, image):
        """Determines the color of the traffic light in the image

        Args:
            image (cv::Mat): image containing the traffic light

        Returns:
            int: ID of traffic light color (specified in styx_msgs/TrafficLight)

        """
        #TODO implement light color prediction
        img = self.preprocess_image(image, True)
        pp = self.net.predict_on_batch(img)
        pp = list(np.squeeze(pp))
        state = pp.index(max(pp))
        return state #TrafficLight.UNKNOWN
