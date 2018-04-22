import tensorflow as tf
import sys,os
sys.path.append(os.pardir)
import numpy as np

from PIL import Image
from PIL import ImageOps

GRAPH_PATH = os.path.join(os.path.dirname(__file__), 'model.pb')

class ImageRec():
    
    def __init__(self):
        self.load_graph()
        self.sess = tf.Session()
        self.softmax = tf.nn.softmax(self.sess.graph.get_tensor_by_name('inference_1:0')) # 1:0は何か
    
    def load_graph(self):
        with tf.gfile.FastGFile(GRAPH_PATH, 'rb') as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())
        tf.import_graph_def(graph_def, name='')
    
    def run(self, image_path):
        img = Image.open(image_path)
        img = img.resize((28, 28), Image.LANCZOS)
        img = img.convert('RGB')
        img = ImageOps.invert(img)
        img = img.convert("L")

        image = np.asarray(img)
        print(image.shape)
        # image = image.reshape(784,) # このままにしたい。
        image = image.reshape(1,784)
        
        predictions = self.sess.run(self.softmax, feed_dict={'x:0': image})
        print('predictions:%s' % predictions)
        
        image_info = []
        for i in range(10):
            label = str(i)
            score = predictions[0][i]
            print('%s (score = %.5f)' % (label, score))
            score_info = {}
            score_info['name']=label
            score_info['score']=round(score * 100.0, 2)
            image_info.append(score_info)

        return image_info