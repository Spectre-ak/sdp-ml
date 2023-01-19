
import time
import cv2
#import mss
import numpy as np
import os
import sys
#os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
import tensorflow as tf
from distutils.version import StrictVersion
from collections import defaultdict
from io import StringIO
from PIL import Image
# from google.colab.patches import cv2_imshow
# ## Env setup
from object_detection.utils import ops as utils_ops
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util
 


# # Model preparation 
PATH_TO_FROZEN_GRAPH = './faster_rcnn_inception_v2_coco_2018_01_28/frozen_inference_graph.pb'
PATH_TO_FROZEN_GRAPH_SSD = './ssd_mobilenet_v1_coco_2018_01_28/frozen_inference_graph.pb'
PATH_TO_FROZEN_GRAPH_faster_rcnn_nas = './faster_rcnn_nas_coco_2018_01_28/frozen_inference_graph.pb'


PATH_TO_FROZEN_GRAPH = PATH_TO_FROZEN_GRAPH_faster_rcnn_nas

# List of the strings that is used to add correct label for each box.
PATH_TO_LABELS = './object_detection/data/mscoco_label_map.pbtxt'
NUM_CLASSES = 90
 
 
# ## Load a (frozen) Tensorflow model into memory.
label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
category_index = label_map_util.create_category_index(categories)
 
detection_graph = tf.Graph()
with detection_graph.as_default():
  od_graph_def = tf.GraphDef()
  with tf.gfile.GFile(PATH_TO_FROZEN_GRAPH, 'rb') as fid:
    serialized_graph = fid.read()
    od_graph_def.ParseFromString(serialized_graph)
    tf.import_graph_def(od_graph_def, name='')
 


def testOverImg(path, file_name):
  print(path)
  writing_path = 'C:/out/trail1/out_faster_rcnn_nas/'+os.path.splitext(file_name)[0]+"-out.jpg"
  print(writing_path)
  # # Detection
  with detection_graph.as_default():
    with tf.Session(graph=detection_graph) as sess:
      while True:
        # Get raw pixels from the screen, save it to a Numpy array
        img=Image.open(path)
        # img=img.resize((1280, 720))

        # img=Image.open('./26.bmp')

        # cv2.imshow('Window', np.array(img))
        # cv2.waitKey(50000)
        print(img.size)
        # img = img.resize((1, 1080, 1920, 3))
        image_np = np.array(img)
        # To get real color we do this:
        #image_np = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)
        # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
        image_np_expanded = np.expand_dims(image_np, axis=0)
        # Actual detection.
        image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
        boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
        scores = detection_graph.get_tensor_by_name('detection_scores:0')
        classes = detection_graph.get_tensor_by_name('detection_classes:0')
        num_detections = detection_graph.get_tensor_by_name('num_detections:0')
        # Visualization of the results of a detection.
        (boxes, scores, classes, num_detections) = sess.run(
            [boxes, scores, classes, num_detections],
            feed_dict={image_tensor: image_np_expanded})
        # print((boxes, scores, classes, num_detections))
        vis_util.visualize_boxes_and_labels_on_image_array(
            image_np,
            np.squeeze(boxes),
            np.squeeze(classes).astype(np.int32),
            np.squeeze(scores),
            category_index,
            use_normalized_coordinates=True,
            line_thickness=2)
        # Show image with detection
        image_np=cv2.cvtColor(image_np,cv2.COLOR_BGR2RGB)
        # cv2.imshow('Marked', image_np)
        cv2.imwrite(writing_path, image_np)
        break
  


def runOverDir():
  base_path = 'C:/out/trail1'
  files = os.listdir(base_path)
  print(files)
  for f in files:
    print("running inference over")
    print(f)
    img_path = base_path+"/"+f
    if "out" in f:
      continue
    testOverImg(img_path, f)
  print("all image processed")



def convertTojpg():
  base_path = 'C:/out/trail1'
  files = os.listdir(base_path)
  print(files)
  for f in files:
    img=Image.open(base_path+"/"+f)
    img=np.array(img)
    name = os.path.splitext(f)[0]
    print(base_path+"/"+name+".jpg")
    cv2.imwrite(base_path+"/"+name+".jpg", img)

runOverDir()
# convertTojpg()

  
