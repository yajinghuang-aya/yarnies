import numpy as np
import os
import six.moves.urllib as urllib
import sys
import pickle


import wget
import cv2

from matplotlib import pyplot as plt
from object_detection.utils import ops as utils_ops
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util
from PIL import Image
import tensorflow as tf

utils_ops.tf = tf.compat.v1
tf.gfile = tf.io.gfile

model_name = 'faster_rcnn_inception_resnet_v2_atrous_oid_2018_01_28'



def load_model(model_name):
	model_dir="models/"   #from https://github.com/tensorflow/models.git 
	model_dir= model_dir+model_name+"/saved_model"
	model = tf.saved_model.load(str(model_dir))
	model = model.signatures['serving_default']

	return model

def run_inference_for_single_image(model, image):
	#image = np.asarray(image)
	# The input needs to be a tensor, convert it using `tf.convert_to_tensor`.
	input_tensor = tf.convert_to_tensor(image)
	# The model expects a batch of images, so add an axis with `tf.newaxis`.
	input_tensor = input_tensor[tf.newaxis,...]

	# Run inference
	output_dict = model(input_tensor)

	# All outputs are batches tensors.
	# Convert to numpy arrays, and take index [0] to remove the batch dimension.
	# We're only interested in the first num_detections.
	num_detections = int(output_dict.pop('num_detections'))
	output_dict = {key:value[0, :num_detections].numpy() for key,value in output_dict.items()}
	output_dict['num_detections'] = num_detections

	# detection_classes should be ints.
	output_dict['detection_classes'] = output_dict['detection_classes'].astype(np.int64)
	print(output_dict)
	# Handle models with masks:
	if 'detection_masks' in output_dict:
	# Reframe the the bbox mask to the image size.
		detection_masks_reframed = utils_ops.reframe_box_masks_to_image_masks(
		         output_dict['detection_masks'], output_dict['detection_boxes'],
		   			image.shape[0], image.shape[1])      
		detection_masks_reframed = tf.cast(detection_masks_reframed > 0.5,
		                           tf.uint8)
		output_dict['detection_masks_reframed'] = detection_masks_reframed.numpy()
	#with open('output_dict.pickle', 'wb') as handle:
	#	pickle.dump(output_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)


	return output_dict


def show_inference(model, image_path):
  # the array based representation of the image will be used later in order to prepare the
  # result image with boxes and labels on it.
  PATH_TO_LABELS = '/home-2/hwang127@jhu.edu/scratch/mie/models/research/object_detection/data/oid_bbox_trainable_label_map.pbtxt'
  category_index = label_map_util.create_category_index_from_labelmap(PATH_TO_LABELS, use_display_name=True)
  image_np = np.array(Image.open(image_path))
  print(image_np.shape)
  # Actual detection.
  output_dict = run_inference_for_single_image(model, image_np)
  # Visualization of the results of a detection.
  vis_util.visualize_boxes_and_labels_on_image_array(
      image_np,
      output_dict['detection_boxes'],
      output_dict['detection_classes'],
      output_dict['detection_scores'],
      category_index,
      instance_masks=output_dict.get('detection_masks_reframed', None),
      use_normalized_coordinates=True,
      line_thickness=8)

  display(Image.fromarray(image_np))


def crop_image(image_np,model, outpath, show=False, save=True):

	labels=[2,68,99,128]
	image = image_np
	hight,width,_ = np.shape(image)
	#with open('output_dict.pickle', 'rb') as handle:
	#	output_dict = pickle.load(handle)
	output_dict = run_inference_for_single_image(model, image)	


	#print(output_dict)
	clothing=[output_dict["detection_scores"][idx] \
				for idx, i in enumerate(output_dict["detection_classes"]) if i in labels]
	if not clothing:
		print('no right labels')
		return False

	max_score =  max(clothing)
	index = list(output_dict["detection_scores"]).index(max_score) 
	[y,x,h,w] = output_dict["detection_boxes"][index]
	cropped = image[int(y*hight):int(h*hight), int(x*width):int(w*width)]
	cropped=flip_rgb(cropped)
	if show:
	    img = Image.fromarray(cropped)#,"RGB")
	    display(img)
	if save:
	    img = Image.fromarray(cropped,"RGB")
	    img.save(outpath)
	    #print('saving cropped image')
	    #np.save(outpath,cropped)
	    return True


	

#image_url='https://images4-f.ravelrycache.com/uploads/goodnightdayknits/641126832/AWxGoodNightDay_July2019_006_small_best_fit.jpg'
#filename = wget.download(image_url)
#image_np = (cv2.imread(filename)).astype(np.uint8)
#filename
#print(np.shape(image_np))
##outpath='cloth_crop.jpg'
#model=load_model(model_name)
#print("model loaded")


#crop_image(image_np,model,outpath)


