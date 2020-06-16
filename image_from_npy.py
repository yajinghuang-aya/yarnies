from PIL import Image
import sys
import numpy as np  
from matplotlib import pyplot as plt


def flip_rgb(image):
	image_new=np.zeros(image.shape,dtype=np.uint8)
	for i in range(image.shape[0]):
	  for j in range(image.shape[1]):
	    image_new[i,j,2]=image[i,j,0]
	    image_new[i,j,1]=image[i,j,1]
	    image_new[i,j,0]=image[i,j,2]
	    #print(cropped_new[i,j])
	image_new=image_new.astype('int')
	#np.shape(image)
	return image_new

filename=sys.argv[-1]
image=np.load(filename)
img = Image.fromarray(image,'RGB')
outpath=filename[:-3]+'png'
img.save(outpath)

#plt.imshow(flip_rgb(image))#, interpolation='nearest')
#plt.show()


