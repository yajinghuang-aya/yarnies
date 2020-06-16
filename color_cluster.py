#from sklearn.cluster import KMeans
import cv2
import wget
#from sklearn_extra.cluster import KMedoids
from yellowbrick.cluster import KElbowVisualizer
from sklearn.cluster import KMeans
from PIL import Image
from color_util import *
import time
import pandas as pd 
import matplotlib.pyplot as plt

def kmedoid_cluster(image,nc=10):

	start_time=time.time()
	#wid=int(image.shape[1]*0.6)
	#height=int(image.shape[0]*0.6)

	#image=cv2.resize(image,(wid,height), interpolation = cv2.INTER_AREA)
	#image=flip_rgb(image)
	#img = Image.fromarray(image,"RGB")
	#img.save('small2.jpg')

	#image=image.reshape([image.shape[0]*image.shape[1],3])
	#image=remove_background(image)
	image=prepare_image(image)
	imagesize=image.shape[0]
	print("compressed image size: ",imagesize)
	cluster = KMedoids(n_clusters=nc,random_state=0)
	label=cluster.fit_predict(image)

	#y_pre=KMedoids.fit_predict(image)
	C = cluster.cluster_centers_
	(unique, counts)=np.unique(label, return_counts=True)
	frequency={u:c for(u,c) in zip(unique, counts)}

	output=[]
	for i in range(nc):
		output.append([counts[i]/imagesize,i,str(C[i])])

	# string to array: [int(s) for s in string[1:-1].split(' ')]

	df = pd.DataFrame(output, columns = ['frequency', 'label','color_center']) 
	end_time=time.time()
	print('time',end_time-start_time)
	return df


def kmeans_simple_cluster(image,nc=10,weight=None):
        model=KMeans(n_clusters=nc)
        kmeans=model.fit(image,sample_weight=weight)
        labels=kmeans.labels_
        colors=np.asarray((kmeans.cluster_centers_))
        (unique, counts)=np.unique(labels, return_counts=True)
        counts=np.array(counts/np.sum(counts))
        counts=counts.reshape((len(counts),1))
        colors=np.hstack((colors,counts))
        return colors

def kmeans_find_n_cluster(image,Nrange=(2,9),weight=None):
	model = KMeans()
	visualizer = KElbowVisualizer(model,
                                  k=Nrange,showbool=False)
	visualizer.fit(image,sample_weight=weight) 

	if visualizer.elbow_value_:
		return visualizer.elbow_value_
	else: 
		print(" ")
		print("no good number of cluster")
		##file_object = open('cluster_error_patID.txt', 'a')
		#file_object.write()
		#file_object.close()
		return False

def kmeans_op(image,Nrange=(2,9),weight=None):
    k=kmeans_find_n_cluster(image,Nrange,weight=weight)
    print(k)
    if k:
    	colors=kmeans_simple_cluster(image,nc=k,weight=weight)
    else:
    	colors=kmeans_simple_cluster(image,nc=4,weight=weight)
    return colors

def color_visualizer(colors):
    count=colors[:,3]
    colors=colors[:,:3]
    plt.barh(np.arange(len(count)),count,color=colors/255.)
    plt.show()
    return True


def remove_background(image):
	index_del=[]
	for i in range(image.shape[0]):
		if image[i][0]>235 and image[i][1]>235 and image[i][2]>235:
			index_del.append(i)
		if image[i][0]<50 and image[i][1]<50 and image[i][2]<50:
			index_del.append(i)

	if index_del:
		image=np.delete(image,index_del,axis=0)
	return image


def prepare_image(image):	
	wid=int(image.shape[1]*0.6)
	height=int(image.shape[0]*0.6)

	image=cv2.resize(image,(wid,height), interpolation = cv2.INTER_AREA)
	wid=image.shape[1]
	height=image.shape[0]
	image=image[int(wid*0.2):int(wid*0.8),int(height*0.2):int(height*0.8)]
	#image=flip_rgb(image)
	#img = Image.fromarray(image,"RGB")
	#img.save('small2.jpg')

	image=image.reshape([image.shape[0]*image.shape[1],3])
	image=remove_background(image)

	return image

#image = cv2.imread('crop_image/rosedalebeauty_medium.jpg')
#df=color_cluster(image)
#print(df)