import numpy as np

def flip_rgb(image):
        image_new=np.zeros(image.shape)
        for i in range(image.shape[0]):
          for j in range(image.shape[1]):
            image_new[i,j,2]=image[i,j,0]
            image_new[i,j,1]=image[i,j,1]
            image_new[i,j,0]=image[i,j,2]
            #print(cropped_new[i,j])
        image_new=image_new.astype('uint8')
        #np.shape(image)
        return image_new
