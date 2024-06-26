import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import requests
from PIL import Image
from io import BytesIO
def conv(image, im_filter):
 """
 :param image: grayscale image as a 2-dimensional numpy array
 :param im_filter:2-dimensional numpy array
 """
 #input dimensions
 height=image.shape[0]
 width=image.shape[1]
 #output image with reduced dimensions
 im_c=np.zeros((height-len(im_filter)+1,width - len(im_filter) + 1))
 # iterate over all rows and columns
 for row in range(len(im_c)):
  for col in range(len(im_c[0])):
    # apply the filter
    for i in range(len(im_filter)):
      for j in range(len(im_filter[0])):
       im_c[row, col] += image[row + i, col +j]*im_filter[i][j]
#fix out-of-bounds values
 im_c[im_c > 255] = 255
 im_c[im_c < 0] = 0
 # plot images for comparison

 plt.figure()
 plt.imshow(image,cmap=cm.Greys_r)
 plt.show()
 plt.imshow(im_c,cmap=cm.Greys_r)
 plt.show()

#load the image
url="https://upload.wikimedia.org/wikipedia/commons/8/88/Commander_Eileen_Collins_-_GPN-2000-001177.jpg"
resp = requests.get(url)
if resp.status_code != 200:
    print(f"Impossible d'obtenir l'image. Code de statut HTTP : {resp.status_code}")
else:
 image_rgb = np.asarray(Image.open(BytesIO(resp.content)).convert("RGB"))
 #convert to grayscale
 image_grayscale=np.mean(image_rgb,axis=2,dtype=np.uint)
 #blur filter
 blur=np.full([10,10],1./100)
 conv(image_grayscale,blur)
 #Sobel edge detectors
 sobel_x=[[-1,-2,-1],[0,0,0], [1,2,1]]
 conv(image_grayscale,sobel_x)
 sobel_y=[[-1,0,1],
 [-2,0,2], [-1,0,1]]

# ---------------------------------------------------------------Code correction----------------------------------------------------------------------------------------------------------------------------
# def conv(image, im_filter):
#  """
#  :param image: grayscale image as a 2-dimensional numpy array
#  :param im_filter:2-dimensional numpy array
#  """
#  # input dimensions
#  height = image.shape[0]
#  width = image.shape[1]
#  # output image with reduced dimensions
#  im_c = np.zeros((height - len(im_filter) + 1, width - len(im_filter) + 1))
#  # iterate over all rows and columns
#  for row in range(len(im_c)):
#   for col in range(len(im_c[0])):
#    # apply the filter
#    for i in range(len(im_filter)):
#     for j in range(len(im_filter[0])):
#      im_c[row, col] += image[row + i, col + j] * im_filter[i][j]
#  # fix out-of-bounds values
#  im_c[im_c > 255] = 255
#  im_c[im_c < 0] = 0
#  # plot images for comparison
#  plt.figure()
#  plt.imshow(image, cmap=cm.Greys_r)
#  plt.show()
#  plt.imshow(im_c, cmap=cm.Greys_r)
#  plt.show()
# # load the image from a local file
# image_path = "grenouille.jpg"
# image_rgb = np.asarray(Image.open(image_path).convert("RGB"))
# # convert to grayscale
# image_grayscale = np.mean(image_rgb, axis=2, dtype=np.uint)
# # blur filter
# blur = np.full([10, 10], 1. / 100)
# conv(image_grayscale, blur)
# # Sobel edge detectors
# sobel_x = [[-1, -2, -1], [0, 0, 0], [1, 2, 1]]
# conv(image_grayscale, sobel_x)
# sobel_y = [[-1, 0, 1],
#            [-2, 0, 2], [-1, 0, 1]]
# conv(image_grayscale, sobel_y)