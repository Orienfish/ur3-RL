import matplotlib.pyplot as plt
import numpy as np
import os
import cv2
PATH = os.path.split(os.path.realpath(__file__))[0]
PATH += '/grp1'
print PATH

def TENG(img):
    	guassianX = cv2.Sobel(img, cv2.CV_64F, 1, 0)
    	guassianY = cv2.Sobel(img, cv2.CV_64F, 1, 0)
    	return np.mean(guassianX * guassianX + 
    					  guassianY * guassianY)

if __name__ == '__main__':
	fList = []
	for i in range(150, 191):
		pic_name = PATH+'/'+str(i)+'.jpg'
		print pic_name
		img = cv2.imread(pic_name, 0)
        	fList.append(TENG(img))
	plt.plot(fList, 'bx--')
	plt.savefig('./focus_show', dpi=1200)
	plt.show()
