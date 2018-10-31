#!/usr/bin/env python
import pyscreenshot as ss
import cv2 as cv
import numpy as np

img = ss.grab()
img = np.array(img)
img = cv.cvtColor(img,cv.COLOR_RGB2BGR)
cv.imwrite('output1.png',img)