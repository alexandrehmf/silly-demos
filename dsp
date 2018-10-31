#!/usr/bin/env python
import pyscreenshot as ss
import numpy as np
import cv2 as cv

png = ss.grab()
img = np.array(png)
img = cv.cvtColor(img,cv.COLOR_RGB2BGR)

cv.imshow('image',img)
cv.waitKey(0)
cv.destroyAllWindows()