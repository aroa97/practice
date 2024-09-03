import cv2
import numpy as np

img_fg = cv2.imread('./img/mask_hannibal.png', cv2.IMREAD_UNCHANGED)
img_bg = cv2.imread('./img/man_face.jpg')

img_fg = cv2.resize(img_fg, (348, 287))

_, mask = cv2.threshold(img_fg[:,:,2], 10, 255, cv2.THRESH_BINARY)
mask_inv = cv2.bitwise_not(mask)

i, j = img_bg.shape[:2][0] - img_fg.shape[:2][0], (img_bg.shape[:2][1] // 2) - (img_fg.shape[:2][1] // 2) + 10

img_fg = cv2.cvtColor(img_fg, cv2.COLOR_BGRA2BGR)
h, w = img_fg.shape[:2]
roi = img_bg[i:i+h, j:j+w ]

masked_fg = cv2.bitwise_and(img_fg, img_fg, mask=mask)
masked_bg = cv2.bitwise_and(roi, roi, mask=mask_inv)

added = masked_fg + masked_bg
img_bg[i:i+h, j:j+w] = added

cv2.imshow('mask', mask)
cv2.imshow('mask_inv', mask_inv)
cv2.imshow('masked_fg', masked_fg)
cv2.imshow('masked_bg', masked_bg)
cv2.imshow('added', added)
cv2.imshow('result', img_bg)
cv2.waitKey()
cv2.destroyAllWindows() 