import cv2 as cv
from matplotlib import pyplot as plt
import numpy as np

fg_num = input("Enter foreground number:")
bg_num = input("Enter background number:")

img = cv.imread(f'foreground/fg-{fg_num}.jpg')
bg = cv.imread(f'background/bg-{bg_num}.jpg')
orignal_mask = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

binary_mask = cv.adaptiveThreshold(
    orignal_mask, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, 11, 2)

binary_mask = cv.bitwise_not(binary_mask)

cv.fastNlMeansDenoising(binary_mask, binary_mask, 35, 10, 25)

binary_mask = cv.erode(binary_mask, (17, 17), iterations=1)

binary_mask = cv.bitwise_not(binary_mask)

height, width, _ = img.shape
left_margin_proportion = 0.3
right_margin_proportion = 0.3
up_margin_proportion = 0.1
down_margin_proportion = 0.1

boundary_rectangle = (
    int(width * left_margin_proportion),
    int(height * up_margin_proportion),
    int(width * (1 - right_margin_proportion)),
    int(height * (1 - down_margin_proportion)),
)
background_model = np.zeros((1, 65), np.float64)
foreground_model = np.zeros((1, 65), np.float64)

mask = np.zeros((height, width), np.uint8)
mask[:] = cv.GC_PR_BGD
mask[binary_mask == 0] = cv.GC_FGD

cv.grabCut(
    img,
    mask,
    boundary_rectangle,
    background_model,
    foreground_model,
    5,
    cv.GC_INIT_WITH_MASK,
)
grabcut_mask = np.where((mask == cv.GC_PR_BGD) | (mask == cv.GC_BGD), 0, 1).astype(
    "uint8"
)

segmented_image = img.copy() * grabcut_mask[:, :, np.newaxis]
bg = cv.resize(bg, (width, height), fx=0, fy=0, interpolation=cv.INTER_CUBIC)
segmented_bg = bg.copy() * (1 - grabcut_mask[:, :, np.newaxis])
newimg = segmented_image + segmented_bg
# newimg = cv.cvtColor(newimg, cv.COLOR_BGR2RGB)

cv.imshow('newimg', newimg)
cv.waitKey(0)
cv.destroyAllWindows()
