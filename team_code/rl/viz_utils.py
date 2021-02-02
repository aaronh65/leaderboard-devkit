import cv2
import numpy as np

def draw_text(image, text, loc, fontColor=(105,105,105)):
    font = cv2.FONT_HERSHEY_SIMPLEX
    fontScale = 0.5
    #fontColor = (139, 0, 139)

    lineType = 2

    cv2.putText(image, text, loc,
        font,
        fontScale,
        fontColor,
        lineType)


