import cv2


def imshow(image):
    cv2.imshow('tmp', image)

    # waits for user to press any key
    # (this is necessary to avoid Python kernel form crashing)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
