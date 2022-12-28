import cv2 as cv
import numpy as np
import skimage


def removeGreen(fg, bg):

    if fg.shape != bg.shape:
        bg = cv.resize(bg, (fg.shape[1], fg.shape[0]),
                       fx=0, fy=0, interpolation=cv.INTER_CUBIC)
    b, g, r = fg[:, :, 0], fg[:, :, 1], fg[:, :, 2]

    mask = (g > 70) & (r < g - 30) & (b < g - 30)
    mask = ~mask

    # mask = skimage.morphology.binary_closing(mask, footprint=np.ones((3, 3)))
    # mask = skimage.morphology.binary_closing(mask, footprint=np.ones((4, 3)))

    fg[mask == False] = bg[mask == False]

    return fg


def videoProcesing(bg, vid) -> None:
    cap = cv.VideoCapture(f'./video/{vid}.mp4')
    # fourcc = cv.VideoWriter_fourcc('h', '2', '6', '4')
    # out_name = name.split('/')[1].split('.')[0]
    """ out = cv.VideoWriter(
        f'output_vid/{model_name}_{name}.mp4', fourcc, 30, (640, 480)) """
    while cap.isOpened():
        ret, frame = cap.read()
        if ret:
            resized_frame = cv.resize(
                frame, (640, 480), fx=0, fy=0, interpolation=cv.INTER_CUBIC)
            processed_frame = removeGreen(resized_frame, bg)
            # out.write(processed_frame)
            cv.imshow(f'Video', processed_frame)
            if cv.waitKey(1) & 0xFF == ord('q'):
                break
        else:
            break
    cap.release()
    # out.release()
    cv.destroyAllWindows()


if __name__ == "__main__":
    fg_num = input("Enter video number:")
    bg_num = input("Enter background number:")

    bg = cv.imread(f'background/bg-{bg_num}.jpg', 1)
    videoProcesing(bg, f'vid-{fg_num}')
