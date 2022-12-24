import cv2 as cv
import numpy as np
import skimage.exposure


def removeGreen(fg, bg):
    if fg.shape != bg.shape:
        bg = cv.resize(bg, (fg.shape[1], fg.shape[0]),
                       fx=0, fy=0, interpolation=cv.INTER_CUBIC)
    lab = cv.cvtColor(fg, cv.COLOR_BGR2LAB)     # Convert to LAB color space

    A = lab[:, :, 1]
    thresh = cv.threshold(A, 0, 255, cv.THRESH_BINARY +
                          cv.THRESH_OTSU)[1]    # Threshold the A channel

    newImg = fg.copy()
    newImg[thresh == 0] = bg[thresh == 0]

    return newImg


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
                frame, (720, 560), fx=0, fy=0, interpolation=cv.INTER_CUBIC)
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
