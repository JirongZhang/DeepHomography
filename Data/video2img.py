import cv2
import os


def save_img(video_path):
    videos = os.listdir(video_path)
    for video_name in videos:
        file_name = video_name.split('.')[0]
        folder_name = os.path.join(video_path, file_name)
        if not os.path.exists(folder_name):
            os.makedirs(folder_name, exist_ok=True)
        vc = cv2.VideoCapture(video_path+video_name)
        c = 0
        rval = vc.isOpened()

        while rval:
            c = c + 1
            rval, frame = vc.read()
            if rval:
                frames = c + 10000
                cv2.imwrite(os.path.join(folder_name, file_name + '_' + str(frames) + '.jpg'), frame)
                cv2.waitKey(1)
            else:
                break
        vc.release()
        print('save_success')
        print(folder_name)
if __name__ == '__main__':
    # path to video folds eg: video_path = './Test/'
    video_path = './Test/'
    save_img(video_path)
    video_path = './Train/'
    save_img(video_path)
    