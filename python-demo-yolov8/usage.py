from onnxinference import detect_from_opencv_frame, detect_from_image_file, visualize
import cv2

def main():
    # usage1：从图片文件中检测
    test_detect_from_image_file("testimages/2222.png") 

    # usage2：从cap.read()读取的帧中检测
    # test_detect_from_opencv_frame("testimages/cut.mp4") 
    
    # usage3：检测视频
    # test_video("testimages/cut.mp4")

def test_detect_from_image_file(image_fn):
    boxes, img = detect_from_image_file(image_fn)
    visual_result = visualize(img, boxes) 
    cv2.waitKey(0)

def test_detect_from_opencv_frame(video_fn):
    cap = cv2.VideoCapture(video_fn)
    ret, frame = cap.read()
    if not ret:
        print("can`t read video") 
    boxes = detect_from_opencv_frame(frame)
    visualize(frame, boxes, show=True)
    cv2.waitKey(0)


def test_video(video_fn):
    cap = cv2.VideoCapture(video_fn)
    while(True):
        ret, frame = cap.read()
        if not ret:
            print("can`t read video") 
        boxes = detect_from_opencv_frame(frame)
        visualize(frame, boxes, show=True)
        cv2.waitKey(100)
    cv2.waitKey(0)
    
if __name__ == "__main__":
    main()
    