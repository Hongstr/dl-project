from cv2 import threshold
import jetson.inference
import jetson.utils
import cv2
import sys

COLORS = [
(255,0,0),
(0,255,0),
(0,0,255),
(255,255,0),
(0,255,255),
(255,0,255),
(128,0,0),
(255,192,203),
(0,128,128)]

def gstreamer_pipeline(
    capture_width=1280,
    capture_height=720,
    display_width=1280,
    display_height=720,
    framerate=24,
    flip_method=0,
):
    return (
        "nvarguscamerasrc ! "
        "video/x-raw(memory:NVMM), "
        "width=(int)%d, height=(int)%d, "
        "format=(string)NV12, framerate=(fraction)%d/1 ! "
        "nvvidconv flip-method=%d ! "
        "video/x-raw, width=(int)%d, height=(int)%d, format=(string)BGRx ! "
        "videoconvert ! "
        "video/x-raw, format=(string)BGR ! appsink"
        % (
            capture_width,
            capture_height,
            framerate,
            flip_method,
            display_width,
            display_height,
        )
    )

print(gstreamer_pipeline(flip_method=0))
cap = cv2.VideoCapture(gstreamer_pipeline(flip_method=0), cv2.CAP_GSTREAMER)
if len(sys.argv) > 1:
    net = jetson.inference.detectNet("ssd-mobilenet-v2",sys.argv,0.5)
else:
    net = jetson.inference.detectNet("ssd-mobilenet-v2",0.5)
while cap.isOpened():
    success, img = cap.read()
    imgCuda = jetson.utils.cudaFromNumpy(img)
    detections = net.Detect(imgCuda)
    for d in detections:
        color = COLORS[d.ClassID%len(COLORS)]
        x1,y1,x2,y2 = int(d.Left), int(d.Top),int(d.Right), int(d.Bottom)
        className = net.GetClassDesc(d.ClassID)
        cv2.rectangle(img,(x1,y1),(x2,y2),color,2)
        cv2.putText(img, className, (x1+5,y1+15),cv2.FONT_HERSHEY_DUPLEX,0.75,color,2)
    #img = jetson.utils.cudaToNumpy(imgCuda)
    cv2.imshow("Image", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
