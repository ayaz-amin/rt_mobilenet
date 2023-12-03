import cv2
import numpy as np
import onnxruntime as rt

def preprocess_frame(frame):
    return np.expand_dims(frame.astype(np.uint8), axis=0)

def get_rect(width, height, d):
    top = d[0] * height
    left = d[1] * width
    bottom = d[2] * height
    right = d[3] * width
    
    top = max(0, np.floor(top + 0.5).astype('int32'))
    left = max(0, np.floor(left + 0.5).astype('int32'))
    bottom = min(height, np.floor(bottom + 0.5).astype('int32'))
    right = min(width, np.floor(right + 0.5).astype('int32'))
    
    return (top, left, bottom, right)

sess = rt.InferenceSession("ssd_mobilenet_v1_13-qdq.onnx", providers=['CPUExecutionProvider'])
outputs = ["num_detections", "detection_boxes", "detection_scores", "detection_classes"]
cam = cv2.VideoCapture(0)
writer = cv2.VideoWriter(
    'record.mp4', cv2.VideoWriter_fourcc(*'MP4V'),
    30, (int(cam.get(3)), int(cam.get(4)))
)

coco_classes = {}
with open("coco-labels-paper.txt", 'r') as f:
    classes = f.read().splitlines()
    for i, cat in enumerate(classes):
        coco_classes[i + 1] = cat

while True:
    ret, frame = cam.read()
    
    processed = preprocess_frame(frame)
    result = sess.run(outputs, {"inputs": processed})
    num_detections, detection_boxes, _, detection_classes = result
    batch_size = num_detections.shape[0]

    for batch in range(batch_size):
        for detection in range(int(num_detections[batch])):
            box = detection_boxes[batch][detection]
            label_idx = detection_classes[batch][detection]
            top, left, bottom, right = get_rect(frame.shape[0], frame.shape[1], box)
            cv2.rectangle(frame, (left, top), (right, bottom), color=(0, 255, 0), thickness=2)
            cv2.putText(frame, coco_classes[label_idx], (left, top-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,255,0), 2)    

    cv2.imshow('frame', frame)
    writer.write(frame)
    if cv2.waitKey(1) == ord('q'):
        break

cam.release()
writer.release()
cv2.destroyAllWindows()
