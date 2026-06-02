from ultralytics import YOLO
import cv2

# Load pretrained YOLOv8 model
model = YOLO("yolov8n.pt")  # small and fast model

# Open image
img = cv2.imread("test.jpg")

# Run detection
results = model(img)

# Show results
for r in results:
    boxes = r.boxes
    for box in boxes:
        cls = int(box.cls[0])
        
        # Class 0 = person in COCO dataset
        if cls == 0:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cv2.rectangle(img, (x1, y1), (x2, y2), (0,255,0), 2)
            cv2.putText(img, "Person", (x1, y1-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,255,0), 2)

cv2.imshow("Person Detection", img)
cv2.waitKey(0)
cv2.destroyAllWindows()