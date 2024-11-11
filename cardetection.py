import cv2
import os
import numpy as np
from datetime import datetime
from discord_webhook import DiscordWebhook

# ตั้งค่า YOLO
YOLO_CONFIG_PATH = 'D:/code/car_detection/cfg/yolov3.cfg'
YOLO_WEIGHTS_PATH = 'D:/code/car_detection/yolov3.weights'
YOLO_CLASSES_PATH = 'D:/code/car_detection/data/coco.names'

# อ่าน classes
with open(YOLO_CLASSES_PATH, 'r') as f:
    classes = f.read().strip().split('\n')

# โหลด YOLO
net = cv2.dnn.readNetFromDarknet(YOLO_CONFIG_PATH, YOLO_WEIGHTS_PATH)
layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]

# สร้างโฟลเดอร์บันทึกภาพ
output_folder = 'captured_images'
os.makedirs(output_folder, exist_ok=True)

# ตั้งค่า Discord Webhook URL
DISCORD_WEBHOOK_URL = 'https://discord.com/api/webhooks/1299796299599581246/XokPrFKXldnrfyRj0l8QSAFe_p-zJDRnfQmNmZCUv6FXkW1M4UPzsWn8QD54vj3wTVZR'

# กำหนดกรอบสี่เหลี่ยมที่ต้องการตรวจจับ
DETECTION_ZONE = (100, 100, 400, 400)  # (x, y, width, height) ของกรอบที่ต้องการ

# ฟังก์ชันตรวจสอบว่ารถอยู่ภายในกรอบที่กำหนดหรือไม่
def is_within_detection_zone(box, detection_zone):
    x, y, w, h = box
    zone_x, zone_y, zone_w, zone_h = detection_zone
    return (x >= zone_x and y >= zone_y and x + w <= zone_x + zone_w and y + h <= zone_y + zone_h)

# ฟังก์ชันตรวจจับรถบนฟุตบาต
def detect_vehicles_on_sidewalk(frame):
    blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outs = net.forward(output_layers)

    height, width = frame.shape[:2]
    boxes = []
    confidences = []
    class_ids = []

    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]

            if classes[class_id] in ["car", "truck", "bus", "motorbike"] and confidence > 0.5:
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                # ตรวจสอบว่ารถอยู่ภายในกรอบที่กำหนดหรือไม่
                if is_within_detection_zone((x, y, w, h), DETECTION_ZONE):
                    boxes.append([x, y, w, h])
                    confidences.append(float(confidence))
                    class_ids.append(class_id)

    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
    return [(boxes[i], classes[class_ids[i]]) for i in indexes]

# ฟังก์ชันบันทึกภาพและส่งไปยัง Discord
def save_and_send_image(frame):
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    image_path = os.path.join(output_folder, f'vehicle_{timestamp}.jpg')
    cv2.imwrite(image_path, frame)

    webhook = DiscordWebhook(url=DISCORD_WEBHOOK_URL, content="Vehicle detected on sidewalk!")
    with open(image_path, "rb") as f:
        webhook.add_file(file=f.read(), filename=image_path)
    webhook.execute()

# เปิดไฟล์วิดีโอ
video_path = 'car.mp4'  # ตั้งชื่อไฟล์วิดีโอที่ต้องการใช้
cap = cv2.VideoCapture(video_path)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # ตรวจจับวัตถุและกรองให้เหลือเฉพาะรถในกรอบที่กำหนด
    detections = detect_vehicles_on_sidewalk(frame)
    
    # วาดกรอบสี่เหลี่ยมสำหรับโซนตรวจจับ
    cv2.rectangle(frame, (DETECTION_ZONE[0], DETECTION_ZONE[1]), 
                  (DETECTION_ZONE[0] + DETECTION_ZONE[2], DETECTION_ZONE[1] + DETECTION_ZONE[3]), 
                  (255, 0, 0), 2)  # สีน้ำเงิน

    for (box, label) in detections:
        x, y, w, h = box
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        save_and_send_image(frame)

    cv2.imshow('Vehicle Detection', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
