import cv2
import RPi.GPIO as GPIO
import time

# Setup GPIO
GPIO.setmode(GPIO.BOARD)
GPIO.setwarnings(False)

# LED pin setup
LED_PIN = 12
GPIO.setup(LED_PIN, GPIO.OUT)
GPIO.output(LED_PIN, GPIO.LOW)

# Ultrasonic sensor pin setup
TRIG = 3
ECHO = 5
GPIO.setup(TRIG, GPIO.OUT)
GPIO.setup(ECHO, GPIO.IN)

# Load class names
classNames = []
classFile = "/home/pavan/Desktop/Object_Detection_Files/coco.names"
with open(classFile, "rt") as f:
    classNames = f.read().rstrip("\n").split("\n")

# Load the object detection model
configPath = "/home/pavan/Desktop/Object_Detection_Files/ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt"
weightsPath = "/home/pavan/Desktop/Object_Detection_Files/frozen_inference_graph.pb"
net = cv2.dnn_DetectionModel(weightsPath, configPath)
net.setInputSize(320, 320)
net.setInputScale(1.0 / 127.5)
net.setInputMean((127.5, 127.5, 127.5))
net.setInputSwapRB(True)

def measure_distance(trig_pin, echo_pin):
    GPIO.output(trig_pin, GPIO.LOW)
    time.sleep(0.000002)
    GPIO.output(trig_pin, GPIO.HIGH)
    time.sleep(0.00001)
    GPIO.output(trig_pin, GPIO.LOW)

    while GPIO.input(echo_pin) == GPIO.LOW:
        start_time = time.time()
    while GPIO.input(echo_pin) == GPIO.HIGH:
        end_time = time.time()

    pulse_duration = end_time - start_time
    distance = pulse_duration * 17150
    distance = round(distance, 2)

    return distance

def getObjects(img, thres, nms, draw=True, objects=["car", "truck", "bus", "train"]):
    classIds, confs, bbox = net.detect(img, confThreshold=thres, nmsThreshold=nms)
    
    object_detected = False

    if len(classIds) != 0:
        for classId, confidence, box in zip(classIds.flatten(), confs.flatten(), bbox):
            className = classNames[classId - 1]
            if className in objects:
                object_detected = True

                if draw:
                    cv2.rectangle(img, box, color=(0, 255, 0), thickness=2)
                    cv2.putText(img, className.upper(), (box[0] + 10, box[1] + 30),
                                cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)
                    cv2.putText(img, str(round(confidence * 100, 2)), (box[0] + 200, box[1] + 30),
                                cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)

    return img, object_detected

if __name__ == "__main__":
    cap = cv2.VideoCapture(0)
    cap.set(3, 640)
    cap.set(4, 480)

    try:
        while True:
            success, img = cap.read()
            if not success:
                break

            # Object detection
            result, object_detected = getObjects(img, 0.45, 0.5, objects=["car", "truck", "bus", "train"])
            cv2.imshow("Output", img)

            # Ultrasonic sensor distance measurement
            distance = measure_distance(TRIG, ECHO)
            print(f"Distance: {distance} cm")

            # Trigger LED based on conditions
            if object_detected and distance < 15:
                GPIO.output(LED_PIN, GPIO.HIGH)
                print("Object detected within 15 cm. Triggering LED.")
            else:
                GPIO.output(LED_PIN, GPIO.LOW)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            time.sleep(0.1)

    except KeyboardInterrupt:
        print("Stopped by user")

    finally:
        cap.release()
        cv2.destroyAllWindows()
        GPIO.cleanup()
