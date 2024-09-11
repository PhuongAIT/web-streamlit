import cv2
import numpy as np

def detect_count_students(st_image):
    focused = (40, 50, 50)  # Giá trị Hue, Saturation và Value của màu focused trong không gian màu HSV
    raising_hand = (20, 255, 255)  # Giá trị Hue, Saturation và Value của màu raising_hand trong không gian màu HSV
    distracted = (100, 150, 150)  # Giá trị Hue, Saturation và Value của màu distracted trong không gian màu HSV
    sleep = (170, 255, 255)  # Giá trị Hue, Saturation và Value của màu sleep trong không gian màu HSV
    using_phone = (90, 100, 100)  # Giá trị Hue, Saturation và Value của màu using_phone trong không gian màu HSV

    file_bytes = np.asarray(bytearray(st_image.read()), dtype=np.uint8)
    opencv_image = cv2.imdecode(file_bytes, 1)

    image = opencv_image.copy()
    image_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    mask_focused = cv2.inRange(image_hsv, focused)
    mask_raising_hand = cv2.inRange(image_hsv, raising_hand)
    mask_distracted = cv2.inRange(image_hsv, distracted)
    mask_sleep = cv2.inRange(image_hsv, sleep)
    mask_using_phone = cv2.inRange(image_hsv, using_phone)
    mask = mask_focused + mask_raising_hand + mask_distracted + mask_sleep + mask_using_phone

    cnts, _ = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    c_num = 0
    for i, c in enumerate(cnts):
        # Tính toán hình chữ nhật bao quanh đối tượng
        x, y, w, h = cv2.boundingRect(c)
        # Lấy màu của đối tượng từ hình ảnh gốc
        object_color = opencv_image[y:y+h, x:x+w]
        # Tính toán màu trung bình của đối tượng
        avg_color = np.mean(object_color, axis=(0, 1)).astype(int)
        # Vẽ hình chữ nhật với màu trung bình của đối tượng
        cv2.rectangle(image, (x, y), (x + w, y + h), tuple(avg_color), 2)
        # Tăng biến đếm số lượng đối tượng
        c_num += 1
        # Thêm số thứ tự của đối tượng
        cv2.putText(image, "#{}".format(c_num), (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

    #convert opencv bgr to rgb
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    return c_num, image 










