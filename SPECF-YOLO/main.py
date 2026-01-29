import cv2
import time
import numpy
import supervision as sv
from ultralytics import YOLO
from collections import Counter
from stereo import camera_configs
from supervision.draw.color import Color


def centerPointDist(p1, p2, PointClouds):
    x, y = int((p1[0] + p2[0]) / 2), int((p1[1] + p2[1]) / 2)
    X = PointClouds[y][x][0] ** 2
    Y = PointClouds[y][x][1] ** 2
    Z = PointClouds[y][x][-1] ** 2
    return "{:.2f}cm".format((X + Y + Z) ** 0.5 / 10)


def coord2dist(p1, p2, PointClouds):
    x, y = int((p1[0] + p2[0]) / 2), int((p1[1] + p2[1]) / 2)
    X = (PointClouds[y][x + 2][0] - PointClouds[y][x - 2][0]) ** 2
    Y = (PointClouds[y][x + 2][1] - PointClouds[y][x - 2][1]) ** 2
    Z = (PointClouds[y][x + 2][-1] - PointClouds[y][x - 2][-1]) ** 2
    return "{:.2f}cm".format((X + Y + Z) ** 0.5 * abs(p1[0] - p2[0]) / 50)


def getPointClouds(images_left, images_right):
    img1_rectified = cv2.remap(images_left, camera_configs.left_map1, camera_configs.left_map2, cv2.INTER_LINEAR)
    img2_rectified = cv2.remap(images_right, camera_configs.right_map1, camera_configs.right_map2, cv2.INTER_LINEAR)

    imgL = cv2.cvtColor(img1_rectified, cv2.COLOR_BGR2GRAY)
    imgR = cv2.cvtColor(img2_rectified, cv2.COLOR_BGR2GRAY)

    stereo = cv2.StereoSGBM.create(minDisparity=32, numDisparities=176, blockSize=16)
    disparity = stereo.compute(imgL, imgR)

    # disp = cv2.normalize(disparity, disparity, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    return cv2.reprojectImageTo3D(disparity.astype(numpy.float32) / 16., camera_configs.Q)


def coord2speed(id, p1, p2, Cnt, PointClouds):
    p = (int((p1[0] + p2[0]) / 2), int((p1[1] + p2[1]) / 2))
    if id == "Tracking":
        return "UN_speed"
    elif id not in Cnt.keys():
        Cnt[id] = {'time': time.time(), 'pos': PointClouds[p[1]][p[0]]}
        return "Calculating"
    else:
        last_time, last_pos = Cnt[id]['time'], Cnt[id]['pos']
        cur_time, cur_pos = time.time(), PointClouds[p[1]][p[0]]
        Cnt[id] = {'time': cur_time, 'pos': cur_pos}
        move = ((cur_pos[0] - last_pos[0]) ** 2 + (cur_pos[1] - last_pos[1]) ** 2 + (cur_pos[-1] - last_pos[-1]) ** 2) ** 0.5
        return "{:.1f}cm/s".format(move / (cur_time - last_time) / 10)


if __name__ == '__main__':
    YOLO = YOLO('.../best.pt')

    camera = cv2.VideoCapture(0)
    camera.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    line_counter = sv.LineZone(start=sv.Point(500, 0), end=sv.Point(500, 480))  # 前面是X后面是Y
    line_annotator = sv.LineZoneAnnotator(thickness=2, text_thickness=1, text_scale=0.5,
                                          color=Color(r=224, g=57, b=151))
    box_annotator = sv.BoxAnnotator(thickness=2, text_thickness=1, text_scale=0.5)

    speed_cnt = Counter()

    while True:
        ret, frame = camera.read()
        if not ret:
            break

        frame1 = frame[:, 0:640]
        frame2 = frame[:, 640:1280]

        pointClouds = getPointClouds(frame1, frame2)

        results = YOLO.track(frame1)
        for result in results:
            detections = sv.Detections.from_yolov8(result)

            if result.boxes.is_track:
                detections.tracker_id = result.boxes.id.cpu().numpy().astype(int)

            position_ids = detections.xyxy
            class_ids = detections.class_id
            confidences = detections.confidence
            tracker_ids = detections.tracker_id

            labels = []
            cnt = Counter()

            for index in range(len(class_ids)):
                p1 = (int(position_ids[index][0]), int(position_ids[index][1]))
                p2 = (int(position_ids[index][2]), int(position_ids[index][3]))

                tracker_id = tracker_ids[index] if result.boxes.is_track else "Tracking"

                cls = YOLO.names[class_ids[index]]
                cnt[cls] += 1

                dist = centerPointDist(p1, p2, pointClouds)
                length = coord2dist(p1, p2, pointClouds)
                speed = coord2speed(tracker_id, p1, p2, speed_cnt,pointClouds)

                labels.append("#{} {} dist:{} length:{} speed:{}".format(tracker_id, cls, dist, length,speed))

            curItems = ""
            for key, val in cnt.items():
                curItems = curItems + "{} {},".format(val, key)

            cur_time = time.time()
            for key,dicts in list(speed_cnt.items()):
                if cur_time - dicts['time'] > 5:
                    del speed_cnt[key]

            cv2.putText(frame1, curItems[:-2], (25, 25), cv2.FONT_HERSHEY_COMPLEX, 0.5, (100, 200, 200), 1)

            frame = box_annotator.annotate(scene=frame1, detections=detections, labels=labels)

            line_counter.trigger(detections=detections)
            line_annotator.annotate(frame=frame1, line_counter=line_counter)

        cv2.imshow("frame1", frame1)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    camera.release()
    cv2.destroyAllWindows()
