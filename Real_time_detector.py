from utils import detector_utils as detector_utils
import cv2
import tensorflow as tf
import multiprocessing
from multiprocessing import Queue, Pool
import time
from utils.detector_utils import WebcamVideoStream
import datetime
import argparse
import keyboard
from utils import pose_classification_utils as classifier
import math
import os

frame_processed = 0
score_thresh = 0.2

# Create a worker thread that loads graph and
# does detection on images in an input queue and puts it on an output queue

os.environ['KERAS_BACKEND'] = 'tensorflow'


def classify(res, model, classification_graph, session, poses):
    # classify
    if res is not None:
        class_res = classifier.classify(model, classification_graph, session, res)
        class_pred = class_res.argmax(axis=-1)
        predicted_label = poses[int(class_pred)]
        return predicted_label


def worker(input_q, output_q, cap_params, frame_processed, poses):
    print(">> loading frozen model for worker")
    detection_graph, sess = detector_utils.load_inference_graph()
    sess = tf.Session(graph=detection_graph)
    print(">> loading keras model for worker")
    try:
        model, classification_graph, session = classifier.load_KerasGraph(
            "F:\Realtime_Hand_tracking\cnn\models\hand_poses_wGarbage_10.h5")
    except Exception as e:
        print(e)

    detection_centres_x = []
    detection_centres_y = []
    is_centres_filled = False
    detected = False
    index = 0
    detection_area = []
    predicted_label = ""
    start_flag = False
    flag_start = 0
    pause_time = 0
    print("Show Start gesture to start reading")
    #file = open("eval_list.csv","w")

    while True:
        # print("> ===== in worker loop, frame ", frame_count)
        frame = input_q.get()
        flag = 0
        if (frame is not None):
            frame_processed += 1
            # Actual detection. Variable boxes contains the bounding box cordinates for hands detected,
            # while scores contains the confidence for each of these boxes.
            # Hint: If len(boxes) > 1 , you may assume you have found atleast one hand (within your score threshold)
            # frame = cv2.normalize(frame, None, 0,255, norm_type=cv2.NORM_MINMAX)
            boxes, scores = detector_utils.detect_objects(frame, detection_graph, sess)

            # print("Hello")
            # print(boxes[0])

            # get region of interest
            res = detector_utils.get_box_image(cap_params['num_hands_detect'], cap_params["score_thresh"],
                                               scores, boxes, cap_params['im_width'], cap_params['im_height'], frame)

            # get boundary box
            if pause_time == 0:
                centre_x, centre_y, area = detector_utils.draw_box_on_image(cap_params['num_hands_detect'],
                                                                        cap_params["score_thresh"],
                                                                        scores, boxes, cap_params['im_width'],
                                                                        cap_params['im_height'],
                                                                        frame)

            if pause_time > 0:
                pause_time -= 1

            if is_centres_filled:
                detection_centres_x = detection_centres_x[1:10]
                detection_centres_y = detection_centres_y[1:10]
                detection_area = detection_area[1:10]

                detection_centres_x.append(centre_x)
                detection_centres_y.append(centre_y)
                detection_area.append(area)
            else:
                detection_centres_x.append(centre_x)
                detection_centres_y.append(centre_y)
                detection_area.append(area)

            if pause_time == 0:
                index += 1
            # print(index)

            if index >= 20:
                index = 0
                is_centres_filled = True
                #start_flag = False

            if index == 0:
                predicted_label = classify(res, model, classification_graph, session, poses)
            # print(predicted_label)
            if predicted_label == "Start" and flag_start == 0:
                print("Start tracking")
                start_flag = True
                flag_start = 1

            elif predicted_label == "Stop" and flag_start == 1 and frame_processed > 30:
                print("Stopped tracking")
                start_flag = False
                flag_start = 0

            if detected:
                detection_centres_x = []
                detection_centres_y = []
                is_centres_filled = False
                index = 0
                detected = False
                detection_area = []
                frame_processed = 0
                pause_time = 10
                #time.sleep(2)
                #start_flag = False

            centres_x = detection_centres_x.copy()
            centres_y = detection_centres_y.copy()

            areas = detection_area.copy()

            centres_x = [v for v in centres_x if v]
            centres_y = [v for v in centres_y if v]

            areas = [a for a in areas if a]
            direction = ""

            # angle_coordinate
            # print(areas)
            if len(centres_x) > 3 and is_centres_filled and len(centres_y) > 3 and len(
                    areas) > 3 and start_flag == True:
                # centres_asc = centres.copy().sort()
                # centres_dsc = centres.copy().sort(reverse=True)
                # print(centres)
                # print("Last x:", centres_x[-1])
                # print("First x: ", centres_x[0])
                # print("Last y:", centres_y[-1])
                # print("First y:", centres_y[0])
                flag = 0
                dX = centres_x[-1] - centres_x[0]
                dY = centres_y[-1] - centres_y[0]

                if dX > 20 and dY > 20:
                    m = dY / dX
                    angle = math.degrees(math.atan(m))
                    if angle < 45:
                        flag = 1
                    elif angle > 45:
                        flag = 2
                    # print(angle)

                if dX > 100 and (abs(dY) < 20 or flag == 1):
                    direction = "Right"
                    #keyboard.press_and_release('left')
                    detected = True
                    print(direction)

                elif -dX > 100 and (abs(dY) < 20 or flag == 1):
                    direction = "Left"
                    #keyboard.press_and_release('right')
                    detected = True
                    print(direction)


                elif dY > 50 and (abs(dX) < 10 or flag == 2):
                    direction = "Down"
                    detected = True
                    print(direction)

                elif -dY > 50 and (abs(dX) < 10 or flag == 2):
                    direction = "Up"
                    detected = True
                    print(direction)

                elif areas[-1] - 3000 > areas[0] and abs(dX) < 30 and abs(dY) < 20:
                    direction = "Zoom in"
                    detected = True
                    print(direction)
                elif areas[-1] < areas[0] - 3000 and abs(dX) < 10 and abs(dY) < 20:
                    direction = "Zoom out"
                    detected = True
                    print(direction)

                # print("hello")


            cv2.putText(frame, direction, (20, 50), cv2.FONT_HERSHEY_SIMPLEX,
                        0.65, (77, 255, 9), 1)

        output_q.put(frame)
    sess.close()


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-src',
        '--source',
        dest='video_source',
        type=int,
        default=0,
        help='Device index of the camera.')
    parser.add_argument(
        '-nhands',
        '--num_hands',
        dest='num_hands',
        type=int,
        default=2,
        help='Max number of hands to detect.')
    parser.add_argument(
        '-fps',
        '--fps',
        dest='fps',
        type=int,
        default=1,
        help='Show FPS on detection/display visualization')
    parser.add_argument(
        '-wd',
        '--width',
        dest='width',
        type=int,
        default=300,
        help='Width of the frames in the video stream.')
    parser.add_argument(
        '-ht',
        '--height',
        dest='height',
        type=int,
        default=200,
        help='Height of the frames in the video stream.')
    parser.add_argument(
        '-ds',
        '--display',
        dest='display',
        type=int,
        default=1,
        help='Display the detected images using OpenCV. This reduces FPS')
    parser.add_argument(
        '-num-w',
        '--num-workers',
        dest='num_workers',
        type=int,
        default=1,
        help='Number of workers.')
    parser.add_argument(
        '-q-size',
        '--queue-size',
        dest='queue_size',
        type=int,
        default=5,
        help='Size of the queue.')
    args = parser.parse_args()

    input_q = Queue(maxsize=args.queue_size)
    output_q = Queue(maxsize=args.queue_size)

    video_capture = WebcamVideoStream(
        src=args.video_source, width=args.width, height=args.height).start()

    cap_params = {}
    frame_processed = 0
    cap_params['im_width'], cap_params['im_height'] = video_capture.size()
    cap_params['score_thresh'] = score_thresh

    # max number of hands we want to detect/track
    cap_params['num_hands_detect'] = args.num_hands

    print(cap_params, args)

    poses = []
    _file = open("poses.txt", "r")
    lines = _file.readlines()
    for line in lines:
        line = line.strip()
        if (line != ""):
            print(line)
            poses.append(line)

    # spin up workers to paralleize detection.
    pool = Pool(args.num_workers, worker,
                (input_q, output_q, cap_params, frame_processed, poses))

    start_time = datetime.datetime.now()
    num_frames = 0
    fps = 0
    index = 0

    cv2.namedWindow('Multi-Threaded Detection', cv2.WINDOW_NORMAL)

    while True:
        frame = video_capture.read()
        frame = cv2.flip(frame, 1)
        index += 1

        input_q.put(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        output_frame = output_q.get()

        output_frame = cv2.cvtColor(output_frame, cv2.COLOR_RGB2BGR)

        elapsed_time = (datetime.datetime.now() - start_time).total_seconds()
        num_frames += 1
        fps = num_frames / elapsed_time
        # print("frame ",  index, num_frames, elapsed_time, fps)

        if (output_frame is not None):
            if (args.display > 0):
                if (args.fps > 0):
                    detector_utils.draw_fps_on_image("FPS : " + str(int(fps)),
                                                     output_frame)
                cv2.imshow('Multi-Threaded Detection', output_frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            else:
                if (num_frames == 400):
                    num_frames = 0
                    start_time = datetime.datetime.now()
                else:
                    print("frames processed: ", index, "elapsed time: ",
                          elapsed_time, "fps: ", str(int(fps)))
        else:
            # print("video end")
            break
    elapsed_time = (datetime.datetime.now() - start_time).total_seconds()
    fps = num_frames / elapsed_time
    print("fps", fps)
    pool.terminate()
    video_capture.stop()
    cv2.destroyAllWindows()
