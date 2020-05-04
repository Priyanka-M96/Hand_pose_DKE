from utils import detector_utils as detector_utils
import cv2
import tensorflow as tf
from multiprocessing import Queue, Pool
from utils.detector_utils_gui import WebcamVideoStream
import datetime
import keyboard
from utils import pose_classification_utils as classifier
import math
import os
import PySimpleGUI as sg

frame_processed = 0
score_thresh = 0.2
gui_sensitivity = 15

sg.theme('LightGreen')

os.environ['KERAS_BACKEND'] = 'tensorflow'
direction = ""


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
            "F:\Realtime_Hand_tracking\cnn\models\hand_poses_wGarbage_100.h5")
    except Exception as e:
        print(e)

    detection_centres_x = []
    detection_centres_y = []
    is_centres_filled = False
    detected = False
    index = 0
    detection_area = []

    start_flag = False
    flag_start = pause_time = 0
    sensitivity = gui_sensitivity
    area = centre_x = centre_y = 0
    detection = ""
    direction = ""
    while True:
        predicted_label = ""
        frame = input_q.get()
        if (frame is not None):
            frame_processed += 1
            boxes, scores = detector_utils.detect_objects(frame, detection_graph, sess)

            # get region of interest
            res = detector_utils.get_box_image(cap_params['num_hands_detect'], cap_params['score_thresh'],
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

            if index >= sensitivity:
                index = 0
                is_centres_filled = True

            if index == 0:
                predicted_label = classify(res, model, classification_graph, session, poses)
                #print(predicted_label)

            if predicted_label == "Start" and flag_start == 0:
                #print("Start")
                detection = "Start tracking"
                start_flag = True
                flag_start = 1

            if detected:
                detection_centres_x = []
                detection_centres_y = []
                is_centres_filled = False
                index = 0
                detected = False
                detection_area = []
                frame_processed = 0
                pause_time = 30

            centres_x = detection_centres_x.copy()
            centres_y = detection_centres_y.copy()

            areas = detection_area.copy()

            centres_x = [v for v in centres_x if v]
            centres_y = [v for v in centres_y if v]

            areas = [a for a in areas if a]

            # angle_coordinate
            if len(centres_x) > 3 and is_centres_filled and len(centres_y) > 3 and len(areas) > 3 and start_flag :
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

                if dX > 100 and (abs(dY) < 20 or flag == 1):
                    direction = "Right"
                    keyboard.press_and_release('right')
                    detected = True
                    #print(direction)

                elif -dX > 100 and (abs(dY) < 20 or flag == 1):
                    direction = "Left"
                    keyboard.press_and_release('left')
                    detected = True
                    #print(direction)


                elif dY > 50 and (abs(dX) < 10 or flag == 2):
                    direction = "Down"
                    detected = True
                    #print(direction)

                elif -dY > 50 and (abs(dX) < 10 or flag == 2):
                    direction = "Up"
                    detected = True
                    #print(direction)

                elif areas[-1] - 3000 > areas[0] and abs(dX) < 30 and abs(dY) < 20:
                    direction = "Zoom in"
                    detected = True
                    #print(direction)
                elif areas[-1] < areas[0] - 3000 and abs(dX) < 10 and abs(dY) < 20:
                    direction = "Zoom out"
                    detected = True
                    #print(direction)

        output_q.put((frame, direction,predicted_label))
    sess.close()


if __name__ == '__main__':
    win_started = False
    video_src = 0
    num_hands = 2
    fps = 1
    width = 300
    height = 200
    display = 1
    num_workers = 1
    queue_size = 5
    image_elem = 0
    direction_elem = ""
    detection_cmd = ""

    input_q = Queue(maxsize=queue_size)
    output_q = Queue(maxsize=queue_size)
    cropped_q = Queue(maxsize=queue_size)

    video_capture = WebcamVideoStream(
        src=video_src, width=width, height=height).start()

    cap_params = {}
    frame_processed = 0
    cap_params['im_width'], cap_params['im_height'] = video_capture.size()
    cap_params['score_thresh'] = score_thresh

    # max number of hands we want to detect/track
    cap_params['num_hands_detect'] = num_hands

    print(cap_params)

    poses = []
    _file = open("poses.txt", "r")
    lines = _file.readlines()
    for line in lines:
        line = line.strip()
        if (line != ""):
            print(line)
            poses.append(line)

    # spin up workers to paralleize detection.
    pool = Pool(num_workers, worker,
                (input_q, output_q, cap_params, frame_processed, poses))


    start_time = datetime.datetime.now()
    num_frames = 0
    fps = 0
    index = 0

    while True:
        frame = video_capture.read()
        frame = cv2.flip(frame, 1)
        index += 1

        input_q.put(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        output = output_q.get()

        output_frame = output[0]
        direction = output[1]
        predicted_label = output[2]
        output_frame = cv2.cvtColor(output_frame, cv2.COLOR_RGB2BGR)

        elapsed_time = (datetime.datetime.now() - start_time).total_seconds()
        num_frames += 1
        fps = num_frames / elapsed_time

        if (output_frame is not None):
            if (display > 0):
                imgbytes = cv2.imencode('.png', output_frame)[1].tobytes()
                # ---------------------------- THE GUI ----------------------------
                if not win_started:
                    win_started = True
                    layout = [
                        [sg.Text('Real Time Motion Tracker', size=(30, 1))],
                        [sg.Text("Show Start gesture to start reading", size=(25, 1))],
                        [sg.Image(data=imgbytes, key='_IMAGE_')],
                        [sg.Button('Motion Tracker')],
                        [sg.Button('Gesture Recognition')],
                        [sg.Column([[sg.Text('The Gesture Shown :',key = 'gest'),sg.Text(predicted_label, key='detection')]],key = '__Col_1__',visible = False)],
                        [sg.Column([[sg.Text('Score_Threshold', key='text_score'),
                         sg.Slider(range=(0, 10), orientation='h', resolution=1, default_value=2, size=(15, 15),
                                   key='score_thresh')],
                        [sg.Text('Sensitivity', key='text_sensitivity'),
                         sg.Slider(range=(0, 30), orientation='h', resolution=1, default_value=10, size=(15, 15),
                                   key='sensitivity')],
                        [sg.Text('Detections delay', key='text_detection'),
                         sg.Slider(range=(10, 30), orientation='h', resolution=1, default_value=10, size=(15, 15),
                                   key='pause')],
                        [sg.Text('The Command is :', key='text_cmd'), sg.Text(direction, key='direction')]],key='__Col_2__',visible = False)],
                        [sg.Exit()]
                    ]
                    win = sg.Window('SSD Webcam Demo', layout, default_element_size=(14, 1), text_justification='right',
                                    auto_size_text=False, finalize=True)


                    image_elem = win['_IMAGE_']
                    direction_elem = win['direction']
                    detection_cmd = win['detection']
                else:
                    image_elem.update(data=imgbytes)
                    if direction != "":
                        direction_elem.update(direction)
                    if predicted_label != "":
                        detection_cmd.update(predicted_label)

                # print(direction)
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
            break

        event, values = win.read(timeout=0)
        if event is None or event == 'Exit':
            break
        elif event == 'Gesture Recognition':
            win.Element('__Col_1__').Update(visible=True)
            win.Element('__Col_2__').Update(visible=False)

        elif event == 'Motion Tracker':
            win.Element('__Col_1__').Update(visible=False)
            win.Element('__Col_2__').Update(visible=True)

        score_thresh = int(values['score_thresh']) / 10
        gui_sensitivity = int(values['sensitivity'])
        pause_time = int(values['pause'])

    elapsed_time = (datetime.datetime.now() - start_time).total_seconds()
    fps = num_frames / elapsed_time
    print("fps", fps)
    win.close()
    pool.terminate()
    video_capture.stop()
    cv2.destroyAllWindows()
