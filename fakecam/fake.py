import os
import cv2
import numpy as np
import requests
import pyfakewebcam
import datetime
from videocaptureasync import VideoCaptureAsync

# fetch the mask with retries (the app needs to warmup and we're lazy)
# e v e n t u a l l y c o n s i s t e n t
def get_mask_bodypix(frame, bodypix_url='http://localhost:9000'):
    mask = None
    while mask is None:
        try:
            _, data = cv2.imencode(".jpg", frame)
            r = requests.post(
                url=bodypix_url,
                data=data.tobytes(),
                headers={'Content-Type': 'application/octet-stream'})
            mask = np.frombuffer(r.content, dtype=np.uint8)
            mask = mask.reshape((frame.shape[0], frame.shape[1]))
        except KeyboardInterrupt:
            raise
        except:
            print("mask request failed, retrying")
    return mask

def get_mask_tf(frame):
    input_height = input_details[0]['shape'][1]
    input_width = input_details[0]['shape'][2]

    img = cv2.resize(frame, (input_width, input_height))

    # add N dim
    input_data = np.expand_dims(img, axis=0)

    # check the type of the input tensor
    floating_model = input_details[0]['dtype'] == np.float32
    if floating_model:
        input_data = (np.float32(input_data) - args.input_mean) / args.input_std

    interpreter.set_tensor(input_details[0]['index'], input_data)

    interpreter.invoke()

    output_data = interpreter.get_tensor(output_details[0]['index'])
    results = np.squeeze(output_data)

    # find the highest-probability class for each pixel (along axis 2)
    out = np.apply_along_axis(np.argmax,2,results)

    # category labels for deeplabv3_257_mv_gpu.tflite
    labels = [ "background", "aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow", "dining table", "dog", "horse", "motorbike", "person", "potted plant", "sheep", "sofa", "train", "tv" ]

    # set pixels with likeliest class == person to 1
    pers_idx = labels.index("person")
    person = np.where(out == pers_idx, 1, 0).astype(np.uint8)

    return cv2.resize(person, (frame.shape[1], frame.shape[0]))

def post_process_mask(mask):
    mask = cv2.dilate(mask, np.ones((10,10), np.uint8) , iterations=1)
    mask = cv2.blur(mask.astype(float), (30,30))
    return mask

def shift_image(img, dx, dy):
    img = np.roll(img, dy, axis=0)
    img = np.roll(img, dx, axis=1)
    if dy>0:
        img[:dy, :] = 0
    elif dy<0:
        img[dy:, :] = 0
    if dx>0:
        img[:, :dx] = 0
    elif dx<0:
        img[:, dx:] = 0
    return img

t = 0
def hologram_effect(img):
    global t
    t += 3
    # add a blue tint
    holo = cv2.applyColorMap(img, cv2.COLORMAP_WINTER)
    # add a halftone effect
    bandLength, bandGap = 2, 3
    for y in range(holo.shape[0]):
        if (y - t) % (bandLength+bandGap) < bandLength:
            holo[y,:,:] = holo[y,:,:] * np.random.uniform(0.1, 0.3)
    # add some ghosting
    holo_blur = cv2.addWeighted(holo, 0.2, shift_image(holo.copy(), 5, 5), 0.8, 0)
    holo_blur = cv2.addWeighted(holo_blur, 0.4, shift_image(holo.copy(), -5, -5), 0.6, 0)
    # combine with the original color, oversaturated
    out = cv2.addWeighted(img, 0.5, holo_blur, 0.6, 0)
    return out

def mask_frame(frame):
    #mask = get_mask_tf(frame)
    mask = get_mask_bodypix(frame)

    # post-process mask and frame
    return post_process_mask(mask)

# composite the foreground and background
def blend_frame(frame, background, mask):
    inv_mask = 1-mask
    if args.enable_hologram:
        inv_mask += 0.1*mask
    result = frame
    for c in range(frame.shape[2]):
        result[:,:,c] = frame[:,:,c]*mask + background[:,:,c]*inv_mask
        #result[:,:,c] = frame[:,:,c]*mask
    #result = cv2.bitwise_and(frame, frame, mask=mask)
    #background = cv2.bitwise_and(background, background, mask=inv_mask)
    #result = cv2.add(background, result)
    #result = cv2.addWeighted(background, 0.6, result, 1, 0)

    return result

try:
    import argparse
    parser = argparse.ArgumentParser(description='Virtual background fake webcam')
    parser.add_argument('-i', '--input', default='/dev/video0', help='real webcam device')
    parser.add_argument('-o', '--output', default='/dev/video20', help='loopback video device')
    parser.add_argument('--width', default=640, help='video width')
    parser.add_argument('--height', default=400, help='video height')
    parser.add_argument('--enable-hologram', action='store_true', help='enable hologram effect')
    parser.add_argument('background', default=['data/*'], nargs='*', help='background files (images or videos)')

    parser.add_argument( '-m', '--model', default='deeplabv3_257_mv_gpu.tflite', help='.tflite model to be executed')
    parser.add_argument('--input_mean', default=127.5, type=float, help='input_mean')
    parser.add_argument('--input_std', default=127.5, type=float, help='input standard deviation')

    args = parser.parse_args()
   
    import tensorflow as tf
    interpreter = tf.lite.Interpreter(model_path=args.model)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
 
    # setup access to the *real* webcam
    print('Opening webcam', args.input, '...')
    #cap = cv2.VideoCapture(args.input)
    cap = VideoCaptureAsync(args.input, args.width, args.height)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    cap.set(cv2.CAP_PROP_FPS, 25)
    
    # setup the fake camera
    fake = None
    if args.output != "imshow":
        print('Writing to loopback device', args.output, '...')
        fake = pyfakewebcam.FakeWebcam(args.output, args.width, args.height)
    
    # load the virtual background
    background_index = 0
    background = None
    bg_cap = None
    
    import glob
    background_filenames = []
    for background_filename in args.background:
        background_filenames += glob.glob(background_filename)
    if len(background_filenames) == 0:
        print("No background files found on", args.background)
        exit(1)

    def change_background():
        global background_index
        global background
        global bg_cap
        background_filename = background_filenames[background_index % len(background_filenames)]

        print("Loading background", background_filename)
        try:
            bg_cap = cv2.VideoCapture(background_filename)
            #bg_cap = VideoCaptureAsync(background_filename, args.width, args.height)
        except:
            background = cv2.imread(background_filename)
            background = cv2.resize(background, (args.width, args.height))

        background_index += 1

    now = datetime.datetime.now()
    nframe = 1
    frame = None
    mask = None
    
    # frames forever
    while True:
        if background is None or datetime.datetime.now() - now > datetime.timedelta(minutes=1):
            change_background()
            now = datetime.datetime.now()

        # Capture background
        if bg_cap:
            _, new_background = bg_cap.read()

            if new_background is None:
                # Try to loop back to the beginning
                bg_cap.set(cv2.CAP_PROP_POS_FRAMES, 1)
                _, new_background = bg_cap.read()

            if new_background is not None:
                background = cv2.resize(new_background, (args.width, args.height))
   
        # Capture webcam 
        nframe += 1
        if frame is None or nframe > 5:
            nframe = 0
            _, frame = cap.read()
            mask = mask_frame(frame)
        else:
            cap.read()

        # Blend webcam image into background
        if background is not None and frame is not None and mask is not None:
            final_frame = frame.copy()
            if args.enable_hologram:
                final_frame = hologram_effect(final_frame)

            final_frame = blend_frame(final_frame, background, mask)

            if fake:
                # fake webcam expects RGB
                final_frame = cv2.cvtColor(final_frame, cv2.COLOR_BGR2RGB)
                fake.schedule_frame(final_frame)
            else:
                cv2.imshow("mask", final_frame)
                cv2.waitKey(1)
    
except KeyboardInterrupt:
    exit(0)