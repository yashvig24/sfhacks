import cv2
camera = cv2.VideoCapture(0)
# number of frames to throw away while the camera adjusts to the lighting
ramp_frames = 25
# number of frames to keep
keep_frames = 50

def get_image():
    # read is the easiest way to get a full image out of a VideoCapture object.
    retval, im = camera.read()
    return im

for i in xrange(ramp_frames):
    temp = get_image()

for i in xrange(keep_frames):
    f = ('dataset/train/2/train_2_' + str(i) + '.png')
    pic = get_image()
    cv2.imwrite(f, pic)

del(camera)