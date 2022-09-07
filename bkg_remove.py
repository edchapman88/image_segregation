import numpy as np
import mediapipe as mp
import cv2
import matplotlib.pyplot as plt

def unsharp_mask(image, kernel_size=(5, 5), sigma=1.0, amount=1.0, threshold=0):
    """Return a sharpened version of the image, using an unsharp mask."""
    blurred = cv2.GaussianBlur(image, kernel_size, sigma)
    sharpened = float(amount + 1) * image - float(amount) * blurred
    sharpened = np.maximum(sharpened, np.zeros(sharpened.shape))
    sharpened = np.minimum(sharpened, 255 * np.ones(sharpened.shape))
    sharpened = sharpened.round().astype(np.uint8)
    if threshold > 0:
        low_contrast_mask = np.absolute(image - blurred) < threshold
        np.copyto(sharpened, image, where=low_contrast_mask)
    return sharpened

BG_COLOR = (192, 192, 192) # gray
MASK_COLOR = (255, 255, 255) # white

# names = ['Ed','Andrew','Chris','David','Dimi','Euan','Jaffer','Nafees','TonyD']
names = ['man']
def augment(path):
  return f'./imgs/{path}.png'
image_paths = map(augment,names)


with mp.solutions.selfie_segmentation.SelfieSegmentation(
    model_selection=0) as selfie_segmentation:
  for idx, file in enumerate(image_paths):
    image = cv2.imread(file)
    image_height, image_width, _ = image.shape

    print(image.shape)
    image_rgba = cv2.cvtColor(image, cv2.COLOR_RGB2RGBA)
    

    # Convert the BGR image to RGB before processing.
    results = selfie_segmentation.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

    smoothed_mask = cv2.bilateralFilter(results.segmentation_mask,50,350,350)

    # Draw selfie segmentation on the background image.
    # To improve segmentation around boundaries, consider applying a joint
    # bilateral filter to "results.segmentation_mask" with "image".
    # condition returns true for foreground
    condition = np.stack((smoothed_mask,) * 4, axis=-1) > 0.9 # default 0.1

    # Generate solid color images for showing the output selfie segmentation mask.
    fg_image = np.zeros(image.shape, dtype=np.uint8)
    fg_image[:] = MASK_COLOR
    bg_image = np.zeros(image_rgba.shape, dtype=np.uint8)
    # bg_image[:,:,:3] = BG_COLOR
    # bg_image[:,:,3] = 0

    output_image = np.where(condition, image_rgba, bg_image)

    sharpened = unsharp_mask(output_image,(5,5),1,5,0)


    cv2.imwrite('out_' + str(names[idx]) + '.png', output_image)



