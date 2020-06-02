import numpy as np
import imageio
from skimage import img_as_ubyte
from PIL import Image

"""
REMEMBER to add an images folder inside the taskgrouping folder.
Place this python file inside the taskgrouping folder.

In the model.py file:
import visualizer

Inside the Network nn.Module class, in the forward(input) function add:
visualizer.visualize(input, outputs)
(before the line that returns outputs)
"""

def visualize(input_img, outputs):
    # Saving the input image
    img_np = input_img[0].cpu().detach().numpy()
    img_np = np.ascontiguousarray(img_np.transpose(1,2,0))
    imageio.imwrite("images/0input.jpg", img_as_ubyte(img_np))

    i = 1
    # Saving the representation and task outputs
    for task, output in outputs.items():
        img_np = output[0].cpu().detach().numpy()
        img_np = np.ascontiguousarray(img_np.transpose(1,2,0))

        if task == "rep":
            continue

        if img_np.shape[2] == 3: 
            if (img_np.min() > -1 and img_np.max() < 1) or True:
                # Per channel rescaling
                for k in range(3):
                    if np.max(np.abs(img_np[:,:,k])) >= 1.0:
                        img_np[:,:,k] /= np.max(np.abs(img_np[:,:,k]))
                imageio.imwrite("images/" + str(i) + task + ".jpg", img_as_ubyte(img_np/np.max(np.abs(img_np))))
                
        elif img_np.shape[2] == 1:
            #img_np = img_np.squeeze(axis = 2)
            imageio.imwrite("images/" + str(i) + task + ".jpg", img_as_ubyte(img_np/np.max(np.abs(img_np))))

        i += 1