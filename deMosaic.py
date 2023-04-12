import torch
import torch.nn as nn
import torch.nn.functional as F
import enum
import numpy as np
import cv2
import numpy as np


 


#define the layout of the image
class Layout(enum.Enum):
    #Bayer pattern layout.
    #The first two values are the row and column of the green pixel in the top left corner.
    RGGB = (0, 1, 1, 2)
    GRBG = (1, 0, 2, 1)
    GBRG = (1, 2, 0, 1)
    BGGR = (2, 1, 1, 0)


#define the demosaic class model

class demosaic (nn.Module):
    #Demosaicing of Bayer images using Malver-He-Cutler algorithm.
    def __init__(self,layout: Layout = Layout.RGGB):
        super(demosaic, self).__init__()
        self.layout = layout
        self.kernels = torch.nn.Parameter(
            torch.tensor(
                [
                    # G at R,B locations
                    # scaled by 16
                    [ 0,  0, -2,  0,  0], # noqa
                    [ 0,  0,  4,  0,  0], # noqa
                    [-2,  4,  8,  4, -2], # noqa
                    [ 0,  0,  4,  0,  0], # noqa
                    [ 0,  0, -2,  0,  0], # noqa

                    # R,B at G in R rows
                    # scaled by 16
                    [ 0,  0,  1,  0,  0], # noqa
                    [ 0, -2,  0, -2,  0], # noqa
                    [-2,  8, 10,  8, -2], # noqa
                    [ 0, -2,  0, -2,  0], # noqa
                    [ 0,  0,  1,  0,  0], # noqa

                    # R,B at G in B rows
                    # scaled by 16
                    [ 0,  0, -2,  0,  0], # noqa
                    [ 0, -2,  8, -2,  0], # noqa
                    [ 1,  0, 10,  0,  1], # noqa
                    [ 0, -2,  8, -2,  0], # noqa
                    [ 0,  0, -2,  0,  0], # noqa

                    # R at B and B at R
                    # scaled by 16
                    [ 0,  0, -3,  0,  0], # noqa
                    [ 0,  4,  0,  4,  0], # noqa
                    [-3,  0, 12,  0, -3], # noqa
                    [ 0,  4,  0,  4,  0], # noqa
                    [ 0,  0, -3,  0,  0], # noqa

                    # R at R, B at B, G at G
                    # identity kernel not shown
                ]
            ).view(4, 1, 5, 5).float() / 16.0,
            requires_grad=False,
        )
        
        self.index = torch.nn.Parameter(
            # Below, note that index 4 corresponds to identity kernel
            self._index_from_layout(layout),
            requires_grad=False,
        )
        
        
    def forward(self, x):
        # Demosaic the input image.
        #Args:
        #    x: Input image. Must be a 4D tensor of shape (B, C, H, W) where C = 1.
        #Returns:
        #  Demosaiced image. A 4D tensor of shape (B, 3, H, W).
        #  The output image is in RGB format.
        #  The output image is clipped to the range [0, 1].
        #  If the shape of x has five dimensions, reshape the x to (B, C, H, W).
         
        print(x.shape)
        B, C, H, W = x.shape

        xpad = F.pad(x, (2, 2, 2, 2), mode="reflect") # Pad the image
        planes = F.conv2d(xpad, self.kernels, stride=1) # Convolve the image with the kernels
        planes = torch.cat( # Concatenate the image with the kernels
            (planes, x), 1
        )  # Concat with input to give identity kernel Bx5xHxW
        rgb = torch.gather( #gather the image
            planes, # Input
            1, 
            self.index.repeat( # Repeat the image
                1, 
                1,
                torch.div(H, 2, rounding_mode="floor"),
                torch.div(W, 2, rounding_mode="floor"),
            ).expand(
                B, -1, -1, -1
            ),  # Expand for singleton batch dimension is faster than repeat
        )
        return torch.clamp(rgb, 0, 1) # Clamp the image
    
    def _index_from_layout(self, layout: Layout) -> torch.Tensor:
        
        # Create the index tensor for the given layout.
        #       ...
        # ... b g b g ...
        # ... g R G r ...
        # ... b G B g ...
        # ... g r g r ...
        #       ...
        # fmt: off
        rggb = torch.tensor(
            [
                # dest channel r
                [4, 1],  # pixel is R,G1
                [2, 3],  # pixel is G2,B
                # dest channel g
                [0, 4],  # pixel is R,G1
                [4, 0],  # pixel is G2,B
                # dest channel b
                [3, 2],  # pixel is R,G1
                [1, 4],  # pixel is G2,B
            ]
        ).view(1, 3, 2, 2)
        # fmt: on
        return {
            Layout.RGGB: rggb,
            Layout.GRBG: torch.roll(rggb, 1, -1),
            Layout.GBRG: torch.roll(rggb, 1, -2),
            Layout.BGGR: torch.roll(rggb, (1, 1), (-1, -2)),
        }.get(layout)
    
    

    
    
    
def demosaicImage(ImageFilename):# Define the demosaic image function
    # Load the image
    # Read the image as grayscale and the image shoud be a single channel image
    image = cv2.imread(ImageFilename, cv2.IMREAD_GRAYSCALE)
    # Print the image name
    print("Image readed: ", ImageFilename)
    #print the image shape
    print ("Image shape: ", image.shape)
    # Convert the image to numpy array
    bayerInput = np.asarray(image)
    # Reshape the input as even rows and columns if the input is not even rows and columns
    # This is achieve by removing the last row and column
    if(bayerInput.shape[0] % 2 != 0):
        bayerInput = bayerInput[:-1, :]
    if(bayerInput.shape[1] % 2 != 0):
        bayerInput = bayerInput[:, :-1]
        
    # Convert the input matrix to a Bx1xHxW, [0..1], torch.float32 RGGB-Bayer tensor    
    bayerInput = torch.tensor(bayerInput).to(torch.float32).unsqueeze(0).unsqueeze(0) / 255.0
    
    
        
    # Detect the cuda device
    print("Detecting the cuda device...")
    if(torch.cuda.is_available()): #check if the cuda device is available
        print("Cuda device detected")
        device = torch.device("cuda")
        # Print the device name
        print("Device name: ", torch.cuda.get_device_name(0))
        
    else:
        print("Cuda device not detected")
        device = torch.device("cpu")
        # Print the device name
        print("Device name: ", torch.cuda.get_device_name(0))
        
    # Transfer the input to the cuda device
    modelInput = bayerInput.to(device) 
        
    # Initialize the fliter
    print("Initializing the fliter...")
    fliter = demosaic()
    
    # Transfer the fliter to the cuda device if the cuda device is available
    if(torch.cuda.is_available()):
        fliter = fliter.to(device)
    
    # Demosaic the image
    with torch.no_grad():
        Output = fliter(modelInput)
    
    # Convert the image to numpy array
    Output = Output.squeeze(0).permute(1, 2, 0).cpu().to(torch.float32).numpy()
    
    
    # Display the image as a RGB image
    cvt = cv2.cvtColor(Output, cv2.COLOR_RGB2BGR)
    
    # Save the image as the original image name with the prefix "demosaic_"
    cv2.imwrite( "Python_Demosaic_" + ImageFilename, cvt*255)
    
    
    

    
    
demosaicImage("test1.png")
demosaicImage("test2.png")
    
    
    
        
        
        
    
         
            