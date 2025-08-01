import os
import cv2
import numpy as np
import random



#I primarily followed the steps outlined in this article: https://www.microsoft.com/en-us/research/publication/ms-asl-a-large-scale-data-set-and-benchmark-for-understanding-american-sign-language/


def uniformFrameTime(totFrames,tarFrames):
    """
    
    We calculate the frame index interval that we need to extract from the video, using a regular interval to cover the entire video

    Args:
        totFrames: Number of frames in the current video 
        tarFrames: The number of frame that we want at the end (here 64)

    Return:
        indices: indices of the frames we have to take in the current video 
    """
    indices=[]
    ratio=(totFrames/tarFrames)
    for i in range(tarFrames):
        indices.append(min(int(round(i*ratio)),totFrames-1))
    return indices


def resizeNx256or256xN(frame):
    """
    Specifically, we resize the image so that its smaller side is 256 pixels, while the other side is resized proportionally
    (The scientific article I based my work on had something quite similar)

    ARgs:
        Frame: the current frame that we want to resize

    Return:
        resizedFrame: The frame resized to dimensions 256xN or Nx256 
    
    """
    h,w,c=frame.shape
    #print(h)
    #print(w)
    #print(c)
    if h<w:
        scale=256/h
        newH,newW =256,int(w*scale)
    else:
        scale=256/w
        newH,newW=int(h*scale),256
    resizedFrame=cv2.resize(frame,(newW,newH))
    return resizedFrame


def crop(frame,cropCoords):
    """
    To crop a frame to a specific area

    ARgs:
        Frame: the frame
        cropCoords: The coordinates of the area 

    Return:
        frame: The frame cropped
    """
    top,left,cropH,cropW=cropCoords
    return frame[top:top+cropH,left:left+cropW]


def calculateCropcoo(frameShape,cropSize=(224,224),randomCrop=False):
    """
    We calculate the coordinates of the cropping area to pass them to the function that will crop the frame using these value

    Args:
        frameShape: the frame

    Return:
        top,left,cropH,cropW: the different coordinates
    """
    h,w,_=frameShape
    cropH,cropW=cropSize

    #IN some cases we will need a random crop and sometimes not
    if randomCrop:
        top=random.randint(0,h-cropH)
        left=random.randint(0,w-cropW)
    else:
        top=(h-cropH)//2
        left=(w-cropW)//2
    return top,left,cropH,cropW


def computeAugmentations(frame):
    """
    We apply standard data augmentations here

    Args:
        Frame: The frame
    
    Return:
        augmentations: The different augmentations (flipped, brightened, and contrasted)
    """
    augmentations = {}
    augmentations['flipped']=cv2.flip(frame,1)
    brightval=random.uniform(0.5,1.5)
    augmentations['brightened']=np.clip(frame*brightval,0,255).astype(np.uint8)
    contrastX=random.uniform(0.5,1.5)
    mean=np.mean(frame)
    augmentations['contrasted']=np.clip((frame-mean)*contrastX+mean,0,255).astype(np.uint8)
    #print(augmentations)
    return augmentations


def extractF(video_path, output_dir, num_frames=64, crop_size=(224, 224)):
    """
    Here, we first calculate the coordinates for a random crop, followed by a centered crop
    We crop the original frame, then the variations, and save them
    """
    os.makedirs(output_dir,exist_ok=True)
    cap=cv2.VideoCapture(video_path)
    total=int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if total==0:
        print(f"Error we don't ave any frames here:  {video_path} ")
        with open("error_log.txt", "a") as log_file:
            log_file.write(f"Error we don't ave any frames here: {video_path} \n")
        cap.release()
        return

    indices=uniformFrameTime(total, num_frames)

    #Fot the original random crop
    originalCropCoords=None 
    #For the other one centering crop
    centerCropCoords=None  

    for i, idx in enumerate(indices):
        cap.set(cv2.CAP_PROP_POS_FRAMES,idx)
        ret,frame=cap.read()

        if ret:
            #SO Firstly Resize to 256xN or Nx256
            resized = resizeNx256or256xN(frame)
            #print(resized)

            #Here it's to define thecropping areas
            if originalCropCoords is None:
                originalCropCoords=calculateCropcoo(resized.shape,crop_size,randomCrop=True)
            if centerCropCoords is None:
                centerCropCoords=calculateCropcoo(resized.shape,crop_size, randomCrop=False)
            #Crop for original
            originaldata=crop(resized,originalCropCoords)

            #The we are saving the original 
            dirFororigin=os.path.join(output_dir,"original")
            os.makedirs(dirFororigin,exist_ok=True)
            original_path=os.path.join(dirFororigin,f"frame_{i:04d}.jpg")
            cv2.imwrite(original_path,originaldata)

            #Crop and augmentation for the others variations
            frameCropped=crop(resized,centerCropCoords)
            #print(frameCropped)
            augmenteddatares=computeAugmentations(frameCropped)
            #print(augmenteddatares)
            #And we have to save these variations
            for aug_type, augFrame in augmenteddatares.items():
                augOutputD = os.path.join(output_dir, aug_type)
                os.makedirs(augOutputD, exist_ok=True)
                #print("test")
                framePath=os.path.join(augOutputD,f"frame_{i:04d}.jpg")
                cv2.imwrite(framePath,augFrame)

    cap.release()
    print(f"Done for {video_path}")
    print("-----------------------")


def processAll(videoDir,framesDir,num_frames=64,crop_size=(224,224)):
    """
    To extract the lists of frames for all videos in each dataset group: train validation and test

    Args:
        crop_size: the final cropping parameter 
        num_frames: the number of frames extracted for each video
        framesDir: The frames directory 
        videoDir: The video Directory
    """
    for classF in os.listdir(videoDir):
        class_path = os.path.join(videoDir, classF)
        outputClassDir = os.path.join(framesDir, classF)
        os.makedirs(outputClassDir, exist_ok=True)

        for video in os.listdir(class_path):
            videoPath=os.path.join(class_path,video)
            outputD=os.path.join(outputClassDir,video.split(".")[0])

            print(f"Extracting for {videoPath} to {outputD}")
            extractF(videoPath,outputD,num_frames,crop_size)


if __name__=="__main__":

    trainVideoDir="ASL20/train"
    valVideoDir="ASL20/val"
    testVideoDir="ASL20/test"

    trainFramesDir="ASL20_frames/train"
    valFramesDir="ASL20_frames/val"
    testFramesDir="ASL20_frames/test"

    numFrames = 64

    processAll(trainVideoDir,trainFramesDir,numFrames)
    processAll(valVideoDir,valFramesDir,numFrames)
    processAll(testVideoDir,testFramesDir,numFrames)
