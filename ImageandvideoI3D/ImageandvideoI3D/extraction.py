import os
import json
import random
import subprocess
from yt_dlp import YoutubeDL
from collections import defaultdict


def jsonReading(file):
    """
    We extract the information from the JSON file

    Args:
        file: File path to load

    Returns:
        int or float: JSON data 
    """
    with open(file, 'r') as f:
        return json.load(f)


def logGenerator(errorName, log_file="error_log.txt"):
    """
    We extract YouTube video download errors and log them in error_log.txt

    Args:
        errorName: The name of the error 
    """
    with open(log_file,"a") as f:
        f.write(errorName+"\n")


def normalizeUrl(urlLink):
    """
    Sometimes the video link URLs are written without "https" but it is necessary to include it for use with Ydl 

    Args:
        urlLink: Url of the video 

    Return:
        urlLink: Normalized URL
    """
    if not urlLink.startswith("http://") and not urlLink.startswith("https://"):
        return "https://"+urlLink
    return urlLink



#Doc of yt_dlb: https://github.com/yt-dlp/yt-dlp
def download_clip(url, output_path):
    """
    Downloads a video in MP4 format using yt-dlp, we can also force the conversion here if it's a necessity

    Args:
        url: The URL of the video to download
        output_path: The file path where the downloaded video will be saved

    Raises:
        Exception: If download process fails
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    url = normalizeUrl(url)

    ydl_opts={
        'outtmpl':output_path,'quiet':False,
        'format':'bestvideo[ext=mp4]+bestaudio[ext=m4a]/best', 
        'merge_output_format':'mp4',
        'postprocessors': [{'key': 'FFmpegVideoConvertor', 'preferedformat': 'mp4'},{'key': 'FFmpegFixupM4a'}]
    }

    try:
        print(f"Downloading the video from {url}")
        with YoutubeDL(ydl_opts) as ydl:
            ydl.download([url])

    except Exception as e:
        logGenerator(f"Error during download ({url}): {e}")

#Doc ffmpeg: https://ffmpeg.org/ffmpeg.html with subprocess import
def Trimming(input,start_time,end,output):
    """
    The video is trimmed to keep only the part corresponding to the label

    Args:
        inpute: link of the video file that we want to trimm
        start: The beginning of the sign in the video 
        end: The end of the sign in the video
        output: Where we want to save the video file 
    """
    videoTrimming = [
        'ffmpeg', '-i', input,
        '-ss', str(start),
        '-to', str(end),
        '-c:v', 'libx264',
        '-c:a', 'aac',
        '-strict', 'experimental',
        '-y', output
    ]
    print(f"Video trimming {input} from {start} to {end} in {output}")
    print("---------------------------------------------------------")
    subprocess.run(videoTrimming)


def organize(data,output_dir):
    """

    Here I organized the tasks in the following order:

    -I create the class folder with its ID if the folder does not already exist
    -I define the name of the final video file that I am about to download
    -I check if the video has already been downloaded and trimmed
    -I download the entire video (temporary file)
    -I check if the temporary file exists after the download attempt
    -I trim the video
    -I delete the temporary files (the .part file and the entire video file that had not yet been trimmed)

    Args:
        data: data from a json file
        output_dir: Path of the folder where we need to download/create class directories
    """
    for sample in data:
        label=sample['label']
        url=sample['url']
        start_time=sample['start_time']
        end_time=sample['end_time']
        signer_id=sample['signer_id']
        text=sample['text']

        #I create the class folder with its ID if the folder does not already exist
        class_dir = os.path.join(output_dir, f"class_{label}")
        os.makedirs(class_dir, exist_ok=True)
        #I define the name of the final video file that I am about to download
        video_name = f"signer{signer_id}_{text}.mp4"
        final_output_path = os.path.join(class_dir, video_name)
        temp_output_path = os.path.join(class_dir, f"{video_name}_temp.mp4")
        #If the final video already exist
        if os.path.exists(final_output_path):
            print(f"Teh video already exist : {final_output_path} so we are skippink this video")
            continue
        #Download the video if the temporary file does not exist
        if os.path.exists(temp_output_path):
            print(f"Temp file already here: {temp_output_path}, so just trimming here")
        else:
            print(f"Dowloading of the video from  {url}...")
            download_clip(url, temp_output_path)
        #Check if the temporary file exists after the download attempt
        if not os.path.exists(temp_output_path):
            logGenerator(f"We don't have the file after doawloading ??!: {temp_output_path}")
            continue
        #Trim the video
        print(f"Trimming : {temp_output_path}")
        Trimming(temp_output_path, start_time, end_time, final_output_path)
        #Delete the temporary file after trimming
        if os.path.exists(temp_output_path):
            os.remove(temp_output_path)
            print(f"Temp file have been deleted : {temp_output_path}")



if __name__ == "__main__":
    #Reading of the JSON files
    train_data = jsonReading('./MSASL/MSASL_train.json')
    val_data = jsonReading('./MSASL/MSASL_val.json')
    test_data = jsonReading('./MSASL/MSASL_test.json')


    #To focus on the 10 forst class <10
    trainDataFiltered = []
    for sample in train_data:
        if sample['label']<10:
            trainDataFiltered.append(sample)
    valDataFiltered = []
    for sample in val_data:
        if sample['label']<10:
            valDataFiltered.append(sample)
    testDataFiltered = []
    for sample in test_data:
        if sample['label']<10:
            testDataFiltered.append(sample)

    organize(trainDataFiltered,"ASL20/train")
    organize(valDataFiltered,"ASL20/val")
    organize(testDataFiltered,"ASL20/test")
