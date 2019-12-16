import cv2
from imageai.Detection import ObjectDetection
from pathlib import Path

"""
Part of an ongoing "Smart Webcam" project.
Cuts out any parts of a video that does not have any humans visible.

In the same folder as this file create 3 sub-folders:
    -> humans   # Stores videos of detected humans
    -> frames   # Stores frames extracted from video
    -> models   # Stores pre-trained models. All should be set-up already, though
    -> temp     # Temporary folder for analysed images.
    
After running the code, please empty the frames and temp folder. This shall be automated in the future.
By 1llusion
"""
class VideoExtraction:
    def start(self, video):
        fps = self.extract(video)
        self.identifyHuman("frames", fps=fps)

    """
    Extracts frames from videos
    """
    @staticmethod
    def extract(video):
        vidcap = cv2.VideoCapture(video)
        fps = vidcap.get(cv2.CAP_PROP_FPS)

        success = True  # Frame read
        count = 0   # Number of frames read

        while success:
            print("Wrote frame ", count)

            # Reading and writing frame
            success, image = vidcap.read()
            if success:
                cv2.imwrite("frames/%d.jpg" % count, image)
            count += 1

        return fps
    """
    Detect all images with humans present and move them from inputFolder to outputFolder
    
    inputFolder = Folder where to search for images
    outputFolder = Where to store detected humans
    probability = What % of confidence should be used for detection?
    modelPath = Path to a pre-trained model
    
    Credit: https://stackabuse.com/object-detection-with-imageai-in-python/
    """
    def identifyHuman(self, inputFolder, outputFolder="./humans/", tempFolder="./temp/", fps=24, probability=30, modelPath="./models/yolo.h5"):
        path = Path(inputFolder)   # To keep path structures tidy

        detector = ObjectDetection()
        detector.setModelTypeAsYOLOv3()

        detector.setModelPath(modelPath)
        detector.loadModel()
        custom = detector.CustomObjects(person=True,
                                        giraffe=True)  # Finding out if humans are in the image. And giraffes, because that would be hilarious.

        files = path.glob("**/*")
        # List of images with a human that will be turned into a video
        human_dict = {}

        for file in [x for x in files if x.is_file()]:
            temp_file = Path(tempFolder + file.name)

            detections = detector.detectCustomObjectsFromImage(custom_objects=custom,
                                                               input_image=file,
                                                               #output_type="array",
                                                               output_image_path=str(temp_file), # Keeping the same filename
                                                               minimum_percentage_probability=probability)
            # Was a human detected?
            if any(d['name'] == "person" for d in detections):
                # Copying the image into the desired human folder
                human_dict[int(file.stem)] = str(file)   # Using the filename as indx (should be integer)
                print("Human found in image:", file.name)

        #  Checking if any humans were found
        if(len(human_dict)):
            print("Found humans! Generating video.")
            # Since the image detection seems to be multi-threaded, the images need to be sorted
            human_list = list(human_dict)
            human_list.sort()
            self.video(human_dict, human_list, outputFolder + "test.avi")

    """
        human_dict = Actual filenames
        human_list = Ordered indexes for human_dics
        outputFile = name of file
    """
    def video(self, human_dict, human_list, outputFile, name="test"):
        # Credit https://theailearner.com/2018/10/15/creating-video-from-images-using-opencv-python/
        human_img = []  # Storing processed human images
        for index in human_list:
            img = cv2.imread(human_dict[index])
            height, width, layers = img.shape
            size = (width, height)
            human_img.append(img)

        out = cv2.VideoWriter(outputFile, cv2.VideoWriter_fourcc(*'DIVX'), 15, size)
        for i in range(len(human_img)):
            out.write(human_img[i])
        out.release()

"""
Example of how to run the class. Just enter a path of a video to be processed and go for a coffee.
"""
if __name__ == "__main__":
    vid = VideoExtraction()
    vid.start("Enter video link here")
    print("Done")
