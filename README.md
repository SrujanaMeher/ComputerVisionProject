# ComputerVisionProject
This is my computer vision project, Passenger Screening for Threat Detection

Object detection is one of the areas of computer vision that has grown as a wide perspective.
Object detection has shown great results in various fields such as medical fields in tumor detection, cancer detection, social networks, face detection. 
Deep learning aligned with object detection can detect almost all objects in a video or an image. 
Images detected has consistency in relative pixel densities that can be modelled by convolution neural networks. 
Multiple object detection and localization using algorithms can be used for image processing.


## Problem solution

Object detection methodology for detecting threats during passenger screening at airports.
This idea is an inspiration from a research wherein, ‘The Department of Homeland Security (DHS)’ has produced several false alarms because of the less accuracy in the scan algorithm. 

So using the feature detection technique, segmentation and classification methods, to identify the threats at airports during body scans aiming at improving the accuracy to avoid false alarms and finding the threats thereby helping the DHS in providing high security to the passengers at airports.

## About the dataset

The dataset consists of 133GB of body scan images in a 3D perspective using HD-AIT systems – High Definition-Advanced Imaging Technology. 

The dataset has images of 976 bodies, each body partitioned into zones from zone 1 to zone 17 corresponding the TSA threat zones. 

Also, it comprises of different images that comprises images of volunteers wearing different clothing types, different body mass indices, different number of threats, different genders and different types of threats such as knifes, guns, blades, cutters, or any sharp metal object.
There were 4 different binary formats of scanner files to identify the presence or absence of threats by body zones, such as: .ahi, .aps, .a3d, .a3daps.
I am using .aps – Projected Image Angle Sequence File as its 10.3MB per file, easy to compute comparatively. This algorithm computes 3D images for 90-degree segments of data that are equally spaced around the region scanned. 
The body zones are cropped from the general body region and the crops are generated by using trail and error method. This ensured that even the threats that are not visible to the human eye can be detected with more precision. Also this gives a concatenated snapshots of four images of the same zone for more approximate threat detection.
These generated .aps files are sent to the preprocess before building the models   and obtaining weights.

![Image description](body_zones.jpg)


To pre-process the data i.e., .aps files we create dictionary of the data. So all the files must be in the .aps dictionary.
The csv files for the training data or test data must be in the current directory.
The other pre-processing steps include converting to grayscale, improving contrast and also normalizing the images.


The segmented body scans are processed and a full projection slices of every .aps file is made. One of the sample is :
### With Threat                                             

![Image description](threat.png)

### Without Threat 
![Image description](nothreat.png)
