# Age Prediction---Classic ML and Deep Learning Approaches

We implement HOG to extract the face region of input images and feed into a MobileNetV2 to produce a result of 10-class classification.
The MobileNetV2 is pre-trained on ImageNet. We add a fully connect layer to it and train all the weights on out dataset.Each prediction
 of the image is in a 10-year range (for example, 21 to 30). The final age is given by the class of image and a function based on the 
 probability of this class.


### Dependencies
python 3.6.7
numpy 1.13.3
dlib 19.4
keras 2.2.4
opencv 3.3.0
imutils 0.4.6

## Running the tests

Please run the main.py in either command line or any python development environment. Follow the instrustion print on screen
to input the image directory to get age prediction. 


## References

[1]’Imdb-Wiki dataset’, https://data.vision.ee.ethz.ch/cvl/rrothe/imdb-wiki/
[2] ‘UTK face dataset’, https://susanqq.github.io/UTKFace/
[3] ‘MobileNetV2: Inverted Residuals and Linear Bottlenecks’, https://arxiv.org/abs/1801.04381


