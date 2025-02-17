# CCTV - Tracking

The main.py consist of the deep learning code required for the person re-identifcation in a security footage.

The data set I have collected is the famous and most popularly used in such scenarios of person re-identification is "Market 1501" where there are multiple cameras covering a single area and looks in different angles helping it understand the features underlying in the images provided .
Thus data need to divided into two main slots those are training ,testing where around 12937 images ,19733 images respectively are given to the model for training and testing.
And there are 751 classifications for the dataset.

For feature Extraction I have used ResNet-50 and model consist of Convulution Layer(Conv2D),Batch Normalization,ReLU,MaxPool.

The learning rate is optimum of '3e-4' which avoided the overfitting of data and underfitting of convergence.

While training the main part is the 'epoch' where it tells how many times we train the data,When given a small number 5 it took around 1 hour and each time accuracy varied.

The range of accuracy is 83.57% - 87.26% ,I tried to increase the epoch number for better accuracy but my system overheated so I stopped the execution of code.



All the relevant information is provided by the Source:'https://asp-eurasipjournals.springeropen.com/articles/10.1186/s13634-024-01139-x'.

Thank You for the oppurtunity for showcasing my skill.
