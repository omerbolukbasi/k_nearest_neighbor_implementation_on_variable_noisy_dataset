## K NEAREST NEIGHBOR IMLEMENTATION ON VARIABLE NOISY DATASETS

### 1. Problem Description and Prior Knowledge
In the assigned problem, there are three data sets retrieved from a system in three different noise modes. Data is split into training and test sets which only the training sets have labels. The first problem is to determine the way for obtaining optimum k while applying KNN. The second is to determine the highest accuracies. 
The following information obtained from Lecture Notes of E Alpaydın 2010 Introduction to Machine Learning, can be used as prior knowledge for these problems:
* When k or h is small, single instances matter; bias is small, variance is large (undersmoothing): High complexity
* As k or h increases, we average over more instances and variance decreases but bias increases (oversmoothing): Low complexity
* Cross-validation is used to finetune k or h

### 2. Explore Data
In the training data there are three nodes of noise: 15,20,25. In each mode’s dataset there are three attributes(0,1,2) and one label (3):<br>
 ![image](https://user-images.githubusercontent.com/44832162/147391723-2900edb3-2f61-4dc5-a919-0bbb95408e90.png)<br>
Labels are integers between 1-8 which gives the information that this problem will be considered as a classification problem. <br>
 ![image](https://user-images.githubusercontent.com/44832162/147391725-f4d0a69e-3113-4c04-bc4f-f73c7d68e0c0.png)<br>
In order to see the distribution of the input attributes, the histogram of each variable needs to be checked.(number of bins=50)<br>
 ![image](https://user-images.githubusercontent.com/44832162/147391728-def3ec29-ffc6-486a-9f19-e69d4e7b74c3.png)<br>
From the above figure the followings are observed:
	Training variables fit with Gaussian Distribution. Attribute values of certain classes are distributed whether µ = 0 or µ = 1. Mean values are same in all noise modes. 
	Variances of attributes are observed same within same noise level. However, in different noise levels, variations are as follows:<br>
![image](https://user-images.githubusercontent.com/44832162/147391742-5ae35392-cb9f-431f-a956-cd83d0c6da93.png)<br>
	For each 3 attributes, the width of the overlappi3ng regions of µ = 0 classes and µ = 1 classes are same within same noise level. However, the classes that have high variance, also have wide overlapping regions. Then the noise levels can be estimated as follows:<br>
![image](https://user-images.githubusercontent.com/44832162/147391749-0af6537e-425d-4852-9666-5f1a3811c8a5.png)<br>
	Same observation can be made from the following 3D visualization. The overlapping areas are more significant in the data that has high noise.<br>
  ![image](https://user-images.githubusercontent.com/44832162/147391766-35158e98-28b2-4b35-8c71-f73ae4956b1c.png)<br>
![image](https://user-images.githubusercontent.com/44832162/147391770-5106b192-463e-491f-bbc1-835b62b86674.png)<br>
![image](https://user-images.githubusercontent.com/44832162/147391771-07b70f6e-2f04-417d-b200-dd2ecbd0c9c7.png)<br>

### 3. Determination of K
If the input dataset has high noise rate and high variance, the model should be as simple as possible. Because if the model is complex then the model may fit with the noise which results with low accuracy. From the prior knowledge, if k is small then the model is complex. And if k is high then the model is simple. So, it can be expected that the noisy model(in this case ds25) should have higher k(for simple model), and the model that has low noise(in this case ds15) should have lower k(for complex model) in order to achieve best accuracy. 
As stated in the prior knowledge, Cross Validation is used to determine best k’s for different noise levels.<br>
![image](https://user-images.githubusercontent.com/44832162/147391779-308c0bff-a169-4cdd-b039-6995b8828ed0.png)<br>
 
Maximum accuracy Dataset15: 0.99921875 at K = 9<br>
Maximum accuracy Dataset20: 0.98359375 at K = 29<br>
Maximum accuracy Dataset25: 0.94124999 at K = 249<br>

### 4. Determination of Hyper Parameters

#### 4.1 Distance Metric
The following distance metrics are tested with corresponding parameter values. Minkowski p=1(manhattan),p=2(euclidean),p=3,5,10,100,1000. The highest accuracy obtained with Minkowski p=3.

#### 4.2 Weights
Weights metric is tested with the following options.<br>
Uniform : uniform weights. All points in each neighborhood are weighted equally.<br>
Distance : weight points by the inverse of their distance. in this case, closer neighbors of a query point will have a greater influence than neighbors which are further away.<br>
The highest performance is obtained with uniform weight.

### 5. Result
The result of Cross Validation on training sets show that low noise data set(ds15) performs better with lower k, while high noise data set(ds25) performs better with higher k. This result validates the previous estimation. High noise data has the best performance with a simpler model(high k value) and vise-versa. Based on these k and other hyper parameter values, the labels of given test sets are predicted and saved in proper files.

