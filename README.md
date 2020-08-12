# Ensemble-Based-Neural-Network-for-theClassification-of-MURA-Dataset
Musculoskeletal Radiographs (MURA) dataset, proposed by Stamford MachineLearning (ML) group, contains 40,561 images of bone X-rays from 14,863 studies.The X-ray images belong to seven body areas of upper extremity- Wrist, Elbow,Finger, Humerus, Forearm, Hand, and Shoulder. The data are classified manually byradiologists into two classes- normal or abnormal. These data samples are labeledusing majority vote by six board-certified Stanford radiologists. The majority votesof these radiologists’ labels are considered as gold standard. The presence of suchrich,complex and diverse labeled dataset inspires to build an accurate but simplermodel for bone anomaly detection. The model proposed by Stamford ML group isa 169 layer deep computationally complex Neural Network (NN), that requires aGraphical Processing Unit (GPU) for implementation. This leads to the necessityof smaller neural network based model that are executable on general purposecomputers. Moreover, the 169 layer deep model works well on par with the goldstandard except for the humerus radiographs, despite the presence of humerusdata labeled with high accuracy. Therefore, in this work we propose an ensembleof smaller neural networks and convolution neural network for highly accurateclassification of MURA study images of humerus.  We use Adaboost algorithmto train this model.  The performance of this model is evaluated using trainingerror, validation error, and Cohen’s kappa coefficients. The model is available in this repo.


Some of the images in the X-rays are given following:

![alt text](xray.JPG?raw=true "Figure 01: X-ray image data samples")

Figure 1: MURA dataset contains 14863 images of the radiography of musculoskeletal studies of theupper extremity. In each of the study multiple views are manually labeled by radiologists. Right sideof the above Figure explains some normally labeled images of Elbow and Wrist, respectively whereinleft side describes some abnormal images from the Humerus and Shoulder, respectively. 

There are some discriptive statistics based on the dataset are given following. 
Figure  2:  Left:  Statistics  of  the  data  in  each  of  the  seven  categories  of  the  studies.

![alt text](std.JPG?raw=true "Figure 01: X-ray image data samples")

![alt text](stdhum.JPG?raw=true "Figure 01: X-ray image data samples") 


Figure  2:  Left:  Statistics  of  the  data  in  each  of  the  seven  categories  of  the  studies.   In  trainset XR_WRIST has maximum number of patients, followed by XR_FINGER, XR_HUMERUS,XR_SHOULDER, XR_HAND, XR_ELBOW and XR_FOREARM. X_FOREARM with 606 patientshas  got  the  least  number.   Similar  pattern  can  be  seen  in  valid  set,  XR_WRIST  has  the  maxi-mum, followed by XR_FINGER, XR_SHOULDER, XR_HUMEROUS, XR_HAND, XR_ELBOW,XR_FOREARM. Here XR_FINGER defines radiographs of Finger upper extremity.  Right: DataStatistics for Humerus Data

<h2> Image Classification Task</h2>
<p1>The task in MURA dataset is to find the binary class of{0,1}.  Each study contains one or moreviews of images and the expected output is then denoted as 0 or 1. We denote 0 as normal and 1 asabnormal. A brief summary of the study data is given below:In the official MURA dataset website (https://stanfordmlgroup.github.io/competitions/mura/) we can see the performance of various authors model in different categories. Some of thesemodel perform well for different categories.  But For upper extremity categories:  Humerus andFinger, almost all of the models perform worst. This may be due to the fact that, the images in thesecategories are not so clear. Also, the number of samples are not sufficiently high in these categories.We can look that in right side of Fig. 1, these images are not so clear to be easily classified by themodels. Thus, for this reason, we concentrate our model to the Humerus cases. In left side of Fig. 2,we can also see that number of studied patients are very few for both training and test dataset inHumerus study. </p1> 

<h2>Modeling</h2>
<p1>In order to investigate the types of abnormalities present in the dataset, we reviewed the radiologistreports to manually label 100 abnormal studies with the abnormality finding: 53 studies were labeledwith fractures, 48 with hardware, 35 with degenerative joint diseases, and 29 with other abnormalitiesincluding lesions and subluxations.
The proposed ensemble model is made up of smaller deep neural networks. These neural networks aretrained on the train data from humerus X-ray images of MURA using the Adaboost algorithm (Freundand Schapire [1999]). Section 3.1 explains the structures and training of each of these smaller neuralnetworks. Section 3.2 explains the overall ensemble training and prediction methods.
  </p1>
<p2>


![alt text](NN_architecture.JPG?raw=true "Figure 01: X-ray image data samples") 
Figure 3: Structure of the small neural network used as weak classifier in the ensemble model.

![alt text](NN_architecture.JPG?raw=true "Figure 01: X-ray image data samples") 
Figure 4: Structure of the small convolution neural network used as weak classifier in the ensemblemodel.

The structural details of each smaller deep neural networks is shown in Fig. 3. Each neural networkhas 10000 nodes in input layers and one node in output layer. There are 10 hidden layers betweeninput and output layer. As the network grows deeper, number of nodes into each new layer reduces.These neural networks are modeled using models available in Keras library.Now, The structural details of each of CNN is shown in Fig. 4. Each of the CNN consists of kernelwindow size of 4x4 with stride one and Maxpooling of 2x2. This auto-encoder network compress thesize of the image to from 100x100 to 9x9 with minimum reconstruction error. After that, the layersare flatten and fully connected to dense layer with 500 nodes and rectified linear activation functionand then another dense layer with 20 nodes and finally the two output layer.

</p2>

