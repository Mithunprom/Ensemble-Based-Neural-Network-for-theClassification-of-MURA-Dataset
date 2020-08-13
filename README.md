<h1>Authors</h1>
<ul><li>Mithun Ghosh</li>  <li>Md Sahil Hassan</li></ul>


This repository contains a Neural Network and Convolution Neural Network based Ensemble model for classifying MURA humerus dataset.

Libraries used-
<ul>
<li>Keras</li>
<li>Tensorflow</li>
<li>Numpy</li>
  </ul>
Dataset is Avaiable in <a href="mura_humerus"> 'here'</a>
Instructions for Running
<ol><li> Make sure you have the necessary libaries installed.</li>
<li> Run the <a href="main.py"> 'main.py'</a> file as-
''' python3 main.py '''. As for the CNN, the juperter file name <a href="Convol_MURA.ipynb"> 'Convol_MURA.ipynb'</a> has been provided. </li>

# Ensemble-Based-Neural-Network-for-theClassification-of-MURA-Dataset
Musculoskeletal Radiographs (MURA) dataset, proposed by Stamford MachineLearning (ML) group, contains 40,561 images of bone X-rays from 14,863 studies.The X-ray images belong to seven body areas of upper extremity- Wrist, Elbow,Finger, Humerus, Forearm, Hand, and Shoulder. The data are classified manually byradiologists into two classes- normal or abnormal. These data samples are labeledusing majority vote by six board-certified Stanford radiologists. The majority votesof these radiologists’ labels are considered as gold standard. The presence of suchrich,complex and diverse labeled dataset inspires to build an accurate but simplermodel for bone anomaly detection. The model proposed by Stamford ML group isa 169 layer deep computationally complex Neural Network (NN), that requires aGraphical Processing Unit (GPU) for implementation. This leads to the necessityof smaller neural network based model that are executable on general purposecomputers. Moreover, the 169 layer deep model works well on par with the goldstandard except for the humerus radiographs, despite the presence of humerusdata labeled with high accuracy. Therefore, in this work we propose an ensembleof smaller neural networks and convolution neural network for highly accurateclassification of MURA study images of humerus.  We use Adaboost algorithmto train this model.  The performance of this model is evaluated using trainingerror, validation error, and Cohen’s kappa coefficients. The model is available in this repo and the details of the project is breifly described <a href="Paper.pdf"> here</a>


Some of the images in the X-rays are given following:

![alt text](xray.JPG?raw=true "Figure 01: X-ray image data samples")

Figure 1: MURA dataset contains 14863 images of the radiography of musculoskeletal studies of theupper extremity. In each of the study multiple views are manually labeled by radiologists. Right sideof the above Figure explains some normally labeled images of Elbow and Wrist, respectively whereinleft side describes some abnormal images from the Humerus and Shoulder, respectively. 

There are some discriptive statistics based on the dataset are given following. 
![alt text](std.JPG?raw=true "Figure 01: X-ray image data samples")

![alt text](stdhum.JPG?raw=true "Figure 01: X-ray image data samples") 

Figure  2:  Upper:  Statistics  of  the  data  in  each  of  the  seven  categories  of  the  studies. Figure  2:  Lower:  Statistics  of  the  data  in  each  of  the  seven  categories  of  the  studies.   In  trainset XR_WRIST has maximum number of patients, followed by XR_FINGER, XR_HUMERUS,XR_SHOULDER, XR_HAND, XR_ELBOW and XR_FOREARM. X_FOREARM with 606 patientshas  got  the  least  number.   Similar  pattern  can  be  seen  in  valid  set,  XR_WRIST  has  the  maxi-mum, followed by XR_FINGER, XR_SHOULDER, XR_HUMEROUS, XR_HAND, XR_ELBOW,XR_FOREARM. Here XR_FINGER defines radiographs of Finger upper extremity.  Right: DataStatistics for Humerus Data

<h2> Image Classification Task</h2>
<p1>The task in MURA dataset is to find the binary class of{0,1}.  Each study contains one or moreviews of images and the expected output is then denoted as 0 or 1. We denote 0 as normal and 1 asabnormal. A brief summary of the study data is given below:In the official MURA dataset website (https://stanfordmlgroup.github.io/competitions/mura/) we can see the performance of various authors model in different categories. Some of thesemodel perform well for different categories.  But For upper extremity categories:  Humerus andFinger, almost all of the models perform worst. This may be due to the fact that, the images in thesecategories are not so clear. Also, the number of samples are not sufficiently high in these categories.We can look that in right side of Fig. 1, these images are not so clear to be easily classified by themodels. Thus, for this reason, we concentrate our model to the Humerus cases. In left side of Fig. 2,we can also see that number of studied patients are very few for both training and test dataset inHumerus study. </p1> 

<h2>Modeling</h2>
<p1>In order to investigate the types of abnormalities present in the dataset, we reviewed the radiologistreports to manually label 100 abnormal studies with the abnormality finding: 53 studies were labeledwith fractures, 48 with hardware, 35 with degenerative joint diseases, and 29 with other abnormalitiesincluding lesions and subluxations.
The proposed ensemble model is made up of smaller deep neural networks. These neural networks aretrained on the train data from humerus X-ray images of MURA using the Adaboost algorithm (Freundand Schapire [1999]). Section 3.1 explains the structures and training of each of these smaller neuralnetworks. Section 3.2 explains the overall ensemble training and prediction methods.
  </p1>
<p2>


![alt text](NN_architecture.JPG?raw=true "Figure 01: X-ray image data samples") 

Figure 3: Structure of the small neural network used as weak classifier in the ensemble model.

![alt text](rgbo1.JPG?raw=true "Figure 01: X-ray image data samples") 

Figure 4: Structure of the small convolution neural network used as weak classifier in the ensemblemodel.

The structural details of each smaller deep neural networks is shown in Fig. 3. Each neural networkhas 10000 nodes in input layers and one node in output layer. There are 10 hidden layers betweeninput and output layer. As the network grows deeper, number of nodes into each new layer reduces.These neural networks are modeled using models available in Keras library.Now, The structural details of each of CNN is shown in Fig. 4. Each of the CNN consists of kernelwindow size of 4x4 with stride one and Maxpooling of 2x2. This auto-encoder network compress thesize of the image to from 100x100 to 9x9 with minimum reconstruction error. After that, the layersare flatten and fully connected to dense layer with 500 nodes and rectified linear activation functionand then another dense layer with 20 nodes and finally the two output layer.

</p2>

<h2> Model Performance</h2>

<table style="width:100%">
  <tr>
    <th>Dataset</th> <th>Proposed Model(NN)(Epoch = 20) </th>
    </th> <th>Proposed Model(NN)(Epoch = 30) </th>
    </th> <th>Proposed Model(CNN)(Epoch = 20) </th>
    </th> <th>Proposed Model(CNN)(Epoch = 30) </th>
  </tr>
  <tr>
  <td>Validation</td><td>.0250</td><td> .050</td><td> 0.119</td><td> 0.123</td>
  </tr>
  <tr>
    <td>Train </td><td> 0.319</td><td> 0.441</td><td>0.4590</td><td> 0.51</td>
  </tr>
</table>

In order to evaluate the performance of our proposed model, we use training error, validation error,and Cohen’s kappa coefficient (κ) for both training and validation data. Table 1 presents theκvaluesof classifications made by the Radiologist with best performance, model by Stamford ML group, andmodel proposed in this work. The results contain model performance for different training epochs.All the mentionedκvalues in this table is calculated by comparing with the gold standard.Table 1 results show that, for classification of training data, the model performs somewhat reliable.With increasing training epochs, theκvalue increases, indicating increasing reliability of the model.For classification on validation data, theκvalue is quite lower compared to the radiologist andStamford ML group model. However, with increasing training epochs,κof our model for validationdata classification increases.  This indicates, with larger training epochs, our model can be madereliable enough to compete with the other two models. However, increasing the number of epochs ncreases training time significantly, which is why we could not present results for higher epochs. Wesuggest using High Performance Computing (HPC) servers to train this model for higher epochs.Along withκvalues, we also measured the training and validation error of our model. For trainingepoch 20, the model yields training and validation error of 34% and 48% respectively. For trainingepoch 30, training and validation errors are 27% and 47%. Finally, for 40 epochs, training error is 27%and validation error is 44%. The study of training and validation errors show that increasing trainingepochs result in decreasing training and validation errors.  This again brings us to the conclusionof training the ensemble model for more epochs to obtain better performance. Moreover, we mayincrease the number of smaller neural networks to get more reliable classification performance. Forthe CNN with five classifiers, we get 25% training and 0.43% testing errors.As, we can see from the table that CNN performs better with comparatively lower epoch. For theCNN, with the increase of the number of classifiers we believe that we will get a betterκvalue andalso training and testing error will be reduced. As mentioned above, due to the lack of resource, wecould not perform much larger number of classifiers. We fixed epoch size to 10 in the CNN model.

<h2>Conclusion</h2>
Early stage detection of anomaly in radiographs is crucial for the patient.  We can notice from the results of Rajpurkar et al. [2018] that even experienced radiologists may sometime misclassify someanomalies. Moreover, human classification is costly, more time consuming and requires more effort.These reasons have made machine learning based classifier model a reliable alternative. Althoughthere are several established models that perform relatively good on MURA dataset, for some upperextremities they are not reliable enough. To add on top of that, these models require compute intensivecomplex Neural Network based models that are difficult to train. We attempt to address these issues in our model by building an ensemble based neural network model, that can perform well on these mages.  The obtained results suggest that this model works with some degree of reliability.  Wefurther notice an increasing trend in model reliability with the increasing number of training epochs.Based on these results we strongly believe that, with the increase in computational resources, ourmodel can be one of the most reliable candidates in MURA dataset classification.
