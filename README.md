# Ensemble-Based-Neural-Network-for-theClassification-of-MURA-Dataset
Musculoskeletal Radiographs (MURA) dataset, proposed by Stamford MachineLearning (ML) group, contains 40,561 images of bone X-rays from 14,863 studies.The X-ray images belong to seven body areas of upper extremity- Wrist, Elbow,Finger, Humerus, Forearm, Hand, and Shoulder. The data are classified manually byradiologists into two classes- normal or abnormal. These data samples are labeledusing majority vote by six board-certified Stanford radiologists. The majority votesof these radiologists’ labels are considered as gold standard. The presence of suchrich,complex and diverse labeled dataset inspires to build an accurate but simplermodel for bone anomaly detection. The model proposed by Stamford ML group isa 169 layer deep computationally complex Neural Network (NN), that requires aGraphical Processing Unit (GPU) for implementation. This leads to the necessityof smaller neural network based model that are executable on general purposecomputers. Moreover, the 169 layer deep model works well on par with the goldstandard except for the humerus radiographs, despite the presence of humerusdata labeled with high accuracy. Therefore, in this work we propose an ensembleof smaller neural networks and convolution neural network for highly accurateclassification of MURA study images of humerus.  We use Adaboost algorithmto train this model.  The performance of this model is evaluated using trainingerror, validation error, and Cohen’s kappa coefficients. The model is available in this repo.


Some of the images in the X-rays are given following:

![alt text](![alt text](http://url/to/img.png))
