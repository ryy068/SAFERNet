 # SAFERNet: Advancing Vision Transformer with Shallow Global Representations coupling Fine-grained Local Features for Student Facial Expression Recognition
 
 <img src="https://github.com/ryy068/SAFERNet/blob/main/models/SAFERNet.png" width="710px">

 
The overall architecture of our SAFERNet. (a) SAFERNet starts with a stem CNN module, followed by four stages for constructing the hierarchical feature maps, next to an adaptive pooling layer, and finally, the head module for classification. (b) The SAFER Block contains two new modules, i.e., SMHSA and DAF. SMHSA module consists of three processes, i.e., Super Token Sampling (STS), Multi-Head Self-Attention (MHSA), and Token Upsampling (TU). DAF fuses spatial and channel dimensional dependencies.

 ## Installation

This project is based on MMClassification, please refer to their repos for installation.

 ## Datasets

 ### Student FER Dataset - SAFERD
We constructed a student FER dataset SAFERD in a classroom setting. The source of the dataset is mainly video recordings of classroom teaching scenes obtained by setting up capture devices in classroom scenarios. In addition, the dataset also contains some videos of film dramas in the classroom setting. By extracting the key frames of the videos and extracting the student faces, a total of 1,401 student face images in png format were obtained. After designing and investigating the semantic labels for the images, the students' faces were finally classified into five categories, which accounted for the following proportions of the dataset: calm (24.3\%), confused (12.5\%), jolly (35.4\%), sleepy (7.4\%), and surprised (20.4\%). Partial samples of the five emotion categories are shown below.

<img src="https://github.com/ryy068/SAFERNet/blob/main/data/SAFERD.png" width="510px">

 ### Other Datasets
We also evaluated our proposed method on other FER datasets. We use the face alignement codes in [face.evl](https://github.com/ZhaoJ9014/face.evoLVe/#Face-Alignment) to align face images. Partial examples of aligned FER datasets are shown below.

<img src="https://github.com/ryy068/SAFERNet/blob/main/data/aligned_FER.png" width="710px">

 ### Terms & Conditions
The dataset is available for non-commercial research purposes only.

You agree not to reproduce, duplicate, copy, sell, trade, resell or exploit for any commercial purposes, any portion of the images and any portion of derived data.

 ### How to get the Dataset
This database is publicly available. It is free for professors and researcher scientists affiliated to a University. Permission to use but not reproduce or distribute our database is granted to all researchers. Send an e-mail to Yan Rong (yrong854@connect.hkust-gz.edu.cn) or Xinlei Li (lixinlei@suibe.edu.cn) to get relevant datasets.



 
