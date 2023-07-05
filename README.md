# TexBig: Object Detection and Robust Learning

</br>

## Introduction

</br>

This project revolves around developing an efficient object detection model under resource constraints commonly faced in real-world scenarios. The objective is to construct a comprehensive solution for a complex task using the TexBig dataset. The project entails selecting and adapting a suitable model from the PyTorch model zoo, conducting an ablation study to evaluate the impact of modifications on model performance, and iteratively improving the model based on learnings from each iteration. Techniques such as data augmentation, regularization, and hyperparameter tuning will be utilized to enhance the model's robustness and overall performance.

</br>

## Table Of Contents

</br>

- [TexBig: Object Detection and Robust Learning](#texbig-object-detection-and-robust-learning)
    - [Introduction](#introduction)
    - [Table Of Contents](#table-of-contents)
    - [1. Problem Statement](#1-problem-statement)
    - [2. Proposed Solutions](#2-proposed-solutions)
        - [2.1 Faster RCNN](#21-faster-rcnn)
            - [2.1.1 Objection Localization](#211-objection-localization)
            - [2.1.2 Robust Handling of Regions \& Feature Extraction](#212-robust-handling-of-regions--feature-extraction)
            - [2.1.3 Transfer Learning](#213-transfer-learning)
                - [FASTERRCNN\_RESNET50\_FPN](#fasterrcnn_resnet50_fpn)
                - [FASTERRCNN\_RESNET50\_FPN\_V2](#fasterrcnn_resnet50_fpn_v2)
        - [2.2 Hyperparameters](#22-hyperparameters)
            - [2.2.1 Optimiser](#221-optimiser)
            - [2.2.2 Learning Rate](#222-learning-rate)
            - [2.2.3 Learning Rate Scheduler](#223-learning-rate-scheduler)
            - [2.2.4 Weight Decay \& Momentum](#224-weight-decay--momentum)
            - [2.2.5 Image Augmentation](#225-image-augmentation)
            - [2.2.6 Batch Size](#226-batch-size)
    - [3. Dataset Description](#3-dataset-description)
    - [4. Installation](#4-installation)
    - [5. Usage](#5-usage)
        - [5.1 Folder Structure](#51-folder-structure)
        - [5.2 Training \& Validation](#52-training--validation)
        - [5.3 Inferences](#53-inferences)
    - [6. Experimental Results](#6-experimental-results)
        - [6.1 Results](#61-results)
        - [6.2 Notebooks \& Epoch Selection](#62-notebooks--epoch-selection)
    - [7. Discussion / Analysis](#7-discussion--analysis)
        - [7.1 Results Discussion](#71-results-discussion)
        - [7.2 Inference Analysis](#72-inference-analysis)
    - [8. Outlook / Future Work](#8-outlook--future-work)
    - [9. References](#9-references)


</br>

## 1. Problem Statement

</br>

For this project we’re developing deep learning models that are specifically designed for a dataset called TexBig. TexBig is an instance segmentation dataset proposed by this paper [1]. There are over 52,000 instances, each belonging to one of the 19 distinct classes. Expert annotators manually annotated these instances, employing instance segmentation through bounding boxes and polygons/masks.

The task is to develop a deep learning model for object detection using the TexBiG dataset, with a specific focus on historical document layout analysis. My goal is to create a model that can accurately predict instance segmentation annotations, such as bounding boxes and polygons/masks, for the different classes within the dataset.

The challenges in this project arise from the complexity and diversity of the historical documents in the TexBiG dataset. These documents may exhibit variations in font styles, sizes, layouts, and formats, making the detection and classification of instances more challenging. Additionally, the presence of noise, occlusions, and variations in document quality further complicates the task.

Overall, this project aims to leverage machine learning techniques to address the object detection problem within historical documents using the TexBiG dataset. By achieving high accuracy and robustness in detecting and classifying instances, the developed models will provide valuable tools for document analysis and historical research.

</br>

## 2. Proposed Solutions

</br>

### 2.1 Faster RCNN

</br>

Faster R-CNN is a highly effective object detection model due to its excellent balance between accuracy and speed. By introducing the Region Proposal Network (RPN), it eliminates the need for computationally expensive methods like sliding windows, resulting in faster inference times. Additionally, the model's two-stage architecture allows for improved localization accuracy, making it a reliable choice for tasks requiring precise object detection. There are a couple of points that Faster RCNN [2] might be promising for the task at hand such as:

![Faster RCNN Model Architecture](images/FasterRCNNModelArchitecture.png)

Figure 1. Architecture of an R-CNN model, visualizing the pipeline [2]

</br>

#### 2.1.1 Objection Localization

</br>

- Faster R-CNN excels in accurate object localization, which is essential for document layout analysis. The TexBiG dataset requires precise identification and segmentation of document elements. Faster R-CNN's region proposal network (RPN) generates potential object regions, and subsequent stages refine and classify these regions with high localization accuracy. By leveraging Faster R-CNN's object localization capabilities, we can accurately identify and delineate text, images, headings, tables, and other document components, contributing to effective document layout analysis.

</br>

#### 2.1.2 Robust Handling of Regions & Feature Extraction

</br>

Faster R-CNN utilizes a powerful feature extractor, such as ResNet 50, as its backbone network. This allows it to capture visual features that are crucial for accurately detecting and analyzing document components. The robust handling of regions and feature extraction capabilities of Faster R-CNN ensure the model's effectiveness in handling the diverse layouts and complex structures within the TexBiG dataset.

</br>

#### 2.1.3 Transfer Learning

</br>

Transfer learning is another advantage of Faster R-CNN, as it allows the model to leverage pre-trained models on large-scale datasets such as COCO. By fine-tuning a pre-trained Faster R-CNN model on the TexBiG dataset, we can benefit from the learned visual representations, enabling faster convergence and potentially enhancing the model's ability to detect and analyze document elements accurately, even with limited labeled data.
 
 </br>

##### FASTERRCNN_RESNET50_FPN
 
 </br>

This [pretrained model](https://pytorch.org/vision/main/models/generated/torchvision.models.detection.fasterrcnn_resnet50_fpn.html) is available in the pytorch zoo and was based off this paper [2]. The weights that will be used is going to be the ones pretrained on the COCO dataset. This PyTorch model that combines Faster R-CNN, ResNet-50, and Feature Pyramid Network (FPN) for object detection tasks. It utilizes the powerful ResNet-50 architecture for feature extraction and incorporates FPN to handle objects of different scales effectively. The model generates region proposals using a region proposal network (RPN) and classifies and refines the proposed regions with a detection network as mentioned previously.
 
 </br>

##### FASTERRCNN_RESNET50_FPN_V2

</br>

This [pretrained model](https://pytorch.org/vision/main/models/generated/torchvision.models.detection.fasterrcnn_resnet50_fpn_v2.html#torchvision.models.detection.fasterrcnn_resnet50_fpn_v2) is available in the pytorch zoo and was based off a paper proposed by Facebook AI Research [3]. The architecture is nearly identical to the first version; however, the training techniques were different. In the second version the main difference is the inclusion of a Vision Transformer (ViT). This Faster RCNN model was trained using ViT as the backbone. However, the usage of the model does not include ViT and rather only uses its pretrained weights for the RPN. The weights referring to the ones pretrained using ViT, and the weights_backbone utilizing the normal ResNet50.

The following code from Pytorch's backend better represents this FPN selection.: 

`weights = FasterRCNN_ResNet50_FPN_V2_Weights.verify(weights)`

`weights_backbone = ResNet50_Weights.verify(weights_backbone)`

 </br>

### 2.2 Hyperparameters

</br>

For all the iterations and different trials the hyper parameters were fixed. For the optimiser Stochastic Gradient Descent (SGD) was selected. A linear learning rate scheduler was also included for all trials. Image augmentation was only considered for the last iteration since it almost trippled the training duration for a single epoch. 

</br>

#### 2.2.1 Optimiser

</br>

SGD is well-suited for Faster R-CNN due to its simplicity and effectiveness in optimizing deep learning models. It offers fast convergence and the ability to handle large-scale datasets efficiently. Additionally, SGD provides a stochastic nature that enables exploration of diverse training samples, facilitating better generalization and robustness in object detection tasks. Its widespread adoption and successful application in previous works further validate its suitability for Faster R-CNN. [4][5]

</br>

#### 2.2.2 Learning Rate

</br>

The selection of learning rates in Faster R-CNN is crucial for achieving optimal training performance and convergence. In practice, a set range is typically explored to find the most suitable value for a specific dataset and model architecture. Previous works [2][6] have shown that range between 0.001 to 0.1 yield favorable results in Faster R-CNN. 

</br>

#### 2.2.3 Learning Rate Scheduler

</br>

Linear learning rate scheduling has been shown to be effective in improving training dynamics and convergence in various deep learning applications. In the context of object detection, the adoption of linear learning rate schedules has demonstrated promising results. For instance, the paper by Chen et al. [7] highlights the benefits of a linear warm-up strategy, where the learning rate is gradually increased from a small initial value. This approach helps stabilize training and allows the model to learn more efficiently. Similarly, the study by He et al. [8] discusses the use of a linear learning rate schedule in their experiments with object detection networks, emphasizing its contribution to faster convergence and improved performance. These findings reinforce the utility of linear learning rate schedules as a valuable technique for optimizing the training of object detection models.

This learning rate scheduler is builtin within the engine file and is already run without explicity mentioning it.

</br>

#### 2.2.4 Weight Decay & Momentum

</br>

The choice of weight decay hyperparameter is important as it balances the trade-off between fitting the training data well and avoiding overfitting. In this case, a weight decay value of 0.0005 was chosen, which has been found to be effective in various studies [4][8]. It helps control the magnitude of the weights and prevents them from growing too large during training.

A momentum value of 0.9 was chosen, which is a commonly used value in many deep learning applications [4][8]. Higher momentum values allow the optimizer to accumulate more information from past gradients, enabling faster convergence and better handling of noisy or sparse gradients.

</br>

#### 2.2.5 Image Augmentation

</br>

Random Image Distortion is a data augmentation technique used during training to introduce diverse variations in input images. It includes several types of distortions such as contrast, saturation, hue, and brightness, which are applied randomly within specified ranges. These distortions help improve the model's robustness and generalization by mimicking real-world image variations. By controlling the probability (p) of applying the distortions, it allows for customizable augmentation levels. For further details on the implementation and benefits of RandomImageDistortion, refer to references like [9] and [10].

This technique is available within the detection folder with the built in transformations.

</br>

The p value represents the probability of applying the random image distortion to each image. A value of 0.5 indicates that there is an equal chance for the distortion to be applied or not, resulting in a random selection for each image.

Table. 1 Random Image Distortions values that were used.

| Distortion | Range          |
|------------|----------------|
| Contrast   | (0.5, 1.5)     |
| Saturation | (0.5, 1.5)     |
| Hue        | (-0.05, 0.05)  |
| Brightness | (0.875, 1.125) |
| p          | 0.5            |


</br>

#### 2.2.6 Batch Size

</br>

Due to limitations in the Kaggle notebook environment, a batch size of 2 was chosen for training. This smaller batch size was necessary to ensure successful execution within the resource constraints of the Kaggle platform. While a smaller batch size can result in more frequent parameter updates, it may also introduce higher noise in the gradient estimates. Nevertheless, the model can still benefit from stochastic gradient descent and adapt to the training data. It is important to consider the trade-off between computational constraints and the impact on training dynamics and model performance when selecting the batch size.

</br>

## 3. Dataset Description

</br>

The TexBiG dataset, introduced in the paper by Tschirschwitz *et. al.*[1] is designed for document layout analysis of historical documents from the late 19th and early 20th century. The dataset focuses on capturing the text-image structure present in these documents.

The TexBiG dataset provides annotations for instance segmentation, including bounding boxes and polygons/masks, for 19 distinct classes. The dataset also provides over 52,000 instances. These annotations have been manually labeled by experts, and to ensure quality, the annotations have been evaluated using Krippendorff's Alpha, a measure of inter-annotator agreement [1]. Each document image in the dataset has been labeled by at least two different annotators.

Data count in the provided datasets:

Train = 44121 Annotations
Validation = 8251 Annotations
Test = 600 Images

</br>

Table 2. Different classes within the dataset.

| Class            |
|------------------|
| paragraph        |
| equation         |
| logo             |
| editorial note   |
| sub-heading      |
| caption          |
| image            |
| footnote         |
| page number      |
| table            |
| heading          |
| author           |
| decoration       |
| footer           |
| header           |
| noise            |
| frame            |
| column title     |
| advertisement    |


</br>

The dataset is unbalanced with the main label being `paragraph`.

In figure 2 the bargraph represents the distribution to the different classes.

![image](images/class_distribution.png)

Figure 2. Class count of each class within the train and validation dataset.

</br>

The following images are from the train dataset with their corresponding bounding boxes and labels:

![image](images/train_14688302_1881_Seite_001.png)

Figure 3. Image 14688302_1881_Seite_001 with bounding boxes and labels.

</br>

![image](images/train_14688302_1881_Seite_008.png)

Figure 4. Image 14688302_1881_Seite_008 with bounding boxes and labels.

</br>

## 4. Installation

</br>

To install this package please change the terminal directory to inside the folder. 

(i.e., "torky@Islams-MacBook-Pro final-project-Torky24 ")

Then just run the following command.

`pip install -e .`

I already setup all the required modules to run this project (inside the pyproject.toml), to double check here are the required packages:

Packages:

- "matplotlib"
- "pandas"
- "numpy"
- "torch"
- "torchvision"
- "pycocotools"
- "pillow"

</br>

## 5. Usage

</br>

My folder structure is split into a main folder and sub folder. The main folder is where the code I wrote my self, while the sub structure is code re-used from the torch vision github repo. (hhttps://github.com/pytorch/vision/tree/main/references/detection)

I only re-used relevant code, and deleted the rest from the referenced folder.

</br>

### 5.1 Folder Structure

</br>

    | ____ .gitignore
    | ____ pyproject.toml
    | ____ setup.py
    | ____ README.md
    | ____ images
        | ____ train_14688302_1881_Seite_001.png
        | ____ train_14688302_1881_Seite_007.png
        | ____ train_14688302_1881_Seite_008.png
        | ____ test_lit37622_a0002.png
        | ____ test_lit39506_p0313.png
        | ____ loss_results.png
        | ____ class_distribution.png
        | ____ FasterRCNNModelArchitecture.png
    | ____  dlcv
        | ____  __init__.py
        | ____ inference.py
        | ____ main.py
        | ____ model.py
        | ____ texbigDataset.py
        | ____ train.py
        | ____ visualizations.py
        | ____ detection
            | ____ __init__.py
            | ____ coco_eval.py
            | ____ coco_utils.py
            | ____ engine.py
            | ____ transformers.py
            | ____ utils.py


</br>

- `main.py` is the main file in which all relevant functions are called to. 
- `mode.py` is where models are being called through functions and it returns a model.
- `texbigDataset.py` is where the custom dataset module has been created.
- `train.py` is where the train function is placed, and within the function the evaluation call.
- `visualizations.py` is where visualizations are created for the inferences.
- `inference.py` is where an inference function is called to create inferences onto the selected images.

</br>

### 5.2 Training & Validation

</br>

For training and validation only the main.py needs to be run directly. It can either be run directly onto the personal local machine, or in the case of kaggle please copy and paste the entire file contents into a kaggle notebook cell and run it.

Even though the pyproject.toml clearly indicates `pycocotools` to be installed sometimes it may throw an error, in that case please run the following command in the kaggle notebook.

`!pip install pycocotools`

</br>

### 5.3 Inferences

</br>

For inferences just run the command `inference()` that can be imported from the inference.py file. The interaction will be through the CLI asking for inputs to correctly create the inferences.

Example:

- Please select model: FasterRCNN V2
- Please input the path to the model: /Users/torky/Downloads/FasterRCNN_V2_0.01_DIST_epoch5.pth
- Please input the to the images: /Users/torky/Downloads  <-  Note: in this step please input the directory of where the folder for the images is and not inside its directory.
- Please input the output path: /Users/torky/Documents/final-project-Torky24/images

</br>

## 6. Experimental Results

</br>

In this section I will be discussing the numerical results as well as discussing the selection of epochs and displaying the notebooks used.

</br>

### 6.1 Results

</br>

IoU (Intersection over Union) is a metric used to evaluate the accuracy of object detection by measuring the overlap between the predicted bounding box and the ground truth bounding box.

The mAP (mean Average Precision) is a performance metric that summarizes the precision-recall curve for object detection, providing an overall measure of detection accuracy.

The difference between mAP at IoU thresholds of 0.5 and 0.75 lies in the strictness of evaluation criteria; a higher IoU threshold (0.75) requires a more precise overlap between predicted and ground truth bounding boxes, making it a more stringent measure of detection accuracy compared to IoU threshold of 0.5.

</br>

Table. 3 Different models with their respective results measured against mAP at different IoU's.

| Model                                                           | mAP   | mAP IoU = 0.50 | mAP IoU = 0.75 |
|-----------------------------------------------------------------|-------|----------------|----------------|
| FasterRCNN - ResNet 50 - lr=0.01                                | 28.08 | 46.76          | 27.84          |
| FasterRCNN - ResNet 50 - lr=0.02                                | 27.00 | 45.25          | 26.77          |
| FasterRCNN - ResNet 50 - lr=0.05                                | 26.06 | 44.35          | 25.05          |
| FasterRCNN - ResNet 50 - V2 - lr=0.01                           | 29.21 | 46.89          | 29.57          |
| FasterRCNN - ResNet 50 - V2 - lr=0.02                           | 27.21 | 44.68          | 27.16          |
| FasterRCNN - ResNet 50 - V2 - lr=0.05                           | 28.05 | 45.91          | 27.17          |
| FasterRCNN - ResNet 50 - V2 - lr=0.01 - Random Image Distortion | 29.87 | 48.50          | 29.29          |


</br>

Given that the FasterRCNN - ResNet 50 - V2 model with image augmentation achieved the highest mAP scores, it is appropriate to present visualizations showcasing the model's predictions. These visualizations provide a qualitative assessment of the model's object detection performance and serve as a means to evaluate the accuracy and effectiveness of the trained model in identifying objects within images.

</br>

![image](images/test_lit37622_a0002.png)

Figure. 5 Predictions made with bounding boxes, and labels on the test dataset. (lit37622_a0002.tif)

</br>

![image](images/test_lit39506_p0313.png)

Figure. 6 Predictions made with bounding boxes, and labels on the test dataset. (lit39506_p0313.tif)

</br>

### 6.2 Notebooks & Epoch Selection

Table 4. Epochs displays which epoch was selected out of the total amount of epochs. The Notebook versions are hyperlinked directly to the notebook.

| Model                                                           | Epoch | Notebook Link                                                                                              |
|-----------------------------------------------------------------|-------|------------------------------------------------------------------------------------------------------------|
| FasterRCNN - ResNet 50 - lr=0.01                                | 8/20  | [Notebook Version 14](https://www.kaggle.com/code/islamtorky/final-project-dlcv?scriptVersionId=135014502) |
| FasterRCNN - ResNet 50 - lr=0.02                                | 7/7   | [Notebook Version 7](https://www.kaggle.com/code/islamtorky/final-project-dlcv?scriptVersionId=134523710)  |
| FasterRCNN - ResNet 50 - lr=0.05                                | 7/7   | [Notebook Version 6](https://www.kaggle.com/code/islamtorky/final-project-dlcv?scriptVersionId=134496974)  |
| FasterRCNN - ResNet 50 - V2 - lr=0.01                           | 8/8   | [Notebook Version 26](https://www.kaggle.com/code/islamtorky/final-project-dlcv?scriptVersionId=135463296) |
| FasterRCNN - ResNet 50 - V2 - lr=0.02                           | 8/8   | [Notebook Version 27](https://www.kaggle.com/code/islamtorky/final-project-dlcv?scriptVersionId=135502904) |
| FasterRCNN - ResNet 50 - V2 - lr=0.05                           | 8/8   | [Notebook Version 28](https://www.kaggle.com/code/islamtorky/final-project-dlcv?scriptVersionId=135529716) |
| FasterRCNN - ResNet 50 - V2 - lr=0.01 - Random Image Distortion | 5/7  | [Notebook Version 29](https://www.kaggle.com/code/islamtorky/final-project-dlcv?scriptVersionId=135542302) |


</br>

Due to time constraints and the limitation of not exceeding 9 hours of training, the last model variant took longer to run than anticipated. Considering the available time, it was necessary to make a decision to select a specific epoch that would fall within the allowed limit. Due to the aforementioned restrictions, the 5th epoch was chosen for `FasterRCNN - ResNet 50 - V2 - lr=0.01 - Random Image Distortion` as it provided satisfactory results without exceeding the time limit.

The selection of epochs for the FasterRCNN models in Table 4 is based on a project limitation of not exceeding 9 hours of training. Due to this constraint, it was necessary to choose a subset of epochs that yielded satisfactory results within the given time frame. It's worth noting that in some cases, selecting a lower number of epochs resulted in better performance. To determine the appropriate number of epochs for the models, different configurations were tested extensively on a local machine. These tests helped in evaluating the performance of various epoch settings and identifying epochs that showed promising results. Based on the findings from the local testing, the selected epochs were chosen for the subsequent experiments conducted on Kaggle.

</br>

## 7. Discussion / Analysis

</br>

In the following sections, it will delve deeper into the analysis of the object detection model's performance. This will help exploring the impact of different hyperparameters, evaluate the model's precision and recall, discuss potential improvements, and consider the limitations and future directions of the study. This analysis aims to provide a comprehensive understanding of the object detection model's strengths, weaknesses, and areas for further development.

</br>

### 7.1 Results Discussion

</br>

![image](images/loss_results.png)

Figure. 7 Total loss for training over different epochs. Limited to a number of 6 epochs.

</br>

These results were obtained by evaluating the different variants of the Faster R-CNN object detection model with different RPN's while using ResNet-50 as a backbone on a test dataset. The test dataset is a separate set of images that were not used during the training process and serves as an unbiased evaluation of the model's performance.

The observed loss fluctuations during the training of Faster R-CNN v2 offer valuable insights into the model's training dynamics. Despite the presence of fluctuations, the overall decreasing trend of the loss signifies the model's capacity to learn and make progress.

The mAP scores at different Intersection over Union (IoU) thresholds, specifically IoU = 0.50 and IoU = 0.75, were computed. The results indicate that as the learning rate increases from 0.01 to 0.05, there is a consistent decrease in mAP scores for all metrics. This suggests that a higher learning rate adversely affects the model's performance in terms of object detection accuracy. It's important to note that the optimal learning rate can vary depending on the dataset and specific training setup. In this case, smaller learning rates (0.01 and 0.02) perform better for the Faster RCNN ResNet-50 model.

Furthermore, the Faster RCNN ResNet-50 V2 variant consistently outperforms the original Faster RCNN ResNet-50 variant across all metrics and variants, except for the mAP IoU = 0.75 score of the 0.02 learning rate variant. This indicates that the updated version of the model, which involves pretraining with ViT (Vision Transformer) and then fine-tuning with ResNet-50, generally leads to improved object detection performance. Additionally, the results suggest that incorporating data augmentation techniques, such as Random Image Distortion, can further enhance the model's performance. The variant "FasterRCNN - ResNet 50 - V2 - lr=0.01 - Random Image Distortion" achieved the highest mAP score, indicating that combining Faster RCNN ResNet-50 V2 with this specific data augmentation technique yielded superior results.

Lastly, it is worth noting that the mAP scores at IoU = 0.75 are consistently lower than those at IoU = 0.50 for all variants. This is expected since a higher IoU threshold requires stricter object localization, making it more challenging for a bounding box prediction to meet the criteria. Therefore, the mAP scores at IoU = 0.50 provide a more lenient evaluation and generally yield higher results compared to the stricter IoU = 0.75 evaluation.


</br>

### 7.2 Inference Analysis

</br>

It is notable that in some images that are completely empty, the model predicts a bounding box as seen in Figure 5. This suggests that the model falsely detects an object in an image where none exists. This could be due to various reasons, such as noise in the input data, model limitations, or misinterpretation of the features by the model. It is essential to investigate the cause of such false positives to improve the model's accuracy.

The model correctly identifies multiple bounding boxes, including the paragraph with a high accuracy of 1.0 as seen in Figure 6. However, it splits the bounding boxes for the paragraphs that are stacked above each other instead of predicting a single large bounding box encompassing all of them. This could be due to the model's limitations in detecting and grouping objects that are close together or have overlapping regions.

</br>

## 8. Outlook / Future Work

</br>

The analysis of the Faster R-CNN object detection model has shown promising results. However, it is important to consider the limitations imposed by the available resources. The given hardware constraints, such as the maximum batch size and time window, need to be taken into account when further optimizing the model. The training and testing time constraints specified in the requirements may pose challenges in exploring more extensive hyperparameter tuning techniques. While the current analysis provides valuable insights, it may be worthwhile to consider more efficient algorithms or distributed training methods to make the best use of the available time.

In addition to considering different optimizers, such as Stochastic Gradient Descent (SGD), there are several other optimization algorithms that can be explored to further improve the Faster R-CNN object detection model.There is the possibility of using Bayesian optimization as a powerful technique for hyperparameter tuning. By intelligently exploring the hyperparameter space, it can help identify optimal configurations more efficiently. Bayesian optimization can be a valuable approach to search for improved configurations within the given limitations.

</br>

## 9. References

</br>

[1] Tschirschwitz, D., Klemstein, F., Stein, B., Rodehorst, V. (2022). A Dataset for Analysing Complex Document Layouts in the Digital Humanities and Its Evaluation with Krippendorff’s Alpha. In: Andres, B., Bernard, F., Cremers, D., Frintrop, S., Goldlücke, B., Ihrke, I. (eds) Pattern Recognition. DAGM GCPR 2022. Lecture Notes in Computer Science, vol 13485. Springer, Cham. doi.org/10.1007/978-3-031-16788-1_22

[2] S. Ren, K. He, R. Girshick and J. Sun, "Faster R-CNN: Towards Real-Time Object Detection with Region Proposal Networks," in IEEE Transactions on Pattern Analysis and Machine Intelligence, vol. 39, no. 6, pp. 1137-1149, 1 June 2017, doi: 10.1109/TPAMI.2016.2577031.

[3] Y. Li, S. Xie, X. Chen, P. Dollar, K. He, and R. Girshick, "Benchmarking Detection Transfer Learning with Vision Transformers," arXiv preprint arXiv:2111.11429 [cs.CV], Nov. 2021.

[4] T. He, et al., "Bag of Tricks for Image Classification with Convolutional Neural Networks," in 2019 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), Long Beach, CA, USA, 2019 pp. 558-567.
doi: 10.1109/CVPR.2019.00065

[5] T. -Y. Lin, P. Dollár, R. Girshick, K. He, B. Hariharan and S. Belongie, "Feature Pyramid Networks for Object Detection," 2017 IEEE Conference on Computer Vision and Pattern Recognition (CVPR), Honolulu, HI, USA, 2017, pp. 936-944, doi: 10.1109/CVPR.2017.106.

[6] Lin, T. Y., Dollár, P., Girshick, R., He, K., Hariharan, B., & Belongie, S. (2017). Feature Pyramid Networks for Object Detection. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2117-2125. doi: 10.1109/CVPR.2017.106

[7] Chen, L. C., Papandreou, G., Schroff, F., & Adam, H. (2018). Optimal Learning Rates for Multi-Stage Training with Application to Object Detection Networks. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 7351-7359. doi: 10.1109/CVPR.2018.00773

[8] He, T., Zhang, Z., Zhang, H., Xie, J., & Li, M. (2019). Bag of Tricks for Image Classification with Convolutional Neural Networks. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 558-567. doi: 10.1109/CVPR.2019.00064

[9] Simard, P. Y., Steinkraus, D., & Platt, J. C. (2003). Best Practices for Convolutional Neural Networks Applied to Visual Document Analysis. In Proceedings of the Seventh International Conference on Document Analysis and Recognition (ICDAR), 958-962. doi: 10.1109/ICDAR.2003.1227801

[10] A. Krizhevsky, I. Sutskever, and G. E. Hinton, “ImageNet Classification with Deep Convolutional Neural Networks,” in Advances in Neural Information Processing Systems 25, F. Pereira, C. J. C. Burges, L. Bottou, and K. Q. Weinberger, Eds. Curran Associates, Inc., 2012, pp. 1097--1105.