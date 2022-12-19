![TensorFlow](https://img.shields.io/badge/TensorFlow-%23FF6F00.svg?style=for-the-badge&logo=TensorFlow&logoColor=white)
![Pandas](https://img.shields.io/badge/pandas-%23150458.svg?style=for-the-badge&logo=pandas&logoColor=white)
![Keras](https://img.shields.io/badge/Keras-%23D00000.svg?style=for-the-badge&logo=Keras&logoColor=white)
![Matplotlib](https://img.shields.io/badge/Matplotlib-%23ffffff.svg?style=for-the-badge&logo=Matplotlib&logoColor=black)
![NumPy](https://img.shields.io/badge/numpy-%23013243.svg?style=for-the-badge&logo=numpy&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-%23F7931E.svg?style=for-the-badge&logo=scikit-learn&logoColor=white)

# Image Classification of Concrete with Cracks or without Cracks by using TensorFlow

## Summary
<p>Common concrete cracks can occur for a variety of reasons and come in many different forms. The causes could be as a result of poor design, which can result in structural cracks. Additionally, settlement cracks can appear when a portion of concrete dips or when the earth beneath the slab isn't compacted properly. Other than that, there are concrete cracks namely hairline cracks which usually develop in concrete foundation as the concrete cures while shrinkage cracks occur while the concrete is curing.</p>

<p>If cracks are not detected in time, they could jeopardise a building's stability and safety. Therefore, by creating this model, it could facilitate the easy identification of cracked objects and aid to save thousands of lives.</p>

<p>1. Data Preparation</p>
  <ol>- Upload the dataset using operating system (os) to fetch the contents in the current directory.</ol>
  <ol>- Load the data as tensorflow dataset by using special method. Split the dataset into train dataset and validation dataset.</ol>
  <ol>- Extract the class names as a list.</ol>
  <ol>- Display some examples to see if the dataset has been uploaded successfully.</ol>
  <p align="center"><img src="image_example.png" alt="image example" width="500"/></p>
  <div align="center"><ol>The figure above shows the random images in the dataset and it's label.</ol></div>
  <ol>-Then, convert the dataset into prefetch dataset</ol>
  
<p>2. Model Development</p>
   <ol>- In this part, I create a model augmentation. I could rotate the image in many form. So it can increase the diversity of the training set by applying random (but realistic) transformations.</ol>
  <ol>-Then, I applied the data augmentation into one image to see the result.</ol>
  
  <ol>- To proceed on doing transfer learning, I create the layer for data normalization by using <strong>tf.keras.applications.mobilenet_v2.preprocess_input.</strong></ol>
  <ol>- Then, I start the transfer learning by instantiate the pretrained model.</ol>
  <ol>- I set the pretrained model as non-trainable to avoid destroying any of the information they contain during future training rounds.</ol>
  <ol>- Create the new classifier by using <strong>layers.GlobalAveragePooling2D()</strong>.</ol>
  <ol>- The <strong>Dense layer</strong> is being used to create an output layer</ol>
  <ol>- Link the layers together</ol>
  <ol>- Initiate the full model pipeline and compile the model</ol>
<ol>- Evaluate the model before combine with testing data. The accuracy we got is 50%.</ol>

<p>3. Model Deployment</p>
   <ol>- Data cleaning need to be done to increase overall productivity and allow for the highest quality information in your decision-making.</ol>
   <ol>- I used interpolate to convert any values that is not in numerical to numericals. Then, the data can be visualized clearly by using graphical method</ol>
   <p align="center"><img src="image_example.png" alt="image example" width="500"/></p>
   
   <div align="center"><ol> As we can see in the graph, there is missing value in between 400 to 500. Thus, the missing value can be filled by using interpolate.</ol></div>
   <p align="center"><img src="model/after_cleaning_graph.png" alt="graph" width="500"/></p>
   <div align="center"><ol> Based on the graph above, the missing value already been filled. </ol></div>

<p>4. Features Selection</p>
   <ol>- In this data, I selected <em>new_cases</em> to do the predictions.</ol>
          
<p>5. Data Pre-processing</p>
   <ol>- <strong>MinMaxScaler</strong> is being used in this part to convert the data into 0 until 1.</ol>
   <ol>- I did train-test-split to split the <em>X_train</em> and <em>y_train</em></ol>
   
<p>Then only we can do <strong>Model Development</strong>.</p>
 <p> a) In Model Development, I used Input as an input layer.</p>
 <p> b) For hidden layers, I used 3 LSTM layers and 2 dropouts.</p>
 <p align="center"><img src="model/model.png" alt="model layers" width="200"/></p>
  
  <p> c) The graph can be visualized by using TensorBoard. The graphs below shows the training data and validation data of my model.</p>
 <p align="center"><img src="model/epoch_loss.png" alt="loss" width="500"/></p>
 <div align="center"><ol> The graph above shows the loss data.</ol></div>
 
 <p>After that, we can proceed to do predictions for testing data.</p>
 <p>1. Data Loading</p>
 <p>- I start uploading the Testing Dataset by using pandas as well.</p>
 
 <p>2. Data Inspection</p>
 <p>- I did <strong>df_test.info()</strong> to look at the datatypes. The datatype for <em>new_cases</em> is float.</p>
 
 <p>3. Data Cleaning</p>
 <p>- I converted the dataype of <em>new_cases</em> to <strong>integer</strong></p>
 
 <p>4. Features Selection</p>
  <p>- In this process, we also choose <em>new_cases</em> only.</p>
  
  <p>5. Data Pre-processing</p>
  <p>- In this part, I combine the training and testing data by using concatenation.</p>
  <p>- Then, I proceed to find the predicted_case.</p>
  
 <p align="center"><img src="model/epoch_mape.png" alt="mape" width="500"/></p>
 <div align="center"><ol> This is the MAPE graph. As we can see in the graph, the training data which is in orange colour starts to overfitting at 12-axis. However, it then went down. It might be due to <em>Dropout layer</em> in the model and also, the node value in LSTM layer before the output layer.</ol></div>
  
 <p> Then, the project is being compiled. This is my MAPE result which is below than 1% and also, the MAPE graphs:</p>
 <p align="center"><img src="model/prediction_graph.png" alt="mape graph" width="500"/></p>
 


## Acknowledgement
Special thanks to ([https://data.mendeley.com/datasets/5y9wdsg2zt/2](https://data.mendeley.com/datasets/5y9wdsg2zt/2)) :smile:
