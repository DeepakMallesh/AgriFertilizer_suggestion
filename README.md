# AgriFertilizer-Recommandation
A user-friendly web app built with Streamlit to recommend the best fertilizer based on soil parameters. It utilizes a trained TensorFlow model to predict the optimal fertilizer and suggests specific products.

## Table of Contents :-
  1.Installation  
2.Usage  
3.Project Structure  
4.Model Training  
5.License  


## Installation :-  
To run this application locally, follow these steps:  
Clone the repository:  
git clone https://github.com/DeepakMallesh/fertilizer-recommendation-system.git
cd fertilizer-recommendation-system  
Install the required dependencies:  
pip install -r requirements.txt  

## Usage :-  
To start the application, run the following command : streamlit run app.py  
This will open a new tab in your default web browser with the Fertilizer Recommendation System.  

## Enter Soil Parameters :-  
In the application, you'll need to enter the following soil parameters:  
Ca (Calcium)  
Mg (Magnesium)  
K (Potassium)  
S (Sulphur)  
N (Nitrogen)  
Lime  
C (Carbon)  
P (Phosphorus)  
Moisture  
Once all parameters are entered, click the "Recommend" button to get the fertilizer suggestion.  

## Project Structure :-  
The main components of this project are:  
app.py : Contains the Streamlit application code.  
model.py : Contains the model training code.  
fertilizer_model.h5 : Pretrained model file.  
scaler.pkl : Scaler used for preprocessing.  
label_encoder.pkl : Label encoder used for encoding class labels.  

## Model Training :-  
If you need to retrain the model, follow these steps:  
Load the dataset : Ensure you have a CSV dataset named FertPredictDataset.csv with appropriate features and class labels.  
Run the model training script : python model.py  
Save the trained model and preprocessing tools: The script will save the model as fertilizer_model.h5, and the preprocessing tools as scaler.pkl and label_encoder.pkl.  

## License :-  
This project is licensed under the MIT License.  
