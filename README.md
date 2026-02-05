Exploratory Analysis of Rainfall Data in India for Agriculture

Project Overview :
This project is a Machine Learning–based Rainfall Prediction System developed using Python and Flask.
The model predicts whether rainfall will occur based on historical weather data.
A trained ML model is integrated into a Flask web application to provide easy and real-time user interaction.

Features :
• Machine learning model for rainfall prediction
• Data preprocessing using encoder and imputer
• Flask-based web interface
• User-friendly HTML frontend
• Real-time prediction output

Technologies Used :
• Python
• Flask
• Pandas
• NumPy
• Scikit-learn
• HTML / CSS
• Pickle (.pkl)


Project Structure :

Rainfall_Prediction_Project/

1.app.py                    # Flask application
2.train_model.py            # Model training script
3.rainfall_prediction.py    # Prediction logic
4.rainfall.csv              # Dataset
5.model.pkl                 # Trained ML model
6.encoder.pkl               # Encoder file
7.imputer.pkl               # Imputer file
8.templates/
 a.index.html
 b.chance.html
 c.nochance.html
9.static/                   # CSS / static files
10.README.md                # Project documentation

 
Machine Learning Model
1.Algorithm: Supervised Learning
2.Data Preprocessing:
 a.Missing values handled using Imputer
 b.Categorical features encoded using Encoder
3.Model Storage: Saved using Pickle (.pkl)

How to Run the Project :

1️⃣ Clone the Repository
git clone https://github.com/jagadala-durga-prasanna/Rainfall_Prediction_Project.git
cd Rainfall_Prediction_Project

2️⃣ Install Required Libraries
pip install flask pandas numpy scikit-learn

3️⃣ Run the Flask Application
python app.py

4️⃣ Open in Browser
“This is a local Flask application and runs on localhost.”
http://127.0.0.1:5000

Output :
The application predicts whether rainfall will occur or not based on the input data and displays the result through the web interface.


requirements.txt :
• flask
• pandas
• numpy
• scikit-learn


Future Scope : 
• Enhance prediction accuracy by training the model on larger and more diverse datasets.
• Integrate real-time weather data APIs for live rainfall prediction.
• Deploy the application on a cloud platform to make it accessible online.
• Extend the system to support region-wise and seasonal rainfall analysis.
• Improve the user interface with advanced visualizations and interactive charts.
• Explore the use of advanced machine learning or deep learning models for better performance.

Team Contribution : 
• This project was developed as a team effort with equal contribution from all members.
• Data collection, preprocessing, and model training were done collaboratively.
• Development of the Flask web application was shared among the team.
• Testing, debugging, and result validation were performed jointly.
• Documentation and GitHub updates were handled collectively.




