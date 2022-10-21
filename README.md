# Disaster-Response-Pipeline Project

### Github Repository
https://github.com/reaganmugonya/Disaster-Response-Pipeline.git

### Motivation
This projects involves using data engineering skills to build a Machine learning pipeline to analyze data from [Appen](https://appen.com/) (formally Figure 8) to classify disaster messages so that they can be sent to the appropriate disaster relief agency.
A web app was developed where emergency workers can input meaages and get classification results in several categories. The web app also contains data visualizations.

### Install
The code was written in Python 3, html, CSS Bootstrap, and Javascript. 
The following Python packages were used: 
1. Sys, 
2. Pandas
3. Sqlalchemy
4. Nltk
5. Re
6. NumPy
7. Pickles
8. Sklearn
9. Plotly
10. Flask
11. Json

### Instructions:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Go to `app` directory: `cd app`

3. Run your web app: `python run.py`

4. Go to http://0.0.0.0:3001/                 Or Go to http://localhost:3001/


### License
This app was completed as part of the Udacity Data Scientist Nanodegree. Code templates and data were provided by Udacity.
