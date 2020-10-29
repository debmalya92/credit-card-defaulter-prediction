# Credit Card Defaulter prediction 

## Table of Content
  * [Demo](#demo)
  * [Overview](#overview)
  * [Motivation](#motivation)
  * [Dataset Information](#dataset-information)
  * [Technical Aspect](#technical-aspect)
  * [Installation](#installation)
  * [Depoyment on Heroku](#deployment_on_heroku)
  * [Directory Tree](#directory-tree)  
  * [Technologies Used](#technologies-used)
  * [Team](#team)
  * [Credits](#credits)

## Demo
Link: [https://credit-card-defaulter-predict.herokuapp.com/](https://credit-card-defaulter-predict.herokuapp.com/)

[![](https://i.imgur.com/4HjhfDQ.png)](https://credit-card-defaulter-predict.herokuapp.com/)


## Overview
This is a classification model for a most common dataset, Credit Card defaulter prediction. Prediction of the next month credit card defaulter based on demographic and last six months behavioral data of customers.

## Motivation
There are times when even a seemingly manageable debt, such as credit cards, goes out of control. Loss of job, medical crisis or business failure are some of the reasons that can impact your finances. In fact, credit card debts are usually the first to get out of hand in such situations due to hefty finance charges (compounded on daily balances) and other penalties.

A lot of us would be able to relate to this scenario. We may have missed credit card payments once or twice because of forgotten due dates or cash flow issues. But what happens when this continues for months? How to predict if a customer will be defaulter in next months?

To reduce the risk of Banks, this model has been developed to predict customer defaulter based on demographic data like gender, age, marital status and behavioral data like last payments, past transactions etc.

## Dataset Information
This dataset contains information on default payments, demographic factors, credit data, history of payment, and bill statements of credit card clients in _Taiwan from April 2005 to September 2005_.

## Technical Aspect
This project is divided into two part:
1. Training a [RandomForestClassifier](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html) classification model using [GridSearchCV](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html) to predict defaulter as accurate as possible.
	- Cleaning the datasets, fixing all features
	- Applying all GridSearchCV to obtain optimal hyperparameters
	- Apply Classification ML model
2. Building and hosting a Flask web app on Heroku.
	- Build the web app using Flask API
	- Upload the project on GitHub
    - Get the customer information from Web app
    - Display the prediction 

## Installation
The Code is written in Python 3.7. If you don't have Python installed you can find it [here](https://www.python.org/downloads/). If you are using a lower version of Python you can upgrade using the pip package, ensuring you have the latest version of pip. To install the required packages and libraries, run this command in the project directory after [cloning](https://www.howtogeek.com/451360/how-to-clone-a-github-repository/) the repository:
```bash
pip install -r requirements.txt
```

## Depoyment on Heroku
Create a new repositoryon [GitHub](https://github.com) and upload the project.

Follow the instruction given on [Heroku Documentation](https://devcenter.heroku.com/articles/getting-started-with-python) to deploy a web app.

## Directory Tree 
```
├── templates 
│   └── index.html
├── app.py
├── credit-card-default.csv
├── credit_default_prediction.py
├── model.pkl
├── Procfile
├── README.md
└── requirements.txt
```

## Technologies Used

![](https://forthebadge.com/images/badges/made-with-python.svg)

[<img target="_blank" src="https://numpy.org/images/logos/numpy.svg" width=100>](https://numpy.org)    [<img target="_blank" src="https://upload.wikimedia.org/wikipedia/commons/thumb/e/ed/Pandas_logo.svg/450px-Pandas_logo.svg.png" width=150>](https://pandas.pydata.org)    [<img target="_blank" src="https://scikit-learn.org/stable/_static/scikit-learn-logo-small.png" width=150>](https://scikit-learn.org/stable)   [<img target="_blank" src="https://www.statsmodels.org/stable/_images/statsmodels-logo-v2-horizontal.svg" width=170>](https://www.statsmodels.org)

[<img target="_blank" src="https://flask.palletsprojects.com/en/1.1.x/_images/flask-logo.png" width=150>](https://flask.palletsprojects.com/en/1.1.x/) [<img target="_blank" src="https://number1.co.za/wp-content/uploads/2017/10/gunicorn_logo-300x85.png" width=200>](https://gunicorn.org) [<img target="_blank" src="https://matplotlib.org/_static/logo2_compressed.svg" width=170>](https://matplotlib.org)      [<img target="_blank" src="https://seaborn.pydata.org/_static/logo-wide-lightbg.svg" width=150>](https://seaborn.pydata.org)

[<img target="_blank" src="https://jupyter.org/assets/nav_logo.svg" width=150>](https://jupyter.org)

## Team
[![Debmalya Ghosal](https://avatars2.githubusercontent.com/u/60285205?s=144&u=45fc55fc21b66ed5ea26153766e3d8e1cc3f4449&v=4)](https://github.com/debmalya92) |
-|
[Debmalya Ghosal](https://github.com/debmalya92) |)

## Credits
- The datasets has been provided by [Kaggle](https://www.kaggle.com/uciml/default-of-credit-card-clients-dataset). The original dataset can be found [here](https://archive.ics.uci.edu/ml/datasets/default+of+credit+card+clients) at the UCI Machine Learning Repository. This project wouldn't have been possible without this dataset.
