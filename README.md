machine-learning-techniques
==============================

You can find documentation about different milestones in docs folder

[DOCS](docs)

Problem Description
------------

In this repository we will use Machine Learning Techniques for studying the problem of the Moriarty attack. The Moriarty malicious agent perpetrate attacks on an user's device while creating labels for the Sherlock dataset. The attack is based on the following attack model: 
* A beningn application is initially given or repackaged to include additional code (malware like spyware). 
* The victim installs the app without know the consequences of the required permissions. 

Firstly, we have a datasheet collected internally by a user’s smartphone each 4 seconds during a month. The purpose is to carry out an analysis of all these data, in order to analyze the behavior of the user's device and detect anomalies caused by a security attack.

In order to make a good interpretation we will apply different techniques. We begin filtering data that provide us more information. Then, we apply statistical procedures (PCA) and clustering algorithms (K-Means) that reduce the variables trying to understand the relationship between them and observing how the data are grouped. 

In the second part of this problem, we have analized when the attack take place in a chosen datasheet. When the attack penetrate in the device we have distinguish between an benign or malicious attack. 


Project Organization
------------

    ├── LICENSE
    ├── Makefile           <- Makefile with commands like `make data` or `make train`
    ├── README.md          <- The top-level README for developers using this project.
    ├── data
    │   ├── external       <- Data from third party sources.
    │   ├── interim        <- Intermediate data that has been transformed.
    │   ├── processed      <- The final, canonical data sets for modeling.
    │   └── raw            <- The original, immutable data dump.
    │
    ├── docs               <- Documentation about milestones
    │
    ├── models             <- Trained and serialized models, model predictions, or model summaries
    │
    ├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
    │                         the creator's initials, and a short `-` delimited description, e.g.
    │                         `1.0-jqp-initial-data-exploration`.
    │
    ├── references         <- Data dictionaries, manuals, and all other explanatory materials.
    │
    ├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
    │   └── figures        <- Generated graphics and figures to be used in reporting
    │
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
    │                         generated with `pip freeze > requirements.txt`
    │
    ├── setup.py           <- makes project pip installable (pip install -e .) so src can be imported
    ├── src                <- Source code for use in this project.
    │   ├── __init__.py    <- Makes src a Python module
    │   │
    │   ├── data           <- Scripts to download or generate data
    │   │   └── make_dataset.py
    │   │
    │   ├── features       <- Scripts to turn raw data into features for modeling
    │   │   └── build_features.py
    │   │
    │   ├── models         <- Scripts to train models and then use trained models to make
    │   │   │                 predictions
    │   │   ├── predict_model.py
    │   │   └── train_model.py
    │   │
    │   └── visualization  <- Scripts to create exploratory and results oriented visualizations
    │       └── visualize.py
    │
    └── tox.ini            <- tox file with settings for running tox; see tox.testrun.org


--------

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>
