# Project Uniqueness Checker
>Under- Graduate final year students can check the uniqueness of their proposed project in comparison to previous years

> Final year project done for award of degree , VTU 
>   CSE, 2020


## Overview
- **Aim**: Check uniqueness of a under-graduate students proposed project by taking in students title and abstract and returning a uniqueness score in comparison to previous year projects
- **Method**: A combination of Siamese Manhattan LSTM + Inception Model is used to compare titles. 
- **Implementation**:
  - **Title similarity**: checked using Siamese Manhattan LSTM + Inception Model
  - **Abstract similarity**: checked using spaCy module
  - **Uniqueness**: calculated as
  
        *Uniqueness=100-max(similarity scores) X 100*
  - **Backend**: SQLite
  - **Frontend**: Flask , Flask-admin, Flask-Login , jQuery

## Installing

1. Create a conda virtual environment
```sh
conda create -n myenv python=3.7
```

2. Install requirements 
```sh
pip install -r requirements.txt
```
3. Run train.py to set up 
```sh
train.py
```

## Use

#### Checking similarity score
Webpage where students can enter proposed project ideas title and abstract to get results
```sh
python checkscore.py
```
paste `http://127.0.0.1:5000/` in the browser to  access the page

#### Project manager dashboard
Dashboard for project manager to perform CRUD operations.
```sh
python projectDashboard.py
```
paste `http://127.0.0.1:5000/` in the browser to  access the page

#### Admin Dashboard
Dashboard where the admin can perform CRUD operations on users and view projects uploaded 
```sh
python adminDashboard.py
```
paste `http://127.0.0.1:5000/` in the browser to  access the page

