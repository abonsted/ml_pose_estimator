# CS 325: Introduction to Machine Learning Project
## Project Description
Our team is building a pose-estimation classification model for exercises based on the [Penn Action dataset](https://dreamdragon.github.io/PennAction/). 

## Repository Breakdown
- **andrew-model.py** : Initial ML Model using Support Vector Classification. This model was not used for the midterm report because it was never able to fully run with the amount of data it was being trained on.
- **dominic-model.py** : ML Model that uses RandomForestClassifier. This model is the main model used in our midterm report and will be refined in the next few weeks for our final project.

- **Goated_Matrix.png** : Confusion matrix made from 'dominic-model.py' with data leakage which results in the near-perfect accuracy.
- **conf-matrix-midterm.png** : Confusion matrix made from 'dominic-model.py' after data leakage problem was solved.

- **data_info.py** : Program used to collect numerical data for each exercise. Used in the data collection 