README for Task 5: Decision Trees and Random Forests

Project Title:

Classification Using Decision Trees and Random Forests

Objective:

The objective of this task is to learn tree-based models for classification. Specifically, we focus on:

1. Training and visualizing a Decision Tree Classifier.


2. Controlling overfitting using max_depth.


3. Training a Random Forest Classifier and comparing its performance with a single decision tree.


4. Interpreting feature importances from the Random Forest model.


5. Evaluating models using cross-validation.



Tools and Libraries:

Python Libraries: pandas, matplotlib, seaborn, scikit-learn

Key Models: DecisionTreeClassifier, RandomForestClassifier

Visualization: plot_tree, seaborn barplot


Dataset:

Name: Heart Disease Dataset

Source: Click here to download dataset

Description: The dataset contains patient features such as age, sex, cholesterol levels, blood pressure, and other heart health indicators. The target column target indicates the presence (1) or absence (0) of heart disease.


Steps Implemented in Code:

1. Load dataset and separate features (X) and target (y).


2. Split the data into training and testing sets (80% train, 20% test).


3. Train a Decision Tree Classifier and evaluate accuracy.


4. Visualize the Decision Tree using plot_tree.


5. Control overfitting by setting max_depth=4 in a second Decision Tree model.


6. Train a Random Forest Classifier and evaluate accuracy.


7. Compute feature importances using Random Forest and visualize them.


8. Perform 5-fold cross-validation to validate the performance of both models.



Key Outputs:

Accuracy of the Decision Tree and Random Forest models.

Decision Tree visualization showing splits and class predictions.

Feature importance bar chart from the Random Forest model.

Cross-validation scores for both models.


Conclusion:

Decision Trees are simple to visualize but can overfit.

Random Forests are ensembles of Decision Trees that generally provide higher accuracy and robustness.

Feature importance helps identify which features contribute most to predicting heart disease.


Next Steps / Improvements:

Tune hyperparameters like n_estimators, max_depth, and min_samples_split for better performance.

Apply other tree-based models such as Gradient Boosting or XGBoost.

Explore feature selection techniques to improve interpretability.# dishap81-task-5
Machine Learning project for the Titanic dataset, focusing on predicting passenger survival. Features data exploration, outlier removal, and model training.
