3 .Regression Implementation
3.1 Methodology
The goal is to predict the lifespan of metal parts using regression models based on production
parameters, material composition, and defect counts. The models were chosen for this task:
Random Forest Regressor, and Gradient Boosting Regressor. Each model was evaluated
for its ability to handle non-linear relationships and interactions in the data.
Model 1: Random Forest Regressor
Random Forest Regressor, an ensemble method, handles non-linear relationships, mixed data
types, and outliers effectively, while providing feature importance and minimizing
overfitting.
Model 2: Gradient Boosting Regressor
Gradient Boosting, a sequential ensemble method, corrects errors iteratively, capturing
complex patterns, with high predictive power, tunable parameters, and robustness to outliers.
Preprocessing Steps
1. Feature Selection: The dataset includes several critical features: coolingRate,
Nickel%, HeatTreatTime, Chromium%, quenchTime, forgeTime, and partType.
These features were chosen based on their potential impact on the lifespan of metal
parts. The feature matrix (X) was constructed by selecting these columns, while the
Lifespan column served as the target variable (Y), representing the output to be
predicted:
2. Categorical Feature Encoding:The categorical feature partType was one-hot
encoded to convert it into a numerical format for compatibility with the regression
models.
3. Feature Scaling: To ensure uniform scaling, all numerical features—coolingRate,
Nickel%, HeatTreatTime, Chromium%, quenchTime, and forgeTime—were
standardized using StandardScaler. This step is crucial for models like Gradient
Boosting that are sensitive to feature magnitudes:
4. Train-Test Split:The dataset is split into training and testing sets to evaluate model
performance. The train_test_split function randomly divides the features (X) and
target (y) into two parts: 80% of the data is used for training (X_train and y_train),
and 20% is reserved for testing (X_test and y_test).
Hyperparameter Tuning Framework
To optimize the performance of the models, GridSearchCV was utilized to fine-tune the
hyperparameters. Common hyperparameters such as max_depth, min_samples_split, and
min_samples_leaf were tuned across all models to control the complexity of the decision
trees and prevent overfitting. These hyperparameters determine the maximum depth of the
trees, the minimum number of samples required to split a node, and the minimum samples
required in a leaf node, respectively.
Model-Specific Hyperparameters
In addition to the common parameters, each model required specific hyperparameters based
on its architecture:
Random Forest Regressor:
I. n_estimators: The number of decision trees in the ensemble. A higher number
typically improves model stability but increases computational cost.
Gradient Boosting Regressor:
I. n_estimators: The number of boosting stages (iterations). Higher values allow the
model to learn more complex patterns but risk overfitting.
II. learning_rate: Controls the contribution of each tree to the overall prediction. Lower
values ensure slower but more stable learning, which helps avoid overfitting.
This structured approach to tuning ensured that each model was optimized for the dataset,
leveraging its strengths while addressing potential weaknesses such as overfitting or
underfitting. The results of the tuning process demonstrated that Gradient Boosting
Regressor performed best due to its ability to iteratively correct errors from previous stages,
achieving an R² score of 0.9684, the highest .
 3.2 Models Evaluation
Experimentation and Final Model Versions
Each model (Random Forest Regressor, Gradient Boosting Regressor) was optimized using
GridSearchCV as described in the hyperparameter tuning framework. Experiments involved
systematically varying hyperparameter combinations and evaluating performance using 5-fold crossvalidation on the training set.
The chosen hyperparameters were applied to build the final versions of each model, which were
subsequently evaluated on a separate test set (20% of the dataset).
Regression Performance Metrics
Model performance was evaluated using Mean Squared Error (MSE) for average squared errors,
Mean Absolute Error (MAE) for absolute differences, and R² Score for explained variance, closer to 1
indicating better performance.
Model Results
Evaluation of Final Models
Random Forest Regressor:
Achieved a strong R² score of 0.9341, indicating that the model explains 93.41% of the variance in
the lifespan data.
The MSE of 6820.92 and MAE of 64.85 demonstrate its robustness in handling the non-linear and
complex relationships in the dataset.
Gradient Boosting Regressor:
Outperformed all models with an R² score of 0.9684, explaining 96.84% of the variance.
Achieved the lowest error metrics (MSE = 3271.30, MAE = 44.94), indicating precise predictions
and strong generalization.
Its iterative approach to minimizing prediction errors ensures that it captures nuanced relationships in
the dataset.
Interpretation and Recommendation
The Gradient Boosting Regressor is the best-performing model, significantly surpassing Random
Forest and Decision Tree regressors in all metrics. Its ability to iteratively correct prediction errors
and effectively balance bias-variance trade-offs makes it the ideal choice for deployment.
The Random Forest Regressor, while competitive, demonstrated slightly higher errors and a lower
R² score, making it a strong alternative but not the top recommendation.
Based on the evaluation metrics and experimental results, the Gradient Boosting Regressor is
recommended for deployment. Its superior performance ensures reliable predictions of metal part
lifespan, supporting the company’s goals of optimizing production processes and reducing defect
rates.
Part 3.3 – Critical Review
Strengths of the Methodology
The methodology effectively combined preprocessing, model selection, and hyperparameter tuning.
The use of ColumnTransformer for scaling numerical features and one-hot encoding categorical
variables ensured consistent data preparation. The inclusion of Random Forest Regressor and
Gradient Boosting Regressor enabled a comprehensive evaluation of performance on the dataset.
Systematic hyperparameter tuning using GridSearchCV improved model accuracy by optimizing
parameters such as max_depth, n_estimators, and learning_rate. Gradient Boosting achieved the
highest performance with an R² score of 0.9684.
Areas for Improvement
The dataset, with only 1,000 entries, is relatively small for ensemble methods, limiting it. Outliers,
such as in forgeTime and defect counts, may have influenced model accuracy. Enhanced feature
engineering, like interaction terms between Nickel% and Chromium%, could improve predictions.
Exploring simpler models like Linear Regression or SVMs could provide benchmarks for
comparison.Future work may explore XGBoost, LightGBM, Neural Networks for larger datasets, and
Bayesian Optimization for efficient hyperparameter tuning.
Conclusion-Gradient Boosting emerged as the most robust model for predicting metal part lifespan.
However, future work should explore alternative preprocessing techniques and model architectures to
enhance accuracy and efficiency further.
4.Classification Implementation
Part 4.1 – Feature Crafting
To classify parts as usable or defective, a binary label, 1500_labels, was introduced. This label
categorizes parts as usable (1) if their Lifespan exceeds 1500 hours, and defective (0) otherwise. The
threshold was selected based on the company’s requirement for a minimum acceptable lifespan. The
addition of this label simplifies the problem into a binary classification task, aligning with the
company’s need for actionable insights to improve production.
An analysis of the 1500_labels feature revealed a little unbalanced dataset:
 Usable Parts (1): Approximately 30% of the dataset.
 Defective Parts (0): Approximately 69% of the dataset.
To confirm the balance of the dataset, a bar chart of the binary label distribution was created. The
chart displayed a uneven distribution.
Two machine learning models, Gradient Boosting Classifier and Random Forest Classifier, were
selected to predict the 1500_labels. These models were chosen for their ability to handle non-linear
relationships and their robustness in classification tasks. Rigorous hyperparameter tuning was
conducted using GridSearchCV, optimizing key parameters such as n_estimators, max_depth, and
learning_rate.
 The Gradient Boosting Classifier achieved a high accuracy of 97%, with strong recall for
Class 1 (usable parts), minimizing the risk of missing usable parts.
 Similarly, the Random Forest Classifier achieved an accuracy of 97%, excelling in
precision for Class 1, reducing false positives for usable parts.
Confusion matrices for both models were visualized to evaluate their predictions. These heatmaps
confirmed the models’ effectiveness in minimizing false negatives and false positives. The dataset,
coupled with robust model performance, ensures the reliability of the classification system.
In conclusion, the binary classification approach with 1500_labels provides a clear and actionable
solution for the company. Gradient Boosting’s higher recall for usable parts makes it the preferred
choice for deployment, while Random Forest offers a reliable alternative. The inclusion of
visualizations enhances confidence in the classification methodology.
Part 4.2 – Methodology
Model Selection
To classify metal parts as usable or defective based on the 1500_labels, two models were selected:
Gradient Boosting Classifier:
Justification: GBC builds models iteratively, optimizing errors from previous iterations. It effectively
captures non-linear relationships and is robust for imbalanced datasets. Its tunable learning rate and
ability to balance bias and variance make it ideal for classification tasks.
Strengths: High accuracy, robust to overfitting, and interpretable feature importance.
Random Forest Classifier:
Justification: RFC is an ensemble learning method that combines multiple decision trees. It is
effective for handling categorical and numerical data and provides insights into feature importance.
RFC is less sensitive to hyperparameters, making it reliable for classification tasks.
Strengths: Handles outliers and noisy data well, robust to overfitting, and easy to interpret.
Preprocessing Routine
1. Feature Selection:
o Excluded Lifespan (target variable) from the input features.
o Selected relevant features: coolingRate, Nickel%, HeatTreatTime, Chromium%,
quenchTime, forgeTime, and partType.
2. Data Splitting:
o Split the data into 80% training and 20% testing subsets using a fixed random_state
to ensure reproducibility.
3. Categorical Encoding:
o Applied One-Hot Encoding to the partType feature to convert it into a numerical
format.
4. Feature Scaling:
o Standardized numerical features (coolingRate, Nickel%, etc.) using StandardScaler
to ensure uniform scaling for both models.
5. Handling Imbalance:
o The label distribution (0: 69.4%, 1: 30.6%) showed imbalance. Models like RFC and
GBC handle such scenarios well, but stratified sampling was employed during data
splitting to maintain label proportions.
Hyperparameter Tuning Framework
To optimize model performance, GridSearchCV was employed with 5-fold cross-validation and
accuracy as the scoring metric. For the Gradient Boosting Classifier, key hyperparameters like
n_estimators (boosting stages), learning_rate (tree contribution), max_depth (tree depth), and
min_samples_split/min_samples_leaf (growth control) were tuned to balance bias and variance.
Similarly, for the Random Forest Classifier, n_estimators (number of trees), max_depth (tree depth),
min_samples_split/min_samples_leaf (splits and leaf nodes), and class_weight (imbalance handling)
were optimized. This process ensured the models were fine-tuned to achieve maximum performance
on preprocessed data.
Conclusion
The combination of Gradient Boosting Classifier and Random Forest Classifier ensures robust
classification of 1500_labels. Preprocessing steps, including feature scaling and encoding, prepared
the data for accurate and fair evaluation. The GridSearchCV tuning framework optimized
hyperparameters to balance bias and variance, ensuring that both models were well-suited for
predicting usable and defective parts.
4.3 Evaluation
To evaluate the chosen classification models, Gradient Boosting Classifier (GBC) and Random
Forest Classifier (RFC), rigorous experimentation was conducted. Each model underwent
hyperparameter tuning using GridSearchCV to optimize performance. Consistent preprocessing,
train-test splits, and evaluation metrics ensured parity between the models.
Experiments and Model Optimization
The experiments focused on optimizing hyperparameters for both Gradient Boosting Classifier (GBC)
and Random Forest Classifier (RFC) to achieve optimal performance. For GBC, parameters such as
the number of boosting stages (n_estimators = 100), learning rate (learning_rate = 0.2), maximum tree
depth (max_depth = 3), and minimum samples for node splits and leaves (min_samples_split = 2,
min_samples_leaf = 1) were fine-tuned. Similarly, RFC optimization involved increasing the number
of trees (n_estimators = 200), setting tree depth (max_depth = 10), and configuring node split
thresholds (min_samples_split = 5, min_samples_leaf = 1). Both models exhibited strong
performance post-tuning.
Model Performance Metrics
The models were evaluated using key metrics: accuracy (correct classifications), precision (true
positive ratio), recall (sensitivity), F1-score (precision-recall balance), and a confusion matrix
(classification visualization).
Hyperparameter Tuning Progression
1. For the Gradient Boosting Classifier, various combinations of hyperparameters were
explored to identify the optimal configuration. Initially, n_estimators was set to 50
with a learning_rate of 0.1 and a max_depth of 3, achieving a validation accuracy
of 94%. Increasing n_estimators to 100 and the learning_rate to 0.2 resulted in a
significant improvement, achieving 97% validation accuracy. Further increasing
n_estimators to 200 and max_depth to 5 provided a slightly lower validation
accuracy of 96%, indicating the best combination was n_estimators=100,
learning_rate=0.2, and max_depth=3.
2. For the Random Forest Classifier, the tuning began with n_estimators=100,
max_depth=10, and min_samples_split=2, yielding a validation accuracy of 95%.
Adjusting n_estimators to 200 and min_samples_split to 5 improved the
validation accuracy to 97%. Increasing max_depth to 20 with the same settings
provided 96% accuracy, suggesting the optimal configuration was
n_estimators=200, max_depth=10, and min_samples_split=5.
Part 4.4 – Critical Review
Strengths of the Methodology
The classification methodology effectively combined robust preprocessing, thoughtful feature
engineering, and hyperparameter tuning to optimize model performance. The use of
Gradient Boosting Classifier and Random Forest Classifier provided a balance of
high accuracy and interpretability. Consistent train-test splits and evaluation metrics ensured
fair comparisons, while the inclusion of regressor-predicted lifespan as an additional feature
enhanced classification performance. Both models demonstrated strong results, with
accuracies exceeding 97%.
The binary classification could be enhanced by exploring multi-class labels or clustering for
nuanced patterns, and SMOTE for imbalanced data handling. Faster optimization methods like
RandomizedSearchCV or Bayesian Optimization may improve tuning efficiency. Future work
could include advanced models like XGBoost or LightGBM and neural networks for capturing
complex relationships.
Part 5 – Conclusions
The experiments and analyses revealed that the Gradient Boosting Regressor and Classifier
excelled in their respective tasks, achieving high accuracy and minimal errors. The regressor
demonstrated precise lifespan predictions, with an R² score of 0.9684 and low error metrics,
outperforming the Random Forest Regressor. The classifiers achieved over 97% accuracy, with
Gradient Boosting providing superior recall, crucial for identifying usable parts. These findings
align with the initial data exploration, where features like Cooling Rate and Nickel% were
identified as key predictors, confirming the models’ ability to capture complex relationships in
the data.
Considering the business context, the Gradient Boosting Classifier is recommended for
deployment due to its actionable framework and high recall, reducing the risk of missing usable
components. While the regressor offers precise lifespan predictions, binary classification better
supports real-time decision-making on production lines, directly addressing the company’s
goal of improving production efficiency. Future work could include advanced models like
XGBoost or LightGBM to refine predictions further and address class imbalance for even
greater robustness. 
