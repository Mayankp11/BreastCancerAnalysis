# Breast Cancer Detection Project

This project focuses on the detection of breast cancer using the k-nearest neighbors (kNN) algorithm. The process involves several steps including data loading, model training, optimization, evaluation, and performance analysis. The project utilizes Python and various libraries for data manipulation, visualization, and machine learning.

## Tasks Completed

### 1. Import the Breast Cancer training and testing datasets

The Breast Cancer training and testing datasets were imported from the provided CSV files: 'data/BreastCancer_trn.csv' and 'data/BreastCancer_tst.csv'. These datasets contain attributes related to breast cancer patients and are used for training and evaluating the kNN model. Both the files are added.

### 2. Apply the kNN model to the training data

A kNN model was applied to the training data using the `KNeighborsClassifier` class from the scikit-learn library. The model was trained to classify breast cancer cases based on the provided attributes. The attributes X0-X8 were separated into X_trn and the class variable into y_trn.

### 3. Train and optimize the kNN model

To optimize the kNN model, the optimal number of neighbors was determined using the `GridSearchCV` class from scikit-learn. This involved training the model with different parameter values and selecting the best one based on cross-validation accuracy.

### 4. Plot the accuracy of the parameters

The accuracy of the kNN model for different parameter values was plotted using matplotlib. The `GridSearchCV` object's `cv_results_` attribute was utilized to compare the performance of different parameter values.

### 5. Graph the confusion matrix

The accuracy of the trained kNN model on the test set was evaluated by plotting the confusion matrix using the `plot_confusion_matrix` function from scikit-learn. This matrix provides insights into true positives, true negatives, false positives, and false negatives.

### 6. Calculate the overall accuracy of the model on the testing data

The overall accuracy of the kNN model on the testing dataset was calculated to assess its performance. This metric indicates the model's ability to correctly classify breast cancer cases based on the provided attributes.

## Libraries Used

- matplotlib.pyplot: For data visualization, including plotting accuracy and confusion matrices.
- seaborn: For enhancing the aesthetics of visualizations.
- pandas: For data manipulation and handling the Breast Cancer datasets.
- scikit-learn:
  - KNeighborsClassifier: For implementing the kNN algorithm.
  - GridSearchCV: For optimizing the model parameters.
  - plot_confusion_matrix: For plotting the confusion matrix.
  
Additionally, `%matplotlib inline` magic command was used to display plots inline in Jupyter Notebook.

## Execution

To replicate the project's results, ensure all required libraries are installed in your Python environment. You can then run the provided code in a Jupyter Notebook or Python script, following the instructions provided in the project's README or notebook file. Make sure to use the provided datasets and adjust file paths accordingly if necessary.

## References

- Documentation for libraries:
  - [Matplotlib](https://matplotlib.org/stable/contents.html)
  - [Seaborn](https://seaborn.pydata.org/)
  - [Pandas](https://pandas.pydata.org/docs/)
  - [scikit-learn](https://scikit-learn.org/stable/documentation.html)

This project was completed as part of a machine learning course and serves as an example of applying the kNN algorithm for breast cancer detection.
