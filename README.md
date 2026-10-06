KDD Project: Math Problem Topic Classifier

A Knowledge Discovery in Databases (KDD) project on the MATH + GSM8K problem dataset. It predicts the topic of a math problem (Algebra, Geometry, Number Theory, etc.) from its text.

KDD steps: Selection → Preprocessing → Transformation (TF-IDF) → Data Mining (4 classifiers + KMeans) → Evaluation

Algorithms: Logistic Regression, Linear SVM, Naive Bayes, Random Forest (about 72-75% accuracy)

Dashboard pages
Overview: dataset size, topic and level distribution
Model Comparison: accuracy, precision, recall, F1, per-topic report, confusion matrix
Predict a Topic: type a problem and get the predicted topic
Top Words per Topic: words each topic is linked to
Clustering: KMeans vs the real topics
