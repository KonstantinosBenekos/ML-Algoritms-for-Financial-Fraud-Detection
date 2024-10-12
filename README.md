# ML-Algoritms-for-Financial-Fraud-Detection
Master Thesis


# Abstract

This here study explores the effectiveness of machine learning algorithms in
detecting credit card fraud. Specifically, the algorithms evaluated include random
forest, logistic regression and support vector machines. The dataset utilized was
obtained from Kaggle and includes data regarding 287,807 credit card transactions which were conducted by European card holders over a two day period in
September of 2013.

After following standard data prepossessing procedures we continue by addressing the issue of imbalance in the data by proposing a resampling technique
which entails under sampling the majority(non-fraud) class in order to create ten
distinct balanced subsets of data, each of which captures a different aspect of the
majority data. The models are then trained on each of the subsets separately and
the final prediction made is determined by majority voting. For comparison, we
also trained the models on the raw-unbalanced data in order for us to determine
the effects our resampling technique had on the models overall performance. We
evaluated the models both from a quantitative perspective leveraging metrics such
as accuracy, recall, precision, F1 scores and most importantly ROC-AUC scores
as well as in business context in the sense of how applicable/useful they would be
in a real world scenario.

The results revealed that while all models trained on the balanced subsets
performed fairly well in terms of accuracy, recall and ROC-AUC scores they were
all severely lacking in precision, something which in the fraud detection context
poses a major issue. For this, it was decided that the our resampling technique
negatively affected the results of our models as it made them too sensitive in
predicting the fraud class which proved to be a drawback when applied to real
world data. Next, when looking at the models trained on the unbalanced data we
observed that our best performer in term of ROC-AUC score, which since we are
talking about unbalanced classification, is our most important metric, was logistic
regression with a score of 99.06%. Furthermore, logistic regression brought to the
table the added advantage of transparency, in the decision making process, as well
as it not needing any prior assumption of the distribution of the data. For those
reason we concluded that logistic regression arose as the most well suited model
for fraud prediction in this particular data set.

Keywords: Credit card fraud, fraud detection, machine learning, logistic regression, random forest, support vector machines, data imbalanc

Load the data set: https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud
