# Project Information
Group name: Teletubbies

Group member: Kiwi Yu, Xin Hu, Yachen Li, Yihan Zhou

Project content: Amazon review data predition

# About data
Amazon Customer Reviews Dataset: https://s3.amazonaws.com/amazon-reviews-pds/readme.html

Description: The dataset collects reviews written in the Amazon.com marketplace and associated metadata from 1995 until 2015. 
(130M+ customer reviews)

# Code File
- [`project v2.ipynb`](https://github.com/ikiwisline/ANLY-502/blob/master/project%20v2.ipynb)

# Executive Summary

In the era of Big Data and Social Computing, the role of customer reviews and ratings from Amazon can be instrumental in predicting the success of businesses. It would be meaningful if we could predict star ratings from text reviews independently since free-text reviews are hard for computer systems to learn and analyze. Hence, our group project focuses on research how predictive are text reviews for star ratings. 

Our plan is to clean the review data using regular expression tokenizer, NLTK English Language stop words measure, and use Count Vectors as word embedding.  Our goal is to predict the review rating, which ranges from 1 to 5, by using several machine learning methods we have learned, and investigate which supervised machine learning methods are best suited to solve it. We expect to achieve the 60% prediction accuracy given the number of five classes. 

# 1.Introduction
The data we used is a collection of reviews written in Amazon.com marketplace and associated metadata from 1995 until 2015. It includes more than 160 million customer reviews, and it’s stored in S3 with both tsv and parquet formats. Each line of this data corresponds to an individual review. The features include marketplace, customer_id, review_id, product_id, product_parent, product_title, product_category, star_rating, helpful_votes, total_votes, vine, verified_purchase, review_headline, review_body, review_date. The product category of reviews contains PC, Kitchen, Home, Wireless, Video, Digital_Video_Games, Sports, Grocery, etc.

# 2.Methods
## 2.1 Preparation & EDA
Before diving into training machine learning methods, we first did some exploratory data analysis. We used filters to check the number of reviews for each marketplace, and decided to focus on the US market, which contains the most of the reviews out of all. Next, a time series has been plotted to observe the growth rate of reviews from 2006 to 2015. It’s clear that there’s a rapid growth of reviews after 2013.

![Figure 1](https://github.com/ikiwisline/ANLY-502/blob/master/images/Figure%201.png)

Figure 1:Ten-year reviews plot


Then, in order to investigate the distribution of different ratings stars in this data, a pie chart was generated. We see that more than half of the reviews are 5 stars, so the data is imbalanced.  In our case of learning imbalanced data, the majority classes might be of our great interest. It’s desirable to have a classifier that gives high prediction accuracy over the majority class, while maintaining reasonable accuracy for the minority classes, therefore, we will leave it as it is.

![Figure 2](https://github.com/ikiwisline/ANLY-502/blob/master/images/Figure%202.png)

Figure 2: Distribution of star ratings for books


Given the size of this dataset and the number of different categories, we decided to focus on books, which has the largest number of customer reviews compared to others.

![Figure 3](https://github.com/ikiwisline/ANLY-502/blob/master/images/Figure%203.png)

Figure 3: Review counts for product_category


Next, some analysis was made on the book category by using spark sql. We took a look at the most review amount books from 2005 to 2015, and we also joined the table of books with top review and the table of books with top ratings to select top books with top 100 review amount. The output is shown below.

![Table 1](https://github.com/ikiwisline/ANLY-502/blob/master/images/Table%201.png)

Table 1: Top rated books with top 100 review amount


## 2.2 Models & Techniques
After doing some exploratory data analysis, we dived into training machine learning methods. By observing the dataset, it is clear that 5 star-rating always corresponds with “great” review words. So we think there is a relationship between star-rating and review words that can be predicted by using machine learning models. And then, different machine learning models were attempted during the following steps.

![Table 2](https://github.com/ikiwisline/ANLY-502/blob/master/images/Table%202.png)

Table 2: Reviews’ characteristic of star-rating and words


First, we did text representation, given that the classifiers and learning algorithms can not directly process the text documents in their original form, as most of them expect numerical feature vectors with a fixed size rather than the raw text documents with variable length. Thus, we tokenized the text to break the text up to words, and used StopWordsRemover to get off the stop words in English, which avoids the noisy features. Then, CountVectorizer was used to transform the text to matrix form. All of the process has been done using the existing SparkML and Pipeline library.

Next step is data splitting, 75% of data is split as a training dataset and 25% of data is assigned as a testing dataset. Then some machine learning models are built, such as logistic regression, naïve bayes, random forest, decision tree, svm and so on. Logistic regression, naïve bayes, random forest and decision tree models were finished with their prediction accuracy. Among these four models, it is easy to find that naïve bayes fits the amazon dataset best, the accuracy is 55.1%.

Given that naïve bayes have the highest prediction accuracy, we tried to tune this model by using ParamGrid and cross validation for parameters in naïve bayes. However, after tuning, the prediction accuracy only increases 0.001%.

![Table 3](https://github.com/ikiwisline/ANLY-502/blob/master/images/Table%203.png)

Table 3: Model Accuracy

However, other models are difficult to fit when doing this project. For svm and neural network models, there was such a long waiting time of more than ten hours with the largest number of master and core, but the model still can’t fit. After thinking about reasons, we think that it would be so hard to apply some models because the data size is massive. We did not use binary and gradient boost models, because the dataset features are not appropriate for these models.


# 3.Conclusion 
The machine learning algorithms increase the success rate for sentiment analysis. Among the models used, the Naive Bayes performs best with the accuracy of 55.1%. By the increase in importance for the Internet, the significance of sentiment analysis increased to interpret huge data from day to day.

For business, they can understand their customers’ opinions and needs, make better decisions and also improve the quality of items. Traditionally businesses relied on surveys, workshops and focused on groups to gain insight into their customers' feelings, but today with modern technology, they are able to harness the power of machine learning to extract meaning from text and dive into opinions of  customers and see why someone might bounce to a competitor or prefer their products.

Moving forward, sentiment analysis is finding a place in marketing and predictive analysis.

# 4.Future Work
A. Adding features: For n-gram tfidf terms, NMF results　　

a)Probably the good starting point is playing with sklearn tfidfvectorizer and using different n-gram settings and running NMF with that tfidf-matrix. 　　

b)Another option is to use textacy (textacy.extract.ngrams).　

c)Run this code with Spark. Apache Spark should give us faster computation time when preprocessing data.

B. Reducing features:

a)We think we can reduce dimensionality and increase overall accuracy by reducing less important features from my dataset.

b)We hope to use a systematic way to find these features and put additional steps to drop these features rather than finding least important features and dropping the last 50 features or 100 features after running xgboost or random forest.

C. Working with bigger dataset/major category:

a)With more products and more reviews, we believe my nlp method works better.

b)We need better AWS instances or set up spark or hadoop to run cluster computing.

# 5.Reference
https://spark.apache.org/docs/2.3.0/ml-classification-regression.html#multinomial-logistic-regression
https://s3.amazonaws.com/amazon-reviews-pds/readme.html

# 6.Divisino of Labor
Xin: Model tuning, EDA, and Writeup;

Kiwi: Model building, Data Analysis, and Writeup;

Yihan: Data cleaning, EDA, Word Tokenizing, and Writeup;

Yachen: Model building, Model tuning, and Writeup
