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
