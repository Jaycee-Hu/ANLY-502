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

