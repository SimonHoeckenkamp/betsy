# betsy

This project tries to use data scraped from kicker.de (a german website for different sport leagues - mostly football) to predict the outcome of the next match day. 
The whole project is written in python.

This project's aim is to get familiar with web scraping as a data mining source and to test some theories, especially with regard to different ML approaches and portfolio management. 

## Data scraping

Currently the features of the model consist of the grades (given by kicker.de) for each individual player and the information, whether it was a home-match or not. 
The targets are the results of the individual matches (1: won, 0: even, -1: lost). For simplicities sake the data is stored in csv files to minimize the requests of the website.
For this approach Beautiful Soup is used.

The current theory states that the outcome of a match is to same grade based on the outcomes of some previous matches. 

## Data handling and  manipulation

The model has the possibility to adjust the number of days and the number of players (per game) which should be investigated for the ML algorithm. Numpy is mostly used 
for data handling during the rearrangement phase. Each data point is constructed as described before. The features of each point consist therefore of player grades and the 
information whether it was a home-match for a variable number of days. The label describes the outcome of the match.

The outcome of the data preparation phase is time series data format. The time steps are the match days which are not neccessarily in constant intervals.

## Model

The easiest and fastest way to test several models was to use the Sikit Learn library. One logistic regression model and one simple neural network is used to classify 
the data. 

The dimension reduction using principal component analysis led not to an improvement. Which might change with a higher number of features (eg. weather data, tipping rates). 

The model tuning phase should consider the usage of Pytorch for more detailed parameter tuning (eg. the no. of nodes and layers or the incluence of the activation function).

The current score of the model (both) is approximately 40%. Which was so far not enough to get profit out of it...

## Output and betting recommendation

For sorting and merging predicted data points Pandas is used. The print function makes it especially comfortable to work with (in comparison with list structures and 
specific libraries). 

The constant of capital and the cash rate indicates that this program can be used to recommend bets on the next match day. But take care if you use it on your own! I am not 
responsible for any losses or harms which might result from taking the results too seriously. 




