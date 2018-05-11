# SVM Digit Recognization Assignment

# All coding are done in MacOS - Sierra 
# Author - Anindya Chatterjee

# Install necassary packages

install.packages("DescTools")
install.packages("caret")
install.packages("doParallel")
install.packages("ggplot2")
install.packages("kernlab")
install.packages("dplyr")
install.packages("readr")
install.packages("gridExtra")
install.packages("caTools")

library(DescTools)
library(caret) 
library(doParallel) 
library(ggplot2) 
library(kernlab)
library(dplyr) 
library(readr) 
library(gridExtra)
library(caTools)


# Since we are already provided with train and test data separately, lets load them in R
train_data<-read.delim("mnist_train.csv",sep=",",stringsAsFactors = FALSE, header = FALSE)
test_data<- read.delim("mnist_test.csv",sep=",",stringsAsFactors = FALSE, header = FALSE)

dim(train_data)
dim(test_data)
# Train data has 60000 rows and 785 attributes
# Test data has 10000 rows and 785 attributes

# Checking the structure of the data set
str(train_data)
str(test_data)

# The first column is the target variable
# Cross verifying by checking the levels using plots

# Renaming the V1 columns with "Number"
colnames(train_data)[1] <- "Number"
colnames(test_data)[1] <- "Number"

# Plotting the distribution of train and test data sets

ggplot(train_data,aes(x = factor(Number),fill=factor(Number)))+geom_bar() 
ggplot(test_data,aes(x = factor(Number),fill=factor(Number)))+geom_bar() 

# Adding an extra column before merging the data sets and performing quality checks
train_data$set<-"train"
test_data$set<-"test"

# Combining train and test data together
complete_data_set<-rbind(train_data,test_data)
dim(complete_data_set)
# Complete data set has 70000 observations with 785 attributes

# Check for NA or missing values
sapply(complete_data_set, function(x) sum(is.na(x))) # No NA values

# Checking for same values for all data points in the complete data set
column_names_same_values <- which(sapply(complete_data_set,function(x) length(unique(x)) == 1))
length(column_names_same_values) # 65 columns 
column_names_same_values

# Removing the above found attributes
# complete_data_set_2<-complete_data_set[-column_names_same_values]
complete_data_set_2 <- complete_data_set %>% select(-column_names_same_values)


# The data set is ready for classification
#Filtering them

train_final <- complete_data_set_2 %>% filter(set == "train") %>% select(-set)
test_final <- complete_data_set_2 %>% filter(set == "test") %>% select(-set)

# Converting the contract to factor to enable regression
train_final$Number<-as.factor(train_final$Number)
test_final$Number<-as.factor(test_final$Number)

# As mentioned sampling the data as operations on this large data set would take a huge computational time
# Sampling the data into 2 parts
# Case 1 - working on 2/3 rd of the data
# Case 2 - working on 1/3 rd of the data
set.seed(100)

case1_train_sample <-sample.split(train_final$Number,SplitRatio=0.6667) 
case1_data_set<-train_final[case1_train_sample,]

case2_train_sample <-sample.split(train_final$Number,SplitRatio=0.3333) 
case2_data_set<-train_final[case2_train_sample,]

# Checking for the distribution of the final data sets as done earlier to check if the sampling is proper
ggplot(case1_data_set,aes(x = factor(Number),fill=factor(Number)))+geom_bar() 
ggplot(case2_data_set,aes(x = factor(Number),fill=factor(Number)))+geom_bar() 

# The plot seems quite even and the data sets are evenly distributed so , we can procees with the SVM
# We will take case2_data_set as the distribution is even and computation will be faster


# Model building and evaluation

# Using Linear Kernel
linear_model <- ksvm(Number~ ., data = case2_data_set, scale = FALSE, kernel = "vanilladot")
linear_model
# parameter : cost C = 1 
# Number of Support Vectors : 4388

# Evaluating the model on the test data set
evaluation <- predict(linear_model, test_final)

#confusion matrix - Linear Kernel - with the test_final data set
confusionMatrix(evaluation,test_final$Number)

# The overall statistics seems good
# Accuracy : 0.9132
# Sensitivity is highest for Class 1 of 0.9859 and lowest for class 8 of 0.8265
# Specificity is very good for all classes , highest for class 6 of 0.9958 and lowest for class 3 of 0.9820


# Lets check the performance of the model using RBF kernel
# RBF kernel
rbf_model <- ksvm(Number~ ., data = case2_data_set, scale = FALSE, kernel = "rbfdot")
rbf_model

# Gaussian Radial Basis kernel function. 
# Hyperparameter : sigma =  1.63691367429906e-07 

# Number of Support Vectors : 6018

evaluation_rbf<- predict(rbf_model, test_final)

#confusion matrix - RBF Kernel
confusionMatrix(evaluation_rbf,test_final$Number)

# The overall statistics seems to be better than the linear model
# Accuracy : 0.9673
# Sensitivity is highest for Class 1 of 0.9894 and lowest for class 9 of 0.9386
# Specificity is very good for all classes , highest for class 1 of 0.9977 and lowest for class 3 of 0.9947

# Considering the accuracy, sensitivity and specificity the RBF model is performing better
# We would do a cross validation on this to see if we get a better 
# performance than the default one


# Hyperparameter tuning and Cross Validation

trainControl <- trainControl(method="cv", number=5)

metric <- "Accuracy"

set.seed(10)
grid <- expand.grid(.sigma=c(0.63e-7,1.63e-7, 2.63e-7), .C=c(-1,0,1,2,3) )

fit.svm <- train(Number~., data=case2_data_set, method="svmRadial", metric=metric, 
                 tuneGrid=grid, trControl=trainControl)

# This model takes a lot of time to run
print(fit.svm)

# There seems to be like little non-linearity in the dataset
# So changing the C and sigma values to get better statistic

rbf_model_changes_1 <- ksvm(Number~.,data=case2_data_set,
                              kernel="rbfdot",
                              scale=FALSE,
                              C=2,
                              kpar=list(sigma=1.63e-7))
rbf_model_changes_1

# Lets check the accuracy of the model

evaluation_rbf_model_changes_1 <- predict(rbf_model_changes_1,test_final)
confusionMatrix(evaluation_rbf_model_changes_1,test_final$Number)

# Accuracy : 0.9716
# Sensitivity is highest for Class 0 of 0.9908 and lowest for class 9 of 0.9485
# Specificity is very good for all classes , highest for class 1 of 0.9983 and lowest for class 3 of 0.9960


# Lets create another model varying the C and sigma values

rbf_model_changes_2 <- ksvm(Number~.,data=case2_data_set,
                            kernel="rbfdot",
                            scale=FALSE,
                            C=3,
                            kpar=list(sigma=2.63e-7))
rbf_model_changes_2

# Lets check the accuracy of the model

evaluation_rbf_model_changes_2 <- predict(rbf_model_changes_2,test_final)
confusionMatrix(evaluation_rbf_model_changes_2,test_final$Number)

# Accuracy : 0.9777
# Sensitivity is highest for Class 0 of 0.9918 and lowest for class 9 of 0.9604
# Specificity is very good for all classes , highest for class 1 of 0.9986 and lowest for class 9 of 0.9967

# Lets check for the accuracy for another model varying C and sigma values

rbf_model_changes_3 <- ksvm(Number~.,data=case2_data_set,
                            kernel="rbfdot",
                            scale=FALSE,
                            C=4,
                            kpar=list(sigma=2.73e-7))
rbf_model_changes_3

# Lets check the accuracy of the model

evaluation_rbf_model_changes_3 <- predict(rbf_model_changes_3,test_final)
confusionMatrix(evaluation_rbf_model_changes_3,test_final$Number)

# Accuracy : 0.9777  - No change in the accuracy        


# We will finalize the model 2 (rbf_model_changes_2) that we built
# Accuracy : 0.9777
# parameter : cost C = 3 
# Hyperparameter : sigma =  2.63e-07 
# Number of Support Vectors : 6318 

