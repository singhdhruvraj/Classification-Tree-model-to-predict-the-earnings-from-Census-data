# Read the dataset
census = read.csv("census.csv")
library(caTools)
set.seed(2000)

#Splitting data into training and testing set
spl = sample.split(census$over50k, SplitRatio=0.6)
train = subset(census, spl==TRUE)
test = subset(census, spl==FALSE)

#Logistics Regression model
incomeLog = glm(over50k ~ ., data=train, family="binomial")
summary(incomeLog)


#Finding the accuracy of the model on the testing set
incomePred = predict(incomeLog, newdata=test, type="response")
table(test$over50k, incomePred>0.5)
(9051+1886)/nrow(test)


# Finding the baseline accuracy for the testing set
table(test$over50k)
9713/(9713+3078)


# Finding the area-under-the-curve (AUC) for this model on the test set
library(ROCR)
ROCRpred = prediction(incomePred, test$over50k)
as.numeric(performance(ROCRpred, "auc")@y.values)



# splits does the tree have in total
library(rpart)
library(rpart.plot)
CARTmodel = rpart(over50k ~ ., data=train)
prp(CARTmodel)


# the accuracy of the model on the testing set
incomePred_2 = predict(CARTmodel, newdata=test, type="class")
table(test$over50k, incomePred_2)
(9243+1596)/nrow(test)

incomePred_2 = predict(CARTmodel, newdata=test)
ROCRpred_2 = prediction(incomePred_2[,2], test$over50k)
# ROC of CART model
perf_2 = performance(ROCRpred_2, "tpr", "fpr")
# ROC of logistic regression model
plot(perf_2)
perf = performance(ROCRpred, "tpr", "fpr")
plot(perf)



# the AUC of the CART model on the test set?
as.numeric(performance(ROCRpred_2, "auc")@y.values)



# the accuracy of the model on the test set, using a threshold of 0.5
set.seed(1)
trainSmall = train[sample(nrow(train), 2000), ]
library(randomForest)
incomeForest = randomForest(over50k ~ ., data=trainSmall)
incomePred_3 = predict(incomeForest, newdata=test)
table(test$over50k, incomePred_3)
(9614+1050)/(9614+99+2028+1050)

vu = varUsed(incomeForest, count=TRUE)
vusorted = sort(vu, decreasing = FALSE, index.return = TRUE)
dotchart(vusorted$x, names(incomeForest$forest$xlevels[vusorted$ix]))

varImpPlot(incomeForest)



# ELECTING CP BY CROSS-VALIDATION

library(caret)
library(e1071)
set.seed(2)
tr.control = trainControl(method = "cv", number = 10)
cartGrid = expand.grid( .cp = seq(0.002,0.1,0.002))
tr = train(over50k ~ ., data = train, method = "rpart", trControl = tr.control, tuneGrid = cartGrid)
tr


# What is the prediction accuracy on the test set?
CARTmodel2 = rpart(over50k ~ ., data=train, cp=0.002)
incomePred_3 = predict(CARTmodel2, newdata=test, type="class")
table(test$over50k, incomePred_3)
(9178+1838)/nrow(test)


# How many splits are there?
prp(CARTmodel2)