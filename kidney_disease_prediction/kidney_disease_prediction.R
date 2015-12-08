# Predicting chronic kidney disease from patient attributes and data

library(RWeka)
library(caret)
library(dplyr)
library(corrplot)

kidney = read.arff("/users/nickbecker/downloads/Chronic_kidney_disease/chronic_kidney_disease_full.arff")

head(kidney)
str(kidney)
summary(kidney$rbc)

for (i in which(sapply(kidney, is.numeric))) {
  kidney[is.na(kidney[, i]), i] <- mean(kidney[, i],  na.rm = TRUE)
}

for (i in which(sapply(kidney, is.factor))) {
  replacement = names(which(table(kidney$appet) == max(table(kidney$appet))))
  kidney[is.na(kidney[, i]), i] <- replacement
}


kidney$su
is.na(kidney$su)

# Creating the train and test set
set.seed(12)
inTrain = createDataPartition(y = kidney$class,
                              p = 0.8,
                              list = FALSE)
str(inTrain)

training = kidney[inTrain,]
validation = kidney[-inTrain,]

head(training)


str(training)
summary(training$class)

##### Model building #####
fitControl = trainControl(method = "repeatedcv",
                          number = 5,
                          repeats = 5,
                          classProbs = TRUE) # 5-fold CV repeated 5 times

set.seed(12)
rfFit1 = train(class ~ ., data = training,
               method = "rf",
               trControl = fitControl)
rfFit1
plot(rfFit1)

glmFit1 = train(class ~ ., data = training,
               method = "glm",
               trControl = fitControl)
glmFit1





