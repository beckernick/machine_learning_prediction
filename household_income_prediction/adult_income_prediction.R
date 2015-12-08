# Predicting whether household income is greater than $50,000 or less than $50,000

library(readr)
library(plyr)
library(dplyr)
library(randomForest)
#library(doMC) 
library(ggplot2)
library(caret)
library(corrplot)
library(stringr)
library(pROC)

adult = read.csv("/users/nickbecker/downloads/adult_data_uci.csv", header=FALSE,
                 stringsAsFactors = FALSE)
head(adult)
colnames(adult) = c("age", "workclass", "fnlwgt", "education", "education_num", "marital_status",
                    "occupation", "relationship", "race", "sex", "capital_gain", "capital_loss",
                    "hours_per_week", "native_country", "income_bucket")

head(adult)
str(adult)

adult$income_bucket = gsub("<=50K", "under_50k", adult$income_bucket)
adult$income_bucket = gsub(">50K", "over_50k", adult$income_bucket)

for(i in colnames(adult)) {
  #print(i)
  if(class(adult[[i]]) == "character") {
    adult[[i]] = gsub("-", "_", adult[[i]])
    adult[[i]] = gsub("\\?", "unknown", adult[[i]])
    adult[[i]] = str_trim(adult[[i]], side = "left")
    adult[[i]] = as.factor(adult[[i]])
  }  
}

str(adult)
summary(adult)

# Let's create dummy variables for the factors
dummies = dummyVars(income_bucket ~ ., data = adult)
features = predict(dummies, newdata = adult)
adult_1 = as.data.frame(cbind(features, income_bucket = adult$income_bucket))
head(adult_1) # over 50K = 1, under 50K = 2

# We'll write this one to a CSV file for python sklearn
write.csv(adult_1, "/Users/nickbecker/Documents/R Workspace/adult_income_dummied.csv", row.names = FALSE)


table(adult$income_bucket)

# Splitting into a train and validation set
set.seed(12)
inTrain = createDataPartition(y = adult$income_bucket,
                              p = 0.8,
                              list = FALSE)
training = adult[inTrain,]
validation = adult[-inTrain,]


# model parameters
fitControl = trainControl(method = "cv",
                          number = 3,
                          #summaryFunction=twoClassSummary,
                          #classProbs=TRUE,
                          verboseIter = TRUE)

gbmGrid = expand.grid(interaction.depth = c(1, 3, 5, 7),
                      n.trees = 300,
                      shrinkage = c(0.1),
                      n.minobsinnode = 20)

# fit a boosted tree model via gbm
set.seed(12)
gbmFit1 = train(x=training[, c(-15)],
                y=training[, 15],
                method = "gbm",
                #metric = "ROC",
                trControl = fitControl,
                tuneGrid = gbmGrid,
                verbose = TRUE)
gbmFit1
varImp(gbmFit1)
plot(gbmFit1)

gbm_preds = predict(gbmFit1, select(validation, -income_bucket))
confusionMatrix(gbm_preds, validation$income_bucket)

# fit a random forest model
set.seed(12)
rfGrid = expand.grid(mtry = c(2:14))
rfFit1 = train(x=training[, -15],
                y=training[, 15],
                method = "rf",
                #metric = "ROC",
                #tuneGrid = rfGrid,
                trControl = fitControl)
rfFit1
plot(rfFit1)


rf_preds = predict(rfFit1, select(validation, -income_bucket))
confusionMatrix(rf_preds, validation$income_bucket)




