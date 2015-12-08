# Predicting telecommunications network disruptions

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
library(tidyr)


train_raw = read.csv("/Users/nickbecker/Documents/R Workspace/telestra_network_disruptions/train.csv")
test_raw = read.csv("/Users/nickbecker/Documents/R Workspace/telestra_network_disruptions/test.csv")
severity_raw = read.csv("/Users/nickbecker/Documents/R Workspace/telestra_network_disruptions/severity_type.csv")
log_feature_raw = read.csv("/Users/nickbecker/Documents/R Workspace/telestra_network_disruptions/log_feature.csv")
event_type_raw = read.csv("/Users/nickbecker/Documents/R Workspace/telestra_network_disruptions/event_type.csv")
resource_type_raw = read.csv("/Users/nickbecker/Documents/R Workspace/telestra_network_disruptions/resource_type.csv")

# Factor the target (fault severity) and relevel
train_raw = mutate(train_raw, fault_severity = as.factor(fault_severity))
head(train_raw, 50)
str(train_raw)
levels(train_raw$fault_severity) = c("zero", "one", "two")
head(train_raw, 50)

# log features
log_feature_raw = arrange(log_feature_raw, id)
head(log_feature_raw)
summary(log_feature_raw)

log_feature_wide = log_feature_raw %>% spread(log_feature, volume)
head(log_feature_wide)
log_feature_wide[is.na(log_feature_wide)] = 0
head(log_feature_wide)


# event type (these are dummy variables with multiple ids)
event_type_raw = arrange(event_type_raw, id)
head(event_type_raw)
str(event_type_raw)
event_type_wide = event_type_raw %>% spread(event_type, event_type)
head(event_type_wide)
event_type_wide = as.data.frame(sapply(event_type_wide, as.numeric))
event_type_wide[is.na(event_type_wide)] = 0
event_type_wide[,c(2:54)][event_type_wide[,c(2:54)] > 0] = 1
str(event_type_wide)
glimpse(event_type_wide)



# resource type (these are dummy variables with multiple ids)
resource_type_raw = arrange(resource_type_raw, id)
head(resource_type_raw)
summary(resource_type_raw)
resource_type_wide = resource_type_raw %>% spread(resource_type, resource_type)

resource_type_wide = as.data.frame(sapply(resource_type_wide, as.numeric))
resource_type_wide[is.na(resource_type_wide)] = 0
resource_type_wide[,c(2:11)][resource_type_wide[,c(2:11)] > 0] = 1
str(resource_type_wide)
glimpse(resource_type_wide)

# severity type (dummies, unique id)
severity_raw = arrange(severity_raw, id)
head(severity_raw)
dummies = dummyVars(id ~ ., data = severity_raw)
features = predict(dummies, newdata = severity_raw)
severity_wide = as.data.frame(cbind(id = severity_raw$id, features))
head(severity_wide) 

train = inner_join(train_raw, log_feature_wide, by = "id")
train = inner_join(train, event_type_wide, by = "id")
train = inner_join(train, resource_type_wide, by = "id")
train = inner_join(train, severity_raw, by = "id") # perfect joins


test = inner_join(test_raw, log_feature_wide, by = "id")
test = inner_join(test, event_type_wide, by = "id")
test = inner_join(test, resource_type_wide, by = "id")
test = inner_join(test, severity_raw, by = "id") # perfect joins


colnames(train) = gsub(" ", "_", colnames(train))
colnames(test) = gsub(" ", "_", colnames(test))


nearZeroVar(train)

# Splitting into a train and validation set
set.seed(12)
inTrain = createDataPartition(y = train$fault_severity,
                              p = 0.8,
                              list = FALSE)
training = train[inTrain,]
validation = train[-inTrain,]


# model parameters
fitControl = trainControl(method = "cv",
                          number = 5,
                          #summaryFunction=twoClassSummary,
                          classProbs=TRUE,
                          verboseIter = TRUE)

gbmGrid = expand.grid(interaction.depth = c(1,4,7),
                      n.trees = 250,
                      shrinkage = c(0.1),
                      n.minobsinnode = c(10,20,30))

# fit a boosted tree model via gbm
set.seed(12)
gbmFit1 = train(x=training[, -c(1,3)],
                y=training[, 3],
                method = "gbm",
                #metric = "ROC",
                trControl = fitControl,
                tuneGrid = gbmGrid,
                verbose = TRUE)
gbmFit1
#The final values used for the model were n.trees = 250, interaction.depth = 7, shrinkage = 0.1
#and n.minobsinnode = 10.

gbm_preds = predict(gbmFit1, validation[, -c(1,3)], type = "prob")
confusionMatrix(gbm_preds, validation$fault_severity)

head(gbm_preds)


# GBM on the entire training data using the previous final values
fitControl_2 = trainControl(method = "none",
                          classProbs=TRUE,
                          verboseIter = TRUE)

gbmGrid_2 = expand.grid(interaction.depth = c(7),
                      n.trees = 250,
                      shrinkage = c(0.1),
                      n.minobsinnode = c(10))
set.seed(12)
gbmFit2 = train(x=train[, -c(1,3)],
                y=train[, 3],
                method = "gbm",
                #metric = "ROC",
                trControl = fitControl_2,
                tuneGrid = gbmGrid_2,
                verbose = TRUE)
gbmFit2

gbm_preds = predict(gbmFit2, test[, -c(1)], type = "prob")
# 0.75034 multiclass loss on Kaggle

head(gbm_preds)




# fit a random forest model
set.seed(12)
rfControl = trainControl(method = "none",
                         #summaryFunction=twoClassSummary,
                         classProbs=TRUE,
                         verboseIter = TRUE)
rfGrid = expand.grid(mtry = 22)

rfFit1 = train(x=training[, -c(1,2,3)],
               y=training[, 3],
               method = "rf",
               #metric = "ROC",
               tuneGrid = rfGrid,
               trControl = rfControl)
rfFit1
plot(rfFit1)


rf_preds = predict(rfFit1, validation[, -c(1,2,3)], type = "prob")
confusionMatrix(rf_preds, validation$fault_severity)


### GBM Predictions

submission_gbm1 = data.frame(id = test$id, gbm_preds)
colnames(submission_gbm1) = c("id", "predict_0",  "predict_1",	"predict_2")
head(submission_gbm1)
write.csv(submission_gbm1, "/users/nickbecker/documents/r workspace/telestra_network_disruptions/gbm_disruptions.csv",
          row.names = FALSE)


### RF Predictions
rf_preds = predict(rfFit1, test[, -c(1,2)], type = "prob")
head(rf_preds)
submission_rf1 = data.frame(test$id, rf_preds)
colnames(submission_rf1) = c("id", "predict_0",	"predict_1",	"predict_2")
head(submission_rf1)
write.csv(submission_rf1, "/users/nickbecker/documents/r workspace/telestra_network_disruptions/rf_disruptions.csv",
          row.names = FALSE)





























