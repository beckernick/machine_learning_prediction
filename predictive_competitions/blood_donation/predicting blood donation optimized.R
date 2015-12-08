# Predicting Blood Donations #
# Using caret

library(caret)
library(plyr)
library(dplyr)

train = read.csv("/Users/nickbecker/Downloads/blood training.csv", header = TRUE)
test = read.csv("/Users/nickbecker/Downloads/blood test.csv", header = TRUE)


head(train)
head(test)
colnames(train) = c("id", "recency", "frequency", "amount", "time", "donated")
colnames(test) = c("id", "recency", "frequency", "amount", "time")

summary(train) # none missing
str(train)

train = mutate(train, donations_per = frequency/time)
test = mutate(test, donations_per = frequency/time)

train$donated[train$donated == 1] <- "Yes"
train$donated[train$donated == 0] <- "No"

str(train)
train$donated = factor(train$donated)

# try partial least squares
ctrl = trainControl(method = "repeatedcv",
                    repeats = 3)

set.seed(12)
plsFit = train(donated ~ .,
               data = train,
               method = "pls",
               tuneLength = 15,
               trControl = ctrl,
               preProc = c("center", "scale"))
plsFit


# Gradient boosting
gbmControl = trainControl(method = "repeatedcv",
                          number = 10,
                          repeats = 10,
                          classProbs = TRUE)

gbmGrid <-  expand.grid(interaction.depth = c(1, 5, 9),
                        n.trees = (1:30)*20,
                        shrinkage = 0.1,
                        n.minobsinnode = 20)
set.seed(12)
gbmFit1 = train(donated ~ ., data = train,
                method = "gbm",
                trControl = gbmControl,
                verbose = FALSE,
                tuneGrid = gbmGrid,
                metric = "Accuracy",
                preProc = c("center", "scale"))
gbmFit1

# make predictions on test
preds_gbm = predict(gbmFit1, newdata = test, type = "prob")
head(preds_gbm)
preds_gbm[,2]
# submission
submission = data.frame(test$id, preds_gbm[,2])
write.csv(submission, "/users/nickbecker/documents/r workspace/caret_gbm.csv",
          row.names = FALSE)

# Random forest model
rfControl = trainControl(method = "repeatedcv",
                          number = 10,
                          repeats = 10,
                          classProbs = TRUE)

set.seed(12)
rfFit1 = train(donated ~ ., data = train,
      method = "rf",
      trControl = rfControl,
      metric = "Accuracy")

rfFit1
preds_rf = predict(rfFit1, newdata = test, type = "prob")
head(preds_rf)



# Neural network model
nnControl = trainControl(method = "repeatedcv",
                          number = 10,
                          repeats = 10,
                          classProbs = TRUE)
neuralGrid <- expand.grid(decay = c(0.5, 0.1, .01), size = c(4, 5, 6, 7, 8),
                          bag = FALSE)

set.seed(12)
nnFit1 = train(donated ~ ., data = train,
               method = "avNNet",
               trControl = nnControl,
               trace = FALSE,
               #maxit = 100,
               preProc = c("center", "scale"),
               tuneGrid = neuralGrid)
nnFit1 # "nnet": .7948848 with size 8 and decay .1 (standard features)
# "avNNet": .7959623 with size 8, decay .1, bag = FALSE (standard features)
# "avNNet": .7997613 with size 6, decay .1, bag = FALSE (std. + donations_per)

# make predictions on test
preds_nn = predict(nnFit1, newdata = test, type = "prob")
head(preds_nn)
preds_nn[,2]
# submission
submission = data.frame(test$id, preds_nn[,2])
write.csv(submission, "/users/nickbecker/documents/r workspace/caret_avNNet.csv",
          row.names = FALSE)




# ensemble submissions by average
submission_rf_gbm = data.frame(test$id, (preds[,2] + preds_rf[,2])/2)
write.csv(submission_rf_gbm, "/users/nickbecker/documents/r workspace/caret_gbm_rf_ensemble.csv",
          row.names = FALSE)

submission_rf_gbm_nn = data.frame(test$id, (preds_gbm[,2] + preds_rf[,2] + preds_nn[,2])/3)
write.csv(submission_rf_gbm_nn, "/users/nickbecker/documents/r workspace/caret_gbm_rf_avNNet_ensemble.csv",
          row.names = FALSE)



























































