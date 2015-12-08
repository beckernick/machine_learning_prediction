### predicting blood donation ###
library(randomForest)
library(dplyr)
library(ggplot2)
library(caret)
library(gbm)
library(Metrics)

train = read.csv("/Users/nickbecker/Downloads/blood training.csv", header = TRUE)
test = read.csv("/Users/nickbecker/Downloads/blood test.csv", header = TRUE)
head(train)
head(test)
colnames(train) = c("id", "recency", "frequency", "amount", "time", "donated")
colnames(test) = c("id", "recency", "frequency", "amount", "time")

# data exploration
summary(train) # none missing
str(train)
train$donated = factor(train$donated)

# lets try creating donations per month
head(train)
train = mutate(train, per.month = frequency / time)
test = mutate(test, per.month = frequency / time)

head(train)

# log amount
train = mutate(train, log.amount = log(amount))
head(train)
test = mutate(test, log.amount = log(amount))

# scale all variables
#train_scale = train[,c(2:5, 7)] %>% mutate_each(funs( as.numeric(scale(.) )))
#colnames(train_scale) = c("recency_scale", "frequency_scale", "amount_scale", "time_scale", "per.month_scale")

#train = cbind(train, train_scale)
head(train)
# split into train and validation
set.seed(100)
index = sample(1:2, size = nrow(train), replace = TRUE, prob = c(0.7, .3))
trainset = train[index == 1,]
validationset = train[index == 2,]

LogLoss <- function(actual, predicted, eps=0.00001) {
  predicted <- pmin(pmax(predicted, eps), 1-eps)
  -1/length(actual)*(sum(actual*log(predicted)+(1-actual)*log(1-predicted)))
}


# random forest
set.seed(100)
rf = randomForest(donated ~ recency + frequency + time, data = trainset)
rf
predictions = predict(rf, test, type = "prob")
head(predictions)
pred_rf = predictions[,2]

LogLoss(as.numeric(validationset$donated)-1, pred_rf) # .70188


## neural network
library(nnet)

set.seed(100)
blood.nnet = nnet(donated ~ recency + frequency + time + amount + log.amount, data = train,
                  size = 10, maxit = 2000)
names(blood.nnet)
predictions = predict(blood.nnet, test, type = "raw")
head(predictions, 50)
pred_nnet = predictions[,1]
LogLoss(as.numeric(validationset$donated)-1, pred_nnet) # .48149 with seed 100 and log amount
head(pred_nnet)

# combine rf and nnet
head(pred_nnet)
head(pred_rf)

pred_nnet_rf = data.frame(pred_nnet, pred_rf)
head(pred_nnet_rf)
pred_nnet_rf = mutate(pred_nnet_rf, avg = (pred_nnet + pred_rf)/2)
str(pred_nnet_rf)

preds_ensemble = pred_nnet_rf$avg
LogLoss(as.numeric(validationset$donated)-1, preds_ensemble) # .497

head(preds_ensemble)
submission = data.frame(test$id, preds_ensemble)
write.csv(submission, "/users/nickbecker/documents/r workspace/blood donation ensemble.csv")





##### deep neural net
library(h2o)
# initialize h2o
localh2o = h2o.init(ip = "localhost", port = 54321, startH2O = TRUE,
                    max_mem_size = '4g')
train.h2o = as.h2o(trainset, localh2o)
validation.h2o = as.h2o(validationset, localh2o)

model.dl = h2o.deeplearning(x = c("recency", "frequency", "time", "log.amount"), y = 6,
                            training_frame = train.h2o)
summary(model.dl)
predictions_dl = h2o.predict(model.dl, validation.h2o)
preds_dl = as.data.frame(predictions_dl)
preds_dl = preds_dl[,3]



