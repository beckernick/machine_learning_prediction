##################################################
#     Water pump Functionality Prediction        #
#               DrivenData Contest               #
##################################################
library(h2o)
library(dplyr)
library(caret)
library(ggplot2)
# data

setwd("/Users/nickbecker/Downloads/water pump prediction")

train_val = read.csv("/Users/nickbecker/Downloads/water pump prediction/training values.csv")
train_lab = read.csv("/Users/nickbecker/Downloads/water pump prediction/training labels.csv")
test = read.csv("/Users/nickbecker/Downloads/water pump prediction/test values.csv")

summary(train_lab)
train_val$label = train_lab$status_group
head(train_val)

# rename to train
train = train_val
str(train)

# remove IDs
train.id = select(train, 1)
test.id = select(test, 1)

train = select(train, -1)
test = select(test, -1)





####### investigating the data ########
colnames(train)
summary(train$label)
unique(train$date_recorded)[1]

# label probabilities
mean(train$label == "functional")
mean(train$label == "functional needs repair")
mean(train$label == "non functional")

colnames(train.5)
# does date affect label? conditional probabilities
a = numeric()
b = numeric()
c = numeric()
for (i in 1:12) {
  a[i] = mean(train.5$label[as.numeric(train.5$month) == i] == "functional")
  b[i] = mean(train.5$label[as.numeric(train.5$month) == i] == "functional needs repair")
  c[i] = mean(train.5$label[as.numeric(train.5$month) == i] == "non functional")
}

func = data.frame(month = factor(1:12), a, b, c)
head(func)
ggplot(data = func, aes(x = month, y = a)) + geom_point()

ggplot(data = train.5, aes(x = ward, y = label)) + geom_point()


# fixing case sensitivity
str(train)
train$installer = factor(tolower(train$installer))
train$scheme_name = factor(tolower(train$scheme_name))

str(train)
test$installer = factor(tolower(test$installer))
test$scheme_name = factor(tolower(test$scheme_name))

# remove num_private and recorded by
table(train$num_private)

train.2 = select(train, -num_private, -recorded_by)
test.2 = select(test, -num_private, -recorded_by)

# replace 0 with NA in amount_tsh
train.2a = train.2
train.2a$amount_tsh[train.2a$amount_tsh == 0] = NA
test.2a = test.2
test.2a$amount_tsh[test.2a$amount_tsh == 0] = NA

head(train.2a)
# replace missing scheme with NA
train.2a$scheme_name[train.2a$scheme_name == ""] = NA
test.2a$scheme_name[test.2a$scheme_name == ""] = NA
#train.2a$scheme_name[train.2a$scheme_name == "none"] = NA
#test.2a$scheme_name[test.2a$scheme_name == "none"] = NA

# impute amount_tsh with average amount_tsh
summary(train.2a$amount_tsh)
sort(table(train.2a$amount_tsh), decreasing = TRUE)[1:5]


# impute public meeting with most common permit by
train.2a$public_meeting[train.2a$public_meeting == ""] = "True"
test.2a$public_meeting[test.2a$public_meeting == ""] = "True"
summary(train.2a$public_meeting)

summary(train.2a)
str(train.2a)

# factor region code and disrict code
train.2a$region_code = factor(train.2a$region_code)
train.2a$district_code = factor(train.2a$district_code)
test.2a$region_code = factor(test.2a$region_code)
test.2a$district_code = factor(test.2a$district_code)

head(train.2a$region_code)
# amount_tsh
# so many 0's, must be missing
# try dropping all with missing value and seeing if prediction improves
str(train.2$amount_tsh)
table(train.2$amount_tsh)

train.3 = filter(train.2, amount_tsh != 0)
test.3 = filter(test.2, amount_tsh != 0)

# try separating dates to days, months, and years
#train.4 = train.3
train.4 = train.2a
train.4$day = factor(substr(train.4$date_recorded, 9, 10))
train.4$month = factor(substr(train.4$date_recorded, 6, 7))
train.4$year = factor(substr(train.4$date_recorded, 1, 4))

test.4 = test.2a
test.4$day = factor(substr(test.4$date_recorded, 9, 10))
test.4$month = factor(substr(test.4$date_recorded, 6, 7))
test.4$year = factor(substr(test.4$date_recorded, 1, 4))


table(train.4$day)
table(train.4$month)
table(train.4$year)

# ridiculous to have 2004 or 2002 when constructed after that
# train.4 = filter(train.4, year != 2002, year != 2004)
# test.4 = filter(test.4, year != 2004, year != 2002)

# remove old recorded_date
train.5 = select(train.4, -date_recorded)
test.5 = select(test.4, -date_recorded)
summary(train.5)


#### randomize train and validation sample
set.seed(100)

index1 = sample(1:2, nrow(train), replace = TRUE, prob = c(0.7, 0.3))
train.1.train = train[index1 == 1, ]
validation.1.train = train[index1 == 2, ]

index2 = sample(1:2, nrow(train.2), replace = TRUE, prob = c(0.7, 0.3))
train.2.train = train.2[index2 == 1, ]
validation.2.train = train.2[index2 == 2, ]

train.2a.train = train.2a[index2 == 1, ]
validation.2a.train = train.2a[index2 == 2, ]


index3 = sample(1:2, nrow(train.3), replace = TRUE, prob = c(0.7, 0.3))
train.3.train = train.3[index3 == 1, ]
validation.3.train = train.3[index3 == 2, ]

index4 = sample(1:2, nrow(train.4), replace = TRUE, prob = c(0.7, 0.3))
train.4.train = train.4[index4 == 1, ]
validation.4.train = train.4[index4 == 2, ]

index5 = sample(1:2, nrow(train.5), replace = TRUE, prob = c(0.7, 0.3))
train.5.train = train.5[index5 == 1, ]
validation.5.train = train.5[index5 == 2, ]

#### initialize h2o ####
localH2O = h2o.init(ip = "localhost", port = 54321, startH2O = TRUE,
                    max_mem_size = '4g')

#### make h2o frames ####
#### splitting into train and test ####

train.h2o = as.h2o(train.1.train, localH2O)
validation.h2o = as.h2o(validation.1.train, localH2O)

train.2.h2o = as.h2o(train.2.train, localH2O)
validation.2.h2o = as.h2o(validation.2.train, localH2O)

train.2a.h2o = as.h2o(train.2a.train, localH2O)
validation.2a.h2o = as.h2o(validation.2a.train, localH2O)

train.3_h2o = as.h2o(train.3.train, localH2O)
validation.3_h2o = as.h2o(validation.3.train, localH2O)

train.4.h2o = as.h2o(train.4.train, localH2O)
validation.4.h2o = as.h2o(validation.4.train, localH2O)

train.5.h2o = as.h2o(train.5.train, localH2O)
validation.5.h2o = as.h2o(validation.5.train[-2], localH2O)

test.5.h2o = as.h2o(test.5, localH2O)

##### deeplearning model ####

#model = h2o.deeplearning(x = 1:39, y = 40, training_frame = train_h2o)
model = h2o.deeplearning(x = c(1:36, 38:40), y = 37, training_frame = train.5.h2o,
                         validation_frame = validation.5.h2o, hidden = c(100,100)
                         )
summary(model)

# using model for prediction
predictions_ = h2o.predict(model, test_h2o)
predictions

preds = as.data.frame(predictions)
head(preds)

labels = preds[,1]
head(labels)
str(labels)


labels = as.character(labels)
str(preds)
str(test)

id = test.id[,1]
head(id)

write.csv(id, "id.csv")
combined = cbind(id, labels)
head(combined)

submission = as.data.frame(combined)
head(submission)
str(submission)
colnames(submission) = c("id", "status_group")


write.csv(submission, "/Users/nickbecker/Downloads/water pump prediction/submissionDeepLearning.csv", quote=FALSE)


#### gbm model ####
model.gbm1 = h2o.gbm(x = c(1:36, 38:40), y = 37, training_frame = train.5.h2o,
                     validation_frame = validation.5.h2o,
                     ntrees = 400, seed = 100)
summary(model.gbm1)
  
predictions_gbm1 = h2o.predict(model.gbm1, test.5.h2o)
predictions_gbm1

predictions_gbm1 = as.data.frame(predictions_gbm1)
head(predictions_gbm1)
str(predictions_gbm1)

labels = preds[,1]
head(labels)
str(labels)


labels = as.character(labels)
str(preds)
str(test)

id = test.id[,1]
head(id)

combined = cbind(id, labels)
head(combined)

submission = as.data.frame(combined)
head(submission)
str(submission)
colnames(submission) = c("id", "status_group")


write.csv(submission, "/Users/nickbecker/Downloads/water pump prediction/submissionGBM.csv", quote=FALSE)


##### random forest model #####
model1 = h2o.randomForest(x = 1:39, y = 40, training_frame = train.h2o,
                          validation_frame = validation.h2o, ntrees = 500, seed = 100)
summary(model1) # standard lowercase: seed 100 validaiton error:  .19517
# totally basic model with lowercase validation error: .19019
# no improvement really from going to lowercase

model2 = h2o.randomForest(x = 1:38, y = 39, training_frame = train.2a.h2o,
                          validation_frame = validation.2a.h2o, ntrees = 300, seed = 100)
summary(model2) # train.2 : validation error .1904762 with 500 trees
# train.2a : validation error 300 trees .18990930

model3 = h2o.randomForest(x = 1:38, y = 39, training_frame = train.3_h2o,
                          validation_frame = validation.3_h2o, ntrees = 500, seed = 100)
summary(model3) # train.3 : validation error .18409482

model4 = h2o.randomForest(x = c(1:38, 40:42), y = 39, training_frame = train.4.h2o,
                          validation_frame = validation.4.h2o, ntrees = 300, seed = 100)
summary(model4) # train.4 dropped : validation error .19254 (factored day, month, year)
# train.4 total : validation error .19171624 300 trees

# same as 4 excluding old date
model5 = h2o.randomForest(x = c(1:36, 38:40), y = 37, training_frame = train.5.h2o,
                          validation_frame = validation.5.h2o, ntrees = 500, seed = 100)
summary(model5) # train.5 dropped vars validation error: .18614719
# train.5 total validation error : .19254 

model5 = h2o.randomForest(x = c(1:36, 38:40), y = 37, training_frame = train.5.h2o,
                          validation_frame = validation.5.h2o, ntrees = 400, seed = 100)
summary(model5) # scored .8115 on drivendata submission



# predictions
predictions_rf5 = h2o.predict(model5, test.5.h2o)
predictions_rf5

predictions_rf5 = as.data.frame(predictions_rf5)
head(predictions_rf5)
labels = predictions_rf5[,1]
head(labels)
str(labels)
labels = as.character(labels)

str(predictions_rf5)
str(test.5)
id = test.id[,1]
head(id)
combined = cbind(id, labels)
head(combined)
submission = as.data.frame(combined)
head(submission)
str(submission)
colnames(submission) = c("id", "status_group")
write.csv(submission, "/Users/nickbecker/Downloads/water pump prediction/submissionRF-5.csv", quote=FALSE)





#### combining gbm1 and rf5
head(predictions_gbm1)
head(predidctions_rf5)

combined = data.frame(gbm_func = predictions_gbm1[,2], gbm_repair = predictions_gbm1[,3],
                      gbm_non = predictions_gbm1[,4], rf5_func = predictions_rf5[,2],
                      rf5_repair = predictions_rf5[,3], rf5_non = predictions_rf5[,4])
head(combined)

gbm_rf_combine = mutate(combined, avg_func = (combined[,1]+combined[,4])/2, avg_repair = (combined[,2]+combined[,5])/2,
       avg_non = (combined[,3]+combined[,6])/2)

head(gbm_rf_combine)
submission = gbm_rf_combine[7:9]
head(submission)

submission = mutate(submission,
        status_group = apply(submission,1,function(x) names(submission)[which(x == max(x))])
)
head(submission, 20)

train.2a$amount_tsh[train.2a$amount_tsh == 0] = NA


submission$status_group[submission$status_group == "avg_func"] = "functional"
submission$status_group[submission$status_group == "avg_non"] = "non functional"
submission$status_group[submission$status_group == "avg_repair"] = "functional needs repair"

submission = 
ensemble_submission = cbind(id, submission$status_group)
ensemble_submission = as.data.frame(ensemble_submission)
head(ensemble_submission)
colnames(ensemble_submission) = c("id", "status_group")
head(ensemble_submission)
write.csv(ensemble_submission, "/Users/nickbecker/Downloads/water pump prediction/submissionRFandGBM.csv", quote=FALSE)

