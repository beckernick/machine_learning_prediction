# Predicting whether a tumor is malignant or benign

library(MASS)
library(caret)
library(DMwR)
library(kernlab)
library(mlbench)
library(dplyr)
library(ROCR)
library(pROC)

data(biopsy)
biopsy = biopsy

colnames(biopsy) = c("ID", "clump_thickness", "cell_size_uniformity", "cell_shape_uniformity",
                     "marginal_adhesion", "single_epithelial_cell_size", "bare_nuclei", "bland_chromatin",
                     "normal_nucleoli", "mitoses", "class")
head(biopsy)
str(biopsy)

summary(biopsy)


# Let's ignore observtions with missing bare nuclei
biopsy_nonmissing = filter(biopsy, !is.na(bare_nuclei))

# Splitting into a train and validation set
set.seed(12)
inTrain = createDataPartition(y = biopsy_nonmissing$class,
                              p = 0.85,
                              list = FALSE)
training = biopsy_nonmissing[inTrain,]
validation = biopsy_nonmissing[-inTrain,]



# model parameters
fitControl = trainControl(method = "cv",
                          number = 3,
                          summaryFunction=twoClassSummary,
                          classProbs=TRUE,
                          verboseIter = TRUE)

rfGrid = expand.grid(mtry = c(1:9))

set.seed(12)
rf_fit = train(x = training[, -c(1,11)], # don't train on the ID number
               y = training[, 11],
               method = "rf",
               tuneGrid = rfGrid,
               trControl = fitControl)
rf_fit

jpeg("/users/nickbecker/documents/github/tumor_biopsy_classification/plots/random_forest_ROC.jpeg")
plot(rf_fit)
dev.off()

rf_preds = predict(rf_fit, validation[, -c(1,11)])
confusionMatrix(rf_preds, validation$class)


### support vector machine
svmGrid = expand.grid(C = seq(.1, 5, .1))

set.seed(12)
svm_fit = train(x = training[, -c(1,11)], # don't train on the ID number
               y = training[, 11],
               method = "svmLinear",
               #metric = "Kappa",
               tuneGrid = svmGrid,
               trControl = fitControl)
svm_fit
jpeg("/users/nickbecker/documents/github/tumor_biopsy_classification/plots/svm_ROC.jpeg")
plot(svm_fit)
dev.off()

svm_preds = predict(svm_fit, validation[, -c(1,11)])
confusionMatrix(svm_preds, validation$class)



# Algorithm is excellent, but we ignored missing observations.
# In real life, we will sometimes have incomplete information.
# Using them could still help


# Let's do knn imputation
filter(biopsy, is.na(bare_nuclei))

biopsy_imputed = knnImputation(biopsy[,-1], k = 5, scale = T, meth = "median",
              distData = NULL)
summary(biopsy)
summary(biopsy_imputed)

# Splitting into a train and validation set
set.seed(12)
inTrain = createDataPartition(y = biopsy_imputed$class,
                              p = 0.85,
                              list = FALSE)
training_imputed = biopsy_imputed[inTrain,]
validation_imputed = biopsy_imputed[-inTrain,]



set.seed(12)
rf_imputed_fit = train(x = training_imputed[, -c(10)], # ID number isnt in this one
               y = training_imputed[, 10],
               method = "rf",
               tuneGrid = rfGrid,
               trControl = fitControl)
rf_imputed_fit

jpeg("/users/nickbecker/documents/github/tumor_biopsy_classification/plots/random_forest_imputed_ROC.jpeg")
plot(rf_imputed_fit)
dev.off()

rf_imputed_preds = predict(rf_imputed_fit, validation_imputed[, -10])
confusionMatrix(rf_imputed_preds, validation_imputed$class)

# No improvement from imputation 


### support vector machine
svmGrid = expand.grid(C = seq(.1, 5, .1))

set.seed(12)
svm_imputed_fit = train(x = training_imputed[, -c(10)], # don't train on the ID number
                y = training_imputed[, 10],
                method = "svmLinear",
                #metric = "Kappa",
                tuneGrid = svmGrid,
                trControl = fitControl)
svm_imputed_fit

jpeg("/users/nickbecker/documents/github/tumor_biopsy_classification/plots/svm_imputed_ROC.jpeg")
plot(svm_imputed_fit)
dev.off()

svm_imputed_preds = predict(svm_imputed_fit, validation_imputed[, -c(10)])
confusionMatrix(svm_imputed_preds, validation_imputed$class)



# Let's optimize ROC curves and plot
fitControl_roc = trainControl(method = "cv",
                          number = 3,
                          summaryFunction=twoClassSummary,
                          savePredictions = TRUE,
                          classProbs=TRUE,
                          verboseIter = TRUE)


set.seed(12)
rf_fit = train(x = training[, -c(1,11)], # don't train on the ID number
               y = training[, 11],
               method = "rf",
               metric = "ROC",
               tuneGrid = rfGrid,
               trControl = fitControl_roc)
rf_fit
plot(rf_fit)

rf_preds = predict(rf_fit, validation[, -c(1,11)])
confusionMatrix(rf_preds, validation$class)


### support vector machine
svmGrid = expand.grid(C = seq(.1, 5, .1))

set.seed(12)
svm_fit = train(x = training[, -c(1,11)], # don't train on the ID number
                y = training[, 11],
                method = "svmLinear",
                metric = "ROC",
                tuneGrid = svmGrid,
                trControl = fitControl_roc)
svm_fit
plot(svm_fit)

svm_preds = predict(svm_fit, validation[, -c(1,11)])
confusionMatrix(svm_preds, validation$class)
















