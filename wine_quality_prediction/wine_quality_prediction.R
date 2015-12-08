library(readr)
library(plyr)
library(dplyr)
library(randomForest)
library(doMC)
library(ggplot2)
library(caret)
library(corrplot)

# Dual core
registerDoMC(cores = 2)


white_wine = read.table("~/downloads/winequality-white.csv", sep = ";", header=TRUE)
red_wine = read.table("~/downloads/winequality-red.csv", sep = ";", header=TRUE)
head(white_wine)
head(red_wine)

white_wine = white_wine %>%
  mutate(color = "White")

red_wine = red_wine %>%
  mutate(color = "Red")

all_wine = rbind(white_wine, red_wine)

head(all_wine)
all_wine = all_wine %>% mutate(color = as.factor(color))
str(all_wine)

# Are there any near zero variance variables?
summary(all_wine)
nearZeroVar(all_wine) # nope

# Let's look at the correlations
str(all_wine)
correlations = cor(all_wine[, c(1:11)]) # numeric variables
corrplot(correlations, method = "number", type = "upper")

# split into train and test set
set.seed(12)
inTrain = createDataPartition(y = all_wine$quality,
                              p = 0.9,
                              list = FALSE)
str(inTrain)
training = all_wine[inTrain,]
test = all_wine[-inTrain,]


# partial least squares discriminant analysis
ctrl = trainControl(method = "repeatedcv",
                    number = 5,
                    repeats = 5)

plsFit = train(quality ~ .,
               data = training,
               method = "pls",
               trControl = ctrl,
               # center and scale predictors for the training set and fuutre samples
               preProc = c("center", "scale"))
plsFit
plot(plsFit)



# fit a simple cross validated boosted tree model via gbm
fitControl = trainControl(method = "repeatedcv",
                          number = 5,
                          repeats = 5) # 5-fold CV repeated 5 times

set.seed(12)
gbmFit1 = train(quality ~ ., data = training,
                method = "gbm",
                trControl = fitControl,
                #preProc = c("center", "scale"),
                # this is for gbm that passes through
                verbose = FALSE)
gbmFit1
plot(gbmFit1)


# Let's do something a little more complicated
# Iterate over several parameters to find the best cross validated combinations
gbmGrid = expand.grid(interaction.depth = c(1:5),
                      n.trees = (1:15)*20,
                      shrinkage = 0.1,
                      n.minobsinnode = 20)

set.seed(12)
gbmFit2 = train(quality ~ ., data = training,
                method = "gbm",
                trControl = fitControl,
                tuneGrid = gbmGrid,
                # this is for gbm that passes through
                verbose = FALSE)
gbmFit2
plot(gbmFit2)


# Let's try a standard regression
fitControl = trainControl(method = "repeatedcv",
                          number = 5,
                          repeats = 5) # 5-fold CV repeated 5 times

linearFit1 = train(quality ~ ., data = training,
                method = "lm",
                trControl = fitControl)
linearFit1



# Choosing the Final Model


# Compare results across models by collecting the resampling results
resamps = resamples(list(Linear = linearFit1,
                         Gbm = gbmFit1,
                         Gbm2 = gbmFit2,
                         pls = plsFit))
resamps
summary(resamps)

# Compute differences (since they are on the same versions of the training data)
difValues = diff(resamps)
difValues
summary(difValues)

bwplot(difValues, layout = c(3,1))



# Plots
p = plot(plsFit, main = "Partial Least Squares Regression")
p1 = plot(gbmFit1, main = "Gradient Boosting Regression 1")
p2 = plot(gbmFit2, main = "Gradient Boosting Regression 2")
p3 = bwplot(difValues, layout = c(2,1), main = "Model Comparison Boxplots")

width = 1200
height = 800

jpeg("Partial_Least_Squares_Regression_Fit.jpeg", width = width, height = height)
p
dev.off()

jpeg("GBM_Fit_1.jpeg", width = width, height = height)
p1
dev.off()

jpeg("GBM_Fit_2.jpeg", width = width, height = height)
p2
dev.off()

jpeg("Model_Comparison_Boxplot.jpeg", width = width, height = height)
p3
dev.off()



