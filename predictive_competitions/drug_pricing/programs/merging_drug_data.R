## Merging Data
library(dplyr)
library(caret)
library(stringi)
library(stringr)
library(tidyr)
library(lubridate)

setwd('/Users/nickbecker/Documents/Github/machine_learning_prediction/predictive_competitions/drug_pricing/')

train_raw = read.csv('data/CAX_Bidding_TRAIN_Molecule_3_4_5.csv')
test_post_raw = read.csv('data/CAX_Bidding_TEST_Molecule_6_Post_LOE.csv')
test_post_raw = select(test_post_raw, -Winning_price_per_standard_unit)


######### Feature Engineering ########

#### Train Data
train_raw = mutate(train_raw, Offer_date = as.character(Offer_date))
glimpse(train_raw)


# fix the dates
train = train_raw %>% separate(Offer_date, c("offer_date", "offer_time"), sep = " ", remove=TRUE, extra = "drop")
train = train %>% separate(offer_date, c("offer_month", "offer_day", "offer_year"), sep = "-", remove = TRUE)
train = train %>% separate(Start_date, c("start_day", "start_month", "start_year"), sep = "-", remove = TRUE)
train = mutate(train,
               end_date_year = as.integer(str_sub(End_date_including_extension, -4, -1)),
               end_date_month = as.integer(str_sub(End_date_including_extension, -7, -6)),
               offer_month = as.integer(offer_month),
               offer_day = as.integer(offer_day),
               start_month = as.integer(start_month),
               start_year = as.integer(start_year),
               start_day = as.integer(start_day),
               end_date_year = as.integer(end_date_year),
               end_date_month = as.integer(end_date_month))
head(train, 3)

table(train$offer_day)
table(train$offer_year)
table(train$start_month)
table(train$start_day)
table(train$start_year)
table(train$end_date_year)
table(train$end_date_month)

# Function to fix the years
two_to_four_year <- function(x){
  temp <- as.integer(x)
  temp <- ifelse(temp > temp %% 100, temp, 2000+temp)
  return(temp)
}

train$offer_year = sapply(train$offer_year, two_to_four_year)
str(train)


### Test Data
test_post_raw = mutate(test_post_raw, Offer_date = as.character(Offer_date))
glimpse(test_post_raw)


# fix the dates
test = test_post_raw %>% separate(Offer_date, c("offer_date", "offer_time"), sep = " ", remove=TRUE, extra = "drop")
test = test %>% separate(offer_date, c("offer_month", "offer_day", "offer_year"), sep = "-", remove = TRUE)
test = test %>% separate(Start_date, c("start_day", "start_month", "start_year"), sep = "-", remove = TRUE)
test = mutate(test,
               end_date_year = as.integer(str_sub(End_date_including_extension, -4, -1)),
               end_date_month = as.integer(str_sub(End_date_including_extension, -7, -6)),
               offer_month = as.integer(offer_month),
               offer_day = as.integer(offer_day),
               start_month = as.integer(start_month),
               start_year = as.integer(start_year),
               start_day = as.integer(start_day),
               end_date_year = as.integer(end_date_year),
               end_date_month = as.integer(end_date_month))
head(test, 3)

test$offer_year = sapply(test$offer_year, two_to_four_year)

glimpse(test)



# take a subset of the training features
train_subset = select(train,
                      Account:offer_year, start_day:start_year,
                      Length_of_contract_in_Months:Winning_Company,
                      end_date_year:end_date_month)

train_subset = na.omit(train_subset)
colnames(train_subset)

test_subset = select(test,
                      Account:offer_year, start_day:start_year,
                      Length_of_contract_in_Months:Winning_Company,
                      end_date_year:end_date_month)

# convert to numeric
train_subset_numeric = data.frame(model.matrix(~ ., train_subset))
head(train_subset_numeric)
colnames(train_subset_numeric)

test_numeric = data.frame(model.matrix(~ ., test_subset))
head(train_subset_numeric)

# model parameters
fitControl = trainControl(method = "cv",
                          number = 10,
                          verboseIter = TRUE)

gbmGrid = expand.grid(interaction.depth = c(1:7),
                      n.trees = 300,
                      shrinkage = c(0.01),
                      n.minobsinnode = c(5, 10, 15, 20))

# fit a gbm to predict winning price per standard unit
set.seed(12)
gbm1 = train(x=train_subset[, -c(14)],
                y=train_subset[, 14],
                method = "gbm",
                tuneGrid = gbmGrid,
                trControl = fitControl)
gbm1

colnames(test)
gbm1_preds = predict(gbm1, test)

head(gbm_preds)




