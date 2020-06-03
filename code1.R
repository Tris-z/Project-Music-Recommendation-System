library(data.table) # read data
library(corrplot) # draw covariance matrix
library(randomForest) # random forest variable selection
library(ROCR) # ROC
library(caret) # confusion matrix

## read data tables
members_dt <- fread("C:/Users/54491/Documents/data/members.csv")
songs_dt <- fread("C:/Users/54491/Documents/data/songs.csv")
train_dt <- fread("C:/Users/54491/Documents/data/train.csv")
test_dt <- fread("C:/Users/54491/Documents/data/test.csv")

## deal with two time features, convert long integer to date format
standard_time <- function(i){
  dd<-as.character(i)
  paste0(substr(dd, 1, 4), "-", 
         substr(dd, 5, 6), "-",
         substr(dd, 7, 8))
}

members_dt[, registration_init_time :=
             as.Date(standard_time(registration_init_time))]
members_dt[, expiration_date :=
             as.Date(standard_time(expiration_date))]

## prepare combined table
train_dt [, id := -1] #originally lack of ID
test_dt [, target := -1] #originally lack of target
both<- rbind(train_dt, test_dt)

## Merge both with songs and members
both <- merge(both, members_dt, by = "msno", all.x=TRUE)
both <- merge(both, songs_dt, by = "song_id", all.x=TRUE)

## Label encode the char columns change char columns into numbers
for (f in names(both)){
  if( class(both[[f]]) == "character"){
    both[is.na(both[[f]]), eval(f) := ""]
    both[, eval(f) := as.integer(
      as.factor(both[[f]]))]
  } else both[is.na(both[[f]]), eval(f) := -999]
}

## There are two date columns left
## change date into numbers, use length_membership instead of date
both[, registration_init_time := julian(registration_init_time)]
both[, expiration_date := julian(expiration_date)]
both[, length_membership := 
       expiration_date - registration_init_time]

## seperate data from "both"
setDF(both)
train_df <- both[both$id == -1,] ## recover into original features
test_df <- both[both$target == -1,]
train_df$id <- NULL
test_df$target <- NULL

rm(both)
rm(songs_dt)
rm(members_dt)
gc()

## define train and test data, using random sampling due to large numbers and memory limitation
train_df1<-train_df[sample(nrow(train_df), floor(nrow(train_df)*0.005)), ]
y <- train_df1$target
train_df1$target <- NULL

test_df$id <- NULL

## covariance matrix
corrplot(cor(train_df1), method="color")

## length_membership & registration_init_time strong correlation, remove one
train_df1$registration_init_time <- NULL
test_df$registration_init_time <- NULL

## retest
corrplot(cor(train_df1), method="color")

## variable selection using Random Forest
set.seed(777)
rf_ntree<- randomForest(y ~ ., data=train_df1, ntree=300, important=TRUE, proximity=TRUE)
plot(rf_ntree)

#模型评估各属性的重要度
importance(rf_ntree)

#绘制变量重要性曲线
varImpPlot(rf_ntree)

#drop unimportance features  
name <- c("source_screen_name", "genre_ids", "city",
          "source_system_tab", "language", "registered_via", "gender")


for (f in names(train_df1)){
  if(names(train_df1[f]) %in% name){
    train_df1[[f]] <- NULL
  } 
}

for (f in names(test_df)){
  if(names(test_df[f]) %in% name){
    test_df[[f]] <- NULL
  } 
}

## train and validation set
T <- nrow(train_df1)
train_70 <- train_df1[1:round(0.7*T), ] #train
train_30 <- train_df1[(round(0.7*T+1):T), ] #validation
y_70 <- y[1:round(0.7*T)] #train
y_30 <- y[round(0.7*T+1):T] #validation


## logistic regression
glm.fit <- glm(y_70 ~ ., data = train_70)
summary(glm.fit)
glm.vali <- predict(glm.fit, train_30, type="response")

## confusion matrix
class <- glm.vali > 0.5
class <- as.numeric(class)
confusionMatrix(factor(y_30), factor(class))

pred <- prediction(glm.vali, y_30)
performance(pred,'auc') #AUC
perf <- performance(pred,'tpr','fpr')
plot(perf)

## prediction result
glm.pred <- predict(glm.fit, test_df, type="response")
target_pre <- ifelse(glm.pred > 0.5, 1, 0)

id <- c(1:nrow(test_df))
test_rst <- cbind(id, target_pre)

# export result
write.csv(test_rst,"test_rst.csv")


## random forest
rf.fit <- randomForest(y_70 ~ ., data=train_70, ntree=200, important=TRUE, proximity=TRUE)
plot(rf.fit)

rf.vali <- predict(rf.fit, train_30, type="response")
class <- rf.vali > 0.5
class <- as.numeric(class)
confusionMatrix(factor(y_30), factor(class))

pred <- prediction(rf.vali, y_30)
performance(pred,'auc') #AUC
perf <- performance(pred,'tpr','fpr')
plot(perf)

## prediction result
rf.pred <- predict(rf.fit, test_df, type="response")
target_pre <- ifelse(rf.pred > 0.5, 1, 0)
test_rst_rf <- cbind(id, target_pre)

# export result
write.csv(test_rst_rf,"test_rst_rf.csv")
