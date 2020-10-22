library(rpart)
library(randomForest)
library(gbm)
library(adabag)
library(caret)
library(partykit)

setwd("/Users/lucasA/Desktop/ML-Class-/Random Forests")

Data <- read.csv("Data.csv", header = TRUE, sep = ';')

n <- nrow(Data)

ggplot(data = Data, aes(x = RELAT))+
  geom_histogram(bins = 50, color = 'black', fill = 'navyblue')+
  theme_minimal()

ggplot(data = Data, aes(x = AGER))+
  geom_histogram(bins = 50, color = 'black', fill = 'navyblue')+
  theme_minimal()

par(mfrow = c(3, 3))

for (i in 4 : 12){
  boxplot(Data[, i] ~ Data$CARVP, main = colnames(Data)[i])
}

# CART

rpart_control <- rpart.control(minbucket = 5, cp = 0.01, xval = 5)

rpart_fit <- rpart(CARVP ~ ., method = 'class', control = rpart_control, data = Data)

summary(rpart_fit)

par(xpd = NA)
plot(rpart_fit, uniform = FALSE)
text(rpart_fit, all = TRUE, use.n = TRUE, cex = 0.8)

partykit::party-plot(as.party(rpart_fit))
detach('package::partykit', unload = TRUE)

plotcp(rpart_fit)
printcp(rpart_fit)

cp <- rpart_fit$cptable[3, 1]
rpart_fit_prune <- prune(rpart_fit, cp = cp)

partykit::party-plot(as.party(rpart_fit_prune))
detach('package::partykit', unload = TRUE)

pred <- predict(rpart_fit_prune, type = 'class')
tab <- table(Data$CARVP, pred)
1 - sum(diag(tab) / sum(tab)) # taux d'erreur

boot <- sample(1 : n, replace = TRUE)
OOB <-  stediff(1 : n, unique(boot))

rpart_control_boot <- rpart.control(minbucket = 5, cp = 0.01, xval = 5)

rpart_fit_boot <- rpart(CARVP ~ ., method = 'class', control = rpart_control_boot, data = boot)

cp <- rpart_fit_boot$cptable[3, 1]
rpart_fit_prune_boot <- prune(rpart_fit_boot, cp = cp)

partykit::party-plot(as.party(rpart_fit_prune_boot))
detach('package::partykit', unload = TRUE)

# Random Forest

pX <- ncol(Data) - 1
valntree <- 2000
valmtry <- floor(sqrt(pX))
valnodedsize <- 1

rdf <- randomForest(CARVP ~ ., data = Data, ntree = valntree,
                    mtry = valmtry, nodesize = valnodedsize, important = TRUE,
                    proximity = TRUE, nPerm = 1)
print(rdf)
on.exit(par)
plot(rdf)

tuneRF(x = Data[, 2 : 13], y = Data$CARVP, mtryStart = 3,
       ntreeTry = 500, stepFactor = 2, improve = 0.001)

fit.control <- trainControl(method = 'repeatedcv', number = 5, repeats = 10,
                            classProbs = TRUE, summaryFunction = twoClassSummary,
                            search = 'grid')

tune.mtry <- expand.grid(.mtry = (1 : 10))

deb <- Sys.time()
rdf_grid <- train(CARVP ~ ., data = Data, method = 'rf', metric = 'Accuracy',
                  tuneGrid = tune.mtry, trControl = fit.control)

fin <- Sys.time()
print(fin - deb)

print(rdf_grid)
plot(rdf_grid)

confusionMatrix(rdf_grid)

pX <- ncol(Data) - 1
valntree <- 500
valmtry <- 2
valnodedsize <- 1

rdf <- randomForest(CARVP ~ ., data = Data, ntree = valntree,
                    mtry = valmtry, nodesize = valnodedsize, important = TRUE,
                    proximity = TRUE, nPerm = 1)

print(rdf)
plot(rdf)

pred <- predict(rdf, newdata = Data)
pred_OOB <- predict(rdf)
tab <- table(Data$CARVP, pred)
tab_OOB <- table(Data$CARVP, pred_OOB)
1 - sum(diag(tab) / sum(tab))
1 - sum(diag(tab_OOB) / sum(tab_OOB))

confusionMatrix(predict(rdf), Data$CARVP, positive = levels(Data$CARVP)[2], mode = 'sens_spec')

varImpPlot(rdf)

importance(rdf, scale = TRUE)

# Boosting

fit_gbm <-gbm(Data$CARVP ~ ., data = Data, distribution = 'multinomial', interaction.depth = 2,
              n.trees = 1000, shrinkage = 0.01, bag.fraction = 0.8, cv.folds = 5) 
fit_gbm

best.iter = gbm.perf(fit_gbm, method = 'cv')
fit_gbm$cv.error[best.iter]
