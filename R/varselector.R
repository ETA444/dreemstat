varselector <- function (df,y,cv,lambda,alpha,model_id = NULL,mode = NULL,split = NULL) {

  # (Specify default values)["mode","split","model_id"]
  mode <- ifelse(is.null(mode),"all",mode) #if "mode" is not specified assign "all"
  split <- ifelse(is.null(split), 0.80,split) #if "split" is not speficied assign 80%-20% split (0.80)
  model_id <- ifelse(is.null(model_id), 1,model_id) #if "model_id" is not specified assign 1

  # (Create train and test) + (Define y)
  df_train <- df[1:((round(nrow(df) * split))),] #we use this to train our models in-sample
  df_test <- df[(((round(nrow(df) * split)))+1):(nrow(df)),] #we use this to test our models out-sample


  # (Fast Forward Regression flow)
  if (mode == "ffr" | mode == "all") { #if mode is (all) or (ffr) => run fast forward regression
    ffr <- caret::train(as.formula(paste(y,'~ .')), data = df_train, method = "leapForward", trControl = cv, preProc = c("center","scale"), tuneGrid = expand.grid(nvmax = seq(2, 30, 1)))
    ffr.coef <- coef(ffr$finalModel, unlist(ffr$bestTune)) ## COEFFICIENTS IN FINAL MODEL
    ## (PREDICTIONS) & (RMSE,MAE) [IN- AND OUT-OF-SAMPLE]
    ffr.pred_insample <- predict(ffr,df_train) #IN-SAMPLE
    ffr.rmse_insample <- RMSE(ffr.pred_insample, unlist(subset(df_train, select = paste0(y))))
    ffr.mae_insample <- MAE(ffr.pred_insample, unlist(subset(df_train, select = paste0(y))))
    ffr.pred_outsample <- predict(ffr,df_test) #OUT-OF-SAMPLE
    ffr.rmse_outsample <- RMSE(ffr.pred_outsample, unlist(subset(df_test, select = paste0(y))))
    ffr.mae_outsample <- MAE(ffr.pred_outsample, unlist(subset(df_test, select = paste0(y)))) #finally, cat prints the results in a readable way + we plot the model:
    #ffr.plot <- plot(ffr) ## PLOT FF REGRESSION
    cat('\n','FAST FORWARD REGRESSION RESULTS:','\n','  FFR-RMSE (INSAMPLE): ',ffr.rmse_insample,'\n','  FFR-MAE (INSAMPLE): ',ffr.mae_insample,'\n','  FFR-RMSE (OUTSAMPLE): ',ffr.rmse_outsample,'\n','  FFR-MAE (OUTSAMPLE): ',ffr.mae_outsample,'\n','\n','FFR COEFFICIENTS:','\n')
    print(ffr.coef)
    #ffr.plot

    # Save objects to global env
    modelnamegenerator <- as.character(paste('ffr.model_id',as.character(model_id),as.character(split),sep = "_"))
    assign(modelnamegenerator,ffr,envir = .GlobalEnv) #save the model object for later testing

    coefnamegenerator <- as.character(paste('ffr.coef_id',as.character(model_id),as.character(split),sep = "_"))
    assign(coefnamegenerator,ffr.coef,envir = .GlobalEnv) #save the coef object for later testing

    cat("Your model was saved! It's called: ",modelnamegenerator,'\n','The coefficients of this model have been saved here: ',coefnamegenerator)
  }

  # (Ridge Regression flow)
  if (mode == "rr" | mode == "all") { #if mode is (all) or (rr) => run ridge regression
    rr <- caret::train(as.formula(paste(y,'~ .')), data = df_train, method = "glmnet", trControl = cv, preProc = c("center","scale"), tuneGrid = expand.grid(alpha = 0, lambda = lambda))
    rr.coef <- coef(rr$finalModel, rr$finalModel$lambdaOpt)  ## COEFFICIENTS IN FINAL MODEL
    ## (PREDICTIONS) & (RMSE,MAE) [IN- AND OUT-OF-SAMPLE]
    rr.pred_insample <- predict(rr,df_train) #IN-SAMPLE
    rr.rmse_insample <- RMSE(rr.pred_insample, unlist(subset(df_train, select = paste0(y))))
    rr.mae_insample <- MAE(rr.pred_insample, unlist(subset(df_train, select = paste0(y))))
    rr.pred_outsample <- predict(rr,df_test) #OUT-OF-SAMPLE
    rr.rmse_outsample <- RMSE(rr.pred_outsample, unlist(subset(df_test, select = paste0(y))))
    rr.mae_outsample <- MAE(rr.pred_outsample, unlist(subset(df_test, select = paste0(y))))
    #rr.plot <- plot(rr)  ## PLOT R REGRESSION
    cat('\n','\n','<-------------------------------------->','\n','RIDGE REGRESSION RESULTS:','\n','  RR-RMSE (INSAMPLE): ',rr.rmse_insample,'\n','  RR-MAE (INSAMPLE): ',rr.mae_insample,'\n','  RR-RMSE (OUTSAMPLE): ',rr.rmse_outsample,'\n','  RR-MAE (OUTSAMPLE): ',rr.mae_outsample,'\n','\n','RR COEFFICIENTS:','\n')
    print(rr.coef)
    #rr.plot

    modelnamegenerator <- as.character(paste('rr.model_id',as.character(model_id),as.character(split),sep = "_"))
    assign(modelnamegenerator,rr,envir = .GlobalEnv) #save the model object for later testing

    coefnamegenerator <- as.character(paste('rr.coef_id',as.character(model_id),as.character(split),sep = "_"))
    assign(coefnamegenerator,rr.coef,envir = .GlobalEnv) #save the coef object for later testing

    cat("Your model was saved! It's called: ",modelnamegenerator,'\n','The coefficients of this model have been saved here: ',coefnamegenerator)
  }

  # (Lasso Regression flow)
  if (mode == "lr" | mode == "all") { #if mode is (all) or (lr) => run lasso regression
    #y <- df_train[,y_colnum]
    lr <- caret::train(as.formula(paste(y,'~ .')), data = df_train, method = "glmnet", trControl = cv, preProc = c("center","scale"), tuneGrid = expand.grid(alpha = 1, lambda = lambda))
    lr.coef <- coef(lr$finalModel, lr$finalModel$lambdaOpt)  ## COEFFICIENTS IN FINAL MODEL
    ## (PREDICTIONS) & (RMSE,MAE) [IN- AND OUT-OF-SAMPLE]
    lr.pred_insample <- predict(lr,df_train) #IN-SAMPLE
    lr.rmse_insample <- RMSE(lr.pred_insample, unlist(subset(df_train, select = paste0(y))))
    lr.mae_insample <- MAE(lr.pred_insample, unlist(subset(df_train, select = paste0(y))))
    lr.pred_outsample <- predict(lr,df_test) #OUT-OF-SAMPLE
    lr.rmse_outsample <- RMSE(lr.pred_outsample, unlist(subset(df_test, select = paste0(y))))
    lr.mae_outsample <- MAE(lr.pred_outsample, unlist(subset(df_test, select = paste0(y))))
    #lr.plot <- plot(lr)  ## PLOT L REGRESSION
    cat('\n','\n','<-------------------------------------->','\n','LASSO REGRESSION RESULTS:','\n','  LR-RMSE (INSAMPLE): ',lr.rmse_insample,'\n','  LR-MAE (INSAMPLE): ',lr.mae_insample,'\n','  LR-RMSE (OUTSAMPLE): ',lr.rmse_outsample,'\n','  LR-MAE (OUTSAMPLE): ',lr.mae_outsample,'\n','\n','LR COEFFICIENTS:','\n')
    print(lr.coef)
    #lr.plot

    modelnamegenerator <- as.character(paste('lr.model_id',as.character(model_id),as.character(split),sep = "_"))
    assign(modelnamegenerator,lr,envir = .GlobalEnv) #save the model object for later testing

    coefnamegenerator <- as.character(paste('lr.coef_id',as.character(model_id),as.character(split),sep = "_"))
    assign(coefnamegenerator,lr.coef,envir = .GlobalEnv) #save the coef object for later testing

    cat("Your model was saved! It's called: ",modelnamegenerator,'\n','The coefficients of this model have been saved here: ',coefnamegenerator)
  }

  # (Elastic Net Regression flow)
  if (mode == "enr" | mode == "all") {
    enr <- caret::train(as.formula(paste(y,'~ .')), data = df_train, method = "glmnet", trControl = cv, preProc = c("center","scale"), tuneGrid = expand.grid(alpha = alpha, lambda = lambda))
    enr.coef <- coef(enr$finalModel, enr$finalModel$lambdaOpt)  ## COEFFICIENTS IN FINAL MODEL
    ## (PREDICTIONS) & (RMSE,MAE) [IN- AND OUT-OF-SAMPLE]
    enr.pred_insample <- predict(enr,df_train) #IN-SAMPLE
    enr.rmse_insample <- RMSE(enr.pred_insample, unlist(subset(df_train, select = paste0(y))))
    enr.mae_insample <- MAE(enr.pred_insample, unlist(subset(df_train, select = paste0(y))))
    enr.pred_outsample <- predict(enr,df_test) #OUT-OF-SAMPLE
    enr.rmse_outsample <- RMSE(enr.pred_outsample, unlist(subset(df_test, select = paste0(y))))
    enr.mae_outsample <- MAE(enr.pred_outsample, unlist(subset(df_test, select = paste0(y))))
    #enr.plot <- plot(enr)  ## PLOT EN REGRESSION
    cat('\n','\n','<-------------------------------------->','\n','ELASTIC NET REGRESSION RESULTS:','\n','  ENR-RMSE (INSAMPLE): ',enr.rmse_insample,'\n','  ENR-MAE (INSAMPLE): ',enr.mae_insample,'\n','  ENR-RMSE (OUTSAMPLE): ',enr.rmse_outsample,'\n','  ENR-MAE (OUTSAMPLE): ',enr.mae_outsample,'\n','\n','ENR COEFFICIENTS:','\n')
    print(enr.coef)
    #enr.plot

    modelnamegenerator <- as.character(paste('enr.model_id',as.character(model_id),as.character(split),sep = "_"))
    assign(modelnamegenerator,enr,envir = .GlobalEnv) #save the model object for later testing

    coefnamegenerator <- as.character(paste('enr.coef_id',as.character(model_id),as.character(split),sep = "_"))
    assign(coefnamegenerator,enr.coef,envir = .GlobalEnv) #save the coef object for later testing

    cat("Your model was saved! It's called: ",modelnamegenerator,'\n','The coefficients of this model have been saved here: ',coefnamegenerator)
  }

  # (RMSE & MAE Comparisons flow - 4 STEPS)
  # Note: We prioritize high score for Out-sample RMSE, therefore we only save the best model
  #       in terms of Out-sample RMSE, which should result in the best model for new data.
  #       However, we still check all other metrics and print the information.

  if (mode == "all") { #ONLY make comparisons in RMSE and MAE values among models if we ran ALL models
    cat('\n','\n','<-------------------------------------->','\n','WHICH MODEL HAS LOWEST RMSE & MAE? (IN-SAMPLE):','\n')

    # > STEP 1: Which model has lowest RMSE? (In-sample)
    if (ffr.rmse_insample < rr.rmse_insample & ffr.rmse_insample < lr.rmse_insample & ffr.rmse_insample < enr.rmse_insample) { #FFR RMSE then MAE VS ALL
      cat("RMSE values (In-sample) of Fast Forward Regression model are lowest!",'\n')
    } else if (rr.rmse_insample < ffr.rmse_insample & rr.rmse_insample < lr.rmse_insample & rr.rmse_insample < enr.rmse_insample) { #RR RMSE then MAE VS ALL
      cat("RMSE values (In-sample) of Ridge Regression model are lowest!",'\n')
    } else if (lr.rmse_insample < ffr.rmse_insample & lr.rmse_insample < rr.rmse_insample & lr.rmse_insample < enr.rmse_insample) { #LR RMSE then MAE VS ALL
      cat("RMSE values (In-sample) of Lasso Regression model are lowest!",'\n')
    } else if (enr.rmse_insample < ffr.rmse_insample & enr.rmse_insample < rr.rmse_insample & enr.rmse_insample < lr.rmse_insample) { #ENR RMSE then MAE VS ALL
      cat("RMSE values (In-sample) of Elastic Net Regression model are lowest!",'\n')
    } else { #ACCOUNT FOR SCENARIO WHERE SOME VALUES ARE EQUAL, SAY WHICH VALUES (TRUE/FALSE)
      cat('There are RMSE values that are equal','\n','FF and RR?:',(ffr.rmse_insample == rr.rmse_insample),'\n','FF and LR?:',(ffr.rmse_insample == lr.rmse_insample),'\n','FF and ENR?:',(ffr.rmse_insample == enr.rmse_insample),'\n',
          '\n','RR and LR?:',(rr.rmse_insample == lr.rmse_insample),'\n','RR and ENR?:',(rr.rmse_insample == enr.rmse_insample),'\n','FF and ENR?:',(ffr.rmse_insample == enr.rmse_insample),'\n',
          '\n','LR and ENR?:',(lr.rmse_insample == enr.rmse_insample),'\n')
    }

    # > STEP 2: Which model has lowest MAE? (In-sample)
    if (ffr.mae_insample < rr.mae_insample & ffr.mae_insample < lr.mae_insample & ffr.mae_insample < enr.mae_insample) {
      cat("MAE values (In-sample) of Fast Forward Regression model are lowest!")
    } else if (rr.mae_insample < ffr.mae_insample & rr.mae_insample < lr.mae_insample & rr.mae_insample < enr.mae_insample) {
      cat("MAE values (In-sample) of Ridge Regression model are lowest!")
    } else if (lr.mae_insample < ffr.mae_insample & lr.mae_insample < rr.mae_insample & lr.mae_insample < enr.mae_insample) {
      cat("MAE values (In-sample) of Lasso Regression model are lowest!")
    } else if (enr.mae_insample < ffr.mae_insample & enr.mae_insample < rr.mae_insample & enr.mae_insample < lr.mae_insample) {
      cat("MAE values (In-sample) of Elastic Net Regression model are lowest!")
    } else { #ACCOUNT FOR SCENARIO WHERE SOME VALUES ARE EQUAL, SAY WHICH VALUES (TRUE/FALSE)
      cat('There are MAE values that are equal','\n','FF and RR?:',(ffr.mae_insample == rr.mae_insample),'\n','FF and LR?:',(ffr.mae_insample == lr.mae_insample),'\n','FF and ENR?:',(ffr.mae_insample == enr.mae_insample),'\n',
          '\n','RR and LR?:',(rr.mae_insample == lr.mae_insample),'\n','RR and ENR?:',(rr.mae_insample == enr.mae_insample),'\n','FF and ENR?:',(ffr.mae_insample == enr.mae_insample),'\n',
          '\n','LR and ENR?:',(lr.mae_insample == enr.mae_insample),'\n')
    }

    cat('\n','\n','<-------------------------------------->','\n','WHICH MODEL HAS LOWEST RMSE & MAE? (OUT-SAMPLE):','\n')

    # > STEP 3: Which model has lowest RMSE? (Out-sample) < we save the best model based on this metric
    if (ffr.rmse_outsample < rr.rmse_outsample & ffr.rmse_outsample < lr.rmse_outsample & ffr.rmse_outsample < enr.rmse_outsample) { #FFR RMSE then MAE VS ALL
      cat("RMSE values (Out-sample) of Fast Forward Regression model are lowest!",'\n')
      modelnamegenerator <- as.character(paste('ffr.rmseout.bestmodel_id',as.character(model_id),as.character(split),sep = "_"))
      assign(modelnamegenerator,ffr,envir = .GlobalEnv) #save the model object for later testing
    } else if (rr.rmse_outsample < ffr.rmse_outsample & rr.rmse_outsample < lr.rmse_outsample & rr.rmse_outsample < enr.rmse_outsample) { #RR RMSE then MAE VS ALL
      cat("RMSE values (Out-sample) of Ridge Regression model are lowest!",'\n')
      modelnamegenerator <- as.character(paste('rr.rmseout.bestmodel_id',as.character(model_id),as.character(split),sep = "_"))
      assign(modelnamegenerator,rr,envir = .GlobalEnv) #save the model object for later testing
    } else if (lr.rmse_outsample < ffr.rmse_outsample & lr.rmse_outsample < rr.rmse_outsample & lr.rmse_outsample < enr.rmse_outsample) { #LR RMSE then MAE VS ALL
      cat("RMSE values (Out-sample) of Lasso Regression model are lowest!",'\n')
      modelnamegenerator <- as.character(paste('lr.rmseout.bestmodel_id',as.character(model_id),as.character(split),sep = "_"))
      assign(modelnamegenerator,lr,envir = .GlobalEnv) #save the model object for later testing
    } else if (enr.rmse_outsample < ffr.rmse_outsample & enr.rmse_outsample < rr.rmse_outsample & enr.rmse_outsample < lr.rmse_outsample) { #ENR RMSE then MAE VS ALL
      cat("RMSE values (Out-sample) of Elastic Net Regression model are lowest!",'\n')
      modelnamegenerator <- as.character(paste('enr.rmseout.bestmodel_id',as.character(model_id),as.character(split),sep = "_"))
      assign(modelnamegenerator,enr,envir = .GlobalEnv) #save the model object for later testing
    } else { #ACCOUNT FOR SCENARIO WHERE SOME VALUES ARE EQUAL, SAY WHICH VALUES (TRUE/FALSE)
      cat('There are RMSE values that are equal','\n','FF and RR?:',(ffr.rmse_outsample == rr.rmse_outsample),ffr.rmse_outsample,rr.rmse_outsample,'\n','FF and LR?:',(ffr.rmse_outsample == lr.rmse_outsample),ffr.rmse_outsample,lr.rmse_outsample,'\n','FF and ENR?:',(ffr.rmse_outsample == enr.rmse_outsample),ffr.rmse_outsample,enr.rmse_outsample,'\n',
          '\n','RR and LR?:',(rr.rmse_outsample == lr.rmse_outsample),rr.rmse_outsample,lr.rmse_outsample,'\n','RR and ENR?:',(rr.rmse_outsample == enr.rmse_outsample),rr.rmse_outsample,enr.rmse_outsample,'\n',
          '\n','LR and ENR:',(lr.rmse_outsample == enr.rmse_outsample),lr.rmse_outsample,enr.rmse_outsample,'\n')
      modelnamegenerator.ffr <- as.character(paste('ffr.rmseout.bestmodel_id',as.character(model_id),as.character(split),sep = "_"))
      assign(modelnamegenerator.ffr,ffr,envir = .GlobalEnv) #save the model object for later testing
      modelnamegenerator.rr <- as.character(paste('rr.rmseout.bestmodel_id',as.character(model_id),as.character(split),sep = "_"))
      assign(modelnamegenerator.rr,rr,envir = .GlobalEnv) #save the model object for later testing
      modelnamegenerator.lr <- as.character(paste('lr.rmseout.bestmodel_id',as.character(model_id),as.character(split),sep = "_"))
      assign(modelnamegenerator.lr,lr,envir = .GlobalEnv) #save the model object for later testing
      modelnamegenerator.enr <- as.character(paste('enr.rmseout.bestmodel_id',as.character(model_id),as.character(split),sep = "_"))
      assign(modelnamegenerator.enr,enr,envir = .GlobalEnv) #save the model object for later testing
    }

    # > STEP 4: Which model has lowest MAE? (Out-sample)
    if (ffr.mae_outsample < rr.mae_outsample & ffr.mae_outsample < lr.mae_outsample & ffr.mae_outsample < enr.mae_outsample) {
      cat("MAE values (Out-sample) of Fast Forward Regression model are lowest!")
    } else if (rr.mae_outsample < ffr.mae_outsample & rr.mae_outsample < lr.mae_outsample & rr.mae_outsample < enr.mae_outsample) {
      cat("MAE values (Out-sample) of Ridge Regression model are lowest!")
    } else if (lr.mae_outsample < ffr.mae_outsample & lr.mae_outsample < rr.mae_outsample & lr.mae_outsample < enr.mae_outsample) {
      cat("MAE values (Out-sample) of Lasso Regression model are lowest!")
    } else if (enr.mae_outsample < ffr.mae_outsample & enr.mae_outsample < rr.mae_outsample & enr.mae_outsample < lr.mae_outsample) {
      cat("MAE values (Out-sample) of Elastic Net Regression model are lowest!")
    } else { #ACCOUNT FOR SCENARIO WHERE SOME VALUES ARE EQUAL, SAY WHICH VALUES (TRUE/FALSE)
      cat('There are MAE values that are equal','\n','FF and RR?:',(ffr.mae_outsample == rr.mae_outsample),'\n','FF and LR?:',(ffr.mae_outsample == lr.mae_outsample),'\n','FF and ENR?:',(ffr.mae_outsample == enr.mae_outsample),'\n',
          '\n','RR and LR?:',(rr.mae_outsample == lr.mae_outsample),'\n','RR and ENR?:',(rr.mae_outsample == enr.mae_outsample),'\n','FF and ENR?:',(ffr.mae_outsample == enr.mae_outsample),'\n',
          '\n','LR and ENR?:',(lr.mae_outsample == enr.mae_outsample))
    }
  }
}
