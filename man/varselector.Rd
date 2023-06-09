\name{varselector}
\alias{varselector}
%- Also NEED an '\alias' for EACH other topic documented here.
\title{
Quickly select important variables.
}
\description{
This function allows you to quickly run (seperately or together) a Fast-forward, Lasso, Ridge and Elastic Net regressions. It uses RMSE and MAE to compare the models from each technique and makes a final recommendation for variable selection.

Beside a final conclusion on variable selection, you are also provided all information that was used for said conclusion. Finally, the created models are saved to your environment, in case they are needed.
}
\usage{
varselector(
  df,
  y,
  cv,
  lambda,
  alpha,
  model_id = NULL,
  mode = NULL,
  split = NULL
)
}
\arguments{
--- Mandatory Arguments ---

These need to be specified, as they have no defaults.

  \item{df}{ Dataframe object that is unsplit in terms of train/test data. Also, subsetted, meaning remove any unwanted columns you don't want in the model.

  --- e.g., df = select(analysis_df, -id)}
  \item{y}{String that represents the name of your dependent variable, as it is called in your dataframe.

  --- e.g., y = 'clicks'}

  \item{cv}{trainControl object that specifies the cross-validation (CV) folds. See documentation of caret package for additional information --- ?caret::train() & ?caret::trainControl()

  --- e.g., cv = trainControl(method = "cv", number = 5)}

  \item{lambda}{Specify the lambda value used for tuning of the ridge and lasso models in the tuneGrid parameter of caret::train().

  --- e.g., lambda = c(seq(0.1, 2, by =0.1) ,  seq(2, 5, 0.5) , seq(5, 25, 5))}

  \item{alpha}{Specify the alpha value used for tuning of the elastic net model in the tuneGrid parameter of caret::train(). This value is 0 by default for Ridge and 1 for Lasso. In the Elastic net it is systematically varied to find the balance between Lasso-Ridge.

  --- e.g., alpha = seq(0.00, 1, 0.1)}

--- Optional Arguments ---

These do not need to be specified, as they have defaults.
  \item{model_id}{Specify a string for naming purposes of the model objects generated by varselector. This is useful varselector is used in a loop, so that model object names generated by the function will remain unique. When used in a loop make sure that each iteration of the loop the model_id value changes (e.g., 1,2,3,4...)}

  \item{mode}{Specify which regression to run --- default is all 4 methods ('all').


  --- Abbreviations:

  'ffr' = run fast-forward;

  'rr' = run ridge regression;

  'lr' = run lasso regression;

  'enr' = elastic net regression

  --- e.g., mode = 'rr'
  }
  \item{split}{A number between 0.01-0.99 specifying the proportion to split your dataframe into  test-train datasets within the function. By default the value is 0.80, meaning 80\% training and 20\% testing split.

  --- e.g., split = 0.75}
}
\details{
This function utilizes multiple features of the "caret" library (Kuhn, 2022) - train(), RMSE() and MAE(). It also uses features from "stats" library (R Core Team, 2022) - predict().
}
\value{
The function returns a comprehensive output in the console, as well as a model objects in your environment.

The console output consists of an initial conclusion on which model has the best RMSE and what the most important predictor variables are. It elaborates by also providing all information that is used to make this conclusion, including: RMSE & MAE values (in- and out-sample) &  coefficient values in each model.
}
\references{
  Kuhn M (2022). _caret: Classification and Regression Training_. R package version
  6.0-93, <https://CRAN.R-project.org/package=caret>.


  R Core Team (2022). R: A language and environment for statistical computing. R
  Foundation for Statistical Computing, Vienna, Austria. URL
  https://www.R-project.org/.
}
\author{
%%  ~~who you are~~
}
\note{
%%  ~~further notes~~
}

%% ~Make other sections like Warning with \section{Warning }{....} ~

\seealso{
?caret::train()

?caret::RMSE()

?stats::predict()
}
