Activity 7 - Linear Discriminant Analysis
================

# Task 1: Load the necessary packages

``` r
library(tidyverse)
```

    ## ── Attaching packages ─────────────────────────────────────── tidyverse 1.3.2 ──
    ## ✔ ggplot2 3.3.6     ✔ purrr   0.3.4
    ## ✔ tibble  3.1.8     ✔ dplyr   1.0.9
    ## ✔ tidyr   1.2.0     ✔ stringr 1.4.1
    ## ✔ readr   2.1.2     ✔ forcats 0.5.2
    ## ── Conflicts ────────────────────────────────────────── tidyverse_conflicts() ──
    ## ✖ dplyr::filter() masks stats::filter()
    ## ✖ dplyr::lag()    masks stats::lag()

``` r
library(tidymodels)
```

    ## ── Attaching packages ────────────────────────────────────── tidymodels 1.0.0 ──
    ## ✔ broom        1.0.0     ✔ rsample      1.1.0
    ## ✔ dials        1.0.0     ✔ tune         1.0.0
    ## ✔ infer        1.0.3     ✔ workflows    1.0.0
    ## ✔ modeldata    1.0.0     ✔ workflowsets 1.0.0
    ## ✔ parsnip      1.0.1     ✔ yardstick    1.0.0
    ## ✔ recipes      1.0.1     
    ## ── Conflicts ───────────────────────────────────────── tidymodels_conflicts() ──
    ## ✖ scales::discard() masks purrr::discard()
    ## ✖ dplyr::filter()   masks stats::filter()
    ## ✖ recipes::fixed()  masks stringr::fixed()
    ## ✖ dplyr::lag()      masks stats::lag()
    ## ✖ yardstick::spec() masks readr::spec()
    ## ✖ recipes::step()   masks stats::step()
    ## • Learn how to get started at https://www.tidymodels.org/start/

``` r
library(discrim)
```

    ## 
    ## Attaching package: 'discrim'
    ## 
    ## The following object is masked from 'package:dials':
    ## 
    ##     smoothness

# Task 2: Load the data and

``` r
resume <- data.frame(readr::read_csv("https://www.openintro.org/data/csv/resume.csv"))
```

    ## Rows: 4870 Columns: 30
    ## ── Column specification ────────────────────────────────────────────────────────
    ## Delimiter: ","
    ## chr (10): job_city, job_industry, job_type, job_ownership, job_req_min_exper...
    ## dbl (20): job_ad_id, job_fed_contractor, job_equal_opp_employer, job_req_any...
    ## 
    ## ℹ Use `spec()` to retrieve the full column specification for this data.
    ## ℹ Specify the column types or set `show_col_types = FALSE` to quiet this message.

# Linear discriminate analysis (LDA)

Linear discriminate analysis (LDA) is another method for classification
problems. For LDA, we make assumptions about the distribution of the
explanatory/independent variable(s) given the response/dependent
variable (e.g., X\|Y \~ Normal). Note that because of this assumption,
LDA will only work with qualitative explanatory variables. Also note
that because of logistic regression, some believe (like the great
Dr. Frank Harrell) that LDA is obsolete. This method then uses Bayes’
theorem to build a classifier for the likelihood of belonging to one of
the levels of the response variable.

When the dependent/target variable has 2 classes, then Logistic
Regressio is more suitable i.e. it is a type of binary classification.
Linear discriminant analysis (LDA) performs well when there is multi
class classification problem at hand i.e. when the dependent/target
variable has more than 2 classes, then Linear discriminate analysis is
suitable.

Dependent/Response variable must be Categorical variable. They can’t be
numbers. If 0 or 1 are there we need to change them to categorical
variable.

# Task 3: LDA

``` r
# Convert received_callback to a factor with more informative labels
resume <- resume %>% 
  mutate(received_callback = factor(received_callback, labels = c("No", "Yes")))

# LDA
lda_years <- discrim_linear() %>% 
  set_mode("classification") %>% 
  set_engine("MASS") %>% 
  fit(received_callback ~ log(years_experience), data = resume)

lda_years
```

    ## parsnip model object
    ## 
    ## Call:
    ## lda(received_callback ~ log(years_experience), data = data)
    ## 
    ## Prior probabilities of groups:
    ##         No        Yes 
    ## 0.91950719 0.08049281 
    ## 
    ## Group means:
    ##     log(years_experience)
    ## No               1.867135
    ## Yes              1.998715
    ## 
    ## Coefficients of linear discriminants:
    ##                            LD1
    ## log(years_experience) 1.638023

Looking at the Group means: portion of the above output, you have the
mean of log(years\_experience) for whether a résumé received a call
back.

Question: 2. What do you notice about these two values? 3. How does this
correspond to the appropriate density plot above? That is, what feature
of these curves are you comparing?

# Task 4: Predictions

``` r
# Predictions for LDA
predict(lda_years, new_data = resume, type = "prob")
```

    ## # A tibble: 4,870 × 2
    ##    .pred_No .pred_Yes
    ##       <dbl>     <dbl>
    ##  1    0.923    0.0769
    ##  2    0.923    0.0769
    ##  3    0.923    0.0769
    ##  4    0.923    0.0769
    ##  5    0.884    0.116 
    ##  6    0.923    0.0769
    ##  7    0.928    0.0724
    ##  8    0.885    0.115 
    ##  9    0.939    0.0612
    ## 10    0.923    0.0769
    ## # … with 4,860 more rows

This output gives us the likelihood for a particular résumé to be in the
“No” or “Yes” callback group. However, looking at this is rather
overwhelming (over 4,000 observations!) so we will now create a
confusion matrix to look at the performance of our model.

``` r
# Creating a confusion matrix
augment(lda_years, new_data = resume) %>% 
  conf_mat(truth = received_callback, estimate = .pred_class)
```

    ##           Truth
    ## Prediction   No  Yes
    ##        No  4478  392
    ##        Yes    0    0

4.  What do you notice? Why do you think this happened?

``` r
# Overall accuracy of our model(i.e., how often did it correctly predict the actual outcome)
augment(lda_years, new_data = resume) %>% 
  accuracy(truth = received_callback, estimate = .pred_class)
```

    ## # A tibble: 1 × 3
    ##   .metric  .estimator .estimate
    ##   <chr>    <chr>          <dbl>
    ## 1 accuracy binary         0.920

So the model was right about 92% of the time… because it never predicted
that someone would receive a callback. Note that this 92% corresponds to
the “No” rate in the response variable

Challenge:

``` r
# Logistic regression requires that the response be a factor variable
resume_model <- logistic_reg() %>%
  set_engine("glm") %>%
  fit(received_callback ~ log(years_experience), data = resume, family = "binomial")

tidy(resume_model) %>% 
  knitr::kable(digits = 3)
```

| term                   | estimate | std.error | statistic | p.value |
|:-----------------------|---------:|----------:|----------:|--------:|
| (Intercept)            |   -3.129 |     0.182 |   -17.157 |       0 |
| log(years\_experience) |    0.358 |     0.088 |     4.082 |       0 |

``` r
tidy(resume_model, exponentiate = TRUE) %>% 
  knitr::kable(digits = 3)
```

| term                   | estimate | std.error | statistic | p.value |
|:-----------------------|---------:|----------:|----------:|--------:|
| (Intercept)            |    0.044 |     0.182 |   -17.157 |       0 |
| log(years\_experience) |    1.431 |     0.088 |     4.082 |       0 |
