# Install only modelling packages that are not already available.

required <- c(
  "tidymodels", "glmnet", "discrim", "klaR", "rpart", "ranger",
  "xgboost", "themis", "future", "rpart.plot", "broom"
)

missing <- required[
  !vapply(required, requireNamespace, logical(1), quietly = TRUE)
]

if (length(missing) == 0L) {
  message("All modelling packages are already installed.")
} else {
  message("Installing: ", paste(missing, collapse = ", "))
  install.packages(missing, repos = "https://cloud.r-project.org")
}

still_missing <- required[
  !vapply(required, requireNamespace, logical(1), quietly = TRUE)
]
if (length(still_missing) > 0L) {
  stop(
    "Installation did not complete for: ",
    paste(still_missing, collapse = ", "),
    call. = FALSE
  )
}

message("All modelling dependencies are available.")
