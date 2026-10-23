# Fit the CV-selected workflow on all development data and evaluate the final
# test set exactly once via last_fit(). Run only after model_comparison.R.

source("model_setup.R")

selection_file <- file.path(RESULTS_DIR, "selected_model.rds")
if (!file.exists(selection_file)) {
  stop(
    "Missing results/selected_model.rds. Run model_comparison.R first.",
    call. = FALSE
  )
}

selection <- readRDS(selection_file)
if (!identical(selection$mode, "full")) {
  stop("The selected model must come from full five-fold tuning.", call. = FALSE)
}
model_key <- selection$model_key
strategy <- selection$strategy
check_model_packages(model_key, strategy)

data_split <- readRDS("data/data_split.rds")
development_data <- rsample::training(data_split)

workflow_spec <- build_workflow(
  model_key = model_key,
  data = development_data,
  strategy = strategy,
  smoke = FALSE,
  final_fit = TRUE
)
final_parameters <- selection$params |>
  dplyr::select(-dplyr::any_of(".config"))
final_workflow <- tune::finalize_workflow(workflow_spec, final_parameters)

set.seed(MODEL_SEEDS[[model_key]] + 9000L)
last_fit_result <- tune::last_fit(
  final_workflow,
  split = data_split,
  metrics = common_metrics,
  control = tune::control_last_fit(
    verbose = TRUE,
    allow_par = FALSE
  )
)

final_predictions <- tune::collect_predictions(last_fit_result)
prediction_validation <- validate_predictions(final_predictions)
common_test_metrics <- compute_common_fold_metrics(final_predictions)
class_test_metrics <- compute_class_fold_metrics(final_predictions)
final_test_metrics <- dplyr::bind_rows(
  common_test_metrics,
  class_test_metrics
) |>
  dplyr::select(.metric, .estimator, class, .estimate) |>
  dplyr::rename(estimate = .estimate) |>
  dplyr::mutate(
    model = selection$model_label,
    strategy = strategy,
    .before = 1
  )
final_confusion <- compute_pooled_confusion(final_predictions)
fitted_workflow <- hardhat::extract_workflow(last_fit_result)

dir.create(RESULTS_DIR, recursive = TRUE, showWarnings = FALSE)
dir.create(FIGURE_DIR, recursive = TRUE, showWarnings = FALSE)

saveRDS(fitted_workflow, file.path(RESULTS_DIR, "final_model.rds"))
saveRDS(last_fit_result, file.path(RESULTS_DIR, "final_last_fit.rds"))
readr::write_csv(
  final_predictions,
  file.path(RESULTS_DIR, "final_test_predictions.csv")
)
readr::write_csv(
  final_test_metrics,
  file.path(RESULTS_DIR, "final_test_metrics.csv")
)
readr::write_csv(
  final_confusion,
  file.path(RESULTS_DIR, "final_confusion_matrix.csv")
)
readr::write_csv(
  prediction_validation,
  file.path(RESULTS_DIR, "final_prediction_validation.csv")
)

confusion_plot <- ggplot2::ggplot(
  final_confusion,
  ggplot2::aes(x = estimate, y = truth, fill = row_proportion)
) +
  ggplot2::geom_tile(colour = "white") +
  ggplot2::geom_text(
    ggplot2::aes(
      label = paste0(
        format(n, big.mark = ","),
        "\n",
        scales::percent(row_proportion, accuracy = 0.1)
      )
    ),
    size = 3.5
  ) +
  ggplot2::scale_fill_gradient(low = "#F7FBFF", high = "#08519C") +
  ggplot2::labs(
    title = paste("Final test confusion matrix:", selection$model_label),
    subtitle = paste("CV-selected strategy:", strategy),
    x = "Predicted health",
    y = "Observed health",
    fill = "Row proportion"
  ) +
  ggplot2::theme_minimal(base_size = 12)
ggplot2::ggsave(
  file.path(FIGURE_DIR, "final_confusion_matrix.png"),
  confusion_plot,
  width = 7,
  height = 5.5,
  dpi = 300,
  bg = "white"
)

save_interpretation <- function(fitted_workflow, model_key, selection) {
  engine_fit <- hardhat::extract_fit_engine(fitted_workflow)
  interpretation <- NULL
  plot_data <- NULL

  if (model_key == "logistic") {
    check_packages("broom", "logistic coefficient extraction")
    interpretation <- broom::tidy(
      engine_fit,
      s = selection$params$penalty[[1]]
    )
    readr::write_csv(
      interpretation,
      file.path(RESULTS_DIR, "final_logistic_coefficients.csv")
    )
    plot_data <- interpretation |>
      dplyr::filter(term != "(Intercept)") |>
      dplyr::mutate(importance = abs(estimate)) |>
      dplyr::group_by(term) |>
      dplyr::slice_max(importance, n = 1L, with_ties = FALSE) |>
      dplyr::ungroup() |>
      dplyr::transmute(feature = term, importance)
  } else if (model_key %in% c("cart", "random_forest")) {
    values <- engine_fit$variable.importance
    interpretation <- tibble::tibble(
      feature = names(values),
      importance = as.numeric(values)
    ) |>
      dplyr::arrange(dplyr::desc(importance))
    readr::write_csv(
      interpretation,
      file.path(RESULTS_DIR, paste0("final_", model_key, "_importance.csv"))
    )
    if (model_key == "cart") {
      check_packages("rpart.plot", "CART tree visualisation")
      saveRDS(engine_fit, file.path(RESULTS_DIR, "final_cart_tree.rds"))
      grDevices::png(
        file.path(FIGURE_DIR, "final_cart_tree.png"),
        width = 2400,
        height = 1600,
        res = 220
      )
      rpart.plot::rpart.plot(
        engine_fit,
        type = 2,
        extra = 104,
        fallen.leaves = TRUE,
        tweak = 0.85,
        main = "Final CART model"
      )
      grDevices::dev.off()
    }
    plot_data <- interpretation
  } else if (model_key == "xgboost") {
    interpretation <- xgboost::xgb.importance(model = engine_fit)
    readr::write_csv(
      interpretation,
      file.path(RESULTS_DIR, "final_xgboost_importance.csv")
    )
    plot_data <- interpretation |>
      dplyr::transmute(feature = Feature, importance = Gain)
  } else if (model_key == "naive_bayes") {
    writeLines(
      c(
        "Naive Bayes has no single engine-native feature-importance measure.",
        "Use its fitted conditional distributions and sensitivity analyses for interpretation."
      ),
      file.path(RESULTS_DIR, "final_naive_bayes_interpretation.txt")
    )
  }

  if (!is.null(plot_data) && nrow(plot_data) > 0L) {
    plot_data <- plot_data |>
      dplyr::slice_max(importance, n = 20L, with_ties = FALSE) |>
      dplyr::mutate(feature = stats::reorder(feature, importance))
    importance_plot <- ggplot2::ggplot(
      plot_data,
      ggplot2::aes(x = feature, y = importance)
    ) +
      ggplot2::geom_col(fill = "#3F6B8A") +
      ggplot2::coord_flip() +
      ggplot2::labs(
        title = paste("Final-model interpretation:", selection$model_label),
        x = NULL,
        y = "Importance magnitude"
      ) +
      ggplot2::theme_minimal(base_size = 12)
    ggplot2::ggsave(
      file.path(FIGURE_DIR, "final_model_importance.png"),
      importance_plot,
      width = 8,
      height = 6,
      dpi = 300,
      bg = "white"
    )
  }

  invisible(interpretation)
}

save_interpretation(fitted_workflow, model_key, selection)
write_modelling_session_info(RESULTS_DIR)

macro_f1 <- final_test_metrics |>
  dplyr::filter(.metric == "macro_f_meas") |>
  dplyr::pull(estimate)
poor_recall <- final_test_metrics |>
  dplyr::filter(.metric == "recall", class == "Poor") |>
  dplyr::pull(estimate)
message(
  "Final test evaluation complete for ", selection$model_label,
  ": Macro-F1 = ", signif(macro_f1, 5),
  ", Poor recall = ", signif(poor_recall, 5), "."
)
