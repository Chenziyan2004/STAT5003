# Compare the five selected model specifications using CV evidence only.
# This script never reads or evaluates the final test set.

source("model_setup.R")

selection_files <- file.path(
  RESULTS_DIR,
  paste0(MODEL_KEYS, "_selection.rds")
)
missing_selection_files <- selection_files[!file.exists(selection_files)]
if (length(missing_selection_files) > 0L) {
  stop(
    "Missing full tuning selections: ",
    paste(missing_selection_files, collapse = ", "),
    ". Finish all five full tuning runs before comparison.",
    call. = FALSE
  )
}

dir.create(RESULTS_DIR, recursive = TRUE, showWarnings = FALSE)
dir.create(FIGURE_DIR, recursive = TRUE, showWarnings = FALSE)

selections <- stats::setNames(lapply(selection_files, readRDS), MODEL_KEYS)
if (any(vapply(selections, function(x) x$mode != "full", logical(1)))) {
  stop("model_comparison.R requires full, not smoke, selections.", call. = FALSE)
}

extract_metric <- function(metrics, metric, class = NA_character_) {
  selected <- metrics |>
    dplyr::filter(.metric == metric)
  if (is.na(class)) {
    selected <- selected |> dplyr::filter(is.na(.data$class))
  } else {
    selected <- selected |> dplyr::filter(.data$class == .env$class)
  }
  if (nrow(selected) != 1L) {
    stop("Could not uniquely identify metric: ", metric, " / ", class)
  }
  selected
}

comparison <- purrr::imap_dfr(selections, function(selection, model_key) {
  metrics <- selection$selected_metrics
  macro <- extract_metric(metrics, "macro_f_meas")
  balanced <- extract_metric(metrics, "bal_accuracy")
  poor_recall <- extract_metric(metrics, "recall", "Poor")
  kappa <- extract_metric(metrics, "kap")
  accuracy <- extract_metric(metrics, "accuracy")
  log_loss <- extract_metric(metrics, "mn_log_loss")

  tibble::tibble(
    model_key = model_key,
    model = selection$model_label,
    strategy = selection$strategy,
    macro_f1 = macro$mean,
    macro_f1_std_err = macro$std_err,
    balanced_accuracy = balanced$mean,
    balanced_accuracy_std_err = balanced$std_err,
    poor_recall = poor_recall$mean,
    poor_recall_std_err = poor_recall$std_err,
    kappa = kappa$mean,
    kappa_std_err = kappa$std_err,
    accuracy = accuracy$mean,
    accuracy_std_err = accuracy$std_err,
    multiclass_log_loss = log_loss$mean,
    multiclass_log_loss_std_err = log_loss$std_err
  )
}) |>
  dplyr::arrange(dplyr::desc(macro_f1)) |>
  dplyr::mutate(cv_rank = dplyr::row_number(), .before = 1)

selected_hyperparameters <- purrr::imap_dfr(
  selections,
  function(selection, model_key) {
    selection$params |>
      dplyr::mutate(
        model_key = model_key,
        model = selection$model_label,
        strategy = selection$strategy,
        .before = 1
      )
  }
)

class_metrics <- purrr::imap_dfr(selections, function(selection, model_key) {
  selection$selected_metrics |>
    dplyr::filter(.metric %in% c("precision", "recall")) |>
    dplyr::mutate(
      model_key = model_key,
      model = selection$model_label,
      strategy = selection$strategy,
      .before = 1
    )
})

prediction_files <- file.path(
  RESULTS_DIR,
  paste0(MODEL_KEYS, "_predictions.rds")
)
pooled_confusions <- purrr::map2_dfr(
  prediction_files,
  MODEL_KEYS,
  function(path, model_key) {
    predictions <- readRDS(path)
    compute_pooled_confusion(predictions) |>
      dplyr::mutate(
        model_key = model_key,
        model = MODEL_LABELS[[model_key]],
        .before = 1
      )
  }
)

baseline_file <- file.path(RESULTS_DIR, "majority_baseline_metrics.csv")
if (!file.exists(baseline_file)) {
  objects <- load_training_objects(smoke = FALSE)
  create_majority_baseline(objects$cv_folds, RESULTS_DIR)
  rm(objects)
  invisible(gc())
}
baseline_metrics <- readr::read_csv(
  baseline_file,
  show_col_types = FALSE
)
baseline_macro <- baseline_metrics |>
  dplyr::filter(.metric == "macro_f_meas") |>
  dplyr::transmute(
    model = "Majority baseline",
    macro_f1 = mean,
    macro_f1_std_err = std_err
  )

comparison_plot_data <- dplyr::bind_rows(
  comparison |>
    dplyr::select(model, macro_f1, macro_f1_std_err),
  baseline_macro
) |>
  dplyr::mutate(model = stats::reorder(model, macro_f1))

comparison_plot <- ggplot2::ggplot(
  comparison_plot_data,
  ggplot2::aes(x = model, y = macro_f1)
) +
  ggplot2::geom_col(fill = "#3F6B8A", width = 0.7) +
  ggplot2::geom_errorbar(
    ggplot2::aes(
      ymin = macro_f1 - macro_f1_std_err,
      ymax = macro_f1 + macro_f1_std_err
    ),
    width = 0.18,
    na.rm = TRUE
  ) +
  ggplot2::coord_flip() +
  ggplot2::labs(
    title = "Grouped cross-validation model comparison",
    subtitle = "Mean strict Macro-F1 ± one standard error",
    x = NULL,
    y = "Macro-F1"
  ) +
  ggplot2::theme_minimal(base_size = 12)

class_plot <- ggplot2::ggplot(
  class_metrics,
  ggplot2::aes(x = model, y = mean, colour = class, group = class)
) +
  ggplot2::geom_point(
    position = ggplot2::position_dodge(width = 0.55),
    size = 2.5
  ) +
  ggplot2::geom_errorbar(
    ggplot2::aes(ymin = mean - std_err, ymax = mean + std_err),
    position = ggplot2::position_dodge(width = 0.55),
    width = 0.18
  ) +
  ggplot2::coord_flip() +
  ggplot2::facet_wrap(~.metric) +
  ggplot2::scale_colour_manual(
    values = c(Good = "#2E7D32", Fair = "#F9A825", Poor = "#C62828")
  ) +
  ggplot2::labs(
    title = "Class-specific CV performance",
    subtitle = "Mean ± one standard error across grouped folds",
    x = NULL,
    y = "Estimate",
    colour = "Health"
  ) +
  ggplot2::theme_minimal(base_size = 12) +
  ggplot2::theme(legend.position = "bottom")

readr::write_csv(comparison, file.path(RESULTS_DIR, "model_comparison.csv"))
readr::write_csv(
  selected_hyperparameters,
  file.path(RESULTS_DIR, "selected_hyperparameters.csv")
)
readr::write_csv(
  class_metrics,
  file.path(RESULTS_DIR, "class_precision_recall_comparison.csv")
)
readr::write_csv(
  pooled_confusions,
  file.path(RESULTS_DIR, "pooled_cv_confusion_matrices.csv")
)
ggplot2::ggsave(
  file.path(FIGURE_DIR, "cv_model_comparison.png"),
  comparison_plot,
  width = 8,
  height = 5,
  dpi = 300,
  bg = "white"
)
ggplot2::ggsave(
  file.path(FIGURE_DIR, "class_precision_recall_comparison.png"),
  class_plot,
  width = 9,
  height = 7,
  dpi = 300,
  bg = "white"
)

winning_key <- comparison$model_key[[1]]
winning_selection <- selections[[winning_key]]
winning_selection$comparison_table <- comparison
winning_selection$selected_by <- "highest mean grouped-CV strict Macro-F1"
saveRDS(
  winning_selection,
  file.path(RESULTS_DIR, "selected_model.rds")
)
write_modelling_session_info(RESULTS_DIR)

message(
  "CV winner: ", winning_selection$model_label,
  " [", winning_selection$strategy, "] with Macro-F1 = ",
  signif(comparison$macro_f1[[1]], 5),
  ". The final test set has not been used."
)
