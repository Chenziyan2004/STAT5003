# Tune one or all candidate models using the fixed grouped CV folds.
#
# Examples:
#   Rscript model_training.R --model=cart --mode=smoke --workers=2
#   Rscript model_training.R --model=cart --mode=full --workers=2
#   Rscript model_training.R --model=all --mode=full --workers=2

source("model_setup.R")

run_strategy_tuning <- function(
    model_key,
    strategy,
    development_data,
    tuning_folds,
    grid,
    smoke,
    workers,
    checkpoint_dir,
    overwrite = FALSE) {
  mode_name <- if (smoke) "smoke" else "full"
  checkpoint_file <- file.path(
    checkpoint_dir,
    paste0(model_key, "_", strategy, "_checkpoint.rds")
  )

  if (file.exists(checkpoint_file) && !overwrite) {
    message("Loading checkpoint: ", checkpoint_file)
    checkpoint <- readRDS(checkpoint_file)
    expected <- list(
      model_key = model_key,
      strategy = strategy,
      mode = mode_name,
      grid_size = nrow(grid)
    )
    matching <- vapply(
      names(expected),
      function(name) identical(checkpoint$metadata[[name]], expected[[name]]),
      logical(1)
    )
    if (!all(matching)) {
      stop(
        "Checkpoint metadata does not match this run. Use --overwrite=true ",
        "or remove the checkpoint.",
        call. = FALSE
      )
    }
    return(checkpoint)
  }

  message(
    "Tuning ", MODEL_LABELS[[model_key]], " [", strategy, "] with ",
    nrow(tuning_folds), " fold(s) and ", nrow(grid), " configuration(s)."
  )

  workflow_spec <- build_workflow(
    model_key = model_key,
    data = development_data,
    strategy = strategy,
    smoke = smoke,
    final_fit = FALSE
  )

  parallel_mode <- if (nrow(tuning_folds) > 1L) "resamples" else "everything"
  engine_packages <- MODEL_ENGINE_PACKAGES[[model_key]]
  worker_packages <- unique(c(
    engine_packages,
    if (strategy == "downsampled") "themis" else character()
  ))

  control <- tune::control_grid(
    verbose = workers == 1L,
    allow_par = workers > 1L,
    save_pred = TRUE,
    save_workflow = TRUE,
    parallel_over = parallel_mode,
    pkgs = worker_packages
  )

  set.seed(MODEL_SEEDS[[model_key]] + match(strategy, c("natural", "downsampled")))
  tuning_result <- tune::tune_grid(
    workflow_spec,
    resamples = tuning_folds,
    grid = grid,
    metrics = common_metrics,
    control = control
  )

  metrics <- tune::collect_metrics(tuning_result, summarize = TRUE) |>
    dplyr::mutate(strategy = strategy, .before = 1)
  macro_rows <- metrics |>
    dplyr::filter(.metric == "macro_f_meas", is.finite(mean))
  if (nrow(macro_rows) == 0L) {
    notes <- tune::collect_notes(tuning_result)
    stop(
      "No successful Macro-F1 estimates for ", model_key, " [", strategy,
      "]. First notes: ", paste(utils::head(notes$note, 3L), collapse = " | "),
      call. = FALSE
    )
  }

  best_params <- tune::select_best(tuning_result, metric = "macro_f_meas")
  best_predictions <- tune::collect_predictions(
    tuning_result,
    parameters = best_params,
    summarize = FALSE
  )
  prediction_validation <- validate_predictions(best_predictions)
  common_fold_metrics <- compute_common_fold_metrics(best_predictions)
  class_fold_metrics <- compute_class_fold_metrics(best_predictions)
  selected_metrics <- summarise_fold_metrics(
    common_fold_metrics,
    class_fold_metrics
  )
  confusion <- compute_pooled_confusion(best_predictions)
  best_macro <- metrics |>
    dplyr::filter(.metric == "macro_f_meas") |>
    dplyr::slice_max(mean, n = 1L, with_ties = FALSE)

  checkpoint <- list(
    metadata = list(
      model_key = model_key,
      strategy = strategy,
      mode = mode_name,
      grid_size = nrow(grid),
      folds = nrow(tuning_folds),
      created_at = as.character(Sys.time())
    ),
    grid = grid,
    compact_tuning_result = compact_tune_result(tuning_result),
    metrics = metrics,
    best_params = best_params,
    best_macro = best_macro,
    predictions = best_predictions,
    prediction_validation = prediction_validation,
    common_fold_metrics = common_fold_metrics,
    class_fold_metrics = class_fold_metrics,
    selected_metrics = selected_metrics,
    confusion = confusion,
    notes = tune::collect_notes(tuning_result)
  )

  # The checkpoint is written as soon as each strategy finishes. It contains
  # only predictions for that strategy's best configuration, avoiding a very
  # large all-grid prediction object while preserving all tuning metrics.
  saveRDS(checkpoint, checkpoint_file)
  message("Saved checkpoint: ", checkpoint_file)
  checkpoint
}

run_one_model <- function(
    model_key,
    mode = c("smoke", "full"),
    strategies = "both",
    workers = DEFAULT_WORKERS,
    grid_size = NULL,
    overwrite = FALSE) {
  mode <- match.arg(mode)
  smoke <- mode == "smoke"
  strategy_values <- normalize_strategies(strategies)
  check_model_packages(model_key, strategy_values)

  output_dir <- if (smoke) file.path(RESULTS_DIR, "smoke") else RESULTS_DIR
  checkpoint_dir <- file.path(output_dir, "checkpoints")
  dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)
  dir.create(checkpoint_dir, recursive = TRUE, showWarnings = FALSE)

  objects <- load_training_objects(smoke = smoke)
  development_data <- objects$development_data
  tuning_folds <- objects$tuning_folds

  if (is.null(grid_size)) {
    grid_size <- if (smoke) {
      DEFAULT_SMOKE_GRID_SIZE
    } else if (model_key == "xgboost") {
      DEFAULT_XGB_GRID_SIZE
    } else {
      DEFAULT_FULL_GRID_SIZE
    }
  }
  grid <- build_tuning_grid(model_key, grid_size, smoke)

  old_future_size <- getOption("future.globals.maxSize")
  options(future.globals.maxSize = 16 * 1024^3)
  on.exit(options(future.globals.maxSize = old_future_size), add = TRUE)

  if (workers > 1L) {
    future::plan(future::multisession, workers = workers)
  } else {
    future::plan(future::sequential)
  }
  on.exit(future::plan(future::sequential), add = TRUE)

  strategy_runs <- stats::setNames(
    purrr::map(strategy_values, function(strategy) {
      run_strategy_tuning(
        model_key = model_key,
        strategy = strategy,
        development_data = development_data,
        tuning_folds = tuning_folds,
        grid = grid,
        smoke = smoke,
        workers = workers,
        checkpoint_dir = checkpoint_dir,
        overwrite = overwrite
      )
    }),
    strategy_values
  )

  selected_strategy_row <- select_strategy_result(strategy_runs)
  selected_strategy <- selected_strategy_row$strategy[[1]]
  selected_run <- strategy_runs[[selected_strategy]]
  prefix <- model_file_prefix(model_key)

  all_metrics <- purrr::map_dfr(strategy_runs, "metrics") |>
    dplyr::mutate(
      model = MODEL_LABELS[[model_key]],
      .before = 1
    )
  selected_metrics <- selected_run$selected_metrics |>
    dplyr::mutate(
      model = MODEL_LABELS[[model_key]],
      strategy = selected_strategy,
      .before = 1
    )
  selected_params <- selected_run$best_params |>
    dplyr::mutate(
      model = MODEL_LABELS[[model_key]],
      model_key = model_key,
      strategy = selected_strategy,
      .before = 1
    )
  selected_predictions <- selected_run$predictions |>
    dplyr::mutate(
      model = MODEL_LABELS[[model_key]],
      strategy = selected_strategy,
      .before = 1
    )

  tuning_archive <- list(
    model_key = model_key,
    model_label = MODEL_LABELS[[model_key]],
    mode = mode,
    recipe_family = MODEL_RECIPE_FAMILIES[[model_key]],
    under_ratio = UNDER_RATIO,
    strategies = purrr::map(strategy_runs, function(x) {
      x[c(
        "metadata", "grid", "compact_tuning_result", "metrics",
        "best_params", "best_macro", "prediction_validation", "notes"
      )]
    })
  )

  selection <- list(
    model_key = model_key,
    model_label = MODEL_LABELS[[model_key]],
    mode = mode,
    strategy = selected_strategy,
    recipe_family = MODEL_RECIPE_FAMILIES[[model_key]],
    params = selected_run$best_params,
    selected_metrics = selected_metrics,
    selected_strategy_comparison = selected_strategy_row,
    under_ratio = UNDER_RATIO
  )

  saveRDS(
    tuning_archive,
    file.path(output_dir, paste0(prefix, "_tuning.rds"))
  )
  saveRDS(
    selected_predictions,
    file.path(output_dir, paste0(prefix, "_predictions.rds"))
  )
  saveRDS(
    selection,
    file.path(output_dir, paste0(prefix, "_selection.rds"))
  )
  readr::write_csv(
    all_metrics,
    file.path(output_dir, paste0(prefix, "_metrics.csv"))
  )
  readr::write_csv(
    selected_metrics,
    file.path(output_dir, paste0(prefix, "_selected_metrics.csv"))
  )
  readr::write_csv(
    selected_params,
    file.path(output_dir, paste0(prefix, "_selected_hyperparameters.csv"))
  )
  readr::write_csv(
    selected_run$confusion,
    file.path(output_dir, paste0(prefix, "_confusion_matrix.csv"))
  )
  readr::write_csv(
    selected_run$prediction_validation,
    file.path(output_dir, paste0(prefix, "_prediction_validation.csv"))
  )
  readr::write_csv(
    dplyr::bind_rows(
      selected_run$common_fold_metrics,
      selected_run$class_fold_metrics
    ),
    file.path(output_dir, paste0(prefix, "_selected_fold_metrics.csv"))
  )
  write_modelling_session_info(output_dir)

  message(
    "Selected ", MODEL_LABELS[[model_key]], " strategy: ",
    selected_strategy, "; CV Macro-F1 = ",
    signif(selected_strategy_row$mean[[1]], 5)
  )
  invisible(selection)
}

main <- function() {
  arguments <- parse_named_arguments(commandArgs(trailingOnly = TRUE))
  model_argument <- arguments$model
  if (is.null(model_argument)) {
    stop(
      "Specify --model=baseline, one model key, or all. Valid model keys: ",
      paste(MODEL_KEYS, collapse = ", "),
      call. = FALSE
    )
  }
  mode <- if (is.null(arguments$mode)) "smoke" else arguments$mode
  if (!mode %in% c("smoke", "full")) {
    stop("--mode must be smoke or full.", call. = FALSE)
  }
  workers <- safe_integer(arguments$workers, DEFAULT_WORKERS, "workers")
  grid_size <- if (is.null(arguments$`grid-size`)) {
    NULL
  } else {
    safe_integer(arguments$`grid-size`, 1L, "grid-size")
  }
  strategies <- if (is.null(arguments$strategies)) {
    "both"
  } else {
    arguments$strategies
  }
  overwrite <- as_flag(arguments$overwrite, FALSE)
  smoke <- mode == "smoke"
  output_dir <- if (smoke) file.path(RESULTS_DIR, "smoke") else RESULTS_DIR

  if (model_argument == "baseline") {
    objects <- load_training_objects(smoke = smoke)
    create_majority_baseline(objects$tuning_folds, output_dir)
    write_modelling_session_info(output_dir)
    message("Majority baseline saved to ", output_dir)
    return(invisible(NULL))
  }

  models <- if (model_argument == "all") MODEL_KEYS else model_argument
  if (any(!models %in% MODEL_KEYS)) {
    stop(
      "Unknown model. Valid values: baseline, all, ",
      paste(MODEL_KEYS, collapse = ", "),
      call. = FALSE
    )
  }

  # The baseline is inexpensive and is regenerated alongside each run so the
  # corresponding output directory always contains its reference metrics.
  objects <- load_training_objects(smoke = smoke)
  create_majority_baseline(objects$tuning_folds, output_dir)
  rm(objects)
  invisible(gc())

  for (model_key in models) {
    run_one_model(
      model_key = model_key,
      mode = mode,
      strategies = strategies,
      workers = workers,
      grid_size = grid_size,
      overwrite = overwrite
    )
  }
}

if (sys.nframe() == 0L) {
  main()
}
