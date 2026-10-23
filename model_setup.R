# Shared setup for STAT5003 model tuning, comparison, and final fitting.
# This file defines functions only. It does not fit models or inspect test data.

source("preprocessing.R")

RESULTS_DIR <- "results"
FIGURE_DIR <- file.path(RESULTS_DIR, "figures")
UNDER_RATIO <- 1
DEFAULT_FULL_GRID_SIZE <- 20L
DEFAULT_XGB_GRID_SIZE <- 15L
DEFAULT_SMOKE_GRID_SIZE <- 2L
DEFAULT_WORKERS <- 2L
PROBABILITY_TOLERANCE <- 1e-6

MODEL_KEYS <- c(
  "logistic", "naive_bayes", "cart", "random_forest", "xgboost"
)

MODEL_LABELS <- c(
  logistic = "Multinomial Logistic",
  naive_bayes = "Naive Bayes",
  cart = "CART",
  random_forest = "Random Forest",
  xgboost = "XGBoost"
)

MODEL_RECIPE_FAMILIES <- c(
  logistic = "scaled",
  naive_bayes = "factor",
  # rpart can accept factors, but exhaustive splitting of the retained
  # 67-level species factor was not computationally viable on 417k analysis
  # rows. The same fold-trained species pooling is retained, then dummy coded.
  cart = "dummy",
  random_forest = "factor",
  xgboost = "dummy"
)

MODEL_SEEDS <- c(
  logistic = 5101L,
  naive_bayes = 5201L,
  cart = 5301L,
  random_forest = 5401L,
  xgboost = 5501L
)

MODEL_ENGINE_PACKAGES <- list(
  logistic = "glmnet",
  naive_bayes = c("discrim", "klaR"),
  cart = "rpart",
  random_forest = "ranger",
  xgboost = "xgboost"
)

MODELLING_CORE_PACKAGES <- c(
  "dplyr", "tidyr", "purrr", "tibble", "readr", "ggplot2",
  "rsample", "recipes", "workflows", "parsnip", "tune", "dials",
  "yardstick", "future", "hardhat"
)

macro_f_meas_impl <- function(
    data,
    truth,
    estimate,
    na_rm = TRUE,
    case_weights = NULL,
    ...) {
  # Define the vector implementation inside the metric function. With a
  # multisession future backend, a top-level helper referenced only by name is
  # not always exported with the yardstick metric object. Keeping this helper
  # local makes the custom metric self-contained on every parallel worker.
  worker_safe_macro_f_meas_vec <- function(
      truth,
      estimate,
      na_rm = TRUE,
      case_weights = NULL,
      estimator = "macro",
      ...) {
    if (!is.factor(truth)) truth <- factor(truth)
    estimate <- factor(estimate, levels = levels(truth))
    weights <- if (is.null(case_weights)) {
      rep(1, length(truth))
    } else {
      as.numeric(case_weights)
    }

    missing <- is.na(truth) | is.na(estimate) | is.na(weights)
    if (any(missing)) {
      if (!na_rm) return(NA_real_)
      truth <- truth[!missing]
      estimate <- estimate[!missing]
      weights <- weights[!missing]
    }

    class_f1 <- vapply(levels(truth), function(class_name) {
      truth_positive <- truth == class_name
      predicted_positive <- estimate == class_name
      true_positive <- sum(weights[truth_positive & predicted_positive])
      false_positive <- sum(weights[!truth_positive & predicted_positive])
      false_negative <- sum(weights[truth_positive & !predicted_positive])
      denominator <- 2 * true_positive + false_positive + false_negative
      if (denominator == 0) 0 else 2 * true_positive / denominator
    }, numeric(1))

    mean(class_f1)
  }

  yardstick::class_metric_summarizer(
    name = "macro_f_meas",
    fn = worker_safe_macro_f_meas_vec,
    data = data,
    truth = {{ truth }},
    estimate = {{ estimate }},
    na_rm = na_rm,
    case_weights = {{ case_weights }},
    estimator = "macro"
  )
}

# yardstick's built-in macro F1 removes classes whose precision is undefined.
# For this project, an unpredicted class must contribute F1 = 0; otherwise the
# always-Good baseline would receive an implausibly high Macro-F1.
macro_f_meas <- yardstick::new_class_metric(
  macro_f_meas_impl,
  direction = "maximize",
  range = c(0, 1)
)

common_metrics <- yardstick::metric_set(
  macro_f_meas,
  yardstick::bal_accuracy,
  yardstick::kap,
  yardstick::accuracy,
  yardstick::mn_log_loss
)

check_packages <- function(packages, context = "this operation") {
  missing_packages <- packages[
    !vapply(packages, requireNamespace, logical(1), quietly = TRUE)
  ]
  if (length(missing_packages) > 0L) {
    stop(
      "Missing packages required for ", context, ": ",
      paste(missing_packages, collapse = ", "),
      ". Run: Rscript install_model_packages.R",
      call. = FALSE
    )
  }
  invisible(TRUE)
}

check_model_packages <- function(model_key, strategies = "both") {
  stopifnot(model_key %in% MODEL_KEYS)
  strategy_values <- normalize_strategies(strategies)
  packages <- c(MODELLING_CORE_PACKAGES, MODEL_ENGINE_PACKAGES[[model_key]])
  if ("downsampled" %in% strategy_values) {
    packages <- c(packages, "themis")
  }
  check_packages(unique(packages), paste("model", model_key))
}

load_training_objects <- function(smoke = FALSE) {
  required_files <- c(
    "data/model_data_master.rds",
    "data/data_split.rds",
    "data/cv_folds.rds"
  )
  missing_files <- required_files[!file.exists(required_files)]
  if (length(missing_files) > 0L) {
    stop(
      "Missing preprocessing objects: ",
      paste(missing_files, collapse = ", "),
      ". Run: Rscript preprocessing.R",
      call. = FALSE
    )
  }

  model_data <- readRDS("data/model_data_master.rds")
  data_split <- readRDS("data/data_split.rds")
  cv_folds <- readRDS("data/cv_folds.rds")

  if (nrow(model_data) != 652172L) {
    stop("The saved master data has an unexpected row count.", call. = FALSE)
  }
  if (nrow(cv_folds) != 5L) {
    stop("The saved grouped CV object must contain exactly five folds.", call. = FALSE)
  }

  development_data <- rsample::training(data_split)
  if (nrow(development_data) != 521746L) {
    stop("The saved development data has an unexpected row count.", call. = FALSE)
  }

  # model_data confirms identity/counts but is not needed during tuning. Removing
  # it avoids duplicating a large object in parallel workers.
  rm(model_data)
  invisible(gc())

  tuning_folds <- if (smoke) {
    rsample::manual_rset(
      splits = cv_folds$splits[1],
      ids = cv_folds$id[1]
    )
  } else {
    cv_folds
  }

  list(
    data_split = data_split,
    cv_folds = cv_folds,
    tuning_folds = tuning_folds,
    development_data = development_data
  )
}

normalize_strategies <- function(x) {
  if (length(x) == 1L && x == "both") {
    return(c("natural", "downsampled"))
  }
  if (length(x) == 1L && grepl(",", x, fixed = TRUE)) {
    x <- strsplit(x, ",", fixed = TRUE)[[1]]
  }
  x <- trimws(x)
  allowed <- c("natural", "downsampled")
  if (length(x) == 0L || any(!x %in% allowed)) {
    stop(
      "strategies must be natural, downsampled, or both.",
      call. = FALSE
    )
  }
  unique(x)
}

build_recipe <- function(data, family, strategy) {
  recipe_args <- list(
    data = data,
    dbh_transform = "log",
    include_user_type = FALSE,
    include_collection_month = FALSE,
    include_geography = TRUE
  )

  if (strategy == "downsampled") {
    return(do.call(
      make_downsampled_recipe,
      c(
        recipe_args,
        list(
          family = family,
          under_ratio = UNDER_RATIO,
          seed = 5005L
        )
      )
    ))
  }

  builder <- switch(
    family,
    factor = make_factor_recipe,
    dummy = make_dummy_recipe,
    scaled = make_scaled_recipe,
    stop("Unknown recipe family: ", family, call. = FALSE)
  )
  do.call(builder, recipe_args)
}

build_model_spec <- function(model_key, smoke = FALSE, final_fit = FALSE) {
  stopifnot(model_key %in% MODEL_KEYS)

  switch(
    model_key,
    logistic = parsnip::multinom_reg(
      penalty = tune::tune(),
      mixture = tune::tune()
    ) |>
      parsnip::set_engine("glmnet") |>
      parsnip::set_mode("classification"),

    # `discrim` registers the klaR engine, while the model constructor is
    # exported by parsnip.
    naive_bayes = parsnip::naive_Bayes(
      smoothness = tune::tune(),
      Laplace = tune::tune()
    ) |>
      parsnip::set_engine("klaR") |>
      parsnip::set_mode("classification"),

    cart = parsnip::decision_tree(
      cost_complexity = tune::tune(),
      tree_depth = tune::tune(),
      min_n = tune::tune()
    ) |>
      parsnip::set_engine(
        "rpart",
        control = rpart::rpart.control(
          xval = 0L,
          maxsurrogate = 0L,
          maxcompete = 0L
        )
      ) |>
      parsnip::set_mode("classification"),

    random_forest = parsnip::rand_forest(
      mtry = tune::tune(),
      min_n = tune::tune(),
      trees = if (smoke) 50L else 500L
    ) |>
      parsnip::set_engine(
        "ranger",
        num.threads = 1L,
        importance = if (final_fit) {
          Sys.getenv("RF_IMPORTANCE", unset = "impurity")
        } else {
          "none"
        }
      ) |>
      parsnip::set_mode("classification"),

    xgboost = parsnip::boost_tree(
      trees = tune::tune(),
      tree_depth = tune::tune(),
      learn_rate = tune::tune(),
      min_n = tune::tune(),
      loss_reduction = tune::tune(),
      sample_size = tune::tune()
    ) |>
      parsnip::set_engine(
        "xgboost",
        nthread = 1L,
        tree_method = "hist",
        eval_metric = "mlogloss"
      ) |>
      parsnip::set_mode("classification")
  )
}

build_tuning_grid <- function(model_key, size, smoke = FALSE) {
  stopifnot(model_key %in% MODEL_KEYS)
  size <- as.integer(size)
  if (is.na(size) || size < 1L) {
    stop("Grid size must be a positive integer.", call. = FALSE)
  }

  set.seed(MODEL_SEEDS[[model_key]])
  switch(
    model_key,
    logistic = dials::grid_space_filling(
      dials::penalty(range = c(-6, 0)),
      dials::mixture(range = c(0, 1)),
      size = size
    ),

    naive_bayes = dials::grid_space_filling(
      dials::smoothness(range = c(0.5, 1.5)),
      dials::Laplace(range = c(0, 3)),
      size = size
    ),

    cart = dials::grid_space_filling(
      dials::cost_complexity(
        range = if (smoke) c(-3, -1) else c(-6, -1)
      ),
      dials::tree_depth(
        range = if (smoke) c(3L, 6L) else c(2L, 20L)
      ),
      dials::min_n(
        range = if (smoke) c(100L, 500L) else c(10L, 500L)
      ),
      size = size
    ),

    random_forest = dials::grid_space_filling(
      dials::mtry(range = c(2L, length(MAIN_PREDICTORS))),
      dials::min_n(range = c(2L, 100L)),
      size = size
    ),

    xgboost = dials::grid_space_filling(
      dials::trees(range = if (smoke) c(20L, 50L) else c(200L, 1200L)),
      dials::tree_depth(range = c(2L, 10L)),
      dials::learn_rate(range = c(-3, -0.5)),
      dials::min_n(range = c(2L, 50L)),
      dials::loss_reduction(range = c(-6, 1)),
      dials::sample_prop(range = c(0.6, 1.0)),
      size = size
    )
  )
}

build_workflow <- function(model_key, data, strategy, smoke = FALSE,
                           final_fit = FALSE) {
  family <- MODEL_RECIPE_FAMILIES[[model_key]]
  recipe_spec <- build_recipe(data, family, strategy)
  model_spec <- build_model_spec(model_key, smoke, final_fit)

  workflows::workflow() |>
    workflows::add_recipe(recipe_spec) |>
    workflows::add_model(model_spec)
}

probability_columns <- function() paste0(".pred_", HEALTH_LEVELS)

validate_predictions <- function(predictions) {
  required <- c("health", ".pred_class", probability_columns())
  missing_columns <- setdiff(required, names(predictions))
  if (length(missing_columns) > 0L) {
    stop(
      "Prediction output is missing: ",
      paste(missing_columns, collapse = ", "),
      call. = FALSE
    )
  }

  probabilities <- as.matrix(predictions[probability_columns()])
  row_sum_error <- max(abs(rowSums(probabilities) - 1), na.rm = TRUE)
  no_missing <- !anyNA(predictions[required])
  truth_has_all_levels <- setequal(
    as.character(unique(predictions$health)),
    HEALTH_LEVELS
  )
  prediction_factor_has_all_levels <- identical(
    levels(predictions$.pred_class),
    HEALTH_LEVELS
  )

  checks <- tibble::tibble(
    check = c(
      "No missing predictions",
      "Probabilities sum to one",
      "Assessment truth contains all classes",
      "Prediction factor defines all classes"
    ),
    passed = c(
      no_missing,
      is.finite(row_sum_error) && row_sum_error <= PROBABILITY_TOLERANCE,
      truth_has_all_levels,
      prediction_factor_has_all_levels
    ),
    detail = c(
      paste(sum(is.na(as.matrix(predictions[required]))), "missing cells"),
      paste("maximum row-sum error =", signif(row_sum_error, 5)),
      paste(sort(unique(as.character(predictions$health))), collapse = "/"),
      paste(levels(predictions$.pred_class), collapse = "/")
    )
  )

  if (!all(checks$passed)) {
    stop(
      "Prediction validation failed: ",
      paste(checks$check[!checks$passed], collapse = "; "),
      call. = FALSE
    )
  }
  checks
}

compute_common_fold_metrics <- function(predictions) {
  grouping_columns <- intersect(c("id", "id2"), names(predictions))
  if (length(grouping_columns) == 0L) {
    predictions$id <- "Assessment"
    grouping_columns <- "id"
  }

  predictions |>
    dplyr::group_by(dplyr::across(dplyr::all_of(grouping_columns))) |>
    dplyr::group_modify(function(.x, .y) {
      common_metrics(
        .x,
        truth = health,
        estimate = .pred_class,
        .pred_Good:.pred_Poor
      )
    }) |>
    dplyr::ungroup() |>
    dplyr::mutate(class = NA_character_)
}

compute_class_fold_metrics <- function(predictions) {
  grouping_columns <- intersect(c("id", "id2"), names(predictions))
  if (length(grouping_columns) == 0L) {
    predictions$id <- "Assessment"
    grouping_columns <- "id"
  }

  predictions |>
    dplyr::group_by(dplyr::across(dplyr::all_of(grouping_columns))) |>
    dplyr::group_modify(function(.x, .y) {
      purrr::map_dfr(HEALTH_LEVELS, function(class_name) {
        truth_positive <- .x$health == class_name
        predicted_positive <- .x$.pred_class == class_name
        true_positive <- sum(truth_positive & predicted_positive)
        predicted_n <- sum(predicted_positive)
        truth_n <- sum(truth_positive)

        tibble::tibble(
          .metric = c("precision", "recall"),
          .estimator = "class",
          .estimate = c(
            if (predicted_n == 0L) NA_real_ else true_positive / predicted_n,
            if (truth_n == 0L) NA_real_ else true_positive / truth_n
          ),
          class = class_name
        )
      })
    }) |>
    dplyr::ungroup()
}

summarise_fold_metrics <- function(common_fold_metrics, class_fold_metrics) {
  dplyr::bind_rows(common_fold_metrics, class_fold_metrics) |>
    dplyr::group_by(.metric, .estimator, class) |>
    dplyr::summarise(
      mean = mean(.estimate, na.rm = TRUE),
      n = sum(!is.na(.estimate)),
      std_err = if (sum(!is.na(.estimate)) > 1L) {
        stats::sd(.estimate, na.rm = TRUE) / sqrt(sum(!is.na(.estimate)))
      } else {
        NA_real_
      },
      .groups = "drop"
    )
}

compute_pooled_confusion <- function(predictions) {
  predictions |>
    dplyr::count(
      truth = health,
      estimate = .pred_class,
      name = "n",
      .drop = FALSE
    ) |>
    dplyr::group_by(truth) |>
    dplyr::mutate(row_proportion = n / sum(n)) |>
    dplyr::ungroup()
}

compact_tune_result <- function(tune_result) {
  # An rset/tune_results object embeds the complete resample data. Keeping it
  # would make each saved strategy hundreds of MB even after predictions were
  # removed. All information needed for reporting/recovery is saved separately
  # (grid, metrics, notes, best parameters and selected OOF predictions).
  id_columns <- intersect(c("id", "id2"), names(tune_result))
  list(
    original_class = class(tune_result),
    resample_ids = tibble::as_tibble(tune_result[id_columns]),
    note = paste(
      "Compact archive: embedded resample data and all-grid predictions",
      "were intentionally omitted."
    )
  )
}

select_strategy_result <- function(strategy_runs) {
  scores <- purrr::imap_dfr(strategy_runs, function(x, strategy) {
    x$metrics |>
      dplyr::filter(.metric == "macro_f_meas") |>
      dplyr::slice_max(mean, n = 1L, with_ties = FALSE) |>
      dplyr::transmute(strategy = strategy, mean, std_err, .config)
  })
  scores |>
    dplyr::arrange(dplyr::desc(mean), std_err) |>
    dplyr::slice(1L)
}

write_modelling_session_info <- function(output_dir = RESULTS_DIR) {
  dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)
  writeLines(
    capture.output(utils::sessionInfo()),
    file.path(output_dir, "modelling_session_info.txt")
  )
}

parse_named_arguments <- function(args) {
  result <- list()
  for (argument in args) {
    if (!startsWith(argument, "--") || !grepl("=", argument, fixed = TRUE)) {
      stop(
        "Arguments must use --name=value syntax. Received: ", argument,
        call. = FALSE
      )
    }
    pieces <- strsplit(sub("^--", "", argument), "=", fixed = TRUE)[[1]]
    result[[pieces[[1]]]] <- paste(pieces[-1], collapse = "=")
  }
  result
}

as_flag <- function(x, default = FALSE) {
  if (is.null(x)) return(default)
  normalized <- tolower(x)
  if (normalized %in% c("true", "1", "yes")) return(TRUE)
  if (normalized %in% c("false", "0", "no")) return(FALSE)
  stop("Boolean values must be true or false.", call. = FALSE)
}

safe_integer <- function(x, default, label) {
  if (is.null(x)) return(as.integer(default))
  value <- suppressWarnings(as.integer(x))
  if (is.na(value) || value < 1L) {
    stop(label, " must be a positive integer.", call. = FALSE)
  }
  value
}

model_file_prefix <- function(model_key) model_key

create_majority_baseline <- function(cv_folds, output_dir = RESULTS_DIR) {
  dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)
  epsilon <- 1e-15

  predictions <- purrr::map2_dfr(
    cv_folds$splits,
    cv_folds$id,
    function(split, fold_id) {
      assessment_data <- rsample::assessment(split)
      tibble::tibble(
        id = fold_id,
        health = assessment_data$health,
        .pred_class = factor("Good", levels = HEALTH_LEVELS),
        .pred_Good = 1 - 2 * epsilon,
        .pred_Fair = epsilon,
        .pred_Poor = epsilon
      )
    }
  )

  validation <- validate_predictions(predictions)
  common_fold <- compute_common_fold_metrics(predictions)
  class_fold <- compute_class_fold_metrics(predictions)
  metrics <- summarise_fold_metrics(common_fold, class_fold) |>
    dplyr::mutate(model = "Majority baseline", strategy = "always Good")

  readr::write_csv(
    metrics,
    file.path(output_dir, "majority_baseline_metrics.csv")
  )
  readr::write_csv(
    validation,
    file.path(output_dir, "majority_baseline_validation.csv")
  )
  invisible(metrics)
}
