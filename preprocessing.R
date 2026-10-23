# STAT5003 Group Project: common preprocessing and grouped resampling
#
# Run once from the project root:
#   Rscript preprocessing.R
#
# Later model scripts should only source this file and load the saved objects:
#   source("preprocessing.R")
#   data_split <- readRDS("data/data_split.rds")
#   cv_folds <- readRDS("data/cv_folds.rds")
#
# This script deliberately fits no classification model.

# -----------------------------------------------------------------------------
# 0. Constants and package checks
# -----------------------------------------------------------------------------

RAW_DATA_FILE <- "2015_Street_Tree_Census_-_Tree_Data.csv"
ARTIFACT_DIR <- "data"
SPLIT_SEED <- 5003L
CV_SEED <- 5004L
SPLIT_PROP <- 0.80
CV_FOLDS <- 5L
SPECIES_RARE_THRESHOLD <- 0.001
MAX_CLASS_PROP_DIFFERENCE <- 0.01
HEALTH_LEVELS <- c("Good", "Fair", "Poor")

UNKNOWN_LEVEL <- "Unknown"
NOVEL_LEVEL <- "Novel"
OTHER_SPECIES_LEVEL <- "Other"

PROBLEM_PREDICTORS <- c(
  "root_stone", "root_grate", "root_other",
  "trunk_wire", "trnk_light", "trnk_other",
  "brch_light", "brch_shoe", "brch_other"
)

SITE_PREDICTORS <- c("curb_loc", "steward", "guards", "sidewalk")
GEOGRAPHY_PREDICTORS <- c("borough", "latitude", "longitude")
MAIN_PREDICTORS <- c(
  "tree_dbh", "spc_common", SITE_PREDICTORS,
  PROBLEM_PREDICTORS, GEOGRAPHY_PREDICTORS
)
SENSITIVITY_VARIABLES <- c("user_type", "collection_month")

RAW_REQUIRED_COLUMNS <- c(
  "status", "health", "block_id", "created_at",
  MAIN_PREDICTORS, "user_type"
)
MASTER_REQUIRED_COLUMNS <- c(
  "health", "resampling_group", MAIN_PREDICTORS, SENSITIVITY_VARIABLES
)
EXCLUDED_RAW_FIELDS <- c(
  "tree_id", "block_id", "address", "postcode", "zip_city",
  "community board", "borocode", "cncldist", "st_assem", "st_senate",
  "nta", "nta_name", "boro_ct", "state", "x_sp", "y_sp",
  "council district", "census tract", "bin", "bbl", "status",
  "stump_diam", "spc_latin", "problems", "created_at"
)

required_packages <- c(
  "readr", "dplyr", "tibble", "purrr", "rsample", "recipes", "generics"
)

check_required_packages <- function() {
  missing_packages <- required_packages[
    !vapply(required_packages, requireNamespace, logical(1), quietly = TRUE)
  ]
  if (length(missing_packages) > 0L) {
    stop(
      "Install the following packages before running preprocessing.R: ",
      paste(missing_packages, collapse = ", "),
      call. = FALSE
    )
  }
  invisible(TRUE)
}

assert_columns <- function(data, required, object_name) {
  missing_columns <- setdiff(required, names(data))
  if (length(missing_columns) > 0L) {
    stop(
      object_name, " is missing required columns: ",
      paste(missing_columns, collapse = ", "),
      call. = FALSE
    )
  }
  invisible(TRUE)
}

# -----------------------------------------------------------------------------
# 1. Master modelling cohort
# -----------------------------------------------------------------------------

read_raw_modelling_columns <- function(path = RAW_DATA_FILE) {
  if (!file.exists(path)) {
    stop("Cannot find the raw data file: ", path, call. = FALSE)
  }

  raw_data <- readr::read_csv(
    path,
    col_select = dplyr::all_of(RAW_REQUIRED_COLUMNS),
    na = c("", "NA", "N/A", "NULL"),
    show_col_types = FALSE,
    progress = FALSE
  )
  assert_columns(raw_data, RAW_REQUIRED_COLUMNS, "Raw data")
  raw_data
}

build_model_data_master <- function(raw_data) {
  assert_columns(raw_data, RAW_REQUIRED_COLUMNS, "Raw data")

  model_data <- raw_data |>
    dplyr::filter(
      .data$status == "Alive",
      .data$health %in% HEALTH_LEVELS
    ) |>
    dplyr::mutate(
      health = factor(.data$health, levels = HEALTH_LEVELS),
      collection_date = as.Date(.data$created_at, format = "%m/%d/%Y"),
      collection_month = factor(
        month.abb[as.integer(format(.data$collection_date, "%m"))],
        levels = month.abb
      )
    ) |>
    dplyr::transmute(
      health = .data$health,
      resampling_group = .data$block_id,
      tree_dbh = .data$tree_dbh,
      spc_common = .data$spc_common,
      curb_loc = .data$curb_loc,
      steward = .data$steward,
      guards = .data$guards,
      sidewalk = .data$sidewalk,
      root_stone = .data$root_stone,
      root_grate = .data$root_grate,
      root_other = .data$root_other,
      trunk_wire = .data$trunk_wire,
      trnk_light = .data$trnk_light,
      trnk_other = .data$trnk_other,
      brch_light = .data$brch_light,
      brch_shoe = .data$brch_shoe,
      brch_other = .data$brch_other,
      borough = .data$borough,
      latitude = .data$latitude,
      longitude = .data$longitude,
      user_type = .data$user_type,
      collection_month = .data$collection_month
    )

  assert_columns(model_data, MASTER_REQUIRED_COLUMNS, "Master modelling data")

  if (!identical(levels(model_data$health), HEALTH_LEVELS)) {
    stop("The outcome factor levels are not in the required order.", call. = FALSE)
  }
  if (anyNA(model_data$health)) {
    stop("The master modelling outcome contains missing values.", call. = FALSE)
  }
  if (anyNA(model_data$resampling_group)) {
    stop("resampling_group contains missing values.", call. = FALSE)
  }
  if (any(model_data$tree_dbh < 0, na.rm = TRUE)) {
    stop("tree_dbh contains negative values; log1p would be invalid.", call. = FALSE)
  }
  if (!setequal(as.character(unique(model_data$health)), HEALTH_LEVELS)) {
    stop("The modelling cohort does not contain exactly Good/Fair/Poor.", call. = FALSE)
  }

  model_data
}

# -----------------------------------------------------------------------------
# 2. Fixed grouped split and grouped cross-validation
# -----------------------------------------------------------------------------

class_proportions <- function(data) {
  counts <- table(factor(data$health, levels = HEALTH_LEVELS))
  stats::setNames(as.numeric(counts / sum(counts)), HEALTH_LEVELS)
}

create_grouped_resamples <- function(
    model_data,
    split_seed = SPLIT_SEED,
    cv_seed = CV_SEED,
    split_prop = SPLIT_PROP,
    v = CV_FOLDS) {
  assert_columns(model_data, MASTER_REQUIRED_COLUMNS, "Master modelling data")

  # health cannot be passed to rsample::strata here because health is not
  # constant within block_id. With more than 100,000 blocks, a grouped random
  # split is expected to preserve the class mix; this is checked below.
  set.seed(split_seed)
  data_split <- rsample::group_initial_split(
    model_data,
    group = resampling_group,
    prop = split_prop
  )

  development_data <- rsample::training(data_split)
  final_test_data <- rsample::testing(data_split)

  split_overlap <- intersect(
    unique(development_data$resampling_group),
    unique(final_test_data$resampling_group)
  )
  if (length(split_overlap) != 0L) {
    stop("Development/test block overlap is not zero.", call. = FALSE)
  }

  split_class_difference <- max(abs(
    class_proportions(development_data) - class_proportions(final_test_data)
  ))
  if (split_class_difference > MAX_CLASS_PROP_DIFFERENCE) {
    stop(
      "Development/test class proportions differ by more than ",
      MAX_CLASS_PROP_DIFFERENCE, ". Choose and document a different split seed.",
      call. = FALSE
    )
  }

  set.seed(cv_seed)
  cv_folds <- rsample::group_vfold_cv(
    development_data,
    group = resampling_group,
    v = v,
    # There are 100,700 groups. In rsample 1.3.2, observation balancing uses
    # an iterative algorithm that is impractical at this scale. Randomly
    # balancing the very large number of groups is fast; observation counts
    # and class proportions are then checked explicitly below.
    balance = "groups"
  )

  cv_overlap <- purrr::map_int(cv_folds$splits, function(split) {
    analysis_groups <- unique(rsample::analysis(split)$resampling_group)
    assessment_groups <- unique(rsample::assessment(split)$resampling_group)
    length(intersect(analysis_groups, assessment_groups))
  })
  if (any(cv_overlap != 0L)) {
    stop("At least one CV fold has overlapping blocks.", call. = FALSE)
  }

  development_proportions <- class_proportions(development_data)
  cv_class_difference <- max(purrr::map_dbl(cv_folds$splits, function(split) {
    max(abs(
      class_proportions(rsample::assessment(split)) - development_proportions
    ))
  }))
  if (cv_class_difference > MAX_CLASS_PROP_DIFFERENCE) {
    stop(
      "At least one CV assessment fold differs from the development class ",
      "proportions by more than ", MAX_CLASS_PROP_DIFFERENCE, ".",
      call. = FALSE
    )
  }

  list(
    data_split = data_split,
    cv_folds = cv_folds,
    split_class_difference = split_class_difference,
    cv_class_difference = cv_class_difference
  )
}

# -----------------------------------------------------------------------------
# 3. Leakage-safe recipe specifications
# -----------------------------------------------------------------------------

validate_recipe_arguments <- function(data, dbh_transform, species_threshold) {
  assert_columns(data, MASTER_REQUIRED_COLUMNS, "Recipe data")
  dbh_transform <- match.arg(dbh_transform, c("log", "raw"))
  if (!is.numeric(species_threshold) || length(species_threshold) != 1L ||
      is.na(species_threshold) || species_threshold <= 0 ||
      species_threshold >= 1) {
    stop("species_threshold must be one number strictly between 0 and 1.", call. = FALSE)
  }
  dbh_transform
}

make_base_recipe <- function(
    data,
    dbh_transform = c("log", "raw"),
    include_user_type = FALSE,
    include_collection_month = FALSE,
    include_geography = TRUE,
    species_threshold = SPECIES_RARE_THRESHOLD) {
  dbh_transform <- validate_recipe_arguments(
    data, dbh_transform, species_threshold
  )

  recipe_spec <- recipes::recipe(health ~ ., data = data) |>
    recipes::update_role(
      resampling_group,
      new_role = "resampling identifier"
    )

  if (!include_user_type) {
    recipe_spec <- recipe_spec |>
      recipes::update_role(user_type, new_role = "sensitivity variable")
  }
  if (!include_collection_month) {
    recipe_spec <- recipe_spec |>
      recipes::update_role(
        collection_month,
        new_role = "sensitivity variable"
      )
  }
  if (!include_geography) {
    recipe_spec <- recipe_spec |>
      recipes::update_role(
        dplyr::all_of(GEOGRAPHY_PREDICTORS),
        new_role = "excluded geography"
      )
  }

  recipe_spec <- recipe_spec |>
    recipes::step_impute_median(recipes::all_numeric_predictors()) |>
    # Pool rare and unseen species into Other. For other nominal predictors,
    # step_novel explicitly handles levels absent from the analysis fold.
    recipes::step_other(
      spc_common,
      threshold = species_threshold,
      other = OTHER_SPECIES_LEVEL
    ) |>
    recipes::step_novel(
      recipes::all_nominal_predictors(),
      new_level = NOVEL_LEVEL
    ) |>
    recipes::step_unknown(
      recipes::all_nominal_predictors(),
      new_level = UNKNOWN_LEVEL
    )

  if (dbh_transform == "log") {
    recipe_spec <- recipe_spec |>
      recipes::step_log(tree_dbh, offset = 1)
  }

  recipe_spec |>
    recipes::step_zv(recipes::all_predictors())
}

make_factor_recipe <- function(data, ...) {
  make_base_recipe(data = data, ...) |>
    recipes::step_string2factor(recipes::all_nominal_predictors()) |>
    recipes::step_zv(recipes::all_predictors())
}

make_dummy_recipe <- function(data, ...) {
  make_base_recipe(data = data, ...) |>
    recipes::step_string2factor(recipes::all_nominal_predictors()) |>
    recipes::step_dummy(
      recipes::all_nominal_predictors(),
      one_hot = FALSE,
      sparse = "auto"
    ) |>
    recipes::step_zv(recipes::all_predictors())
}

make_scaled_recipe <- function(data, ...) {
  # Normalisation precedes dummy creation so binary dummy columns are not
  # standardised. Means/SDs are learned only when each resample is prepped.
  make_base_recipe(data = data, ...) |>
    recipes::step_normalize(recipes::all_numeric_predictors()) |>
    recipes::step_string2factor(recipes::all_nominal_predictors()) |>
    recipes::step_dummy(
      recipes::all_nominal_predictors(),
      one_hot = FALSE,
      sparse = "auto"
    ) |>
    recipes::step_zv(recipes::all_predictors())
}

make_downsampled_recipe <- function(
    data,
    family = c("factor", "dummy", "scaled"),
    under_ratio = 1,
    seed = 5005L,
    ...) {
  if (!requireNamespace("themis", quietly = TRUE)) {
    stop(
      "make_downsampled_recipe() requires the optional 'themis' package. ",
      "The base recipes preserve the natural class distribution.",
      call. = FALSE
    )
  }
  family <- match.arg(family)
  builder <- switch(
    family,
    factor = make_factor_recipe,
    dummy = make_dummy_recipe,
    scaled = make_scaled_recipe
  )
  recipe_spec <- builder(data = data, ...)

  # This is deliberately the final training step and skip = TRUE prevents any
  # change to CV assessment data or the final test set.
  themis::step_downsample(
    recipe_spec,
    health,
    under_ratio = under_ratio,
    skip = TRUE,
    seed = seed
  )
}

# -----------------------------------------------------------------------------
# 4. Diagnostics and validation
# -----------------------------------------------------------------------------

partition_class_diagnostics <- function(data_split) {
  partitions <- list(
    Development = rsample::training(data_split),
    `Final test` = rsample::testing(data_split)
  )

  purrr::imap_dfr(partitions, function(data, partition_name) {
    data |>
      dplyr::count(health, .drop = FALSE, name = "class_n") |>
      dplyr::mutate(
        partition = partition_name,
        observations = nrow(data),
        unique_blocks = dplyr::n_distinct(data$resampling_group),
        class_proportion = .data$class_n / sum(.data$class_n),
        .before = 1
      )
  })
}

cv_class_diagnostics <- function(cv_folds) {
  purrr::map2_dfr(cv_folds$splits, cv_folds$id, function(split, fold_id) {
    assessment_data <- rsample::assessment(split)
    assessment_data |>
      dplyr::count(health, .drop = FALSE, name = "class_n") |>
      dplyr::mutate(
        fold = fold_id,
        assessment_observations = nrow(assessment_data),
        assessment_blocks = dplyr::n_distinct(
          assessment_data$resampling_group
        ),
        class_proportion = .data$class_n / sum(.data$class_n),
        .before = 1
      )
  })
}

validate_recipes_on_first_fold <- function(cv_folds) {
  first_split <- cv_folds$splits[[1]]
  analysis_data <- rsample::analysis(first_split)
  assessment_data <- rsample::assessment(first_split)

  builders <- list(
    factor = make_factor_recipe,
    dummy = make_dummy_recipe,
    scaled = make_scaled_recipe
  )

  recipe_diagnostics <- purrr::imap_dfr(builders, function(builder, family) {
    recipe_spec <- builder(analysis_data)
    trained_recipe <- recipes::prep(
      recipe_spec,
      training = analysis_data,
      retain = FALSE,
      verbose = FALSE
    )
    baked_assessment <- recipes::bake(
      trained_recipe,
      new_data = assessment_data,
      composition = "tibble"
    )
    variable_summary <- summary(trained_recipe, original = FALSE)
    predictor_names <- variable_summary$variable[
      variable_summary$role == "predictor"
    ]
    predictor_names <- intersect(predictor_names, names(baked_assessment))
    missing_predictors <- predictor_names[
      vapply(
        baked_assessment[predictor_names],
        anyNA,
        logical(1)
      )
    ]

    if (length(missing_predictors) > 0L) {
      stop(
        family, " recipe left missing predictor values: ",
        paste(missing_predictors, collapse = ", "),
        call. = FALSE
      )
    }

    tibble::tibble(
      recipe_family = family,
      predictors_before = length(MAIN_PREDICTORS),
      predictors_after = length(predictor_names),
      assessment_rows_processed = nrow(baked_assessment),
      unexpected_missing_predictors = length(missing_predictors)
    )
  })

  factor_recipe <- recipes::prep(
    make_factor_recipe(analysis_data),
    training = analysis_data,
    retain = FALSE,
    verbose = FALSE
  )

  other_step <- which(vapply(
    factor_recipe$steps,
    inherits,
    logical(1),
    what = "step_other"
  ))
  retained_species <- generics::tidy(
    factor_recipe,
    number = other_step[[1]]
  )$retained
  retained_species_levels <- length(unique(retained_species)) + 1L

  novel_row <- assessment_data[1, , drop = FALSE]
  novel_row$curb_loc <- "__NOVEL_LEVEL_VALIDATION__"
  novel_result <- recipes::bake(
    factor_recipe,
    new_data = novel_row,
    composition = "tibble"
  )
  novel_level_handled <- identical(
    as.character(novel_result$curb_loc), NOVEL_LEVEL
  )

  novel_species_row <- assessment_data[1, , drop = FALSE]
  novel_species_row$spc_common <- "__NOVEL_SPECIES_VALIDATION__"
  novel_species_result <- recipes::bake(
    factor_recipe,
    new_data = novel_species_row,
    composition = "tibble"
  )
  novel_species_handled <- identical(
    as.character(novel_species_result$spc_common), OTHER_SPECIES_LEVEL
  )

  list(
    recipe_diagnostics = recipe_diagnostics,
    retained_species_levels = retained_species_levels,
    novel_level_handled = novel_level_handled,
    novel_species_handled = novel_species_handled
  )
}

build_validation_checks <- function(
    model_data,
    data_split,
    cv_folds,
    recipe_validation,
    split_class_difference,
    cv_class_difference) {
  development_data <- rsample::training(data_split)
  final_test_data <- rsample::testing(data_split)

  split_overlap <- length(intersect(
    unique(development_data$resampling_group),
    unique(final_test_data$resampling_group)
  ))
  cv_overlap <- purrr::map_int(cv_folds$splits, function(split) {
    length(intersect(
      unique(rsample::analysis(split)$resampling_group),
      unique(rsample::assessment(split)$resampling_group)
    ))
  })

  role_summary <- summary(make_factor_recipe(development_data), original = TRUE)
  group_role <- role_summary$role[
    role_summary$variable == "resampling_group"
  ]

  tibble::tribble(
    ~check, ~passed, ~detail,
    "Outcome has exactly Good/Fair/Poor",
    identical(levels(model_data$health), HEALTH_LEVELS) &&
      setequal(as.character(unique(model_data$health)), HEALTH_LEVELS),
    paste(levels(model_data$health), collapse = "/"),
    "Outcome has no missing values",
    !anyNA(model_data$health),
    paste(sum(is.na(model_data$health)), "missing"),
    "Final test not used to estimate transformations",
    TRUE,
    "Recipe validation was trained on CV Fold 1 analysis data only",
    "resampling_group is not a predictor",
    length(group_role) == 1L && group_role != "predictor",
    paste("role =", group_role),
    "Excluded identifiers absent from modelling table",
    length(intersect(EXCLUDED_RAW_FIELDS, names(model_data))) == 0L,
    paste("found:", paste(intersect(EXCLUDED_RAW_FIELDS, names(model_data)), collapse = ", ")),
    "All required master variables present",
    length(setdiff(MASTER_REQUIRED_COLUMNS, names(model_data))) == 0L,
    paste(length(MASTER_REQUIRED_COLUMNS), "required variables"),
    "Development/test block overlap is zero",
    split_overlap == 0L,
    paste(split_overlap, "overlapping blocks"),
    "Every CV analysis/assessment overlap is zero",
    all(cv_overlap == 0L),
    paste(cv_overlap, collapse = "/"),
    "Development/test class proportions are similar",
    split_class_difference <= MAX_CLASS_PROP_DIFFERENCE,
    paste("maximum absolute difference =", signif(split_class_difference, 4)),
    "CV assessment class proportions are similar",
    cv_class_difference <= MAX_CLASS_PROP_DIFFERENCE,
    paste("maximum absolute difference =", signif(cv_class_difference, 4)),
    "All recipe variants process one complete CV assessment fold",
    all(recipe_validation$recipe_diagnostics$unexpected_missing_predictors == 0L),
    paste(recipe_validation$recipe_diagnostics$assessment_rows_processed, collapse = "/"),
    "Novel categorical levels are handled",
    recipe_validation$novel_level_handled,
    paste("unseen curb_loc ->", NOVEL_LEVEL),
    "Novel species are pooled safely",
    recipe_validation$novel_species_handled,
    paste("unseen spc_common ->", OTHER_SPECIES_LEVEL)
  )
}

# -----------------------------------------------------------------------------
# 5. Build and save all pre-model artefacts
# -----------------------------------------------------------------------------

build_preprocessing_artifacts <- function(
    raw_data_file = RAW_DATA_FILE,
    artifact_dir = ARTIFACT_DIR) {
  check_required_packages()
  dir.create(artifact_dir, recursive = TRUE, showWarnings = FALSE)

  message("Reading required raw columns...")
  raw_data <- read_raw_modelling_columns(raw_data_file)
  message("Constructing the fixed modelling cohort...")
  model_data_master <- build_model_data_master(raw_data)
  rm(raw_data)
  invisible(gc())

  message("Creating the grouped development/test split and CV folds...")
  resamples <- create_grouped_resamples(model_data_master)
  data_split <- resamples$data_split
  cv_folds <- resamples$cv_folds

  message("Validating recipes on CV Fold 1...")
  recipe_validation <- validate_recipes_on_first_fold(cv_folds)
  validation_checks <- build_validation_checks(
    model_data_master,
    data_split,
    cv_folds,
    recipe_validation,
    resamples$split_class_difference,
    resamples$cv_class_difference
  )

  if (!all(validation_checks$passed)) {
    failed <- validation_checks$check[!validation_checks$passed]
    stop(
      "Preprocessing validation failed: ",
      paste(failed, collapse = "; "),
      call. = FALSE
    )
  }

  partition_diagnostics <- partition_class_diagnostics(data_split)
  fold_diagnostics <- cv_class_diagnostics(cv_folds)
  preprocessing_diagnostics <- dplyr::mutate(
    recipe_validation$recipe_diagnostics,
    retained_species_levels = recipe_validation$retained_species_levels
  )

  message("Saving fixed data, resamples, and diagnostics...")
  saveRDS(
    model_data_master,
    file.path(artifact_dir, "model_data_master.rds")
  )
  saveRDS(data_split, file.path(artifact_dir, "data_split.rds"))
  saveRDS(cv_folds, file.path(artifact_dir, "cv_folds.rds"))
  readr::write_csv(
    partition_diagnostics,
    file.path(artifact_dir, "partition_diagnostics.csv")
  )
  readr::write_csv(
    fold_diagnostics,
    file.path(artifact_dir, "cv_fold_diagnostics.csv")
  )
  readr::write_csv(
    preprocessing_diagnostics,
    file.path(artifact_dir, "preprocessing_diagnostics.csv")
  )
  readr::write_csv(
    validation_checks,
    file.path(artifact_dir, "preprocessing_validation_checks.csv")
  )
  writeLines(
    capture.output(utils::sessionInfo()),
    file.path(artifact_dir, "preprocessing_session_info.txt")
  )

  message(
    "Preprocessing complete: ", nrow(model_data_master),
    " observations, ", dplyr::n_distinct(model_data_master$resampling_group),
    " blocks, all ", nrow(validation_checks), " checks passed."
  )

  invisible(list(
    model_data_master = model_data_master,
    data_split = data_split,
    cv_folds = cv_folds,
    validation_checks = validation_checks
  ))
}

if (sys.nframe() == 0L) {
  build_preprocessing_artifacts()
}
