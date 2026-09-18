# STAT5003 Group Project: Phase 1 EDA and Planning
# Dataset: 2015 Street Tree Census - Tree Data (NYC Open Data)
#
# Proposed classification question:
# Can observable tree, site and geographic characteristics classify the
# perceived health of living NYC street trees as Good, Fair or Poor?
#
# Run this file from the project root with:
#   Rscript main.r
#
# The script creates report-ready tables and figures in eda_outputs/.

# -----------------------------------------------------------------------------
# 0. Setup
# -----------------------------------------------------------------------------

required_packages <- c(
  "readr", "dplyr", "tidyr", "ggplot2", "forcats", "scales"
)

missing_packages <- required_packages[
  !vapply(required_packages, requireNamespace, logical(1), quietly = TRUE)
]

if (length(missing_packages) > 0) {
  stop(
    "Install the following packages before running this script: ",
    paste(missing_packages, collapse = ", "),
    call. = FALSE
  )
}

suppressPackageStartupMessages({
  library(readr)
  library(dplyr)
  library(tidyr)
  library(ggplot2)
  library(forcats)
  library(scales)
})

options(dplyr.summarise.inform = FALSE)

data_file <- "2015_Street_Tree_Census_-_Tree_Data_20260912.csv"
output_dir <- "eda_outputs"
figure_dir <- file.path(output_dir, "figures")
table_dir <- file.path(output_dir, "tables")

if (!file.exists(data_file)) {
  stop("Cannot find the data file: ", data_file, call. = FALSE)
}

dir.create(figure_dir, recursive = TRUE, showWarnings = FALSE)
dir.create(table_dir, recursive = TRUE, showWarnings = FALSE)

health_colours <- c(
  "Good" = "#2E7D32",
  "Fair" = "#F9A825",
  "Poor" = "#C62828"
)

theme_set(
  theme_minimal(base_size = 12) +
    theme(
      plot.title.position = "plot",
      plot.title = element_text(face = "bold"),
      panel.grid.minor = element_blank(),
      legend.position = "bottom"
    )
)

save_plot <- function(plot, filename, width = 8, height = 5) {
  ggsave(
    filename = file.path(figure_dir, filename),
    plot = plot,
    width = width,
    height = height,
    dpi = 300,
    bg = "white"
  )
}

print_table <- function(x, title) {
  cat("\n", title, "\n", strrep("-", nchar(title)), "\n", sep = "")
  print(x, n = Inf)
}

# -----------------------------------------------------------------------------
# 1. Read and audit the raw data
# -----------------------------------------------------------------------------

trees_raw <- read_csv(
  data_file,
  na = c("", "NA", "N/A", "NULL"),
  show_col_types = FALSE,
  progress = FALSE
)

data_overview <- tibble(
  metric = c(
    "Rows",
    "Columns",
    "Unique tree IDs",
    "Duplicated tree IDs",
    "Numeric columns",
    "Non-numeric columns"
  ),
  value = c(
    nrow(trees_raw),
    ncol(trees_raw),
    n_distinct(trees_raw$tree_id),
    sum(duplicated(trees_raw$tree_id)),
    sum(vapply(trees_raw, is.numeric, logical(1))),
    sum(!vapply(trees_raw, is.numeric, logical(1)))
  )
)

variable_types <- tibble(
  variable = names(trees_raw),
  imported_class = vapply(
    trees_raw,
    function(x) paste(class(x), collapse = "/"),
    character(1)
  ),
  distinct_values = vapply(trees_raw, n_distinct, integer(1), na.rm = FALSE)
)

status_distribution <- trees_raw %>%
  count(status, name = "n", sort = TRUE) %>%
  mutate(proportion = n / sum(n))

write_csv(data_overview, file.path(table_dir, "data_overview.csv"))
write_csv(variable_types, file.path(table_dir, "variable_types.csv"))
write_csv(status_distribution, file.path(table_dir, "status_distribution.csv"))

print_table(data_overview, "Raw data overview")
print_table(status_distribution, "Tree status distribution")

# -----------------------------------------------------------------------------
# 2. Missingness and data-quality patterns
# -----------------------------------------------------------------------------

missing_summary <- trees_raw %>%
  summarise(across(everything(), ~ sum(is.na(.x)))) %>%
  pivot_longer(
    cols = everything(),
    names_to = "variable",
    values_to = "missing_n"
  ) %>%
  mutate(missing_pct = missing_n / nrow(trees_raw)) %>%
  arrange(desc(missing_n))

write_csv(missing_summary, file.path(table_dir, "missing_summary.csv"))
print_table(filter(missing_summary, missing_n > 0), "Variables with missing values")

p_missing <- missing_summary %>%
  filter(missing_n > 0) %>%
  mutate(variable = fct_reorder(variable, missing_pct)) %>%
  ggplot(aes(x = missing_pct, y = variable)) +
  geom_col(fill = "#3F6B8A", width = 0.72) +
  geom_text(
    aes(label = percent(missing_pct, accuracy = 0.1)),
    hjust = -0.1,
    size = 3.4
  ) +
  scale_x_continuous(
    labels = label_percent(),
    expand = expansion(mult = c(0, 0.14))
  ) +
  labs(
    title = "Missingness in the raw census data",
    subtitle = "Several missing fields are structurally related to dead trees and stumps",
    x = "Missing records",
    y = NULL
  )

save_plot(p_missing, "01_missingness_overall.png", width = 8, height = 5.5)

# Health, species and condition fields were not collected for dead trees/stumps.
# This plot separates structural missingness from missingness among living trees.
missing_pattern_variables <- c(
  "health", "spc_common", "steward", "guards", "sidewalk",
  "problems", "bin", "bbl", "council district", "census tract"
)

missing_by_status <- trees_raw %>%
  group_by(status) %>%
  summarise(
    across(all_of(missing_pattern_variables), ~ mean(is.na(.x))),
    .groups = "drop"
  ) %>%
  pivot_longer(
    cols = -status,
    names_to = "variable",
    values_to = "missing_pct"
  )

write_csv(missing_by_status, file.path(table_dir, "missingness_by_status.csv"))

p_missing_status <- ggplot(
  missing_by_status,
  aes(x = status, y = fct_rev(variable), fill = missing_pct)
) +
  geom_tile(colour = "white", linewidth = 0.5) +
  geom_text(
    aes(label = percent(missing_pct, accuracy = 0.1)),
    size = 3
  ) +
  scale_fill_gradient(
    low = "#F7FBFF",
    high = "#08519C",
    labels = label_percent()
  ) +
  labs(
    title = "Missingness pattern depends on tree status",
    x = "Tree status",
    y = NULL,
    fill = "Missing"
  )

save_plot(p_missing_status, "02_missingness_by_status.png", width = 8, height = 5.5)

# -----------------------------------------------------------------------------
# 3. Define the classification cohort and engineer EDA features
# -----------------------------------------------------------------------------

issue_variables <- c(
  "root_stone", "root_grate", "root_other",
  "trunk_wire", "trnk_light", "trnk_other",
  "brch_light", "brch_shoe", "brch_other"
)

cohort_flow <- tibble(
  step = c(
    "All census records",
    "Living trees",
    "Living trees with Good/Fair/Poor health"
  ),
  n = c(
    nrow(trees_raw),
    sum(trees_raw$status == "Alive", na.rm = TRUE),
    sum(
      trees_raw$status == "Alive" &
        trees_raw$health %in% c("Good", "Fair", "Poor"),
      na.rm = TRUE
    )
  )
) %>%
  mutate(retained_pct = n / first(n))

trees <- trees_raw %>%
  filter(status == "Alive", health %in% c("Good", "Fair", "Poor")) %>%
  mutate(
    health = factor(health, levels = c("Good", "Fair", "Poor")),
    created_at = as.Date(created_at, format = "%m/%d/%Y"),
    collection_month = factor(
      format(created_at, "%m"),
      levels = sprintf("%02d", 1:12),
      labels = month.abb
    ),
    problem_count = rowSums(
      across(all_of(issue_variables), ~ .x == "Yes"),
      na.rm = TRUE
    ),
    problem_group = factor(
      if_else(problem_count >= 3, "3+", as.character(problem_count)),
      levels = c("0", "1", "2", "3+")
    )
  )

write_csv(cohort_flow, file.path(table_dir, "cohort_flow.csv"))
print_table(cohort_flow, "Analytic cohort flow")

# -----------------------------------------------------------------------------
# 4. Outcome distribution and imbalance
# -----------------------------------------------------------------------------

class_distribution <- trees %>%
  count(health, .drop = FALSE, name = "n") %>%
  mutate(
    proportion = n / sum(n),
    plot_label = paste0(comma(n), "\n", percent(proportion, accuracy = 0.1))
  )

majority_class_accuracy <- max(class_distribution$proportion)

write_csv(
  select(class_distribution, -plot_label),
  file.path(table_dir, "class_distribution.csv")
)
print_table(select(class_distribution, -plot_label), "Health class distribution")
cat(
  "\nMajority-class benchmark accuracy: ",
  percent(majority_class_accuracy, accuracy = 0.1),
  "\n",
  sep = ""
)

p_class <- ggplot(class_distribution, aes(x = health, y = n, fill = health)) +
  geom_col(width = 0.68, show.legend = FALSE) +
  geom_text(aes(label = plot_label), vjust = -0.25, size = 3.6) +
  scale_fill_manual(values = health_colours) +
  scale_y_continuous(
    labels = label_comma(),
    expand = expansion(mult = c(0, 0.14))
  ) +
  labs(
    title = "Perceived health of living street trees is imbalanced",
    subtitle = "Overall accuracy alone would conceal weak performance on Fair and Poor trees",
    x = "Health class",
    y = "Number of trees"
  )

save_plot(p_class, "03_health_class_distribution.png", width = 7, height = 5)

# -----------------------------------------------------------------------------
# 5. Univariate analysis: tree diameter and species
# -----------------------------------------------------------------------------

dbh_quantiles <- quantile(
  trees$tree_dbh,
  probs = c(0, 0.01, 0.25, 0.50, 0.75, 0.95, 0.99, 0.999, 1),
  na.rm = TRUE
)

dbh_q1 <- unname(quantile(trees$tree_dbh, 0.25, na.rm = TRUE))
dbh_q3 <- unname(quantile(trees$tree_dbh, 0.75, na.rm = TRUE))
dbh_iqr <- dbh_q3 - dbh_q1
dbh_upper_iqr_fence <- dbh_q3 + 1.5 * dbh_iqr

dbh_summary <- tibble(
  metric = c(
    "Minimum", "1st percentile", "Median", "Mean", "99th percentile",
    "99.9th percentile", "Maximum", "Zero-diameter living trees",
    "Living trees over 100 inches", "Upper 1.5-IQR fence"
  ),
  value = c(
    min(trees$tree_dbh, na.rm = TRUE),
    unname(dbh_quantiles["1%"]),
    median(trees$tree_dbh, na.rm = TRUE),
    mean(trees$tree_dbh, na.rm = TRUE),
    unname(dbh_quantiles["99%"]),
    unname(dbh_quantiles["99.9%"]),
    max(trees$tree_dbh, na.rm = TRUE),
    sum(trees$tree_dbh == 0, na.rm = TRUE),
    sum(trees$tree_dbh > 100, na.rm = TRUE),
    dbh_upper_iqr_fence
  )
)

write_csv(dbh_summary, file.path(table_dir, "dbh_summary.csv"))
print_table(dbh_summary, "Diameter-at-breast-height audit")

# The x-axis is restricted to the 99.9th percentile for legibility. Extreme
# values are retained in the audit table and should be reviewed, not silently
# deleted merely because they satisfy an automated outlier rule.
dbh_plot_limit <- unname(quantile(trees$tree_dbh, 0.999, na.rm = TRUE))

p_dbh <- ggplot(trees, aes(x = tree_dbh, fill = health)) +
  geom_histogram(binwidth = 2, boundary = 0, colour = "white", linewidth = 0.15) +
  facet_wrap(~ health, ncol = 1, scales = "free_y") +
  coord_cartesian(xlim = c(0, dbh_plot_limit)) +
  scale_fill_manual(values = health_colours, guide = "none") +
  scale_y_continuous(labels = label_comma()) +
  labs(
    title = "Tree diameter distribution by health class",
    subtitle = paste0(
      "Display capped at the 99.9th percentile (",
      number(dbh_plot_limit),
      " inches); extreme values remain in the audit"
    ),
    x = "Diameter at breast height (inches)",
    y = "Number of trees"
  )

save_plot(p_dbh, "04_dbh_distribution_by_health.png", width = 8, height = 7)

top_species <- trees %>%
  count(spc_common, name = "n", sort = TRUE) %>%
  slice_head(n = 15) %>%
  mutate(proportion = n / nrow(trees))

write_csv(top_species, file.path(table_dir, "top_species.csv"))

p_species <- top_species %>%
  mutate(spc_common = fct_reorder(spc_common, n)) %>%
  ggplot(aes(x = n, y = spc_common)) +
  geom_col(fill = "#4C956C", width = 0.72) +
  scale_x_continuous(labels = label_comma()) +
  labs(
    title = "Fifteen most common species among living trees",
    x = "Number of trees",
    y = NULL
  )

save_plot(p_species, "05_top_species.png", width = 8, height = 6)

# -----------------------------------------------------------------------------
# 6. Bivariate analysis: predictors versus the outcome
# -----------------------------------------------------------------------------

health_by_borough <- trees %>%
  count(borough, health, .drop = FALSE, name = "n") %>%
  group_by(borough) %>%
  mutate(proportion = n / sum(n), borough_total = sum(n)) %>%
  ungroup()

write_csv(health_by_borough, file.path(table_dir, "health_by_borough.csv"))

p_borough <- ggplot(
  health_by_borough,
  aes(x = fct_reorder(borough, proportion, .fun = max), y = proportion, fill = health)
) +
  geom_col(width = 0.72) +
  coord_flip() +
  scale_fill_manual(values = health_colours) +
  scale_y_continuous(labels = label_percent()) +
  labs(
    title = "Health composition differs across boroughs",
    subtitle = "Descriptive association only; borough is not a causal explanation",
    x = NULL,
    y = "Within-borough proportion",
    fill = "Health"
  )

save_plot(p_borough, "06_health_by_borough.png", width = 8, height = 5)

species_health <- trees %>%
  count(spc_common, health, .drop = FALSE, name = "n") %>%
  group_by(spc_common) %>%
  mutate(species_total = sum(n), proportion = n / species_total) %>%
  ungroup()

# Requiring at least 1,000 trees avoids ranking extremely rare species on a
# handful of observations. The threshold is an EDA display rule, not a model rule.
species_poor_share <- species_health %>%
  filter(health == "Poor", species_total >= 1000) %>%
  arrange(desc(proportion))

write_csv(species_poor_share, file.path(table_dir, "poor_share_by_species.csv"))
print_table(
  slice_head(species_poor_share, n = 15),
  "Species with the highest observed Poor share (at least 1,000 trees)"
)

p_species_poor <- species_poor_share %>%
  slice_head(n = 15) %>%
  mutate(spc_common = fct_reorder(spc_common, proportion)) %>%
  ggplot(aes(x = proportion, y = spc_common)) +
  geom_col(fill = health_colours[["Poor"]], width = 0.72) +
  scale_x_continuous(labels = label_percent()) +
  labs(
    title = "Poor-health share varies substantially by species",
    subtitle = "Restricted to species with at least 1,000 observed living trees",
    x = "Observed Poor-health share",
    y = NULL
  )

save_plot(p_species_poor, "07_poor_share_by_species.png", width = 8, height = 6)

health_by_problem_count <- trees %>%
  count(problem_group, health, .drop = FALSE, name = "n") %>%
  group_by(problem_group) %>%
  mutate(group_total = sum(n), proportion = n / group_total) %>%
  ungroup()

write_csv(
  health_by_problem_count,
  file.path(table_dir, "health_by_problem_count.csv")
)

p_problems <- ggplot(
  health_by_problem_count,
  aes(x = problem_group, y = proportion, fill = health)
) +
  geom_col(width = 0.72) +
  scale_fill_manual(values = health_colours) +
  scale_y_continuous(labels = label_percent()) +
  labs(
    title = "Trees with more recorded problems have poorer observed health",
    subtitle = "Counts of 3 or more are pooled because the higher counts are sparse",
    x = "Number of recorded root, trunk and branch problems",
    y = "Within-group proportion",
    fill = "Health"
  )

save_plot(p_problems, "08_health_by_problem_count.png", width = 8, height = 5)

observer_health <- trees %>%
  count(user_type, health, .drop = FALSE, name = "n") %>%
  group_by(user_type) %>%
  mutate(group_total = sum(n), proportion = n / group_total) %>%
  ungroup()

write_csv(observer_health, file.path(table_dir, "health_by_user_type.csv"))

p_observer <- ggplot(
  observer_health,
  aes(x = user_type, y = proportion, fill = health)
) +
  geom_col(width = 0.72) +
  scale_fill_manual(values = health_colours) +
  scale_y_continuous(labels = label_percent()) +
  labs(
    title = "Recorded health varies by observer type",
    subtitle = "This may reflect location mix and/or measurement differences",
    x = "Census observer type",
    y = "Within-observer proportion",
    fill = "Health"
  )

save_plot(p_observer, "09_health_by_user_type.png", width = 8, height = 5)

# -----------------------------------------------------------------------------
# 7. Feature-feature and spatial exploration
# -----------------------------------------------------------------------------

# Species and diameter are two predictors. This plot assesses their association
# without using the outcome variable.
top_species_names <- top_species$spc_common[1:12]
dbh_species_plot_limit <- unname(quantile(trees$tree_dbh, 0.99, na.rm = TRUE))

p_species_dbh <- trees %>%
  filter(spc_common %in% top_species_names) %>%
  mutate(spc_common = fct_reorder(spc_common, tree_dbh, .fun = median)) %>%
  ggplot(aes(x = tree_dbh, y = spc_common)) +
  geom_boxplot(
    fill = "#A7C7A1",
    colour = "#31572C",
    outlier.shape = NA,
    linewidth = 0.35
  ) +
  coord_cartesian(xlim = c(0, dbh_species_plot_limit)) +
  labs(
    title = "Diameter distributions differ across common species",
    subtitle = "Feature-feature relationship; display capped at the overall 99th percentile",
    x = "Diameter at breast height (inches)",
    y = NULL
  )

save_plot(p_species_dbh, "10_species_vs_dbh.png", width = 8, height = 6)

set.seed(5003)
map_pool <- trees %>%
  filter(!is.na(longitude), !is.na(latitude))

map_sample <- map_pool %>%
  slice_sample(n = min(30000, nrow(map_pool)))

p_map <- ggplot(map_sample, aes(x = longitude, y = latitude, colour = health)) +
  geom_point(alpha = 0.28, size = 0.35) +
  coord_equal() +
  scale_colour_manual(values = health_colours) +
  labs(
    title = "Spatial distribution of tree health across New York City",
    subtitle = "Random sample of 30,000 living trees for legibility",
    x = "Longitude",
    y = "Latitude",
    colour = "Health"
  )

save_plot(p_map, "11_spatial_health_sample.png", width = 8, height = 7)

# -----------------------------------------------------------------------------
# 8. Candidate modelling data and leakage controls
# -----------------------------------------------------------------------------

# Excluded from modelling:
# - health is the outcome, not a predictor.
# - status is constant after restricting the cohort to living trees.
# - tree_id, block_id, address, BIN and BBL are identifiers rather than
#   generalisable tree attributes. block_id should still define resampling groups.
# - stump_diam is not applicable to living trees.
# - spc_latin duplicates the information in spc_common.
# - problems duplicates the nine issue indicators in a combined text field.
# - administrative geography fields strongly overlap. Keep borough plus
#   latitude/longitude in the first specification; compare a sensitivity model
#   with less geographic information.

model_data <- trees %>%
  transmute(
    health,
    resampling_group = block_id,
    tree_dbh,
    curb_loc,
    spc_common,
    steward,
    guards,
    sidewalk,
    user_type,
    collection_month,
    borough,
    latitude,
    longitude,
    across(all_of(issue_variables))
  )

feature_plan <- tibble::tribble(
  ~field_or_group, ~role, ~planned_treatment,
  "health", "Outcome", "Three-class factor: Good, Fair, Poor",
  "block_id", "Resampling only", "Keep blocks together across train/test splits; do not predict with the ID",
  "tree_dbh", "Predictor", "Audit implausible values; compare raw and log1p forms inside resampling",
  "spc_common", "Predictor", "Use common name only and pool rare species within each training fold",
  "curb_loc/steward/guards/sidewalk", "Predictors", "Treat as categorical and explicitly handle unknown values",
  "root/trunk/branch indicators", "Predictors", "Use the nine Yes/No fields; do not also use raw problems text",
  "user_type", "Sensitivity predictor", "Assess observer bias by comparing models with and without this field",
  "collection_month", "Sensitivity predictor", "Check seasonal/collection-process effects",
  "borough/latitude/longitude", "Predictors", "Use a parsimonious geography set and test sensitivity to its removal",
  "IDs/address/redundant geographies", "Excluded", "High-cardinality identifiers or duplicated location information",
  "status/stump_diam", "Excluded", "Constant or not applicable after selecting living trees"
)

write_csv(feature_plan, file.path(table_dir, "feature_plan.csv"))

cat(
  "\nModel-data object created in memory: ",
  comma(nrow(model_data)),
  " rows x ",
  ncol(model_data),
  " columns (including outcome and resampling group).\n",
  sep = ""
)

# Save reproducibility information without writing a second copy of the large data.
writeLines(capture.output(sessionInfo()), file.path(output_dir, "session_info.txt"))

cat(
  "\nEDA complete. Report-ready files are in: ",
  normalizePath(output_dir),
  "\n",
  sep = ""
)
