#'
#' Size estaimtes for JT and DD tests in principal study
#'
#' This script produces a table of estimated size values reproduced in 
#' manuscript Table 2.

library(here)
library(fs)
library(tidyverse)


# --- Script registries and locations ---

# Distributon identifiers and labels
DIST_KEYS <- c("F1","F2","F3","F4","F5","F6","F7","F8")
DIST_VALUES <- c(
  "F1. Normal",
  "F2. Uniform",
  "F3. Normal with outliers",
  "F4. Normal plus uniform",
  "F5. Logistic",
  "F6. Cauchy",
  "F7. Gumbel",
  "F8. Exponential"
)
DIST_LABELS <- setNames(DIST_VALUES, DIST_KEYS)

# Test identifiers and labels
TEST_KEYS <- c("DD_test", "JT_test")
TEST_VALUES <- c("Directional Difference", "Jonckheere-Terpstra")
TEST_LABELS <- setNames(TEST_VALUES, TEST_KEYS)

# Specify input filenames
principal_symmetric_data_file <- here("data", "principal_symmetric.csv")
principal_asymmetric_data_file <- here("data", "principal_asymmetric.csv")

# Specify output filename, building directory structure if needed
out_dir <- here("outputs", "tables", "size", "principal_study")
if (!dir_exists(out_dir)) { dir_create(out_dir) }
out_table_file <- here(out_dir, "test_size.csv")


# --- Utility functions for significance indication ---

#' Compute p-value for hypothesis test of rejection rate equality to level
#' 
#' This function compares accepts a vector of rejection rates and compares
#' each value to a hypothesized level to compute a two-sided p-value for a
#' test of equality to that level. Hypothesis testing assumes a reasonably
#' large sample size, which is satisfied, here.
#' 
#' @param rejection_rate Numeric vector; observed rejection rates.
#' @param alpha Numeric; nominal level of test against which rejection rates
#'   are being compared.
#' @param num_simulations Numeric; number of simulations used in computing
#'   observed rejection rates.
#'
#' @return Numeric vector; p-values of respective two-sided hypothesis tests
#'   of rejection rate equality to level.
#' 
#' @note Because the hypothesis of a proportion value also implies hypothesized 
#'   variance, a null variance based on the hypothesized level is preferred to 
#'   a sample variance in this setting.
size_pval <- function(rejection_rate, alpha, num_simulations = 500000) {

  null_var <- alpha * (1 - alpha)
  z_score <- (rejection_rate - alpha) / sqrt(null_var / num_simulations)
  
  pval <- 2 * pnorm(abs(z_score), lower.tail = FALSE)
  
  return(pval)

}

#' Conversion of p-values to stars
#' 
#' @param pval Numeric vector; p-values to be converted to stars.
#'
#' @return String vector; stars like "***" or "*".
pval_stars <- function(pval) {

  stars <- rep("", length(pval))
  stars[pval >= 0 & pval < 0.01] <- "***"
  stars[pval >= 0.01 & pval < 0.05] <- "**"
  stars[pval >= 0.05 & pval < 0.10] <- "*"
  
  return(stars)
}

# --- Ingest data ---

if (file.exists(principal_symmetric_data_file)) {
  sym_data <- read.csv(
    principal_symmetric_data_file, 
    header = TRUE,
    strip.white = TRUE
  ) %>% 
    as_tibble() %>%
    mutate(dist = DIST_LABELS[dist]) %>%
    mutate(test_name = TEST_LABELS[test_name])
}

if (file.exists(principal_asymmetric_data_file)) {
  asy_data <- read.csv(
    principal_asymmetric_data_file,
    header = TRUE,
    strip.white = TRUE
  ) %>% 
    as_tibble() %>%
    mutate(dist = DIST_LABELS[dist]) %>%
    mutate(test_name = TEST_LABELS[test_name])
}

# Since symmetric and asymmetric treatments are equivalent when the underlying
# shift is exactly zero, we can combine the no-shift measurements of the 
# symmetric and asymmetric studies to double the sample size used in computing
# size measurements. Each sample involves 250,000 simulations, so the combined
# sample consists of 500,000 simulations.
all_data <- bind_rows(
    sym_data %>% filter(treatment_shift == 0) %>% select(-treatment_shift), 
    asy_data %>% filter(treatment_shift == 0) %>% select(-treatment_shift)
  ) %>%
  group_by(k, n, dist, test_name, alpha) %>%
  summarise(rejection_rate = mean(rejection_rate, na.rm = FALSE), 
            .groups = "drop")

# --- Transform study data into size table ---

size_table <- all_data %>%
  
  # Pivot to wide format
  pivot_wider(names_from = alpha,
              values_from = rejection_rate) %>%
  arrange(test_name, dist, k, n) %>%
  select(test_name, dist, n, `0.01`, `0.05`, `0.1`) %>%

  # Annotate size estiamtes with significance indicators
  mutate(`0.01` = paste(
    format(round(`0.01`,4), nsmall=4),
    size_pval(`0.01`, 0.01) %>% pval_stars(),
    sep=""
  )) %>%
  mutate(`0.05` = paste(
    format(round(`0.05`,4), nsmall=4),
    size_pval(`0.05`, 0.05) %>% pval_stars(),
    sep=""
  )) %>%
  mutate(`0.1` = paste(
    format(round(`0.1`,4), nsmall=4),
    size_pval(`0.1`, 0.1) %>% pval_stars(),
    sep=""
  )) %>%

  # Replace true null values with dashes
  mutate(`0.01` = if_else(`0.01` == "0.0000***", "-", `0.01`))


# --- Save table to file ---

write.csv(size_table, out_table_file)