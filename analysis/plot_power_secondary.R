#'
#' Plot power-variance curves for JT and DD tests in the secondary study
#'
#' This script produces the power-variance plot depicted in manuscript Figure 3.

library(here)
library(fs)
library(tidyverse)

# Imports -----------------------------------------------------------------

# --- Script registries and locations ---

# Distributon identifiers and labels
DIST_KEYS <- c("F1","F2","F3","F4","F5","F6","F7","F8")
DIST_VALUES <- c(
  'F[1](sigma^2)*"  Normal family"',
  'F[2](sigma^2)*"  Uniform family"',
  'F[3](sigma^2)*"  Normal with outliers family"',
  'F[4](sigma^2)*"  Normal plus uniform family"',
  'F[5](sigma^2)*"  Logistic family"',
  'F[6](sigma^2)*"  Cauchy family"',
  'F[7](sigma^2)*"  Gumbel family"',
  'F[8](sigma^2)*"  Exponential family"'
)
DIST_LABELS <- setNames(DIST_VALUES, DIST_KEYS)

# Shift identifiers and labels
SHIFT_VALUES <- c(
  '" (with shift d=7.5)"',
  '" (with shift d=15)"',
  '" (with shift d=3)"',
  '" (with shift d=15)"',
  '" (with shift d=7.5)"',
  '" (with shift d=7.5)"',
  '" (with shift d=7.5)"',
  '" (with shift d=7.5)"'
)
SHIFT_LABELS <- setNames(SHIFT_VALUES, DIST_KEYS)

# Test identifiers and labels
TEST_KEYS <- c("DD_test", "JT_test")
TEST_VALUES <- c("Directional Difference", "Jonckheere-Terpstra")
TEST_LABELS <- setNames(TEST_VALUES, TEST_KEYS)

# Specify input filenames
secondary_symmetric_data_file <- here("data", "secondary_symmetric.csv")

# Specify output filename, building directory structure if needed
out_dir <- here("outputs", "plots", "power", "secondary_study")
if (!dir_exists(out_dir)) { dir_create(out_dir) }
out_file <- here(out_dir, "power_variance.pdf")


# --- Ingest data ---

if (file.exists(secondary_symmetric_data_file)) {
  secondary_data <- read.csv(
    secondary_symmetric_data_file, 
    header = TRUE,
    strip.white = TRUE
  ) %>% 
    as_tibble() %>%
    mutate(dist = paste0(DIST_LABELS[dist], "*", SHIFT_LABELS[dist])) %>%
    mutate(test_name = TEST_LABELS[test_name])
}

  
#' Generate and save a 4x2 grid of power-variance plots
#'
#' Parameters
#' @param data A data frame with columns c("k", "n", "treatment", 
#'                 "variance", "test", "alpha", "power")
#' @param filename A character string used to name the resulting plot to a
#'                     pdf file.
construct_var_power_plot <- function(data, filename) {

  # Generate plot
  plot <- ggplot(data) +
    theme_minimal() +
    theme(
      panel.border = element_rect(fill = NA, color = "gray70", linewidth = NULL),
      legend.position = "top",
      legend.title = element_blank(),
      axis.title.x = element_text(margin = margin(t = 15)),
      axis.title.y = element_text(margin = margin(r = 10))
    ) +
    labs(
      x = expression("Variance; median absolute deviation for " * F[6](sigma^2)),
      y = "Power"
    ) +
    # Power values are a bit noisy, even with 100,000 simulations, a slight 
    # smoothing filter reduces nuisance noise while preserving shape.
    geom_line(aes(x = parameter, y = rejection_rate, color = test_name, 
                  linetype = test_name), linewidth = 0.9) +
    scale_color_manual(values=c('black', 'gray60')) +
    facet_wrap(~ dist, nrow = 4, scales = "free_x", labeller = label_parsed)

  # Save plot to pdf
  ggsave(filename = filename, plot = plot, width = 8.5, height = 11, device = "pdf")
    
}


# Power by variance plots -------------------------------------------------

construct_var_power_plot(secondary_data, out_file)