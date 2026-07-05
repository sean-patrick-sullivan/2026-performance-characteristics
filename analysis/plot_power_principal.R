#'
#' Plot power curves for JT and DD tests in the principal study
#'
#' This script produces power plots depicted in manuscript Figures 1 and 2 and 
#' numerous figures in Part 2 of the online appendix.


library(here)
library(fs)
library(tidyverse)

# --- Script registries and locations ---

# Distributon identifiers and labels
DIST_KEYS <- c("F1","F2","F3","F4","F5","F6","F7","F8")
DIST_VALUES <- c(
  'F[1]*"  Normal"',
  'F[2]*"  Uniform"',
  'F[3]*"  Normal with outliers"',
  'F[4]*"  Normal plus uniform"',
  'F[5]*"  Logistic"',
  'F[6]*"  Cauchy"',
  'F[7]*"  Gumbel"',
  'F[8]*"  Exponential"'
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
out_dir <- here("outputs", "plots", "power", "principal_study")
if (!dir_exists(out_dir)) { dir_create(out_dir) }


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

# Plotting utilitiy functions ---------------------------------------------

#' Generate and save a 4x2 grid of power plots
#'
#' @param data Tibble; A tidyverse dataframe populated with Monte Carlo study 
#'   data with columns \code{k}, \code{n}, \code{dist}, \code{treatment}, 
#'   \code{test_name}, \code{alpha}, and \code{rejection_rate}.
#' @param n Numeric; used in a filter that selects only records with \code{n} 
#'   observations per group.
#' @param alpha Numeric; used in a filter that selects selects only records 
#'   involving comparison against \code{alpha} level.
#' @param filename String; location to which plot will be output. Defined using
#'   here::here().
#'
#' @return Invisibly returns \code{NULL}. Called for side effect of saving the
#'   resulting plot as a PDF file with name \code{filename}.
plot_power_grid <- function(data, filter_n, filter_alpha, filename) {
  
  # Collect data
  plot_data = data %>% 
    filter(n == filter_n, alpha == filter_alpha)

  # Build plot
  plot <- ggplot(plot_data) +
    theme_minimal() +
    theme(
      panel.border = element_rect(fill = NA, color = "gray70"),
      legend.position = "top",
      legend.title = element_blank(),
      axis.title.x = element_text(margin = margin(t = 15)),
      axis.title.y = element_text(margin = margin(r = 10))
    ) +
    labs(
      x = expression("Size of locational shift, "* d>=0),
      y = "Power"
    ) +
    geom_line(aes(x = treatment_shift, y = rejection_rate, 
                  color = test_name, linetype = test_name), 
              linewidth = 0.9) +
    scale_colour_manual(values=c('black', 'gray60')) +
    facet_wrap(~ dist, nrow = 4, scales = "free_x", labeller = label_parsed)
  
  # Save plot to pdf
  ggsave(filename = filename, plot = plot, width = 8.5, height = 11, device = "pdf")
   
}


# --- Power plots for principal study, symmetric treatment shifts ---

# Size 0.01
plot_power_grid(sym_data, 2, 0.01, 
                here(out_dir, "symmetric_alpha01_n2.pdf"))
plot_power_grid(sym_data, 3, 0.01, 
                here(out_dir, "symmetric_alpha01_n3.pdf"))
plot_power_grid(sym_data, 4, 0.01, 
                here(out_dir, "symmetric_alpha01_n4.pdf"))

# Size 0.05
plot_power_grid(sym_data, 2, 0.05, 
                here(out_dir, "symmetric_alpha05_n2.pdf"))
plot_power_grid(sym_data, 3, 0.05, 
                here(out_dir, "symmetric_alpha05_n3.pdf"))
plot_power_grid(sym_data, 4, 0.05, 
                here(out_dir, "symmetric_alpha05_n4.pdf"))

# Size 0.1
plot_power_grid(sym_data, 2, 0.1, 
                here(out_dir, "symmetric_alpha10_n2.pdf"))
plot_power_grid(sym_data, 3, 0.1, 
                here(out_dir, "symmetric_alpha10_n3.pdf"))
plot_power_grid(sym_data, 4, 0.1, 
                here(out_dir, "symmetric_alpha10_n4.pdf"))


# --- Power plots for principal study, asymmetric treatment shifts ---

# Size 0.01
plot_power_grid(asy_data, 2, 0.01, 
                here(out_dir, "asymmetric_alpha01_n2.pdf"))
plot_power_grid(asy_data, 3, 0.01, 
                here(out_dir, "asymmetric_alpha01_n3.pdf"))
plot_power_grid(asy_data, 4, 0.01, 
                here(out_dir, "asymmetric_alpha01_n4.pdf"))

# Size 0.05
plot_power_grid(asy_data, 2, 0.05, 
                here(out_dir, "asymmetric_alpha05_n2.pdf"))
plot_power_grid(asy_data, 3, 0.05, 
                here(out_dir, "asymmetric_alpha05_n3.pdf"))
plot_power_grid(asy_data, 4, 0.05, 
                here(out_dir, "asymmetric_alpha05_n4.pdf"))

# Size 0.1
plot_power_grid(asy_data, 2, 0.1, 
                here(out_dir, "asymmetric_alpha10_n2.pdf"))
plot_power_grid(asy_data, 3, 0.1, 
                here(out_dir, "asymmetric_alpha10_n3.pdf"))
plot_power_grid(asy_data, 4, 0.1, 
                here(out_dir, "asymmetric_alpha10_n4.pdf"))