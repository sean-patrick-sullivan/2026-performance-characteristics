#'
#' Density illustrations for F1 through F8
#'
#' This script produces the density curves depicted in online appendix Figure 1.

library(here)
library(fs)
library(ordinal)
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

# Specify output filename, building directory structure if needed
out_dir <- here("outputs", "plots", "densities")
if (!dir_exists(out_dir)) { 
  dir_create(out_dir)
}
out_filename <- here(out_dir, "dist_densities_F1-F8.pdf")

# --- Sampling distributions ---

#' Common function signature for F1 through F8
#' 
#' @param size Integer; number of random draws to take from distribution.
#' 
#' @return Numeric vector; random draws from distribution.
#' 
#' @note The definitions of F1 through F8 in this script are *only* used to 
#'   plot illustrative densities. Equivalent definitions of F1 through F8 in 
#'   \code{monte_carlo/dists/fixed_dists.py} are used when running Monte Carlo 
#'   studies. The python versions of the distribution functions are commented 
#'   with additional details.

F1 <- function(size) {
  rnorm(size, mean = 0, sd = sqrt(50))
}

F2 <- function(size) {
  runif(size, min = -10*sqrt(6), max = 10*sqrt(6))
}

F3 <- function(size) {
  primaryVar <- rnorm(size, mean = 0, sd = sqrt(5))
  outlierVar <- rnorm(size, mean = 0, sd = sqrt(25))
  binomialChoice <- rbinom(size, size = 1, prob = 0.8)
  
  result <- (primaryVar * binomialChoice) + (outlierVar * (1 - binomialChoice))
  return(result)
}

F4 <- function(size) {
  norm_var <- rnorm(size, mean = 0, sd = sqrt(25))
  unif_var <- runif(size, min = -10*sqrt(6), max = 10*sqrt(6))
  
  result <- norm_var + unif_var
  return(result)
}

F5 <- function(size) {
  rlogis(size, location = 0, scale = 5*sqrt(6)/pi)
}

F6 <- function(size) {
  rcauchy(size, location = 0, scale = 1)
}

F7 <- function(size) {
  beta <- 10*sqrt(3)/pi
  mu <- -beta * 0.57721566490153286060651209008240243
  result <- rgumbel(size, location = mu, scale = beta)
  return(result)
}

F8 <- function(size) {
  exp_var <- rexp(size, rate = sqrt(2)/10)
  result <- exp_var - 1/(sqrt(2)/10)
  return(result)
}


# --- Data generation ---

num_simulations <- 3000000
set.seed(926375)

# Generate sample data for empirical density plots
sample_1 <- F1(num_simulations)
sample_2 <- F2(num_simulations)
sample_3 <- F3(num_simulations)
sample_4 <- F4(num_simulations)
sample_5 <- F5(num_simulations)
sample_6 <- F6(num_simulations)
sample_7 <- F7(num_simulations)
sample_8 <- F8(num_simulations)

# Convert sample data to ggplot2-friendly data frame
data <- data.frame(
  value = c(sample_1, sample_2, sample_3, sample_4, sample_5, sample_6, 
    sample_7, sample_8),
  distribution = factor(rep(c("F1", "F2", "F3", "F4", "F5", "F6", "F7", "F8"), 
    each = num_simulations))
) %>% as_tibble()

# Manually restrict specific samples to improve plot legibility. (This allows
# automatic scales to be used in density plots while preventing low-probability
# extreme values from causing large and distracting scale disparities between
# facets for different distributions.)
data <- data %>%
  mutate(value = ifelse(distribution == "F1" & abs(value) > 30, NA, value)) %>%
  mutate(value = ifelse(distribution == "F5" & abs(value) > 50, NA, value)) %>%
  mutate(value = ifelse(distribution == "F6" & abs(value) > 10, NA, value)) %>%
  mutate(value = ifelse(distribution == "F7" & value > 40, NA, value)) %>%
  mutate(value = ifelse(distribution == "F8" & value > 30, NA, value))

# Define theoretical density function for the uniform distribution to better
# illustrate sharp edges of this distribution.
F2_pdf <- function(x) {
  a = -10*sqrt(6)
  b = 10*sqrt(6)
  ifelse(x >= a & x <= b, 1 / (b - a), 0)
}

# Rewrite data values with plot vocabulary
data <- data %>%
  mutate(distribution = DIST_LABELS[distribution])


# --- Plot densities ---

plot <- ggplot(data, aes(x = value)) +

  # Set theme parameters
  theme_minimal() +
  theme(
    panel.border = element_rect(fill = NA, color = "gray70"),
    axis.title.x = element_text(margin = margin(t = 15)),
    axis.title.y = element_text(margin = margin(r = 10))
  ) +

  # For most distributions, plot empirical density curve
  geom_density(data = data %>% filter(distribution != 'F[2]*"  Uniform"'), 
    aes(y = after_stat(density)), fill = "lightgray", color = "darkgray", adjust = 3, 
      linewidth = 0.75) +

  # For unifrom distribution, substitute theoretical density curve
  geom_area(
    data = data %>% filter(distribution == 'F[2]*"  Uniform"'),
    stat = "function",
    fun = F2_pdf,
    fill = "lightgray"
  ) +
  stat_function(
    fun = F2_pdf,
    data = data %>% filter(distribution == 'F[2]*"  Uniform"'),
    geom = "area",
    fill = "lightgray", color = "darkgray", linewidth = 0.75
  ) +

  # Allow both axes to be free across facets, specify 4 rows
  facet_wrap(~ distribution, nrow = 4, scales = "free", labeller = label_parsed) + 
  labs(
       x = "Value",
       y = "Density"
  )

# --- Save plot to file ---

ggsave(filename = out_filename, plot = plot, width = 8.5, height = 11, 
       device = "pdf")