summary(Olympics_dataset_Final_product)
colnames(Olympics_dataset_Final_product)

library(dplyr)

vars_to_keep <- c(
  "year", "team", "NOC", "gapminder_countryname",
  "medal", "sport", "event", "season",
  "GDP per capita, PPP (constant 2021 international $)",
  "GDP total, PPP (constant 2021 international $)",
  "GDP per capita growth (%)", "Population", "Gini"
)


unique(df_country_model$season)
df_country_model_summer <- df_country_model %>%
  filter(season == "Summer Olympics")



df_country_model_summer <- df_country_model_summer %>%
  mutate(points = case_when(
    medal == "Gold" ~ 3,
    medal == "Silver" ~ 2,
    medal == "Bronze" ~ 1,
    TRUE ~ 0
  ))

events_per_country <- df_country_model_summer %>%
  group_by(team, year) %>%
  summarise(event_count = n_distinct(event), .groups = "drop")

df_country_summary <- df_country_model_summer %>%  
  left_join(events_per_country, by = c("team", "year"))

str(df_country_summary)

library(dplyr)

df_country_summary2 <- df_country_summary %>%
  distinct(year, medal, event, team, .keep_all = TRUE)

library(dplyr)

df_aggregated2 <- df_country_summary2 %>%
  group_by(team, year) %>%
  summarise(
    total_points = sum(points, na.rm = TRUE),
    event_count = first(event_count),
    GDP_per_capita = first(`GDP per capita, PPP (constant 2021 international $)`),
    GDP_total = first(`GDP total, PPP (constant 2021 international $)`),
    GDP_growth = first(`GDP per capita growth (%)`),
    Population = first(Population),
    Gini = first(Gini),
    gapminder_countryname = first(gapminder_countryname),
    NOC = first(NOC),
    .groups = "drop"
  )

df_aggregated2 <- df_aggregated2 %>%
  filter(!is.na(Population))

df_aggregated2 <- df_aggregated2 %>%
  filter(GDP_per_capita != "no_data")

df_aggregated2 <- df_aggregated2 %>%
  filter(GDP_total != "no_data")

df_aggregated2 <- df_aggregated2 %>%
  filter(Gini != "no_data")

str(df_aggregated)

df_aggregated2 <- df_aggregated2 %>%
  mutate(
    GDP_per_capita = as.numeric(GDP_per_capita),
    GDP_total      = as.numeric(GDP_total),
    GDP_growth     = as.numeric(GDP_growth),
    Population     = as.numeric(Population),
    Gini           = as.numeric(Gini)
  )


df_with_na <- df_aggregated[!complete.cases(df_aggregated), ]


write.csv(df_aggregated, "df_aggregated.csv", row.names = FALSE)

str(df_aggregated)
#error in amount of points and events

df_usa_1984_2 <- df_country_summary2 %>%
  filter(year == 1984, team == "USA")


df_usa_1984_2 <- df_country_summary2 %>%
  filter(year == 1984, team == "USA")

total_points_usa_1984 <- sum(df_usa_1984_2$points, na.rm = TRUE)

medal_counts <- df_usa_1984_2 %>%
  count(medal)

print(medal_counts)



#predict expected points


library(dplyr)
library(randomForest)
library(Metrics)

set.seed(123)

# Filter and prepare data (remove rows with NA)
df_model <- df_aggregated2 %>%
  select(team, year, total_points, event_count, GDP_per_capita, GDP_total, GDP_growth, Population, Gini) %>%
  filter(!is.na(total_points)) %>%
  mutate(year = as.integer(year))  # convert year to integer for filtering

# Define train and test sets based on year
train_data <- df_model %>% filter(year < 2020)
test_data  <- df_model %>% filter(year >= 2020)

# Train random forest on training data
rf_model <- randomForest(total_points ~ event_count + GDP_per_capita + GDP_total + GDP_growth + Population + Gini,
                         data = train_data, ntree = 500)

# Predict on test data
test_data$predictions <- predict(rf_model, newdata = test_data)

# Evaluate model performance
rmse_val <- rmse(test_data$total_points, predictions)
cat("RMSE on test set (2020 and 2024):", rmse_val, "\n")

options(scipen = 999)

#steps to do fix -> points -> points kinda fixed
#make more features -> distance, previous winner

sum(is.na(Olympics_dataset_Final_product$height_in_cm))
