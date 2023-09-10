# mumbai_house


## Introduction

### Aim
Create a machine learning model to predict house prices in Mumbai based on area and conduct geospatial analysis to understand price variations across localities. Investigate the reasons behind these variations and estimate the overall real estate pricing trends in the area.

### Significance
Housing Price Prediction: Developing a model to predict house prices helps potential buyers and sellers make informed decisions in the Mumbai real estate market.

Geospatial Analysis: Understanding how house prices vary across different localities in Mumbai provides valuable insights for investors, developers, and urban planners.

Identifying Influencing Factors: By digging into the reasons for price variations, we can uncover critical factors such as proximity to amenities, transportation, and demand-supply dynamics.

Estate Pricing Estimate: Estimating overall real estate pricing trends in Mumbai helps stakeholders gauge market conditions and assess investment opportunities.

Data Collection and Preprocessing
Data set: Kaggle dataset containing scraped data with the following information:

House Prices and Locations: Data includes house prices and their geographical locations across Mumbai.

House Attributes: Information on the condition of houses (new or resale) and their respective areas.

Amenities: Details about various amenities provided with the houses.

The data was collected through web scraping from online real estate websites, and the basics of web scraping were learned and reviewed. The process of web scraping for the project was tested and could be implemented in future iterations.

## Data Preprocessing:

Feature Engineering: New variables like "price per sqft" were introduced. Statistical analysis was performed on numerical data, including price and area.

Location Grouping: Locations and localities were grouped based on their pincodes, mapping regions to their respective pincode areas.

Cleaning and Formatting: Locations were cleaned and formatted for consistency.

Overall Area Names: Each area was assigned an overall name based on the pincode, mapping places with the same pincode to a common location.

Null Value Check: A check for null values was conducted, and no null values were found in the dataset.

Amenities Scale Rating: Categorical variables related to amenities were analyzed, and they were assigned an amenities scale rating based on the number of amenities provided. This rating likely helps quantify the quality or desirability of the amenities offered with each property, making it easier to incorporate into the analysis or modeling process.

These preprocessing steps helped prepare the dataset for further analysis and modeling, ensuring data quality and consistency.

## Exploratory Data Analysis (EDA)
Data Visualization: Data was visualized through various plots, including displot for sale price, heat maps for feature correlations, scatterplots for numerical data (e.g., price, area, price per sqft) with respect to sale price, and the relationship between the number of bedrooms and localities with respect to sale price.

Correlation Visualization: Heatmaps were used to visualize correlations between different features, providing insights into the relationships between variables.

Scatterplots: Scatterplots were created to visualize the relationships between numerical data, such as price, area, and price per sqft, in relation to sale price.

Number of Bedrooms and Localities: Visualizations were used to analyze how the number of bedrooms and localities affect sale prices.

Skewness and Kurtosis:

Initial analysis revealed that the original data had a skewness of 0.75 and a kurtosis of 88.12, indicating some degree of non-normality and heavy-tailedness.
Outlier Analysis:

Outliers were analyzed using various techniques:
Basic size criteria for bedrooms (120 sqft) were applied, and no cases were found below this threshold.

Outliers were identified using the 1.5 * IQR rule, removing extreme values both above and below this range.

Special consideration was given to cases where a 2BHK property had a higher price than a 3BHK property in the same locality, addressing these rare anomalies.

After Outlier Removal: Following outlier removal, the data's skewness reduced to 1.50, and kurtosis decreased to 3.30. These values suggest that the data became more normally distributed and less heavy-tailed after addressing outliers.

## Model Selection
Regression Modeling: To predict house prices in Mumbai based on the given dataset, several regression algorithms were applied, including:

Linear Regression: A straightforward model used to establish a linear relationship between the input features and the target variable (house prices). It's a good baseline model for regression tasks.
Support Vector Regression (SVR): SVR is a regression technique that uses support vector machines to find the optimal hyperplane that best fits the data. It's useful for capturing complex relationships in the data.

Random Forest Regressor: Random forests are an ensemble method that combines multiple decision trees to improve predictive accuracy. They handle non-linearity well and are robust against overfitting.

AdaBoost Regressor: AdaBoost is an ensemble technique that combines weak learners to create a strong learner. In regression, it can be used to improve model performance by focusing on the data points that are difficult to predict.

Gradient Boosting Regressor: Gradient boosting is another ensemble method that builds a strong predictive model by iteratively improving upon the weaknesses of the previous model. It's known for its high accuracy.

CatBoost Regressor: CatBoost is a gradient boosting library that is known for its ease of use and ability to handle categorical features well. It often provides competitive results with minimal hyperparameter tuning.

Each of these regression algorithms was applied to the dataset to build predictive models for house prices. The choice of algorithms provides a diverse set of modeling approaches, allowing for an exploration of which one performs best for this specific prediction task. Performance metrics and validation techniques were likely used to evaluate and compare these models.

## Model Evaluation

The models were rigorously evaluated using various performance metrics in the following order:

R2 Score: The main metric used for evaluation, indicating the proportion of variance in the target variable (house prices) that the model can explain. Higher R2 scores are preferred.

Adjusted R2 Score: A modified R2 score that accounts for the number of predictors in the model, offering a more robust measure of goodness-of-fit.

Mean Absolute Error (MAE): Measures the average absolute difference between predicted and actual prices. Lower MAE values indicate better model accuracy.

Mean Squared Error (MSE): Measures the average squared difference between predicted and actual prices. Lower MSE values indicate better precision in predicting house prices.

Root Mean Squared Error (RMSE): The square root of MSE, providing an easily interpretable metric in the same unit as the target variable.
The preferred models were those with the:

Highest R2 Score, indicating the best explanatory power.
Highest Adjusted R2 Score, considering model complexity.
Lowest MAE, indicating minimal prediction error.
Lowest MSE, reflecting accurate predictions with smaller errors.
Lowest RMSE, providing a more interpretable metric.
These evaluation criteria ensure that the selected model not only fits the data well but also generalizes effectively to make accurate predictions on unseen data, which is essential for a regression task like predicting house prices.

Why R2 was the primary metric
The R2 score was a good choice as the primary metric for house price prediction in the Mumbai dataset because:

Interpretability: R2 measures the proportion of variance in the target variable (house prices) explained by the model. It provides a straightforward and interpretable assessment of predictive performance.
Comparison: R2 allows for easy comparison between different models. Higher R2 scores indicate models that better capture the underlying relationships in the data.
The Adjusted R2 score is useful in combination with R2 because:

Model Complexity: Adjusted R2 considers the number of predictors in the model. It penalizes the inclusion of irrelevant or redundant features, promoting simpler and more interpretable models.
Balanced Evaluation: While R2 may increase with the addition of more features (even if they don't improve predictive power), Adjusted R2 corrects for this by accounting for model complexity. It offers a balanced evaluation of model fit.

Together, R2 and Adjusted R2 provide a comprehensive assessment of model performance, taking into account both predictive accuracy and model simplicity, which is crucial for making meaningful predictions in real-world scenarios like house price prediction in Mumbai.

## Model Development and Training

Why Catboost was the best model for the data?
The CatBoostRegressor's impressive performance with an R2 score of 0.999496 and an adjusted R2 score of 0.999408 . Several factors that may have contributed to its success:

Handling Categorical Variables: CatBoost is well-known for its robust handling of categorical variables without requiring extensive preprocessing. In real estate data, categorical features like location or property type can be crucial, and CatBoost's ability to handle them effectively could have provided an advantage.

Gradient Boosting: CatBoost is a gradient boosting algorithm, which is inherently powerful for capturing complex nonlinear relationships in data. It excels at fitting the data well, which is evident from the high R2 score.

Hyperparameter Tuning: CatBoost often performs well with minimal hyperparameter tuning, making it a practical choice. Its default hyperparameter settings may have suited the data characteristics.
Robustness Against Overfitting: CatBoost is known for its ability to handle overfitting, which is important in regression tasks. The high R2 and adjusted R2 scores indicate that the model likely achieved a good balance between fitting the training data and generalizing to new data.

Ensemble Learning: CatBoost is an ensemble method that combines multiple trees (base models) to improve predictive accuracy. This ensemble approach can lead to robust and accurate predictions.
Data Quality: The quality of the data, including its cleanliness and consistency, can significantly impact model performance. If the data was well-preprocessed and suited CatBoost's requirements, it would contribute to its success.

Amount of Data: Gradient boosting algorithms like CatBoost can benefit from larger datasets. If the dataset was sufficiently large, it could have allowed CatBoost to leverage its capacity for learning complex patterns.


## Model Evaluation and Fine Tuning

### Grid Search Fine-Tuning:

The CatBoostRegressor model was fine-tuned using a grid search to optimize its hyperparameters. Here are the results:

Before Fine-Tuning (Normal CatBoost Model):

R2 Score: 0.999496
Adjusted R2 Score: 0.999408
Best Hyperparameters After Fine-Tuning:

Number of Estimators (n_estimators): 500
Learning Rate: 0.1
After Fine-Tuning (Optimized CatBoost Model):

R2 Score: 0.9993896204336425
Adjusted R2 Score: 0.9992824149573984
How Fine Tuning Increased the Metrics and Why:

Fine-tuning improved the model's performance by optimizing its hyperparameters. Here's why it led to higher metrics:

Hyperparameter Optimization: The initial "normal" CatBoost model had default hyperparameters, which may not have been the most suitable for the specific dataset. Fine-tuning allowed the model to find the best hyperparameter values for this particular data, such as the number of estimators and learning rate.

Increased Model Fit: By optimizing hyperparameters, the model became better tailored to the dataset's characteristics. This resulted in a slightly improved fit to the training data, reflected in the R2 and adjusted R2 scores. It means the model better explained the variance in house prices.

Reduced Overfitting: Fine-tuning can help mitigate overfitting, ensuring that the model generalizes well to unseen data. Although the improvement in metrics is modest, it indicates a better balance between fitting the training data and making accurate predictions on new data.



## Model Explainability

The SHAP (SHapley Additive exPlanations) framework was used to explain how the CatBoostRegressor was making predictions in the house price prediction task. Here's a summary of the insights gained:

Price per Area: The price per area (price per square foot) emerged as a crucial factor in determining house prices. This indicates that buyers and sellers in the Mumbai real estate market are highly sensitive to the cost relative to the size of the property.

Locality Significance: The model highlighted the significance of locality or location in predicting house prices. Different areas or neighborhoods in Mumbai have varying levels of demand, amenities, and desirability, which directly influence property prices.

Area Size: The area of the house was also identified as an important feature. Larger properties tend to have higher prices, which aligns with common real estate market trends.

Number of Bedrooms: The number of bedrooms in a property was a contributing factor. This suggests that the size and layout of a property play a role in its pricing, as more bedrooms often indicate larger homes.

Amenities Influence: Amenities provided with a property were found to be influential. Properties offering more amenities tended to command higher prices, as these features enhance the overall value and comfort of a property.

## Geospatial Analysis
Spatial analysis of Mumbai's localities was conducted using Shapely and Plotly, incorporating location and price data from the dataset. Here's a concise summary of the process:

Grouping by Median Price: Localities were grouped based on the median house price in each area. This categorization likely helped identify areas with different pricing dynamics.

GeoJSON Data: Location coordinates were obtained and represented as a GeoJSON file using Geopandas. This file provided the spatial context for mapping.

Data Merging: The GeoJSON data was merged with the price and location data from the dataset. This linkage allowed for spatial analysis of house prices in specific areas.

Latitude and Longitude Analysis: Latitude and longitude coordinates were analyzed in relation to house prices and locality. This analysis could provide insights into how location affects property values.

Color-Coded Mapping: The data was plotted on a map using Plotly, with color-coding to represent house prices. This visual representation likely allowed for a quick understanding of how prices vary across different areas of Mumbai.

Spatial analysis is valuable in understanding the geographical distribution of house prices, helping investors, buyers, and urban planners make informed decisions based on location-specific insights.

## Real-world applications and benefits of the project's insights
Importance of Accurate House Price Prediction in Mumbai:

Accurate house price prediction is of paramount importance in Mumbai for several reasons:

Investment Decisions: Potential buyers, sellers, and real estate investors rely on accurate price predictions to make informed decisions. Mumbai's dynamic real estate market necessitates precise pricing information.

Urban Planning: Government agencies and urban planners use pricing data to guide infrastructure development, zoning, and public services allocation.

Economic Indicator: Real estate prices are often considered an economic indicator, reflecting the city's economic health. Accurate predictions contribute to macroeconomic assessments.
Real-World Applications and Benefits:

The insights and models developed in this project have real-world applications and benefits:

Property Valuation: Homeowners and real estate professionals can use the models for property valuation, aiding in fair pricing strategies.

Investment Strategies: Investors can leverage accurate predictions to identify lucrative investment opportunities in various localities.

Policy Development: Urban planners and policymakers can utilize the spatial analysis to inform development policies, infrastructure projects, and housing initiatives.

Market Transparency: Providing accurate pricing data enhances market transparency, fostering trust among stakeholders.

In conclusion, accurate house price prediction and spatial analysis of Mumbai's real estate market offer valuable insights that extend beyond property transactions. They empower individuals, organizations, and government bodies to make informed decisions, contribute to economic growth, and promote efficient urban development in this bustling metropolis.

## Future Directions

Time Series Analysis: If you have access to historical data, consider performing time series analysis to identify any trends or seasonal patterns in house prices over time. This can provide valuable insights and improve the accuracy of your predictions.

Data Augmentation: Consider augmenting your dataset with additional relevant features, like nearby amenities, crime rates, school ratings, or public transportation availability. These factors can significantly impact house prices.

Time-to-Travel Analysis: Integrate data on travel time to key locations like business districts, schools, hospitals, etc. Commute times can significantly influence property prices.

Weather Data: Integrate weather data into your analysis to determine if weather patterns affect house prices. For example, proximity to the coastline might be desirable during pleasant weather but less so during monsoons.


### Libraries used: Numpy, Seaborn, Matplotlib, Shap, shapely, plotly, Pandas, Scipy, and many more.
### Frameworks: Scikit Learn.
