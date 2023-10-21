## Inspiration

In large organizations, employees often grapple with the daunting task of sifting through extensive paperwork to understand and adhere to various rules and regulations. This project was inspired by the need to alleviate this challenge. The goal is to categorize regulations into meaningful categories, making it easier for businesses to identify crucial updates, automate their assignment to relevant individuals or departments, and efficiently manage compliance policies.

The project specifically addresses regulatory updates pertinent to providers of banking, insurance, and financial services, as these updates are often issued by various regulatory bodies. Natural Language Processing (NLP) models are employed to analyze the content of regulatory updates and assign tags based on this analysis to facilitate classification.

## How we built it

**Dataset:** We sourced the dataset from Kaggle, containing "Law" and "Category Names."

**Pre-processing:** To prepare the text data, we performed the following pre-processing steps:
1. Conversion of text to lowercase.
2. Removal of white spaces.
3. Elimination of punctuation marks.
4. Extraction of digits.
5. Removal of stop words.

**Feature Engineering:** We used a TFIDF vectorizer to convert the pre-processed text data into numerical vectors.

**Modeling:** Initially, we applied basic machine learning models, including Logistic Regression, K-Nearest Neighbors (KNN), and Naive Bayes. However, these models did not yield satisfactory results. Subsequently, we explored tree-based models such as LightGBM (LGBM), Random Forest, and Gradient Boosting. After extensive experimentation, we found that an ensemble of LGBM with its optimal parameters and Random Forest using a One Vs Rest Classifier produced the best results.

**Demonstration:** To showcase our project, we utilized the Gradio library for creating a user-friendly interface.

## Challenges we ran into

Several challenges were encountered during the development of this project, including:
- Time-consuming hyperparameter tuning using exhaustive Grid Search CV to find the best parameters.
- Extensive model training time, particularly for ensemble learning.
- The need for regular data updates due to policy amendments.

## Accomplishments that we're proud of

Our model has demonstrated remarkable performance, achieving a classification accuracy of 90% and an F1 score of 0.903 in categorizing regulatory updates. This success underlines the effectiveness of our approach and the potential for automating regulatory compliance management.

**Key Takeaways from the Hackathon:**
- An ensemble of a Tree-Based Model and a Linear Model provides a balanced approach, addressing the high variance in tree models and high bias in linear models.
- Consider training time as a critical parameter, especially when optimizing resource usage.
- LGBM (OvR) stands out as an optimized model, offering minimal training time and competitive results.

## Future Scope

Our project has promising avenues for further development:
- Regular retraining of models with newly updated data to maintain accuracy.
- Customization options for companies to classify their specific by-laws, enhancing relevance and applicability.

---

*Note: This README provides an overview of the project. Detailed technical documentation and code can be found in the project repository.*
