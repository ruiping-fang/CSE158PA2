### **1. Dataset Identification and Exploratory Analysis**

- **Dataset**:

  - The dataset is **Rent the Runway**, containing 192,544 rows with columns like `fit`, `age`, `weight`, `size`, `category`, `body type`, and `review_text`.
  - The target variable is `fit`, which indicates how well the clothing fits (`Small`, `Fit`, `Large`).

- **Exploratory Analysis**:

  - The dataset was preprocessed by handling missing values and extracting meaningful numerical data (e.g., extracting numerical weights from strings like "137lbs").
  - Categorical features (`category`, `body type`) were one-hot encoded, while text reviews (`review_text`) were transformed using TF-IDF (500 features).
  - **Basic Statistics**:
    - Numeric features (`age`, `size`, `weight`) were scaled using standard scaling.
    - Text data was vectorized to capture important terms for predicting fit.

- **Motivation**:
  - Exploring these features helps understand how age, weight, and textual feedback affect fit predictions, motivating a model that combines numerical, categorical, and textual inputs.

---

### **2. Predictive Task**

- **Task**: Predict the clothing fit (`Small`, `Fit`, `Large`) based on features like `age`, `weight`, `size`, `category`, `body type`, and `review_text`.
- **Evaluation**:
  - The model's performance is evaluated using metrics like accuracy and categorical cross-entropy loss.
  - Data is split into training and testing sets (80-20 split), with a validation split during training (20% of training data).
- **Baseline Models**:
  - Logistic Regression serve as baselines for this task, focusing on numeric and categorical features only.
- **Feature Processing**:
  - Numeric features (`age`, `size`, `weight`) were scaled.
  - Categorical features (`category`, `body type`) were one-hot encoded.
  - Textual features (`review_text`) were transformed into 500-dimensional TF-IDF vectors.

---

### **3. Model Description**

- **Proposed Model**:

  - A **fully connected neural network** processes the combined feature set:
    - Numeric input: Scaled values for `age`, `weight`, and `size`.
    - Categorical input: One-hot encoded vectors for `category` and `body type`.
    - Text input: TF-IDF vectorized `review_text` (500 features).
  - The model uses dense layers with ReLU activations and dropout for regularization.
  - The output layer is a softmax function predicting one of three classes (`Small`, `Fit`, `Large`).

- **Optimization**:

  - The model is optimized using the Adam optimizer with a learning rate of 0.001 and categorical cross-entropy loss.
  - Dropout (30%) prevents overfitting.

- **Challenges**:

  - **Scalability**: Processing TF-IDF with 500 features and combining with numeric and categorical features can be computationally expensive.
  - **Class Imbalance**: Possible imbalance in `fit` labels may require weighted loss functions or resampling.

- **Alternative Models Considered**:（Placeholder）

  - Logistic Regression or Random Forest (simpler but may not effectively handle text data).
  - A transformer-based model (e.g., BERT) for text, though computationally more intensive.

- **Strengths and Weaknesses**:

  - Strengths:

    - Can capture complex relationships in the data
    - Incorporates both structured and unstructured (text) data
    - Uses dropout for regularization to combat overfitting

  - Weaknesses:
    - May require more data and computational resources compared to simpler models
    - Less interpretable than linear models or decision trees

---

### **4. Related Literature**

- **Dataset Origin**:

  - The dataset is publicly available and used for predictive modeling tasks in e-commerce.[link needed here, cite needed]

- **Studies**:

  - Our approach builds on the work of Misra et al. (2018) [Decomposing Fit Semantics for Product Size Recommendation](https://dl.acm.org/doi/10.1145/3240323.3240398).

    Recommends product sizes by learning embeddings that reflect semantic notions of "fit" in metric spaces. Uses user profiles and product embeddings to predict fit preferences, focusing on latent factors.

  - [Deep learning-based collaborative filtering recommender systems: a comprehensive and systematic review](https://link.springer.com/article/10.1007/s00521-023-08958-3)

    This review examines the application of deep learning methods in collaborative filtering, emphasising their effectiveness in capturing complex user-item interactions.

- **State-of-the-Art Methods**:
  - Neural networks and transformer-based models (e.g., BERT) are commonly used for analyzing customer reviews.
  - Logistic Regression and Random Forest are traditional baselines for structured data.

---

### **5. Results and Conclusions**

- **Model Performance**:
  - The neural network achieved good performance, balancing numeric, categorical, and text features effectively.
  - Validation accuracy (around 81%) shows the model generalizes well to unseen data.
- **Feature Representations**:

  - Numeric features (`age`, `weight`, `size`) were informative, with text (`review_text`) adding meaningful context for predicting fit.
  - One-hot encoding worked well for categorical features like `category` and `body type`.

- **Success Factors**:

  - The integration of diverse features (numeric, categorical, and text) allowed the model to learn complex relationships.
  - Regularization (dropout) prevented overfitting despite the large input space.

- **Failures**:
  - The model may struggle with extremely short reviews or missing data.
  - Imbalanced class distribution could affect predictions for minority labels (`Small` or `Large`).
