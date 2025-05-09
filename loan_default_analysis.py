import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

class LoanDefaultPredictor:
    def __init__(self, recovery_rate=0.10):
        """
        Initialize the loan default predictor.
        
        Args:
            recovery_rate (float): Expected recovery rate in case of default (default: 0.10)
        """
        self.recovery_rate = recovery_rate
        self.scaler = StandardScaler()
        self.lr_model = LogisticRegression(random_state=42)
        self.rf_model = RandomForestClassifier(random_state=42)
        self.feature_columns = [
            'credit_lines_outstanding',
            'loan_amt_outstanding',
            'total_debt_outstanding',
            'income',
            'years_employed',
            'fico_score'
        ]
    
    def load_and_prepare_data(self, file_path):
        """Load and prepare the loan data."""
        df = pd.read_csv(file_path)
        
        # Split features and target
        X = df[self.feature_columns]
        y = df['default']
        
        # Split into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Scale the features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        return X_train_scaled, X_test_scaled, y_train, y_test
    
    def train_models(self, X_train, y_train):
        """Train both logistic regression and random forest models."""
        # Train logistic regression
        self.lr_model.fit(X_train, y_train)
        
        # Train random forest
        self.rf_model.fit(X_train, y_train)
    
    def evaluate_models(self, X_test, y_test):
        """Evaluate both models and print performance metrics."""
        # Get predictions
        lr_pred_proba = self.lr_model.predict_proba(X_test)[:, 1]
        rf_pred_proba = self.rf_model.predict_proba(X_test)[:, 1]
        
        # Calculate ROC AUC scores
        lr_auc = roc_auc_score(y_test, lr_pred_proba)
        rf_auc = roc_auc_score(y_test, rf_pred_proba)
        
        print("\nModel Performance:")
        print(f"Logistic Regression ROC AUC: {lr_auc:.4f}")
        print(f"Random Forest ROC AUC: {rf_auc:.4f}")
        
        # Print classification reports
        print("\nLogistic Regression Classification Report:")
        print(classification_report(y_test, self.lr_model.predict(X_test)))
        
        print("\nRandom Forest Classification Report:")
        print(classification_report(y_test, self.rf_model.predict(X_test)))
        
        return lr_auc, rf_auc
    
    def predict_default_probability(self, loan_data, model_type='rf'):
        """
        Predict the probability of default for a given loan.
        
        Args:
            loan_data (dict): Dictionary containing loan features
            model_type (str): 'lr' for logistic regression or 'rf' for random forest
        
        Returns:
            float: Probability of default
        """
        # Convert input to DataFrame
        input_df = pd.DataFrame([loan_data])
        
        # Scale the features
        input_scaled = self.scaler.transform(input_df[self.feature_columns])
        
        # Get prediction based on model type
        if model_type.lower() == 'lr':
            return self.lr_model.predict_proba(input_scaled)[0, 1]
        else:
            return self.rf_model.predict_proba(input_scaled)[0, 1]
    
    def calculate_expected_loss(self, loan_data, loan_amount, model_type='rf'):
        """
        Calculate the expected loss for a loan.
        
        Args:
            loan_data (dict): Dictionary containing loan features
            loan_amount (float): Amount of the loan
            model_type (str): 'lr' for logistic regression or 'rf' for random forest
        
        Returns:
            float: Expected loss
        """
        pd = self.predict_default_probability(loan_data, model_type)
        lgd = 1 - self.recovery_rate  # Loss Given Default
        return pd * lgd * loan_amount
    
    def plot_feature_importance(self):
        """Plot feature importance for the random forest model."""
        importance = self.rf_model.feature_importances_
        features = self.feature_columns
        
        # Create DataFrame for plotting
        importance_df = pd.DataFrame({
            'Feature': features,
            'Importance': importance
        }).sort_values('Importance', ascending=False)
        
        # Plot
        plt.figure(figsize=(10, 6))
        sns.barplot(x='Importance', y='Feature', data=importance_df)
        plt.title('Feature Importance in Random Forest Model')
        plt.tight_layout()
        plt.show()

def main():
    # Initialize the predictor
    predictor = LoanDefaultPredictor()
    
    # Load and prepare data
    X_train, X_test, y_train, y_test = predictor.load_and_prepare_data('Task 3 and 4_Loan_Data.csv')
    
    # Train models
    predictor.train_models(X_train, y_train)
    
    # Evaluate models
    predictor.evaluate_models(X_test, y_test)
    
    # Plot feature importance
    predictor.plot_feature_importance()
    
    # Example loan data
    example_loan = {
        'credit_lines_outstanding': 2,
        'loan_amt_outstanding': 5000,
        'total_debt_outstanding': 15000,
        'income': 75000,
        'years_employed': 5,
        'fico_score': 650
    }
    
    # Calculate expected loss
    loan_amount = 10000
    expected_loss = predictor.calculate_expected_loss(example_loan, loan_amount)
    
    print("\nExample Loan Analysis:")
    print(f"Loan Amount: ${loan_amount:,.2f}")
    print(f"Probability of Default: {predictor.predict_default_probability(example_loan):.2%}")
    print(f"Expected Loss: ${expected_loss:,.2f}")

if __name__ == "__main__":
    main() 