import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from sklearn.linear_model import LinearRegression
from scipy.interpolate import interp1d
from typing import List, Tuple, Dict

def load_data(file_path='Nat_Gas.csv'):
    """Load and preprocess the natural gas price data."""
    df = pd.read_csv(file_path)
    df['Dates'] = pd.to_datetime(df['Dates'])
    df['Prices'] = df['Prices'].astype(float)
    return df

def prepare_model_data(df):
    """Prepare data for the linear regression model."""
    # Convert dates to numeric values (days since start)
    start_date = df['Dates'].min()
    df['days'] = (df['Dates'] - start_date).dt.days
    
    # Create and fit the model
    X = df['days'].values.reshape(-1, 1)
    y = df['Prices'].values
    model = LinearRegression()
    model.fit(X, y)
    
    return model, start_date

def estimate_price(date_str, model, start_date, df):
    """
    Estimate the natural gas price for a given date.
    
    Args:
        date_str (str): Date in format 'MM/DD/YY' or 'YYYY-MM-DD'
        model: Fitted linear regression model
        start_date: Reference start date for the model
        df: Original dataframe with actual data
    
    Returns:
        float: Estimated price
    """
    try:
        # Convert input date string to datetime
        if '/' in date_str:
            date = datetime.strptime(date_str, '%m/%d/%y')
        else:
            date = datetime.strptime(date_str, '%Y-%m-%d')
        
        # Calculate days since start date
        days = (date - start_date).days
        
        # If date is within the range of actual data, use interpolation
        if df['Dates'].min() <= date <= df['Dates'].max():
            # Create interpolation function
            f = interp1d(df['days'], df['Prices'], kind='linear', fill_value='extrapolate')
            return float(f(days))
        
        # For dates outside the range, use the linear regression model
        return float(model.predict([[days]])[0])
    
    except ValueError as e:
        raise ValueError(f"Invalid date format. Please use 'MM/DD/YY' or 'YYYY-MM-DD' format. Error: {str(e)}")

class StorageContract:
    def __init__(self, 
                 injection_dates: List[str],
                 withdrawal_dates: List[str],
                 injection_rates: List[float],
                 withdrawal_rates: List[float],
                 max_storage: float,
                 storage_cost_per_unit: float,
                 model: LinearRegression,
                 start_date: datetime,
                 price_data: pd.DataFrame):
        """
        Initialize a storage contract with all necessary parameters.
        
        Args:
            injection_dates: List of dates for gas injection
            withdrawal_dates: List of dates for gas withdrawal
            injection_rates: List of injection rates (units per day)
            withdrawal_rates: List of withdrawal rates (units per day)
            max_storage: Maximum storage capacity
            storage_cost_per_unit: Cost per unit of storage per day
            model: Price prediction model
            start_date: Reference start date for the model
            price_data: Historical price data
        """
        self.injection_dates = [datetime.strptime(d, '%Y-%m-%d') if '-' in d 
                              else datetime.strptime(d, '%m/%d/%y') for d in injection_dates]
        self.withdrawal_dates = [datetime.strptime(d, '%Y-%m-%d') if '-' in d 
                               else datetime.strptime(d, '%m/%d/%y') for d in withdrawal_dates]
        self.injection_rates = injection_rates
        self.withdrawal_rates = withdrawal_rates
        self.max_storage = max_storage
        self.storage_cost_per_unit = storage_cost_per_unit
        self.model = model
        self.start_date = start_date
        self.price_data = price_data
        
        # Validate inputs
        self._validate_inputs()
    
    def _validate_inputs(self):
        """Validate the contract parameters."""
        if len(self.injection_dates) != len(self.injection_rates):
            raise ValueError("Number of injection dates must match number of injection rates")
        if len(self.withdrawal_dates) != len(self.withdrawal_rates):
            raise ValueError("Number of withdrawal dates must match number of withdrawal rates")
        if any(rate <= 0 for rate in self.injection_rates + self.withdrawal_rates):
            raise ValueError("All rates must be positive")
        if self.max_storage <= 0:
            raise ValueError("Maximum storage must be positive")
        if self.storage_cost_per_unit < 0:
            raise ValueError("Storage cost cannot be negative")
    
    def calculate_contract_value(self) -> Dict:
        """
        Calculate the value of the storage contract.
        
        Returns:
            Dictionary containing:
            - total_value: Net present value of the contract
            - cash_flows: List of all cash flows
            - storage_levels: List of storage levels over time
        """
        # Sort all dates and create a timeline
        all_dates = sorted(set(self.injection_dates + self.withdrawal_dates))
        storage_level = 0
        cash_flows = []
        storage_levels = []
        
        for date in all_dates:
            # Calculate storage cost for the period
            if storage_levels:
                days_since_last = (date - all_dates[all_dates.index(date)-1]).days
                storage_cost = storage_level * self.storage_cost_per_unit * days_since_last
                cash_flows.append(('storage_cost', date, -storage_cost))
            
            # Handle injections
            if date in self.injection_dates:
                idx = self.injection_dates.index(date)
                rate = self.injection_rates[idx]
                price = estimate_price(date.strftime('%Y-%m-%d'), self.model, self.start_date, self.price_data)
                injection_cost = rate * price
                storage_level += rate
                
                if storage_level > self.max_storage:
                    raise ValueError(f"Storage capacity exceeded on {date}")
                
                cash_flows.append(('injection', date, -injection_cost))
            
            # Handle withdrawals
            if date in self.withdrawal_dates:
                idx = self.withdrawal_dates.index(date)
                rate = self.withdrawal_rates[idx]
                price = estimate_price(date.strftime('%Y-%m-%d'), self.model, self.start_date, self.price_data)
                withdrawal_revenue = rate * price
                storage_level -= rate
                
                if storage_level < 0:
                    raise ValueError(f"Storage level cannot go below 0 on {date}")
                
                cash_flows.append(('withdrawal', date, withdrawal_revenue))
            
            storage_levels.append((date, storage_level))
        
        total_value = sum(cf[2] for cf in cash_flows)
        
        return {
            'total_value': total_value,
            'cash_flows': cash_flows,
            'storage_levels': storage_levels
        }

def main():
    # Load and prepare data
    df = load_data()
    model, start_date = prepare_model_data(df)
    
    # Example usage
    print("\nNatural Gas Storage Contract Pricing")
    print("===================================")
    
    # Example contract parameters
    contract = StorageContract(
        injection_dates=['2023-01-01', '2023-02-01', '2023-03-01'],
        withdrawal_dates=['2023-04-01', '2023-05-01'],
        injection_rates=[2000, 1500, 1000],
        withdrawal_rates=[2000, 2500],
        max_storage=10000,
        storage_cost_per_unit=0.015,
        model=model,
        start_date=start_date,
        price_data=df
    )
    
    try:
        result = contract.calculate_contract_value()
        print(f"\nContract Value: ${result['total_value']:,.2f}")
        print("\nCash Flows:")
        for flow_type, date, amount in result['cash_flows']:
            print(f"{date.strftime('%Y-%m-%d')}: {flow_type:12} ${amount:,.2f}")
        print("\nStorage Levels:")
        for date, level in result['storage_levels']:
            print(f"{date.strftime('%Y-%m-%d')}: {level:,.0f} units")
    except ValueError as e:
        print(f"Error: {str(e)}")

if __name__ == "__main__":
    main() 