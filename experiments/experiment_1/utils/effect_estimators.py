"""
Causal effect estimators for Experiment 1
"""

import numpy as np
import pandas as pd
from typing import List, Optional, Dict
from abc import ABC, abstractmethod
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
import warnings


class BaseEstimator(ABC):
    """Base class for causal effect estimators"""
    
    def __init__(self):
        self.is_fitted = False
        self.impact_value = None
        
    @abstractmethod
    def fit(self, data: pd.DataFrame, treatment: str, outcome: str, 
            adjustment_set: List[str]) -> 'BaseEstimator':
        """Fit the estimator"""
        pass
    
    @abstractmethod
    def estimate_impact(self, delta: float = 1.0) -> float:
        """Estimate the impact of delta-unit change in treatment"""
        pass


class LinearRegressionEstimator(BaseEstimator):
    """
    Estimate causal effect using standardized linear regression.
    
    Impact is measured as the standardized regression coefficient.
    """
    
    def __init__(self, use_robust_se: bool = False):
        super().__init__()
        self.use_robust_se = use_robust_se
        self.model = None
        self.scaler_X = None
        self.scaler_y = None
        self.coefficient = None
        self.std_error = None
        
    def fit(self, data: pd.DataFrame, treatment: str, outcome: str, 
            adjustment_set: List[str]) -> 'LinearRegressionEstimator':
        """
        Fit linear regression model.
        
        Parameters
        ----------
        data : pd.DataFrame
            Dataset
        treatment : str
            Treatment variable name
        outcome : str
            Outcome variable name
        adjustment_set : List[str]
            Variables to adjust for
            
        Returns
        -------
        self
        """
        # Prepare features: treatment + adjustment set
        features = [treatment] + list(adjustment_set)
        
        X = data[features].values
        y = data[outcome].values
        
        # Standardize features and outcome
        self.scaler_X = StandardScaler()
        self.scaler_y = StandardScaler()
        
        X_scaled = self.scaler_X.fit_transform(X)
        y_scaled = self.scaler_y.fit_transform(y.reshape(-1, 1)).ravel()
        
        # Fit linear regression
        self.model = LinearRegression()
        self.model.fit(X_scaled, y_scaled)
        
        # Extract coefficient for treatment (first variable)
        self.coefficient = self.model.coef_[0]
        
        # Calculate standard error
        y_pred = self.model.predict(X_scaled)
        residuals = y_scaled - y_pred
        n = len(y_scaled)
        p = X_scaled.shape[1]
        
        # Residual standard error
        rse = np.sqrt(np.sum(residuals**2) / (n - p - 1))
        
        # Standard error of coefficient
        X_var = np.sum((X_scaled[:, 0] - np.mean(X_scaled[:, 0]))**2)
        self.std_error = rse / np.sqrt(X_var)
        
        self.is_fitted = True
        self.impact_value = abs(self.coefficient)  # Use absolute value
        
        return self
    
    def estimate_impact(self, delta: float = 1.0) -> float:
        """
        Estimate impact of delta standard deviations change in treatment.
        
        Parameters
        ----------
        delta : float
            Number of standard deviations to change treatment
            
        Returns
        -------
        float
            Estimated impact on outcome (in standard deviations)
        """
        if not self.is_fitted:
            raise RuntimeError("Model not fitted. Call fit() first.")
        
        return abs(self.coefficient * delta)
    
    def get_confidence_interval(self, alpha: float = 0.05) -> tuple:
        """
        Get confidence interval for the coefficient.
        
        Parameters
        ----------
        alpha : float
            Significance level (default: 0.05 for 95% CI)
            
        Returns
        -------
        tuple
            (lower_bound, upper_bound)
        """
        if not self.is_fitted:
            raise RuntimeError("Model not fitted. Call fit() first.")
        
        from scipy import stats
        
        # t-statistic for confidence interval
        t_val = stats.t.ppf(1 - alpha/2, df=len(self.model.coef_) - 1)
        
        margin = t_val * self.std_error
        lower = self.coefficient - margin
        upper = self.coefficient + margin
        
        return (abs(lower), abs(upper))


class PartialDependenceEstimator(BaseEstimator):
    """
    Estimate causal effect using Partial Dependence Plots (PDP).
    
    Impact is measured as the slope of the PDP.
    """
    
    def __init__(self, model_type: str = 'rf', n_estimators: int = 100):
        """
        Parameters
        ----------
        model_type : str
            'rf' for Random Forest or 'gb' for Gradient Boosting
        n_estimators : int
            Number of trees in the ensemble
        """
        super().__init__()
        self.model_type = model_type
        self.n_estimators = n_estimators
        self.model = None
        self.treatment_col = None
        self.pdp_slope = None
        
    def fit(self, data: pd.DataFrame, treatment: str, outcome: str, 
            adjustment_set: List[str]) -> 'PartialDependenceEstimator':
        """
        Fit the model and calculate PDP slope.
        
        Parameters
        ----------
        data : pd.DataFrame
            Dataset
        treatment : str
            Treatment variable name
        outcome : str
            Outcome variable name
        adjustment_set : List[str]
            Variables to adjust for
            
        Returns
        -------
        self
        """
        # Prepare features
        features = [treatment] + list(adjustment_set)
        self.treatment_col = treatment
        
        X = data[features].values
        y = data[outcome].values
        
        # Fit model
        if self.model_type == 'rf':
            self.model = RandomForestRegressor(
                n_estimators=self.n_estimators,
                random_state=42,
                n_jobs=-1
            )
        elif self.model_type == 'gb':
            self.model = GradientBoostingRegressor(
                n_estimators=self.n_estimators,
                random_state=42
            )
        else:
            raise ValueError(f"Unknown model_type: {self.model_type}")
        
        self.model.fit(X, y)
        
        # Calculate PDP slope
        self.pdp_slope = self._calculate_pdp_slope(data, features, treatment)
        self.is_fitted = True
        self.impact_value = abs(self.pdp_slope)
        
        return self
    
    def _calculate_pdp_slope(self, data: pd.DataFrame, features: List[str], 
                            treatment: str, n_points: int = 50) -> float:
        """
        Calculate the slope of the partial dependence plot.
        
        Parameters
        ----------
        data : pd.DataFrame
            Dataset
        features : List[str]
            Feature names
        treatment : str
            Treatment variable name
        n_points : int
            Number of points to evaluate PDP
            
        Returns
        -------
        float
            Average slope of PDP
        """
        X = data[features].values
        treatment_idx = features.index(treatment)
        
        # Get range of treatment values
        treatment_vals = data[treatment].values
        treatment_min = np.percentile(treatment_vals, 5)
        treatment_max = np.percentile(treatment_vals, 95)
        
        # Create grid of treatment values
        grid_values = np.linspace(treatment_min, treatment_max, n_points)
        
        # Calculate PDP
        pdp_values = []
        
        for val in grid_values:
            # Create copy of X with treatment set to val
            X_modified = X.copy()
            X_modified[:, treatment_idx] = val
            
            # Predict and average
            predictions = self.model.predict(X_modified)
            pdp_values.append(np.mean(predictions))
        
        # Calculate slope using linear regression on PDP
        pdp_values = np.array(pdp_values)
        
        # Fit line to PDP
        coeffs = np.polyfit(grid_values, pdp_values, 1)
        slope = coeffs[0]
        
        return slope
    
    def estimate_impact(self, delta: float = 1.0) -> float:
        """
        Estimate impact of delta-unit change in treatment.
        
        Parameters
        ----------
        delta : float
            Change in treatment variable
            
        Returns
        -------
        float
            Estimated change in outcome
        """
        if not self.is_fitted:
            raise RuntimeError("Model not fitted. Call fit() first.")
        
        return abs(self.pdp_slope * delta)


class DoCalculusEstimator(BaseEstimator):
    """
    Estimate causal effect using do-calculus (backdoor adjustment).
    
    This implements the backdoor adjustment formula:
    E[Y | do(X = x)] = sum_z E[Y | X = x, Z = z] * P(Z = z)
    """
    
    def __init__(self, n_bins: int = 10):
        """
        Parameters
        ----------
        n_bins : int
            Number of bins for discretizing continuous adjustment variables
        """
        super().__init__()
        self.n_bins = n_bins
        self.effect = None
        
    def fit(self, data: pd.DataFrame, treatment: str, outcome: str, 
            adjustment_set: List[str]) -> 'DoCalculusEstimator':
        """
        Estimate causal effect using backdoor adjustment.
        
        Parameters
        ----------
        data : pd.DataFrame
            Dataset
        treatment : str
            Treatment variable name
        outcome : str
            Outcome variable name
        adjustment_set : List[str]
            Variables to adjust for (confounders)
            
        Returns
        -------
        self
        """
        if len(adjustment_set) == 0:
            # No confounders, simple difference
            self.effect = self._estimate_no_confounders(data, treatment, outcome)
        else:
            # Use backdoor adjustment
            self.effect = self._estimate_with_adjustment(
                data, treatment, outcome, adjustment_set
            )
        
        self.is_fitted = True
        self.impact_value = abs(self.effect)
        
        return self
    
    def _estimate_no_confounders(self, data: pd.DataFrame, treatment: str, 
                                 outcome: str) -> float:
        """Estimate effect when there are no confounders"""
        # Simple regression coefficient
        from scipy import stats
        
        X = data[treatment].values.reshape(-1, 1)
        y = data[outcome].values
        
        # Calculate correlation-based effect
        slope, intercept, r_value, p_value, std_err = stats.linregress(
            X.ravel(), y
        )
        
        return slope
    
    def _estimate_with_adjustment(self, data: pd.DataFrame, treatment: str, 
                                  outcome: str, adjustment_set: List[str]) -> float:
        """Estimate effect with backdoor adjustment"""
        # Use stratification approach for simplicity
        # Bin adjustment variables and stratify
        
        df = data.copy()
        
        # Create strata based on adjustment set
        strata_cols = []
        for adj_var in adjustment_set:
            col_name = f"{adj_var}_bin"
            df[col_name] = pd.qcut(
                df[adj_var], 
                q=self.n_bins, 
                labels=False, 
                duplicates='drop'
            )
            strata_cols.append(col_name)
        
        # Calculate weighted average effect across strata
        effects = []
        weights = []
        
        for strata_vals, group_df in df.groupby(strata_cols):
            if len(group_df) < 5:  # Skip small strata
                continue
            
            # Estimate effect within stratum
            X = group_df[treatment].values
            y = group_df[outcome].values
            
            if len(np.unique(X)) > 1:  # Need variation in treatment
                from scipy import stats
                slope, _, _, _, _ = stats.linregress(X, y)
                effects.append(slope)
                weights.append(len(group_df))
        
        if len(effects) == 0:
            # Fallback to simple regression
            return self._estimate_no_confounders(data, treatment, outcome)
        
        # Weighted average
        effects = np.array(effects)
        weights = np.array(weights)
        weighted_effect = np.average(effects, weights=weights)
        
        return weighted_effect
    
    def estimate_impact(self, delta: float = 1.0) -> float:
        """
        Estimate impact of delta-unit change in treatment.
        
        Parameters
        ----------
        delta : float
            Change in treatment variable
            
        Returns
        -------
        float
            Estimated change in outcome
        """
        if not self.is_fitted:
            raise RuntimeError("Model not fitted. Call fit() first.")
        
        return abs(self.effect * delta)


class CausalForestEstimator(BaseEstimator):
    """
    Placeholder for Causal Forest estimator.
    
    Note: This requires econml or causalml packages.
    For now, falls back to Random Forest variable importance.
    """
    
    def __init__(self):
        super().__init__()
        self.feature_importance = None
        self.model = None
        warnings.warn(
            "CausalForestEstimator is using Random Forest variable importance "
            "as a proxy. Install econml for true causal forest implementation."
        )
        
    def fit(self, data: pd.DataFrame, treatment: str, outcome: str, 
            adjustment_set: List[str]) -> 'CausalForestEstimator':
        """
        Fit using Random Forest and extract variable importance.
        
        Parameters
        ----------
        data : pd.DataFrame
            Dataset
        treatment : str
            Treatment variable name
        outcome : str
            Outcome variable name
        adjustment_set : List[str]
            Variables to adjust for
            
        Returns
        -------
        self
        """
        features = [treatment] + list(adjustment_set)
        
        X = data[features].values
        y = data[outcome].values
        
        # Fit Random Forest
        self.model = RandomForestRegressor(
            n_estimators=200,
            max_depth=10,
            random_state=42,
            n_jobs=-1
        )
        self.model.fit(X, y)
        
        # Extract importance for treatment variable (first feature)
        self.feature_importance = self.model.feature_importances_[0]
        
        self.is_fitted = True
        self.impact_value = self.feature_importance
        
        return self
    
    def estimate_impact(self, delta: float = 1.0) -> float:
        """
        Return the feature importance as impact measure.
        
        Parameters
        ----------
        delta : float
            Ignored for this estimator
            
        Returns
        -------
        float
            Feature importance value
        """
        if not self.is_fitted:
            raise RuntimeError("Model not fitted. Call fit() first.")
        
        return self.feature_importance


if __name__ == "__main__":
    # Test with synthetic data
    print("Testing effect estimators...")
    print()
    
    np.random.seed(42)
    n = 1000
    
    # Create synthetic data: Z -> X -> Y, Z -> Y
    Z = np.random.randn(n)
    X = 0.5 * Z + np.random.randn(n) * 0.5
    Y = 0.7 * X + 0.3 * Z + np.random.randn(n) * 0.3
    
    data = pd.DataFrame({'X': X, 'Y': Y, 'Z': Z})
    
    print("Testing Linear Regression Estimator:")
    lr_est = LinearRegressionEstimator()
    lr_est.fit(data, 'X', 'Y', ['Z'])
    print(f"  Coefficient: {lr_est.coefficient:.3f}")
    print(f"  Impact: {lr_est.estimate_impact():.3f}")
    print(f"  95% CI: {lr_est.get_confidence_interval()}")
    
    print("\nTesting Partial Dependence Estimator:")
    pd_est = PartialDependenceEstimator(model_type='rf')
    pd_est.fit(data, 'X', 'Y', ['Z'])
    print(f"  PDP Slope: {pd_est.pdp_slope:.3f}")
    print(f"  Impact: {pd_est.estimate_impact():.3f}")
    
    print("\nTesting Do-Calculus Estimator:")
    do_est = DoCalculusEstimator()
    do_est.fit(data, 'X', 'Y', ['Z'])
    print(f"  Effect: {do_est.effect:.3f}")
    print(f"  Impact: {do_est.estimate_impact():.3f}")
    
    print("\nTesting Causal Forest Estimator (RF proxy):")
    cf_est = CausalForestEstimator()
    cf_est.fit(data, 'X', 'Y', ['Z'])
    print(f"  Feature Importance: {cf_est.feature_importance:.3f}")
    print(f"  Impact: {cf_est.estimate_impact():.3f}")
