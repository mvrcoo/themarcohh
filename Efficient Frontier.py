import numpy as np
import pandas as pd
import plotly.graph_objects as go

# Data: Portfolio comparison
portfolio_data = pd.DataFrame({
    "Portfolio": ["MVP", "P* (Short Allowed)", "P* (No Short)"],
    "Expected Return (%)": [1.25, 4.86, 3.11],
    "Standard Deviation (%)": [5.47, 11.37, 8.63],
    "Sharpe Ratio": [0.1990, 0.4134, 0.3416]
})

# Risk-free rate (annualized in %)
risk_free_rate = 0.156602035237182  # Example: 0.15%

# Generate the Efficient Frontier Curve
weights = np.linspace(0, 1, 100)  # Linearly spaced weights
mvp_return = 1.25  # MVP Expected Return
p_star_return = 4.86  # P* Expected Return
mvp_std = 5.47  # MVP Std Dev
p_star_std = 11.37  # P* Std Dev

# Efficient Frontier formula assuming two portfolios
efficient_returns = weights * p_star_return + (1 - weights) * mvp_return
efficient_stds = np.sqrt((weights * p_star_std) ** 2 + ((1 - weights) * mvp_std) ** 2)

# Apply constraints: Filter efficient frontier points
filtered_indices = (efficient_returns <= 5) & (efficient_stds <= 12)
efficient_returns = efficient_returns[filtered_indices]
efficient_stds = efficient_stds[filtered_indices]

# Compute Sharpe Ratios for all points on the filtered efficient frontier
sharpe_ratios = (efficient_returns - risk_free_rate) / efficient_stds

# Find the tangency portfolio (maximum Sharpe Ratio)
max_sharpe_idx = np.argmax(sharpe_ratios)
market_portfolio_return = efficient_returns[max_sharpe_idx]
market_portfolio_std = efficient_stds[max_sharpe_idx]

# Create the Capital Market Line (CML)
cml_std = np.linspace(0, max(efficient_stds) + 5, 100)  # Extend beyond market portfolio std
cml_slope = (market_portfolio_return - risk_free_rate) / market_portfolio_std
cml_returns = risk_free_rate + cml_slope * cml_std

# Create the Risk-Return Tradeoff Curve with Efficient Frontier and Risk-Free Rate
fig = go.Figure()

# Plot the efficient frontier
fig.add_trace(go.Scatter(
    x=efficient_stds,
    y=efficient_returns,
    mode='lines',
    name="Efficient Frontier",
    line=dict(color='blue', width=2)
))

# Plot individual portfolios
fig.add_trace(go.Scatter(
    x=portfolio_data["Standard Deviation (%)"],
    y=portfolio_data["Expected Return (%)"],
    mode='markers+text',
    text=portfolio_data["Portfolio"],
    name="Portfolios",
    marker=dict(size=10, color='orange')
))

# Add the risk-free rate
fig.add_trace(go.Scatter(
    x=[0],
    y=[risk_free_rate],
    mode='markers+text',
    text=["Risk-Free Rate"],
    name="Risk-Free Rate",
    marker=dict(size=10, color='green', symbol='diamond')
))

# Add the market portfolio (tangency point)
fig.add_trace(go.Scatter(
    x=[market_portfolio_std],
    y=[market_portfolio_return],
    mode='markers+text',
    text=["Market Portfolio"],
    name="Market Portfolio",
    marker=dict(size=12, color='red', symbol='circle')
))

# Add the Capital Market Line (CML)
fig.add_trace(go.Scatter(
    x=cml_std,
    y=cml_returns,
    mode='lines',
    name="Capital Market Line (CML)",
    line=dict(color='black', dash='dash', width=2)
))

# Chart Layout
fig.update_layout(
    title="Efficient Frontier with Market Portfolio and Capital Market Line",
    xaxis_title="Standard Deviation (%) (Risk)",
    yaxis_title="Expected Return (%)",
    showlegend=True,
    legend_title="Legend",
    xaxis_range=[0, 12],  # Set Standard Deviation range (0 to 12%)
    yaxis_range=[0, 5],   # Set Expected Return range (0 to 5%)
)

# Show the interactive chart
fig.show()