!pip install streamlit

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

# Load the data
data = pd.read_csv('food_orders_new_delhi.csv')

# Convert date and time columns to datetime
data['Order Date and Time'] = pd.to_datetime(data['Order Date and Time'])
data['Delivery Date and Time'] = pd.to_datetime(data['Delivery Date and Time'])

# Create a function to extract numeric values from the 'Discounts and Offers' string
def extract_discount(discount_str):
    if 'off' in discount_str:
         # Fixed amount off
        return float(discount_str.split(' ')[0])
    elif '%' in discount_str:
#         # Percentage off

       return float(discount_str.split('%')[0])

    else:
#         # No discount
        return 0.0

# Apply the function to create a new 'Discount Value' column
data['Discount Percentage'] = data['Discounts and Offers'].apply(lambda x: extract_discount(x))

# For percentage discounts, calculate the discount amount based on the order value
data['Discount Amount'] = data.apply(lambda x: (x['Order Value'] * x['Discount Percentage'] / 100)
                                                    if x['Discount Percentage'] > 1

                                                   else x['Discount Percentage'], axis=1)

# Adjust 'Discount Amount' for fixed discounts directly specified in the 'Discounts and Offers' column
data['Discount Amount'] = data.apply(lambda x: x['Discount Amount'] if x['Discount Percentage'] <= 1

                                                   else x['Order Value'] * x['Discount Percentage'] / 100, axis=1)

# Calculate total costs and revenue per order
data['Total Costs'] = data['Delivery Fee'] + data['Payment Processing Fee'] + data['Discount Amount']
data['Revenue'] = data['Commission Fee']
data['Profit'] = data['Revenue'] - data['Total Costs']

# Aggregate data to get overall metrics
total_orders = data.shape[0]
total_revenue = data['Revenue'].sum()
total_costs = data['Total Costs'].sum()
total_profit = data['Profit'].sum()

overall_metrics = {
    "Total Orders": total_orders,
    "Total Revenue": total_revenue,
    "Total Costs": total_costs,
    "Total Profit": total_profit
}

# Create a Streamlit dashboard
st.title('Food Delivery Cost and Profitability Analysis')

# Display overall metrics
st.subheader('Overall Metrics')
st.write(overall_metrics)

# Histogram of profits per order
st.subheader('Profit Distribution per Order')
st.pyplot(plt.hist(data['Profit'], bins=50, color='skyblue', edgecolor='black'))

# Pie chart for the proportion of total costs
costs_breakdown = data[['Delivery Fee', 'Payment Processing Fee', 'Discount Amount']].sum()
st.subheader('Proportion of Total Costs')
st.pyplot(plt.pie(costs_breakdown, labels=costs_breakdown.index, autopct='%1.1f%%', startangle=140, colors=['tomato', 'gold', 'lightblue']))

# Bar chart for total revenue, costs, and profit
totals = ['Total Revenue', 'Total Costs', 'Total Profit']
values = [total_revenue, total_costs, total_profit]
st.subheader('Total Revenue, Costs, and Profit')
st.pyplot(plt.bar(totals, values, color=['green', 'red', 'blue']))

# Filter the dataset for profitable orders
profitable_orders = data[data['Profit'] > 0]

# Calculate the average commission percentage for profitable orders
profitable_orders['Commission Percentage'] = (profitable_orders['Commission Fee'] / profitable_orders['Order Value']) * 100

# Calculate the average discount percentage for profitable orders
profitable_orders['Effective Discount Percentage'] = (profitable_orders['Discount Amount'] / profitable_orders['Order Value']) * 100

# Calculate the new averages
new_avg_commission_percentage = profitable_orders['Commission Percentage'].mean()
new_avg_discount_percentage = profitable_orders['Effective Discount Percentage'].mean()

st.subheader('Average Commission and Discount Percentages for Profitable Orders')
st.write(new_avg_commission_percentage, new_avg_discount_percentage)

# Simulate profitability with recommended discounts and commissions
recommended_commission_percentage = 30.0  # 30%
recommended_discount_percentage = 6.0    # 6%

# Calculate the simulated commission fee and discount amount using recommended percentages
data['Simulated Commission Fee'] = data['Order Value'] * (recommended_commission_percentage / 100)
data['Simulated Discount Amount'] = data['Order Value'] * (recommended_discount_percentage / 100)

# Recalculate total costs and profit with simulated values
data['Simulated Total Costs'] = (data['Delivery Fee'] + data['Payment Processing Fee'] + data['Simulated Discount Amount'])

# data['Simulated Profit'] = (data['Simulated Commission Fee'] - data['Simulated Total Costs'])

# # Visualizing the comparison
# st.subheader('Comparison of Profitability: Actual vs. Recommended Discounts and Commissions')
# st.pyplot(sns.kdeplot(data['Profit'], label='Actual Profitability', fill=True, alpha=0.5, linewidth=2))
# st.pyplot(sns.kdeplot(data['Simulated Profit'], label='Estimated Profitability with Recommended Rates', fill=True, alpha=0.5, linewidth=2))

# # Step 3: Run the Streamlit app
# !streamlit run app.py
# The above code creates a Streamlit app that displays the overall metrics, profit distribution, costs breakdown, and profitability comparison. You can run the app by executing the command !streamlit run app.py in a code cell.
