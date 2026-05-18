import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from babel.numbers import format_currency
sns.set_theme(style='dark')

def create_orders_reviews_merged_df(df):
    df['order_delivered_customer_date'] = pd.to_datetime(
        df['order_delivered_customer_date']
    )

    df['order_purchase_timestamp'] = pd.to_datetime(
        df['order_purchase_timestamp']
    )

    df['delivery_time (days)'] = (df['order_delivered_customer_date'] - df['order_purchase_timestamp']).dt.days

    orders_reviews_relation = df.groupby(by="review_score").agg({
       "delivery_time (days)": 'mean'
    }).reset_index().round(2)

    return orders_reviews_relation

def create_products_reviews_top10_df(df):
    product_reviews_top10 = df.groupby(by="product_category_name").review_score.mean().round(2).sort_values(ascending=False).reset_index().head(10)

    return product_reviews_top10

def create_products_reviews_bottom10_df(df):
    products_reviews_bottom10 = df.groupby(by="product_category_name").review_score.mean().round(2).sort_values(ascending=True).reset_index().head(10)

    return products_reviews_bottom10

def create_orders_payments_merged_df(df):
    df['order_purchase_timestamp'] = pd.to_datetime(
        df['order_purchase_timestamp']
    )

    monthly_orders_df = df.resample(rule='ME', on='order_purchase_timestamp').agg({
    "order_id": "nunique",
    "payment_value": "sum"
    })
    monthly_orders_df.index = monthly_orders_df.index.strftime('%Y-%m')
    monthly_orders_df = monthly_orders_df.reset_index()
    monthly_orders_df.rename(columns={
        "order_id": "order_count",
        "payment_value": "revenue"
    }, inplace=True)
    last_year_order = monthly_orders_df.sort_values(by='order_purchase_timestamp', ascending=False).head(12)

    return last_year_order

def create_customer_states_df(df):
    top10_customer_states = df.groupby(by='customer_state').customer_unique_id.nunique().sort_values(ascending=False).reset_index().head(10)
    top10_customer_states.rename(columns={'customer_unique_id': 'customer count'}, inplace=True)

    return top10_customer_states

def create_customer_cities_df(df):
    top10_customer_cities = df.groupby(by='customer_city').customer_unique_id.nunique().sort_values(ascending=False).reset_index().head(10)
    top10_customer_cities.rename(columns={'customer_unique_id': 'customer count'}, inplace=True)

    return top10_customer_cities

orders_payments_merged_df = pd.read_csv('../data/orders_payments_merged_df.csv')
orders_reviews_merged_df = pd.read_csv('../data/orders_reviews_merged_df.csv')
products_reviews_df = pd.read_csv('../data/products_reviews_df.csv')
customer_df = pd.read_csv('../data/olist_customers_dataset.csv')

orders_reviews_relation =create_orders_reviews_merged_df(orders_reviews_merged_df)
product_reviews_top10 = create_products_reviews_top10_df(products_reviews_df)
products_reviews_bottom10 = create_products_reviews_bottom10_df(products_reviews_df)
last_year_order = create_orders_payments_merged_df(orders_payments_merged_df)
top10_customer_states = create_customer_states_df(customer_df)
top10_customer_cities = create_customer_cities_df(customer_df)

st.header('Olist Data Analysis Dashboard')

st.subheader('Delivery Time vs Review Score')

fig, ax = plt.subplots(figsize=(10, 5))
ax.scatter(x=orders_reviews_relation['delivery_time (days)'], y=orders_reviews_relation['review_score'])
ax.set_xlabel('Average Delivery Time (days)')
ax.set_ylabel('Review Score')
ax.set_title('Relationship between Delivery Time and Review Score')
st.pyplot(fig)

st.subheader('Best and Worst Performing Product by Review Score')

fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(30, 15))

colors = ["#72BCD4", "#D3D3D3", "#D3D3D3", "#D3D3D3", "#D3D3D3"]

sns.barplot(x="review_score", y="product_category_name", data=product_reviews_top10.head(10), palette=colors, ax=ax[0])
ax[0].set_ylabel(None)
ax[0].set_xlabel(None)
ax[0].set_title("Best Performing Product", loc="center", fontsize=35)
ax[0].tick_params(axis ='y', labelsize=25)

sns.barplot(x="review_score", y="product_category_name", data=products_reviews_bottom10.head(10), palette=colors, ax=ax[1])
ax[1].set_ylabel(None)
ax[1].set_xlabel(None)
ax[1].invert_xaxis()
ax[1].yaxis.set_label_position("right")
ax[1].yaxis.tick_right()
ax[1].set_title("Worst Performing Product", loc="center", fontsize=35)
ax[1].tick_params(axis='y', labelsize=25)

st.pyplot(fig)

st.subheader('Last Year Monthlty Revenue')

fig, ax = plt.subplots(figsize=(20, 5))
ax.plot(
    last_year_order.sort_values(ascending=True, by='order_purchase_timestamp')["order_purchase_timestamp"],
    last_year_order.sort_values(ascending=True, by='order_purchase_timestamp')["revenue"],
    marker='o',
    linewidth=2,
    color="#72BCD4"
)
ax.set_title("Last Year Revenue", loc="center", fontsize=20)
ax.tick_params(axis='x', labelsize=15)
ax.tick_params(axis='y', labelsize=15)

st.pyplot(fig)

st.subheader('Top 10 Customer States')

fig, ax = plt.subplots(figsize=(24, 6))

colors = ["#72BCD4", "#D3D3D3", "#D3D3D3", "#D3D3D3", "#D3D3D3"]

sns.barplot(x="customer count", y="customer_state", data=top10_customer_states, palette=colors)
ax.set_ylabel(None)
ax.set_xlabel(None)
ax.set_title("Top 10 Customer States", loc="center", fontsize=15)
ax.tick_params(axis ='y', labelsize=12)
st.pyplot(fig)

st.subheader('Top 10 Customer Cities')

fig, ax = plt.subplots(figsize=(24, 6))

colors = ["#72BCD4", "#D3D3D3", "#D3D3D3", "#D3D3D3", "#D3D3D3"]

sns.barplot(x="customer count", y="customer_city", data=top10_customer_cities, palette=colors)
ax.set_ylabel(None)
ax.set_xlabel(None)
ax.set_title("Top 10 Customer Cities", loc="center", fontsize=15)
ax.tick_params(axis ='y', labelsize=12)
st.pyplot(fig)

