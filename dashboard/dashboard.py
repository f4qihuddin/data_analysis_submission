import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from babel.numbers import format_currency
import numpy as np
sns.set_theme(style='dark')
from pathlib import Path

def create_orders_reviews_relation_df(df):
    df['order_delivered_customer_date'] = pd.to_datetime(
        df['order_delivered_customer_date']
    )

    df['order_purchase_timestamp'] = pd.to_datetime(
        df['order_purchase_timestamp']
    )

    df['delivery_time (days)'] = (df['order_delivered_customer_date'] - df['order_purchase_timestamp']).dt.days

    orders_reviews_relation = df.resample(rule='ME', on='order_purchase_timestamp').agg({
    "delivery_time (days)": 'mean',
    "review_score": 'mean'
    })

    orders_reviews_relation.index = orders_reviews_relation.index.strftime('%Y-%m')
    orders_reviews_relation = orders_reviews_relation.reset_index()
    orders_reviews_relation.rename(columns={
        "delivery_time (days)": "average_delivery_time (days)",
        "review_score": "average_review_score"
    }, inplace=True)
    orders_reviews_relation = orders_reviews_relation.sort_values(by='order_purchase_timestamp', ascending=False).head(12)

    return orders_reviews_relation

def create_products_reviews_top10_df(df):
    df['order_month'] = df['order_purchase_timestamp'].dt.to_period('M')

    monthly_category_counts = df.groupby(['order_month', 'product_category_name']).size().reset_index(name='total_order')

    idx = monthly_category_counts.groupby('order_month')['total_order'].idxmax()

    product_reviews_top10 = monthly_category_counts.loc[idx].copy()

    product_reviews_top10['order_purchase_timestamp'] = product_reviews_top10['order_month'].dt.strftime('%Y-%m')

    product_reviews_top10.drop(columns=['order_month'], inplace=True)

    product_reviews_top10.rename(columns={
        "product_category_name": "most_purchased_product_category",
    }, inplace=True)

    product_reviews_top10 = product_reviews_top10.sort_values(by='order_purchase_timestamp', ascending=False).head(12)

    total_top10_product_order = product_reviews_top10.groupby('most_purchased_product_category').total_order.sum().reset_index().sort_values(by='total_order', ascending=False)
    return total_top10_product_order

def create_products_reviews_bottom10_df(df):
    df['order_month'] = df['order_purchase_timestamp'].dt.to_period('M')

    monthly_category_counts_bottom = df.groupby(['order_month', 'product_category_name']).size().reset_index(name='total_order')

    idx_least = monthly_category_counts_bottom.groupby('order_month')['total_order'].idxmin()

    product_reviews_bottom10 = monthly_category_counts_bottom.loc[idx_least].copy()

    product_reviews_bottom10['order_purchase_timestamp'] = product_reviews_bottom10['order_month'].dt.strftime('%Y-%m')

    product_reviews_bottom10.drop(columns=['order_month'], inplace=True)

    product_reviews_bottom10.rename(columns={
        "product_category_name": "less_purchased_product_category",
    }, inplace=True)

    product_reviews_bottom10 = product_reviews_bottom10.sort_values(by='order_purchase_timestamp', ascending=False).head(12)

    total_bottom10_product_order = product_reviews_bottom10.groupby('less_purchased_product_category').total_order.sum().reset_index().sort_values(by='total_order', ascending=True)
    return total_bottom10_product_order

def create_last_year_order_df(df):
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
    # Temukan tanggal terbaru dalam DataFrame
    latest_date = df['order_purchase_timestamp'].max()

    # Hitung tanggal satu tahun sebelum tanggal terbaru
    one_year_ago = latest_date - pd.DateOffset(years=1)

    # Filter DataFrame untuk data 1 tahun terakhir
    customer_order_last_year_df = df[df['order_purchase_timestamp'] >= one_year_ago]

    top10_customer_states = customer_order_last_year_df.groupby(by='customer_state').customer_unique_id.nunique().sort_values(ascending=False).reset_index().head(10)
    top10_customer_states.rename(columns={'customer_unique_id': 'customer count'}, inplace=True)

    return top10_customer_states

def create_customer_cities_df(df):
    # Temukan tanggal terbaru dalam DataFrame
    latest_date = df['order_purchase_timestamp'].max()

    # Hitung tanggal satu tahun sebelum tanggal terbaru
    one_year_ago = latest_date - pd.DateOffset(years=1)

    # Filter DataFrame untuk data 1 tahun terakhir
    customer_order_last_year_df = df[df['order_purchase_timestamp'] >= one_year_ago]

    top10_customer_cities = customer_order_last_year_df.groupby(by='customer_city').customer_unique_id.nunique().sort_values(ascending=False).reset_index().head(10)
    top10_customer_cities.rename(columns={'customer_unique_id': 'customer count'}, inplace=True)

    return top10_customer_cities

def create_rfm_df(df):
    # Temukan tanggal terbaru dalam DataFrame
    latest_date = df['order_purchase_timestamp'].max()

    # Hitung tanggal satu tahun sebelum tanggal terbaru
    one_year_ago = latest_date - pd.DateOffset(years=1)

    # Filter DataFrame untuk data 1 tahun terakhir
    customer_order_payment_last_year_df = df[df['order_purchase_timestamp'] >= one_year_ago]

    rfm_df = customer_order_payment_last_year_df.groupby(by="customer_unique_id", as_index=False).agg({
        "order_purchase_timestamp": "max", # mengambil tanggal order terakhir
        "order_id": "nunique", # menghitung jumlah order
        "payment_value": "sum" # menghitung jumlah revenue yang dihasilkan
    })
    rfm_df.columns = ["customer_unique_id", "max_order_timestamp", "frequency", "monetary"]

    # Calculate the most recent purchase timestamp in the entire dataset (as a full datetime object)
    recent_overall_timestamp = df["order_purchase_timestamp"].dropna().max()

    # Calculate recency in hours
    # x here is a pandas Timestamp object (from rfm_df["max_order_timestamp"])
    rfm_df["recency"] = rfm_df["max_order_timestamp"].apply(
        lambda x: (recent_overall_timestamp - x).total_seconds() / 3600 if pd.notna(x) else np.nan
    )

    rfm_df.drop("max_order_timestamp", axis=1, inplace=True)

    return rfm_df

BASE_DIR = Path(__file__).resolve().parent

DATA_DIR = BASE_DIR.parent / "data"
LOGO_DIR = BASE_DIR.parent / "logo"

customer_order_payment_df = pd.read_csv(DATA_DIR / "customer_order_payment_df.csv")
orders_customers_df = pd.read_csv(DATA_DIR / "orders_customers_df.csv")
orders_payments_merged_df = pd.read_csv(DATA_DIR / "orders_payments_merged_df.csv")
orders_reviews_merged_df = pd.read_csv(DATA_DIR / "orders_reviews_merged_df.csv")
products_reviews_df = pd.read_csv(DATA_DIR / "products_reviews_df.csv")

# SIDEBAR DATE INPUT
# customer_order_payment_df filter
customer_order_payment_df["order_purchase_timestamp"] = pd.to_datetime(
    customer_order_payment_df["order_purchase_timestamp"]
)

min_date = customer_order_payment_df["order_purchase_timestamp"].min()
max_date = customer_order_payment_df["order_purchase_timestamp"].max()

# orders_customers_df filter
orders_customers_df["order_purchase_timestamp"] = pd.to_datetime(
    orders_customers_df["order_purchase_timestamp"]
)

min_date = orders_customers_df["order_purchase_timestamp"].min()
max_date = orders_customers_df["order_purchase_timestamp"].max()

# orders_payments_merged_df filter
orders_payments_merged_df["order_purchase_timestamp"] = pd.to_datetime(
    orders_payments_merged_df["order_purchase_timestamp"]
)

min_date = orders_payments_merged_df["order_purchase_timestamp"].min()
max_date = orders_payments_merged_df["order_purchase_timestamp"].max()

# orders_reviews_merged_df filter
orders_reviews_merged_df["order_purchase_timestamp"] = pd.to_datetime(
    orders_reviews_merged_df["order_purchase_timestamp"]
)

min_date = orders_reviews_merged_df["order_purchase_timestamp"].min()
max_date = orders_reviews_merged_df["order_purchase_timestamp"].max()

# products_reviews_df filter
products_reviews_df["order_purchase_timestamp"] = pd.to_datetime(
    products_reviews_df["order_purchase_timestamp"]
)

min_date = products_reviews_df["order_purchase_timestamp"].min()
max_date = products_reviews_df["order_purchase_timestamp"].max()

with st.sidebar:
    st.image(LOGO_DIR / "ecommerce_6220945.png")

    # Mengambil start_date & end_date dari date_input
    start_date, end_date = st.date_input(
        label='Rentang Waktu',min_value=min_date,
        max_value=max_date,
        value=[min_date, max_date]
    )

customer_order_payment_df_filtered = customer_order_payment_df[(customer_order_payment_df["order_purchase_timestamp"] >= pd.Timestamp(start_date)) & 
                (customer_order_payment_df["order_purchase_timestamp"] <= pd.Timestamp(end_date))]

orders_customers_df_filtered = orders_customers_df[(orders_customers_df["order_purchase_timestamp"] >= pd.Timestamp(start_date)) & 
                (orders_customers_df["order_purchase_timestamp"] <= pd.Timestamp(end_date))]

orders_payments_merged_df_filtered = orders_payments_merged_df[(orders_payments_merged_df["order_purchase_timestamp"] >= pd.Timestamp(start_date)) & 
                (orders_payments_merged_df["order_purchase_timestamp"] <= pd.Timestamp(end_date))]

orders_reviews_merged_df_filtered = orders_reviews_merged_df[(orders_reviews_merged_df["order_purchase_timestamp"] >= pd.Timestamp(start_date)) & 
                (orders_reviews_merged_df["order_purchase_timestamp"] <= pd.Timestamp(end_date))]

products_reviews_df_filtered = products_reviews_df[(products_reviews_df["order_purchase_timestamp"] >= pd.Timestamp(start_date)) & 
                (products_reviews_df["order_purchase_timestamp"] <= pd.Timestamp(end_date))]

orders_reviews_relation = create_orders_reviews_relation_df(orders_reviews_merged_df_filtered)
product_reviews_top10 = create_products_reviews_top10_df(products_reviews_df_filtered)
products_reviews_bottom10 = create_products_reviews_bottom10_df(products_reviews_df_filtered)
last_year_order = create_last_year_order_df(orders_payments_merged_df_filtered)
top10_customer_states = create_customer_states_df(orders_customers_df_filtered)
top10_customer_cities = create_customer_cities_df(orders_customers_df_filtered)
rfm_df = create_rfm_df(customer_order_payment_df_filtered)

# MAIN VIEW

st.header('Olist Data Analysis Dashboard')

st.subheader('Delivery Time vs Review Score')

col1, col2 = st.columns(2)

with col1:
    highest_average_review_score = orders_reviews_relation['average_review_score'].max()
    st.metric("Highest Average Review Score", value=round(highest_average_review_score, 2))

with col2:
    highest_average_delivery_time = orders_reviews_relation['average_delivery_time (days)'].max()
    st.metric("Highest Average Delivery Time (days)", value=round(highest_average_delivery_time, 2))

col3, col4 = st.columns(2)

with col3:
    lowest_average_review_score = orders_reviews_relation['average_review_score'].min()
    st.metric("Lowest Average Review Score", value=round(lowest_average_review_score, 2))

with col4:
    lowest_average_delivery_time = orders_reviews_relation['average_delivery_time (days)'].min()
    st.metric("Lowest Average Delivery Time (days)", value=round(lowest_average_delivery_time, 2))      

fig, ax = plt.subplots(nrows=2, ncols=1, figsize=(15, 5))
ax[0].plot(orders_reviews_relation['order_purchase_timestamp'], orders_reviews_relation['average_review_score'])
ax[0].set_xlabel('Month')
ax[0].set_ylabel('Average Review Score')
ax[0].set_title('Review Score Trend (1-5)')

plt.plot(orders_reviews_relation['order_purchase_timestamp'], orders_reviews_relation['average_delivery_time (days)'])
ax[1].set_xlabel('Month')
ax[1].set_ylabel('Average Delivery Time')
ax[1].set_title('Delivery Time Trend')

plt.tight_layout()
st.pyplot(fig)

st.subheader('Best and Worst Performing Product by Total Order')

col5, col6 = st.columns(2)

with col5:
    best_product = product_reviews_top10['most_purchased_product_category'][0]
    best_product_total_orders = product_reviews_top10['total_order'][0]
    st.metric("Best Product", value=best_product)
    st.metric("Total Orders", value=best_product_total_orders)

with col6:
    worst_product = products_reviews_bottom10['less_purchased_product_category'][0]
    worst_product_total_orders = products_reviews_bottom10['total_order'][0]
    st.metric("Worst Product", value=worst_product)
    st.metric("Total Orders", value=worst_product_total_orders)

fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(24, 6))

colors = ["#72BCD4", "#D3D3D3", "#D3D3D3", "#D3D3D3", "#D3D3D3"]

# Prepare data for the first plot (Best Performing Products)
sns.barplot(x="total_order", y="most_purchased_product_category", data=product_reviews_top10, palette=colors, ax=ax[0])
ax[0].set_ylabel(None)
ax[0].set_xlabel(None)
ax[0].set_title("Most Purchased Products", loc="center", fontsize=15)
ax[0].tick_params(axis ='y', labelsize=12)

# Prepare data for the second plot (Worst Performing Products)
sns.barplot(x="total_order", y="less_purchased_product_category", data=products_reviews_bottom10, palette=colors, ax=ax[1])
ax[1].set_ylabel(None)
ax[1].set_xlabel(None)
ax[1].invert_xaxis()
ax[1].yaxis.set_label_position("right")
ax[1].yaxis.tick_right()
ax[1].set_title("Less Purchased Products", loc="center", fontsize=15)
ax[1].tick_params(axis='y', labelsize=12)

plt.suptitle("Most and Less Purchased Product per Month", fontsize=20)
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
st.pyplot(fig)

st.subheader('Last Year Monthlty Revenue')

col7, col8 = st.columns(2)

with col7:
    total_orders = last_year_order['order_count'].sum()
    st.metric("Total Orders", value=total_orders)

with col8:
    total_revenue = last_year_order['revenue'].sum()
    formatted_revenue = format_currency(total_revenue, 'BRL', locale='pt_BR')
    st.metric("Total Revenue", value=formatted_revenue)

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

plt.tight_layout()
st.pyplot(fig)

st.subheader('Top 10 Customer States and Cities')

col9, col10 = st.columns(2)

with col9:
    best_state = top10_customer_states['customer_state'][0]
    st.metric("Best State", value=best_state)

with col10:

    best_city = top10_customer_cities['customer_city'][0]
    st.metric("Best City", value=best_city)

fig, ax = plt.subplots(figsize=(24, 6))

colors = ["#72BCD4", "#D3D3D3", "#D3D3D3", "#D3D3D3", "#D3D3D3"]

sns.barplot(x="customer count", y="customer_state", data=top10_customer_states, palette=colors)
ax.set_ylabel(None)
ax.set_xlabel(None)
ax.set_title("Top 10 Customer States", loc="center", fontsize=15)
ax.tick_params(axis ='y', labelsize=12)

plt.tight_layout()
st.pyplot(fig)

fig, ax = plt.subplots(figsize=(24, 6))

colors = ["#72BCD4", "#D3D3D3", "#D3D3D3", "#D3D3D3", "#D3D3D3"]

sns.barplot(x="customer count", y="customer_city", data=top10_customer_cities, palette=colors)
ax.set_ylabel(None)
ax.set_xlabel(None)
ax.set_title("Top 10 Customer Cities", loc="center", fontsize=15)
ax.tick_params(axis ='y', labelsize=12)

plt.tight_layout()
st.pyplot(fig)

st.subheader('RFM Analysis')

fig, ax = plt.subplots(nrows=3, ncols=1, figsize=(10, 20))

colors = ["#72BCD4",  "#72BCD4",  "#72BCD4",  "#72BCD4",  "#72BCD4"]

sns.barplot(y="customer_unique_id", x="recency", data=rfm_df.sort_values(by="recency", ascending=True).head(10), palette=colors, ax=ax[0])
ax[0].set_ylabel(None)
ax[0].set_xlabel(None)
ax[0].set_title("By Recency (hours)", loc="center", fontsize=18)
ax[0].tick_params(axis ='x', labelsize=15)

sns.barplot(y="customer_unique_id", x="frequency", data=rfm_df.sort_values(by="frequency", ascending=False).head(10), palette=colors, ax=ax[1])
ax[1].set_ylabel(None)
ax[1].set_xlabel(None)
ax[1].set_title("By Frequency", loc="center", fontsize=18)
ax[1].tick_params(axis='x', labelsize=15)

sns.barplot(y="customer_unique_id", x="monetary", data=rfm_df.sort_values(by="monetary", ascending=False).head(10), palette=colors, ax=ax[2])
ax[2].set_ylabel(None)
ax[2].set_xlabel(None)
ax[2].set_title("By Monetary", loc="center", fontsize=18)
ax[2].tick_params(axis='x', labelsize=15)

plt.suptitle("Best Customer Based on RFM Parameters", fontsize=20, y=0.95)
plt.tight_layout()
st.pyplot(fig)
