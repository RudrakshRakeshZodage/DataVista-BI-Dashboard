import pandas as pd

def generate_kpis(orders_df: pd.DataFrame, products_df: pd.DataFrame) -> dict:
    # Merge both dataframes on product_id
    df = pd.merge(orders_df, products_df, on='product_id', how='inner')

    # Total Revenue
    df['revenue'] = df['quantity'] * df['price']
    total_revenue = df['revenue'].sum()

    # Total Orders
    total_orders = df['order_id'].nunique()

    # Total Products Sold
    total_quantity = df['quantity'].sum()

    # Unique Customers
    total_customers = df['customer_id'].nunique() if 'customer_id' in df.columns else 'N/A'

    return {
        "Total Revenue": f"₹ {total_revenue:,.2f}",
        "Total Orders": total_orders,
        "Products Sold": total_quantity,
        "Unique Customers": total_customers
    }