def recommend_similar_products(selected_product, df):
    product_info = df[df['name'] == selected_product].iloc[0]
    category = product_info['category']
    price = product_info['price']
    similar = df[(df['category'] == category) & (df['price'].between(price*0.8, price*1.2))]
    return similar['name'].unique().tolist()
