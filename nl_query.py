from langchain_community.llms import Ollama
from langchain_core.prompts import PromptTemplate
import pandas as pd

def run_local_llm_query(query, orders_df, products_df):
    try:
        # ✅ Merge actual data to compute real trends
        merged_df = pd.merge(orders_df, products_df, on='product_id', how='left')

        # ✅ Precompute category-wise total quantity ordered
        category_trends = (
            merged_df.groupby("category")["quantity"]
            .sum()
            .sort_values(ascending=False)
            .reset_index()
        )

        # ✅ Add product count per category
        category_counts = (
            products_df.groupby("category")["product_id"]
            .count()
            .reset_index()
            .rename(columns={"product_id": "product_count"})
        )

        # ✅ Final trend DataFrame with actual numbers
        trend_df = pd.merge(category_trends, category_counts, on="category")
        trend_df = trend_df.rename(columns={
            "category": "Category",
            "quantity": "Total Quantity Ordered",
            "product_count": "Product Count"
        })

        # ✅ Convert trend_df to markdown table
        markdown_table = trend_df.to_markdown(index=False)

        # ✅ Create LLM prompt with real summary and user query
        llm = Ollama(model="mistral")  # Or "llama2", "gemma", etc.
        prompt = PromptTemplate.from_template("""
You are a data analyst AI.

Below is the actual data summary of product categories with total quantity ordered and product count:

{markdown_table}

Now answer this user question based on the data above in **clear text** and, if possible, highlight top/bottom categories:

🔎 Question: {query}
""")

        final_prompt = prompt.format(markdown_table=markdown_table, query=query)
        return llm.invoke(final_prompt)

    except Exception as e:
        return f"❌ Error: {e}"
