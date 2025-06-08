import pandas as pd
from langchain_core.documents import Document

def dataconverter():
    # Load CSV file
    product_data = pd.read_csv("C:/Users/varun/OneDrive/Desktop/Product_Rec_Chatbot/data/Flipkart_Reviews.csv")

    # Select relevant columns
    data = product_data[["product_title", "review"]]

    product_list = []

    # Iterate over DataFrame rows and store each product/review pair
    for _, row in data.iterrows():
        object = {
            "product_name": row["product_title"],
            "review": row["review"]
        }
        product_list.append(object)  # ✅ Fix: Append inside loop

    docs = []

    # Convert product list to LangChain-compatible Document objects
    for entry in product_list:
        metadata = {"product_name": entry['product_name']}
        doc = Document(page_content=entry['review'], metadata=metadata)
        docs.append(doc)

    print(f"Total Documents Created: {len(docs)}")  # ✅ Debugging line
    return docs