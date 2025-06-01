import streamlit as st
import os
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser

# Load environment variables
load_dotenv()
os.environ['GROQ_API_KEY'] = os.getenv("GROQ_API_KEY")
os.environ["LANGCHAIN_TRACING_V2"] = "true"

# Define Pydantic model for product
class Product(BaseModel):
    product_name: str = Field(description="Product name")
    product_details: str = Field(description="Product details")
    tentative_price: float = Field(description="Tentative price of product in USD")

# Initialize JSON parser and LLM
json_output_parser = JsonOutputParser(pydantic_object=Product)
llm = ChatGroq(model="llama-3.3-70b-versatile")

# Create prompt template
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a chatbot which gives details of product in JSON format. Strictly follow provided output format instructions. \n{format_instructions}"),
    ("user", "{input}")
])

# Create LangChain pipeline
chain = prompt | llm | json_output_parser

# Streamlit app layout
st.set_page_config(page_title="Product Details Finder", page_icon="🔍", layout="centered")
st.title("Product Details Finder")
st.markdown("Enter a product name to view its details in a formatted layout.")

# Input field for product name
product_input = st.text_input("Product Name", placeholder="e.g., Samsung Galaxy F41", key="product_input")

# Button to fetch details
if st.button("Get Product Details", key="fetch_button"):
    if product_input:
        with st.spinner("Fetching product details..."):
            try:
                # Invoke the chain with user input
                response = chain.invoke({
                    "input": f"Give me details of {product_input}",
                    "format_instructions": json_output_parser.get_format_instructions()
                })
                # Display formatted output in a styled container
                st.markdown("### Product Information")
                with st.container():
                    st.markdown(f"**Product Name:** {response['product_name']}")
                    st.markdown(f"**Details:** {response['product_details']}")
                    st.markdown(f"**Tentative Price:** ${response['tentative_price']:.2f} USD")
            except Exception as e:
                st.error(f"An error occurred: {str(e)}")
    else:
        st.warning("Please enter a product name.")


# Footer
st.markdown("---")
st.markdown("Built with Streamlit for the Agentic AI Course by Krish Naik Academy")