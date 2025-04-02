import streamlit as st
import os
import pandas as pd
from typing import List, Dict
from neo4j import GraphDatabase
from langchain_groq import ChatGroq
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Neo4jVector
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import PyPDFLoader
from langchain.chains import RetrievalQA
import networkx as nx
from sklearn.linear_model import LinearRegression
from pyvis.network import Network
import tempfile
import numpy as np
import plotly.graph_objects as go
from streamlit import plotly_chart
import base64
import re

# Streamlit page configuration
st.set_page_config(layout="wide", page_title="Knowledge Graph with Crop Price Prediction")

# Neo4j Configuration
NEO4J_URI = "neo4j+s://947f4e44.databases.neo4j.io"
NEO4J_USER = "neo4j"
NEO4J_PASSWORD = "R7T40DT9nt7bUkcTkNj0qybHD-zBW2BAWgVN7nt8F6k"

class KnowledgeGraphRAG:
    def __init__(self):
        self.driver = GraphDatabase.driver(
            NEO4J_URI,
            auth=(NEO4J_USER, NEO4J_PASSWORD)
        )
        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-mpnet-base-v2"
        )
    
    def delete_database(self):
        """Delete all nodes and relationships in the database"""
        with self.driver.session() as session:
            session.run("MATCH (n) DETACH DELETE n")
            try:
                session.run("CALL db.index.vector.drop('document_vectors')")
            except Exception as e:
                st.warning("Vector index might not exist or was already deleted.")
    
    def retrieve_existing_graph(self):
        """Retrieve existing knowledge graph relationships"""
        with self.driver.session() as session:
            result = session.run("""
            MATCH (e1:Entity)-[r:RELATES]->(e2:Entity)
            RETURN e1.name as source, r.type as relationship, e2.name as target
            """)
            return [dict(record) for record in result]
    
    def create_vector_store(self, documents: List):
        """Create vector store in Neo4j"""
        vector_store = Neo4jVector.from_documents(
            documents,
            self.embeddings,
            url=NEO4J_URI,
            username=NEO4J_USER,
            password=NEO4J_PASSWORD,
            index_name="document_vectors",
            node_label="Document",
            embedding_node_property="embedding",
            text_node_property="text"
        )
        return vector_store
    
    def query(self, question: str) -> Dict:
        """Retrieve relevant document passages and relationships"""
        vector_store = Neo4jVector(
            self.embeddings,
            url=NEO4J_URI,
            username=NEO4J_USER,
            password=NEO4J_PASSWORD,
            index_name="document_vectors"
        )
        retriever = vector_store.as_retriever()
        qa_chain = RetrievalQA.from_chain_type(
            llm=ChatGroq(api_key="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", model_name="llama3-70b-8192"),
            chain_type="stuff",
            retriever=retriever
        )
        rag_answer = qa_chain.run(question)
        kg_data = self.retrieve_existing_graph()
        return {
            "rag_answer": rag_answer,
            "knowledge_graph": kg_data
        }

def main():
    st.title("🌟 Advanced RAG System with Knowledge Graph Retrieval")
    rag_system = KnowledgeGraphRAG()
    
    question = st.text_input("Ask a question:", placeholder="Type your question here...")
    if question:
        with st.spinner("🤔 Analyzing..."):
            results = rag_system.query(question)
        st.markdown("### 📝 Answer")
        st.markdown(f"```
{results['rag_answer']}
```")
        st.markdown("### 🔗 Knowledge Graph Connections")
        for rel in results["knowledge_graph"]:
            st.markdown(f"🔹 {rel['source']} → *{rel['relationship']}* → {rel['target']}")

if __name__ == "__main__":
    main()
