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
st.set_page_config(layout="wide", page_title="Knowledge Graph with RAG and Retrieval")

# Neo4j Configuration
NEO4J_URI = "neo4j+s://947f4e44.databases.neo4j.io"
NEO4J_USER = "neo4j"
NEO4J_PASSWORD = "R7T40DT9nt7bUkcTkNj0qybHD-zBW2BAWgVN7nt8F6k"

# Initialize Groq
GROQ_API_KEY = "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"
llm = ChatGroq(api_key=GROQ_API_KEY, model_name="llama3-70b-8192")

class KnowledgeGraphRAG:
    def __init__(self):
        self.driver = GraphDatabase.driver(
            NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD)
        )
        self.embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")

    def create_vector_store(self, documents: List):
        print("📌 Creating vector store...")  # Debug print
        try:
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
            print("✅ Vector store created successfully!")  # Debug print
        except Exception as e:
            print(f"❌ Error creating vector store: {e}")


    def create_knowledge_graph(self, documents: List):
        with self.driver.session() as session:
            for doc in documents:
                prompt = f"""
                Extract key entities and their relationships from this text.
                Format: (entity1)-[relationship]->(entity2)
                Text: {doc.page_content}
                """
                response = llm.predict(prompt)
                relationships = self._parse_relationships(response)
                for rel in relationships:
                    session.run("""
                    MERGE (e1:Entity {name: $entity1})
                    MERGE (e2:Entity {name: $entity2})
                    MERGE (e1)-[:RELATES {type: $relationship}]->(e2)
                    """, rel)
    
    def retrieve_graph_data(self):
        with self.driver.session() as session:
            result = session.run("""
            MATCH (e1:Entity)-[r:RELATES]->(e2:Entity)
            RETURN e1.name as source, r.type as relationship, e2.name as target
            """
            )
            return [dict(record) for record in result]

    def query(self, question: str) -> Dict:

        vector_store = Neo4jVector(
            self.embeddings,
            url=NEO4J_URI,
            username=NEO4J_USER,
            password=NEO4J_PASSWORD,
            index_name="document_vectors"
        )
        retriever = vector_store.as_retriever()
        qa_chain = RetrievalQA.from_chain_type(
            llm=llm, chain_type="stuff", retriever=retriever
        )
        rag_answer = qa_chain.run(question)

        kg_data = self.retrieve_graph_data()
        return {"rag_answer": rag_answer, "knowledge_graph": kg_data}

    def visualize_graph(kg_data):

        net = Network(height="600px", width="100%", directed=True)

        # Add nodes and edges
        for rel in kg_data:
            net.add_node(rel["source"], label=rel["source"], color="#1E88E5")
            net.add_node(rel["target"], label=rel["target"], color="#D81B60")
            net.add_edge(rel["source"], rel["target"], label=rel["relationship"])

        # Save and display the graph
        path = "graph.html"
        net.save_graph(path)

        with open(path, "r", encoding="utf-8") as f:
            html_code = f.read()
        
        st.components.v1.html(html_code, height=600)


def main():
    st.title("🌟 Advanced RAG with Knowledge Graph Retrieval")
    rag_system = KnowledgeGraphRAG()
    
    # Sidebar
    with st.sidebar:
        st.header("Controls")
        uploaded_file = st.file_uploader("Upload PDF", type=["pdf"])
        if uploaded_file:
            with st.spinner("Processing document..."):
                with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp:
                    tmp.write(uploaded_file.getvalue())
                    loader = PyPDFLoader(tmp.name)
                    documents = loader.load()
                
                text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
                splits = text_splitter.split_documents(documents)
                
                rag_system.create_vector_store(splits)
                rag_system.create_knowledge_graph(splits)
            st.success("✅ Document processed!")
    
    # Query Interface
    st.markdown("### 🔍 Ask a Question")
    question = st.text_input("Your Question:", placeholder="Type here...")
    
    if question:
        with st.spinner("Retrieving answers..."):
            results = rag_system.query(question)
        
        st.markdown("### 📝 Answer")
        st.markdown(f"""
```
{results['rag_answer']}
```""")
        
        st.markdown("### 🔗 Related Knowledge Graph Data")
        
        if results["knowledge_graph"]:
            visualize_graph(results["knowledge_graph"])
        else:
            st.warning("No relationships found in the knowledge graph.")

if __name__ == "__main__":
    main()
