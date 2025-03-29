import streamlit as st
import os
import tempfile
import networkx as nx
import plotly.graph_objects as go
from neo4j import GraphDatabase
from transformers import pipeline, AutoTokenizer, Trainer, TrainingArguments
from datasets import Dataset
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import PyPDFLoader
import time
import spacy  # For improved relationship extraction
from spacy.lang.en.stop_words import STOP_WORDS
import torch

# ✅ Load spaCy NLP Model
nlp = spacy.load("en_core_web_sm")

# ✅ Load Agriculture-BERT Model
MODEL_NAME = "recobo/agriculture-bert-uncased"
fill_mask = pipeline("fill-mask", model=MODEL_NAME, tokenizer=MODEL_NAME)

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

# ✅ Fine-tuning AgriBERT
class AgriDataset(torch.utils.data.Dataset):
    def __init__(self, texts, labels):
        self.encodings = tokenizer(texts, truncation=True, padding=True, max_length=512)
        self.labels = labels
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx])
        return item

# ✅ Neo4j Configuration
NEO4J_URI = "neo4j+s://f859410b.databases.neo4j.io"
NEO4J_USER = "neo4j"
NEO4J_PASSWORD = "UDAQaAOeey_dYNj2krLzpTeTINjnPgRgcJ4XeLtotpE"

class KnowledgeGraph:
    def __init__(self):
        self.driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))

    def delete_database(self):
        with self.driver.session() as session:
            session.run("MATCH (n) DETACH DELETE n")
            st.success("✅ Database cleared successfully!")

    def extract_entities_and_relationships(self, text):
        """ Extract refined entities and relationships using dependency parsing + Agriculture-BERT """
        doc = nlp(text)
        relationships = []
        entities = set()

        for ent in doc.ents:
            if ent.label_ in ["ORG", "GPE", "PRODUCT", "EVENT", "LOC", "NORP", "FAC", "WORK_OF_ART", "LAW"]:
                entities.add(ent.text)

        for token in doc:
            if token.text.lower() not in STOP_WORDS and token.pos_ in ["VERB", "NOUN", "ADJ"]:
                subject = [child.text for child in token.lefts if child.dep_ in ["nsubj", "nsubjpass"] and child.text.lower() not in STOP_WORDS]
                obj = [child.text for child in token.rights if child.dep_ in ["dobj", "pobj", "attr"] and child.text.lower() not in STOP_WORDS]

                if subject and obj:
                    masked_sentence = f"{subject[0]} {token.lemma_} [MASK] {obj[0]}"
                    try:
                        predictions = fill_mask(masked_sentence)
                        for pred in predictions[:10]:  # Take top 10 refined relationships
                            if "token_str" in pred and pred["score"] > 0.6:  # Lower confidence threshold
                                relationships.append({
                                    "entity1": subject[0],
                                    "relationship": pred["token_str"],
                                    "entity2": obj[0],
                                    "confidence": pred["score"]
                                })
                    except Exception as e:
                        st.error(f"🚨 Error in fill-mask pipeline: {e}")

        return entities, relationships

    def create_knowledge_graph(self, text):
        """ Create nodes and relationships in Neo4j """
        entities, relationships = self.extract_entities_and_relationships(text)
        with self.driver.session() as session:
            for entity in entities:
                session.run("MERGE (e:Entity {name: $name})", {"name": entity})
            for rel in relationships:
                session.run("""
                    MERGE (e1:Entity {name: $entity1})
                    MERGE (e2:Entity {name: $entity2})
                    MERGE (e1)-[r:RELATIONSHIP {type: $relationship, confidence: $confidence}]->(e2)
                """, rel)

    def create_3d_graph(self):
        """ Generate a 3D visualization of the knowledge graph """
        G = nx.DiGraph()
        with self.driver.session() as session:
            result = session.run("""
                MATCH (e1:Entity)-[r]->(e2:Entity)
                RETURN e1.name AS source, r.type AS relationship, e2.name AS target, r.confidence AS confidence
            """)
            records = list(result)

        if not records:
            return None

        for record in records:
            G.add_edge(record["source"], record["target"],
                       label=f"{record['relationship']} ({record['confidence']:.2f})")

        pos = nx.spring_layout(G, dim=3)
        edge_x, edge_y, edge_z = [], [], []
        edge_text = []

        for edge in G.edges():
            x0, y0, z0 = pos[edge[0]]
            x1, y1, z1 = pos[edge[1]]
            edge_x.extend([x0, x1, None])
            edge_y.extend([y0, y1, None])
            edge_z.extend([z0, z1, None])
            edge_text.append(G.edges[edge]["label"])  # Relationship label

        edges_trace = go.Scatter3d(
            x=edge_x, y=edge_y, z=edge_z,
            line=dict(width=2, color='#888'),
            hoverinfo='text',
            text=edge_text,
            mode='lines+text',
            textposition='middle center'
        )

        node_x, node_y, node_z = [], [], []
        node_text = list(G.nodes())

        for node in G.nodes():
            x, y, z = pos[node]
            node_x.append(x)
            node_y.append(y)
            node_z.append(z)

        nodes_trace = go.Scatter3d(
            x=node_x, y=node_y, z=node_z,
            mode='markers+text',
            hovertext=node_text,
            hoverinfo='text',
            marker=dict(size=10, color='#00ff00', line_width=2),
            text=node_text,
            textposition='top center'
        )

        fig = go.Figure(data=[edges_trace, nodes_trace])
        fig.update_layout(width=1300, height=1200)
        return fig

# ✅ Streamlit UI
st.title("🌿 Agricultural Knowledge Graph using Fine-Tuned Agriculture-BERT")

kg = KnowledgeGraph()

# 🔹 Sidebar Controls
with st.sidebar:
    st.header("⚙️ Controls")
    if st.button("🗑️ Delete Database"):
        kg.delete_database()

uploaded_files = st.file_uploader("📄 Upload PDF documents", type=["pdf"], accept_multiple_files=True)
if uploaded_files:
    with st.spinner("⏳ Processing documents..."):
        for uploaded_file in uploaded_files:
            with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp:
                tmp.write(uploaded_file.getvalue())
                loader = PyPDFLoader(tmp.name)
                documents = loader.load()
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=400, separators=["\n\n", ".", "?"])
            splits = text_splitter.split_documents(documents)
            for doc in splits:
                kg.create_knowledge_graph(doc.page_content)
        st.success("✅ Documents processed successfully!")

st.markdown("### 🌐 Knowledge Graph Visualization")
fig = kg.create_3d_graph()
if fig:
    st.plotly_chart(fig, use_container_width=True)
else:
    st.info("ℹ️ No relationships to visualize yet. Upload documents to create the knowledge graph.")
