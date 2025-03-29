import streamlit as st
import os
import tempfile
import networkx as nx
import plotly.graph_objects as go
from neo4j import GraphDatabase
from transformers import pipeline, set_seed
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import PyPDFLoader

# ✅ Set random seed for reproducibility
set_seed(42)

# 🔹 Load AgriBERT-based text generation model
generator = pipeline('text-generation', model='benkimz/agbrain')

# 🔹 Load BERT-based Named Entity Recognition (NER) model
ner_pipeline = pipeline("ner", model="dslim/bert-base-NER")

# ✅ Neo4j Configuration
NEO4J_URI = "neo4j+s://f859410b.databases.neo4j.io"
NEO4J_USER = "neo4j"
NEO4J_PASSWORD = "UDAQaAOeey_dYNj2krLzpTeTINjnPgRgcJ4XeLtotpE"

class KnowledgeGraph:
    def __init__(self):
        """ Initialize Neo4j database connection """
        self.driver = GraphDatabase.driver(
            NEO4J_URI,
            auth=(NEO4J_USER, NEO4J_PASSWORD)
        )

    def delete_database(self):
        """ Delete all nodes and relationships from Neo4j database """
        with self.driver.session() as session:
            session.run("MATCH (n) DETACH DELETE n")
            st.success("✅ Database cleared successfully!")

    def extract_relationships(self, text):
        """ Extract relationships using AgriBERT and Named Entity Recognition (NER) """
        entities = ner_pipeline(text)
        relationships = []

        # 🔹 Generate related agricultural text using AgriBERT
        generated_texts = generator(text, max_new_tokens=50, num_return_sequences=2)

        for gen_text in generated_texts:
            gen_entities = ner_pipeline(gen_text['generated_text'])

            # 🔹 Extract entity relationships from generated text
            for i in range(len(gen_entities) - 1):
                if gen_entities[i]["entity"].startswith("B-") and gen_entities[i+1]["entity"].startswith("B-"):
                    relationships.append({
                        "entity1": gen_entities[i]["word"],
                        "relationship": "RELATED_TO",
                        "entity2": gen_entities[i+1]["word"]
                    })

        return relationships

    def create_knowledge_graph(self, text):
        """ Create nodes and relationships in Neo4j """
        relationships = self.extract_relationships(text)
        with self.driver.session() as session:
            for rel in relationships:
                session.run("""
                    MERGE (e1:Entity {name: $entity1})
                    MERGE (e2:Entity {name: $entity2})
                    MERGE (e1)-[:RELATED_TO]->(e2)
                """, rel)

    def create_3d_graph(self):
        """ Generate a 3D visualization of the knowledge graph """
        G = nx.DiGraph()

        with self.driver.session() as session:
            result = session.run("""
                MATCH (e1:Entity)-[r:RELATED_TO]->(e2:Entity)
                RETURN e1.name AS source, r.type AS relationship, e2.name AS target
            """)
            records = list(result)

        if not records:
            return None

        for record in records:
            G.add_edge(record["source"], record["target"], label=record["relationship"])

        pos = nx.spring_layout(G, dim=3)
        edge_x, edge_y, edge_z = [], [], []
        for edge in G.edges():
            x0, y0, z0 = pos[edge[0]]
            x1, y1, z1 = pos[edge[1]]
            edge_x.extend([x0, x1, None])
            edge_y.extend([y0, y1, None])
            edge_z.extend([z0, z1, None])

        edges_trace = go.Scatter3d(
            x=edge_x, y=edge_y, z=edge_z,
            line=dict(width=1, color='#888'),
            hoverinfo='none',
            mode='lines'
        )

        node_x, node_y, node_z = [], [], []
        for node in G.nodes():
            x, y, z = pos[node]
            node_x.append(x)
            node_y.append(y)
            node_z.append(z)

        nodes_trace = go.Scatter3d(
            x=node_x, y=node_y, z=node_z,
            mode='markers',
            hovertext=list(G.nodes()),
            hoverinfo='text',
            marker=dict(size=8, color='#00ff00', line_width=2)
        )

        fig = go.Figure(data=[edges_trace, nodes_trace])
        return fig

# ✅ Streamlit UI
st.title("🌿 Agricultural Knowledge Graph using AgriBERT")

# ✅ Initialize Knowledge Graph
kg = KnowledgeGraph()

# 🔹 Sidebar Controls
with st.sidebar:
    st.header("⚙️ Controls")
    if st.button("🗑️ Delete Database"):
        kg.delete_database()

# 🔹 File Upload
uploaded_file = st.file_uploader("📄 Upload a PDF document", type=["pdf"])
if uploaded_file:
    with st.spinner("⏳ Processing document..."):
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp:
            tmp.write(uploaded_file.getvalue())
            loader = PyPDFLoader(tmp.name)
            documents = loader.load()

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        splits = text_splitter.split_documents(documents)

        for doc in splits:
            kg.create_knowledge_graph(doc.page_content)

        st.success("✅ Document processed successfully!")

# 🔹 Graph Visualization
st.markdown("### 🌐 Knowledge Graph Visualization")
fig = kg.create_3d_graph()
if fig:
    st.plotly_chart(fig, use_container_width=True)
else:
    st.info("ℹ️ No relationships to visualize yet. Upload a document to create the knowledge graph.")
