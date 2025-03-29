import streamlit as st
import tempfile
from neo4j import GraphDatabase
import networkx as nx
import plotly.graph_objects as go
from langchain_groq import ChatGroq
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Neo4jVector
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import PyPDFLoader
import re

# Neo4j Configuration
NEO4J_URI = "neo4j+s://f859410b.databases.neo4j.io"
NEO4J_USER = "neo4j"
NEO4J_PASSWORD = "UDAQaAOeey_dYNj2krLzpTeTINjnPgRgcJ4XeLtotpE"

# Initialize Groq LLM
GROQ_API_KEY = "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"
llm = ChatGroq(api_key=GROQ_API_KEY, model_name="mixtral-8x7b-32768")

class KnowledgeGraph:
    def __init__(self):
        self.driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))

    def delete_database(self):
        with self.driver.session() as session:
            session.run("MATCH (n) DETACH DELETE n")
            try:
                session.run("CALL db.index.vector.drop('document_vectors')")
            except:
                pass  # Ignore if index does not exist

    def _parse_relationships(self, llm_response: str):
        relationships = []
        pattern = r'\(([^)]+)\)-\[([^\]]+)\]->\(([^)]+)\)'
        for line in llm_response.split('\n'):
            matches = re.findall(pattern, line.strip())
            for match in matches:
                if len(match) == 3:
                    relationships.append({'entity1': match[0].strip(), 'relationship': match[1].strip(), 'entity2': match[2].strip()})
        return relationships

    def create_knowledge_graph(self, documents):
        with self.driver.session() as session:
            for doc in documents:
                prompt = f"""
                Extract key entities and their relationships from this text. 
                Format each relationship exactly as: (entity1)-[relationship]->(entity2)
                Only include clear, explicit relationships from the text.
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

    def create_3d_graph(self):
        G = nx.DiGraph()
        with self.driver.session() as session:
            result = session.run("""
                MATCH (e1:Entity)-[r:RELATES]->(e2:Entity)
                RETURN e1.name as source, r.type as relationship, e2.name as target
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
            mode='lines')
        
        node_x, node_y, node_z, hover_text = [], [], [], []
        for node in G.nodes():
            x, y, z = pos[node]
            node_x.append(x)
            node_y.append(y)
            node_z.append(z)
            hover_text.append(node)
        
        nodes_trace = go.Scatter3d(
            x=node_x, y=node_y, z=node_z,
            mode='markers',
            hovertext=hover_text,
            hoverinfo='text',
            marker=dict(size=8, color='#00ff00', line_width=2))
        
        fig = go.Figure(data=[edges_trace, nodes_trace])
        fig.update_layout(
            showlegend=False,
            scene=dict(
                xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                zaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            ),
            margin=dict(l=0, r=0, t=0, b=0),
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)'
        )
        return fig

def main():
    st.title("📌 Knowledge Graph Visualization")
    kg = KnowledgeGraph()
    
    with st.sidebar:
        st.header("Controls")
        if st.button("🗑️ Delete Database"):
            with st.spinner("Deleting database..."):
                kg.delete_database()
            st.success("Database cleared!")
    
    st.markdown("### 📄 Upload a Document")
    uploaded_file = st.file_uploader("Upload PDF", type=["pdf"])
    
    if uploaded_file:
        with st.spinner("Processing document..."):
            with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp:
                tmp.write(uploaded_file.getvalue())
                loader = PyPDFLoader(tmp.name)
                documents = loader.load()
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
            splits = text_splitter.split_documents(documents)
            kg.create_knowledge_graph(splits)
        st.success("Document processed!")
    
    st.markdown("### 🌐 Knowledge Graph Visualization")
    fig = kg.create_3d_graph()
    if fig:
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No relationships to visualize. Upload a document first.")

if __name__ == "__main__":
    main()
