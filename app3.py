import random
import streamlit as st
from neo4j import GraphDatabase
import networkx as nx
import plotly.graph_objects as go
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage
from langchain.chains import RetrievalQA
from langchain.vectorstores import Neo4jVector
from langchain.embeddings import HuggingFaceEmbeddings
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from pyvis.network import Network
import streamlit.components.v1 as components

# Streamlit page configuration
st.set_page_config(layout="wide", page_title="Knowledge Graph Explorer")

# Neo4j Configuration
NEO4J_URI = "neo4j+s://947f4e44.databases.neo4j.io"
NEO4J_USER = "neo4j"
NEO4J_PASSWORD = "R7T40DT9nt7bUkcTkNj0qybHD-zBW2BAWgVN7nt8F6k"

# Groq API Configuration
GROQ_API_KEY = "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"

class KnowledgeGraphExplorer:
    def __init__(self):
        self.driver = GraphDatabase.driver(
            NEO4J_URI,
            auth=(NEO4J_USER, NEO4J_PASSWORD))
        
        # Initialize Groq client
        self.llm = ChatGroq(
            api_key=GROQ_API_KEY,
            model_name="llama3-8b-8192",
            temperature=0.3
        )
          # Initialize LLaMA model for XAI explanations
        self.xai_model_name = "meta-llama/Llama-2-7b-chat-hf"  # Using a smaller variant
        self.xai_tokenizer = None
        self.xai_model = None
        self._init_xai_model()
        
        # Initialize embeddings for RAG
        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-mpnet-base-v2"
        )
    
    def _init_xai_model(self):
        """Initialize the XAI model on first use"""
        try:
            self.xai_tokenizer = AutoTokenizer.from_pretrained(self.xai_model_name)
            self.xai_model = AutoModelForCausalLM.from_pretrained(
                self.xai_model_name,
                device_map="auto",
                torch_dtype=torch.float16
            )
        except Exception as e:
            print(f"Could not load XAI model: {str(e)}")
            self.xai_model = None

    def generate_content(self, content_type, graph_content, language="English"):
        """Generate different types of content in specified language"""
        if graph_content == "No relationships found in the graph.":
            return graph_content
        
        max_chars = 3000
        if len(graph_content) > max_chars:
            graph_content = graph_content[:max_chars] + "\n\n[Content truncated due to size limits]"
        
        # Handle summaries normally
        if content_type == "summary":
            prompts = {
                "English": f"""Provide a concise summary of this knowledge graph in English:
                Focus on main entities and relationships.
                Graph Content: {graph_content}
                English Summary:""",
                "Tamil": f"""Provide a Tamil summary (தமிழ் சுருக்கம்):
                Focus on main entities and relationships.
                Graph Content: {graph_content}
                Summary:""",
                "Hindi": f"""Provide a Hindi summary (हिंदी सारांश):
                Focus on main entities and relationships.
                Graph Content: {graph_content}
                Summary:"""
            }
            
            try:
                response = self.llm.invoke(
                    [HumanMessage(content=prompts[language])],
                    model="llama3-8b-8192"
                )
                return response.content.strip()
            except Exception as e:
                return f"Error generating {language} summary: {str(e)}"
        
        # Special handling for XAI explanations
        elif content_type == "explanation":
            if self.xai_model is None:
                return self._fallback_explanation(graph_content, language)
            
            try:
                # Extract key relationships for explanation
                relationships = self._extract_key_relationships(graph_content)
                
                # Prepare input for XAI model
                input_text = (
                    f"Knowledge Graph Context:\n{relationships}\n\n"
                    f"Explain the structure and relationships in this knowledge graph "
                    f"in {language} using Explainable AI (XAI) principles:\n"
                )
                
                # Generate explanation
                input_ids = self.xai_tokenizer(input_text, return_tensors="pt").input_ids.to("cuda")
                output = self.xai_model.generate(
                    input_ids,
                    max_new_tokens=300,
                    temperature=0.7,
                    do_sample=True
                )
                explanation = self.xai_tokenizer.decode(output[0], skip_special_tokens=True)
                
                # Post-process to remove the input text from output
                explanation = explanation[len(input_text):].strip()
                return explanation
                
            except Exception as e:
                print(f"XAI explanation failed: {str(e)}")
                return self._fallback_explanation(graph_content, language)
    
    def _extract_key_relationships(self, graph_content):
        """Extract the most important relationships for explanation"""
        # Simple extraction - you can enhance this
        lines = graph_content.split("\n")
        relationships = []
        for line in lines:
            if "--[" in line:  # Basic pattern matching for relationships
                relationships.append(line.split("\n")[0])  # Just the relationship line
        return "\n".join(relationships[:10])  # Limit to top 10 relationships
    
    def _fallback_explanation(self, graph_content, language):
        """Fallback explanation using the standard LLM"""
        prompts = {
            "English": f"""Explain this knowledge graph using XAI principles:
            Graph Content: {graph_content}
            Explanation:""",
            "Tamil": f"""Explain this knowledge graph in Tamil (தமிழ் விளக்கம்):
            Graph Content: {graph_content}
            Explanation:""",
            "Hindi": f"""Explain this knowledge graph in Hindi (हिंदी व्याख्या):
            Graph Content: {graph_content}
            Explanation:"""
        }
        
        try:
            response = self.llm.invoke(
                [HumanMessage(content=prompts[language])],
                model="llama3-8b-8192"
            )
            return response.content.strip()
        except Exception as e:
            return f"Error generating {language} explanation: {str(e)}"
        
    def get_graph_titles(self, limit=10):
        """Extract notable titles/labels from the graph for sidebar display"""
        titles = set()
        
        with self.driver.session() as session:
            # Get node labels with counts
            result = session.run("""
            CALL db.labels() YIELD label
            CALL apoc.cypher.run('MATCH (n:`'+label+'`) RETURN count(n) AS count', {})
            YIELD value
            RETURN label, value.count AS count
            ORDER BY count DESC
            LIMIT $limit
            """, {'limit': limit})
            
            for record in result:
                titles.add(f"{record['label']} ({record['count']} nodes)")
            
            # Get relationship types with counts
            result = session.run("""
            CALL db.relationshipTypes() YIELD relationshipType
            CALL apoc.cypher.run('MATCH ()-[r:`'+relationshipType+'`]->() RETURN count(r) AS count', {})
            YIELD value
            RETURN relationshipType, value.count AS count
            ORDER BY count DESC
            LIMIT $limit
            """, {'limit': limit})
            
            for record in result:
                titles.add(f"{record['relationshipType']} ({record['count']} relationships)")
            
            # Get some notable named entities
            result = session.run("""
            MATCH (n)
            WHERE n.name IS NOT NULL
            RETURN DISTINCT labels(n) AS labels, n.name AS name
            ORDER BY rand()
            LIMIT $limit
            """, {'limit': limit})
            
            for record in result:
                label = ", ".join(record["labels"])
                titles.add(f"{label}: {record['name']}")
        
        return sorted(titles)
    #XAI implementation using llm
    def extract_graph_content(self, limit=50):
        """Extract nodes and relationships with size limit"""
        content = []
        with self.driver.session() as session:
            result = session.run(f"""
            MATCH (n)-[r]->(m)
            RETURN labels(n) as source_labels, properties(n) as source_props, 
                   type(r) as rel_type, properties(r) as rel_props,
                   labels(m) as target_labels, properties(m) as target_props
            LIMIT {limit}
            """)
            
            for record in result:
                source_label = ", ".join(record["source_labels"])
                target_label = ", ".join(record["target_labels"])
                rel_type = record["rel_type"]
                
                relationship = f"{source_label} --[{rel_type}]--> {target_label}"
                
                if record["source_props"]:
                    relationship += f"\n  Source: {str({k: v for k, v in record['source_props'].items() if k in ['name', 'type', 'label']})}"
                if record["rel_props"]:
                    relationship += f"\n  Relationship: {str({k: v for k, v in record['rel_props'].items() if k in ['type', 'weight']})}"
                if record["target_props"]:
                    relationship += f"\n  Target: {str({k: v for k, v in record['target_props'].items() if k in ['name', 'type', 'label']})}"
                
                content.append(relationship)
        
        return "\n\n".join(content) if content else "No relationships found in the graph."

    def generate_content(self, content_type, graph_content, language="English"):
        """Generate different types of content in specified language"""
        if graph_content == "No relationships found in the graph.":
            return graph_content
        
        max_chars = 3000
        if len(graph_content) > max_chars:
            graph_content = graph_content[:max_chars] + "\n\n[Content truncated due to size limits]"
        
        # Content-specific prompts
        prompts = {
            "summary": {
                "English": f"""
                Provide a summary of this knowledge graph in English.
                Focus on main entities and their key relationships.
                
                Graph Content:
                {graph_content}
                
                English Summary:
                """,
                "Tamil": f"""
                Provide a detailed summary of this knowledge graph in Tamil (தமிழில்).
                - Maintain all technical terms in English
                - Focus on main entities and their key relationships
                - Use simple, clear Tamil language
                
                Graph Content:
                {graph_content}
                
                Tamil Summary (தமிழ் சுருக்கம்):
                """,
                "Hindi": f"""
                Provide a detailed summary of this knowledge graph in Hindi (हिंदी में).
                - Maintain all technical terms in English
                - Focus on main entities and their key relationships
                - Use simple, clear Hindi language
                
                Graph Content:
                {graph_content}
                
                Hindi Summary (हिंदी सारांश):
                """
            },
            "explanation": {
                "English": f"""
                Provide a detailed explanation of this knowledge graph in English using Explainable AI (XAI) principles.
                - Explain the graph structure, key patterns, and important relationships
                - Highlight any notable insights or anomalies
                - Use clear, simple language suitable for non-technical users
                
                Graph Content:
                {graph_content}
                
                English Explanation:
                """,
                "Tamil": f"""
                Provide a detailed explanation of this knowledge graph in Tamil (தமிழில்) using Explainable AI (XAI) principles.
                - Maintain all technical terms in English
                - Explain the graph structure, key patterns, and important relationships
                - Highlight any notable insights or anomalies
                - Use simple, clear Tamil language suitable for non-technical users
                
                Graph Content:
                {graph_content}
                
                Tamil Explanation (தமிழ் விளக்கம்):
                """,
                "Hindi": f"""
                Provide a detailed explanation of this knowledge graph in Hindi (हिंदी में) using Explainable AI (XAI) principles.
                - Maintain all technical terms in English
                - Explain the graph structure, key patterns, and important relationships
                - Highlight any notable insights or anomalies
                - Use simple, clear Hindi language suitable for non-technical users
                
                Graph Content:
                {graph_content}
                
                Hindi Explanation (हिंदी व्याख्या):
                """
            }
        }
        
        try:
            response = self.llm.invoke(
                [HumanMessage(content=prompts[content_type][language])],
                model="gemma2-9b-it"
            )
            return response.content.strip()
        except Exception as e:
            return f"Error generating {language} {content_type}: {str(e)}"
    def visualize_graph(self, limit=50):
        """Interactive knowledge graph visualization using PyVis"""
        # Create a PyVis network
        net = Network(
            height="700px",
            width="100%",
            bgcolor="#ffffff",
            font_color="#333333",
            directed=True,
            notebook=False
        )
        
        # Physics configuration for better layout
        net.barnes_hut(
            gravity=-2000,
            central_gravity=0.3,
            spring_length=200,
            spring_strength=0.05,
            damping=0.09,
            overlap=0.5
        )
        
        node_details = {}
        edge_details = {}
        label_colors = {
            "Person": "#FF6B6B",
            "Organization": "#4ECDC4",
            "Location": "#45B7D1",
            "Event": "#FFA07A",
            "Document": "#98D8C8",
            "default": "#6A8D92"
        }
        
        with self.driver.session() as session:
            # Get nodes and relationships
            result = session.run(f"""
            MATCH (n)-[r]->(m)
            RETURN n, r, m
            LIMIT {limit}
            """)
            
            for record in result:
                node1 = record["n"]
                node2 = record["m"]
                rel = record["r"]
                
                # Process nodes
                for node in [node1, node2]:
                    if node.id not in node_details:
                        label = next(iter(node.labels), "default")[0]
                        name = node.get("name", label)
                        
                        node_details[node.id] = {
                            "label": name,
                            "type": label,
                            "color": label_colors.get(label, label_colors["default"]),
                            "properties": dict(node)
                        }
                        
                        # Add to network
                        net.add_node(
                            node.id,
                            label=name,
                            color=label_colors.get(label, label_colors["default"]),
                            shape="dot" if label == "Person" else "diamond",
                            size=25 if label == "Person" else 20,
                            title=self._create_node_tooltip(dict(node), label)
                        )
                
                # Add relationship
                edge_details[(node1.id, node2.id, rel.id)] = {
                    "type": rel.type,
                    "properties": dict(rel)
                }
                
                net.add_edge(
                    node1.id,
                    node2.id,
                    title=self._create_edge_tooltip(rel.type, dict(rel)),
                    label=rel.type,
                    color="#888888",
                    width=2,
                    arrows="to"
                )
        
        # Generate HTML
        net.save_graph("temp_graph.html")
        
        # Read HTML file and display in Streamlit
        with open("temp_graph.html", "r", encoding="utf-8") as f:
            html = f.read()
        
        # Make responsive
        html = html.replace(
            'width: 100%', 
            'width: 100%; height: 700px; border: 1px solid #ddd; border-radius: 8px;'
        )
        
        # Display in Streamlit
        components.html(html, height=700)
        
        # Store graph data
        st.session_state.graph_data = {
            'node_details': node_details,
            'edge_details': edge_details
        }

    def _create_node_tooltip(self, properties, label):
        """Generate HTML tooltip for nodes"""
        tooltip = f"""
        <div style="font-family: Arial; padding: 10px;">
            <h4 style="margin: 0; color: #333; border-bottom: 1px solid #eee; padding-bottom: 5px;">
                {properties.get('name', label)}
            </h4>
            <p style="margin: 5px 0; color: #666;"><em>{label}</em></p>
            <table style="width: 100%; border-collapse: collapse; margin-top: 10px;">
        """
        
        for key, value in properties.items():
            if key != "name":
                tooltip += f"""
                <tr>
                    <td style="padding: 3px 0; border-bottom: 1px dotted #eee; font-weight: bold; color: #555;">
                        {key}:
                    </td>
                    <td style="padding: 3px 0; border-bottom: 1px dotted #eee; color: #333;">
                        {str(value)[:50] + '...' if len(str(value)) > 50 else str(value)}
                    </td>
                </tr>
                """
        
        tooltip += """
            </table>
        </div>
        """
        return tooltip

    def _create_edge_tooltip(self, rel_type, properties):
        """Generate HTML tooltip for edges"""
        tooltip = f"""
        <div style="font-family: Arial; padding: 10px;">
            <h4 style="margin: 0; color: #333; border-bottom: 1px solid #eee; padding-bottom: 5px;">
                {rel_type}
            </h4>
            <table style="width: 100%; border-collapse: collapse; margin-top: 10px;">
        """
        
        for key, value in properties.items():
            tooltip += f"""
            <tr>
                <td style="padding: 3px 0; border-bottom: 1px dotted #eee; font-weight: bold; color: #555;">
                    {key}:
                </td>
                <td style="padding: 3px 0; border-bottom: 1px dotted #eee; color: #333;">
                    {str(value)[:50] + '...' if len(str(value)) > 50 else str(value)}
                </td>
            </tr>
            """
        
        tooltip += """
            </table>
        </div>
        """
        return tooltip
        
    def setup_rag_chain(self):
        """Setup RAG chain for querying the graph"""
        vector_store = Neo4jVector.from_existing_index(
            embedding=self.embeddings,
            url=NEO4J_URI,
            username=NEO4J_USER,
            password=NEO4J_PASSWORD,
            index_name="vector",
            node_label="Document",
            text_node_property="text",
            embedding_node_property="embedding"
        )
        
        return RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=vector_store.as_retriever(),
            return_source_documents=True
        )
    
    def generate_suggested_queries(self, graph_content, language="English"):
        """Generate relevant queries based on the graph content"""
        if graph_content == "No relationships found in the graph.":
            return []
        
        prompt = f"""
        Analyze this knowledge graph content and suggest 3-5 specific, useful queries that would help explore this graph.
        Focus on the main entities and relationships present in the graph.
        Return only the queries as a bulleted list, no additional explanation.
        
        Graph Content:
        {graph_content[:3000]}  # Limit to first 3000 chars to avoid token limits
        
        Suggested queries:
        - """
        
        try:
            response = self.llm.invoke(
                [HumanMessage(content=prompt)],
                model="llama3-8b-8192"
            )
            # Parse the response into a list of queries
            queries = [q.strip() for q in response.content.split("-") if q.strip()]
            return queries[:5]  # Return max 5 queries
        except Exception as e:
            print(f"Error generating queries: {str(e)}")
            return [
                "Show all Person nodes",
                "What relationships exist between Company and Person?",
                "Find all nodes connected to [specific entity]"
            ]

    def query_graph(self, question, language="English"):
        """Query the graph using RAG with language support"""
        try:
            qa_chain = self.setup_rag_chain()
            result = qa_chain({"query": question})
            
            # Translate the answer if needed
            if language != "English":
                translation_prompt = f"""
                Translate this answer to {language} while maintaining all technical accuracy:
                
                Original Answer:
                {result["result"]}
                
                {language} Translation:
                """
                translated = self.llm.invoke(
                    [HumanMessage(content=translation_prompt)],
                    model="llama3-8b-8192"
                )
                answer = translated.content.strip()
            else:
                answer = result["result"]
            
            return {
                "answer": answer,
                "sources": result["source_documents"]
            }
        except Exception as e:
            return {
                "answer": f"Error querying graph: {str(e)}",
                "sources": []
            }
        
def main():
    st.title("🌐 Neo4j Knowledge Graph Explorer")
    
    # Initialize all session state variables at the start
    if 'explorer' not in st.session_state:
        with st.spinner("Initializing system..."):
            st.session_state.explorer = KnowledgeGraphExplorer()
    
    # Initialize graph-related session state variables
    if 'graph_content' not in st.session_state:
        st.session_state.graph_content = None
    if 'current_language' not in st.session_state:
        st.session_state.current_language = "English"
    if 'summaries' not in st.session_state:
        st.session_state.summaries = {"English": None, "Tamil": None, "Hindi": None}
    if 'explanations' not in st.session_state:
        st.session_state.explanations = {"English": None, "Tamil": None, "Hindi": None}
    if 'suggested_query' not in st.session_state:
        st.session_state.suggested_query = ""
    
    # Sidebar content
    with st.sidebar:
        st.subheader("Graph Controls")
        graph_limit = st.slider("Max nodes to display", 10, 400, 50)
        
        # Graph content highlights - only show if we have content
        st.subheader("🔍 Graph Content Preview")
        
        # Only try to show preview if we have graph content
        if st.session_state.graph_content:
            with st.spinner("Loading graph highlights..."):
                try:
                    titles = st.session_state.explorer.get_graph_titles()
                    if titles:
                        st.markdown("**Graph Content Highlights:**")
                        # Group the titles by type
                        node_titles = [t for t in titles if "nodes)" in t]
                        rel_titles = [t for t in titles if "relationships)" in t]
                        entity_titles = [t for t in titles if ": " in t]
                        
                        if node_titles:
                            st.markdown("**Node Types:**")
                            for title in node_titles[:5]:
                                st.markdown(f"- {title}")
                        
                        if rel_titles:
                            st.markdown("**Relationship Types:**")
                            for title in rel_titles[:5]:
                                st.markdown(f"- {title}")
                        
                        if entity_titles:
                            st.markdown("**Sample Entities:**")
                            for title in entity_titles[:10]:
                                st.markdown(f"- {title}")
                        
                        st.markdown("---")
                        st.markdown("💡 **Suggested Queries:**")
                        
                        # Generate and show suggested queries
                        suggested_queries = st.session_state.explorer.generate_suggested_queries(
                            st.session_state.graph_content
                        )
                        
                        for query in suggested_queries:
                            if st.button(query, key=f"suggested_{query[:20]}"):
                                st.session_state.suggested_query = query
                                # This will rerun the app and the query will be used in the Query tab

                except Exception as e:
                    st.error(f"Couldn't load graph preview: {str(e)}")
        else:
            st.info("Analyze the graph to see content preview")

    # Display graph visualization
    st.markdown("### 📊 Graph Visualization")
    with st.spinner("Loading graph..."):
        fig = st.session_state.explorer.visualize_graph(limit=graph_limit)
    
    if fig:
        st.plotly_chart(fig, use_container_width=True)

    
    # Initialize session state
    if 'graph_content' not in st.session_state:
        st.session_state.graph_content = None
    if 'current_language' not in st.session_state:
        st.session_state.current_language = "English"
    if 'summaries' not in st.session_state:
        st.session_state.summaries = {"English": None, "Tamil": None, "Hindi": None}
    if 'explanations' not in st.session_state:
        st.session_state.explanations = {"English": None, "Tamil": None, "Hindi": None}
    
    # Generate content button
    if st.button("Analyze Graph"):
        with st.spinner("Extracting graph content..."):
            st.session_state.graph_content = st.session_state.explorer.extract_graph_content(limit=graph_limit)
            st.session_state.current_language = "English"
    
    # Tab interface
    tab1, tab2, tab3 = st.tabs(["Summary", "Explanation", "Query"])
    
    with tab1:
        # Summary Tab
        st.markdown("### 📝 Graph Summary")
        
        if st.session_state.graph_content:
            # Language selector for summary
            summary_lang = st.radio(
                "Summary Language:",
                ["English", "Tamil", "Hindi"],
                index=["English", "Tamil", "Hindi"].index(st.session_state.current_language),
                key="summary_lang"
            )
            
            # Generate summary if not available
            if not st.session_state.summaries[summary_lang]:
                with st.spinner(f"Generating {summary_lang} summary..."):
                    st.session_state.summaries[summary_lang] = st.session_state.explorer.generate_content(
                        "summary",
                        st.session_state.graph_content,
                        language=summary_lang
                    )
            
            # Display summary
            st.markdown(f"#### Graph Insights ({summary_lang}):")
            st.write(st.session_state.summaries[summary_lang])
            
            with st.expander("View raw graph data"):
                st.text(st.session_state.graph_content)
        else:
            st.info("Please analyze the graph first using the button above")
    
    with tab2:
        # Explanation Tab (XAI)
        st.markdown("### 🔍 Graph Explanation (XAI)")
        
        if st.session_state.graph_content:
            # Language selector for explanation
            explain_lang = st.radio(
                "Explanation Language:",
                ["English", "Tamil", "Hindi"],
                index=["English", "Tamil", "Hindi"].index(st.session_state.current_language),
                key="explain_lang"
            )
            
            # Generate explanation if not available
            if not st.session_state.explanations[explain_lang]:
                with st.spinner(f"Generating {explain_lang} explanation..."):
                    st.session_state.explanations[explain_lang] = st.session_state.explorer.generate_content(
                        "explanation",
                        st.session_state.graph_content,
                        language=explain_lang
                    )
            
            # Display explanation
            st.markdown(f"#### Detailed Explanation ({explain_lang}):")
            st.write(st.session_state.explanations[explain_lang])
        else:
            st.info("Please analyze the graph first using the button above")
    
    with tab3:
        # Query Tab (RAG)
        st.markdown("### ❓ Query the Knowledge Graph")
        
        if st.session_state.graph_content:
            # Language selector for query
            query_lang = st.radio(
                "Query Language:",
                ["English", "Tamil", "Hindi"],
                index=["English", "Tamil", "Hindi"].index(st.session_state.current_language),
                key="query_lang"
            )
            
            # Query input with examples
            query_examples = [
                "Show all Person nodes",
                "What relationships exist between Company and Person?",
                "Find all nodes connected to [specific entity]"
            ]
            
            user_question = st.text_input(
                "Ask a question about the knowledge graph:",
                placeholder="e.g. " + random.choice(query_examples)
            )
            
            if user_question:
                with st.spinner("Searching for answers..."):
                    result = st.session_state.explorer.query_graph(user_question, language=query_lang)
                    
                    st.markdown(f"#### Answer ({query_lang}):")
                    st.write(result["answer"])
                    
                    if result["sources"]:
                        with st.expander("View sources"):
                            for i, doc in enumerate(result["sources"]):
                                st.markdown(f"**Source {i+1}:**")
                                st.write(doc.page_content)
                                st.write("---")
        else:
            st.info("Please analyze the graph first using the button above")

if __name__ == "__main__":
    main()
