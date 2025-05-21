from langchain.agents import initialize_agent, Tool
from langchain.agents.agent_types import AgentType
from langchain.chat_models import ChatOpenAI
from SPARQLWrapper import SPARQLWrapper, JSON
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

# Load vector model for FAISS
embedder = SentenceTransformer("all-MiniLM-L6-v2")

# Sample skills corpus (to simulate vector DB)
skills = ["Machine Learning", "Deep Neural Networks", "Support Vector Machines",
          "Model Evaluation", "Gradient Boosting", "Natural Language Processing"]

# Vectorize and build FAISS index
skill_embeddings = embedder.encode(skills, normalize_embeddings=True)
index = faiss.IndexFlatL2(skill_embeddings.shape[1])
index.add(skill_embeddings)

# DBpedia Query Function
def query_dbpedia(query: str) -> str:
    sparql = SPARQLWrapper("https://dbpedia.org/sparql")
    sparql.setQuery(query)
    sparql.setReturnFormat(JSON)
    try:
        results = sparql.query().convert()
        bindings = results["results"]["bindings"]
        output = []
        for result in bindings:
            row = [f"{k}: {v['value']}" for k, v in result.items()]
            output.append(", ".join(row))
        return "\n".join(output) if output else "No results found."
    except Exception as e:
        return f"Error querying DBpedia: {e}"


# LangChain tool definition
tools = [
    Tool(
        name="DBpediaSearch",
        func=query_dbpedia,
        description="Use this tool to query DBpedia with SPARQL to find structured information about concepts."
    )
]

# 5. Initialize ReAct agent
llm = ChatOpenAI(temperature=0, model="gpt-4o", openai_api_key = api_key) # Add openai key
agent = initialize_agent(
    tools,
    llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True
)

# 6. User query (needs to be optimized to avoid agent bypassing tool use)
user_query = "Find subtopics related to machine learning in DBpedia. Return the results as a Python list of strings. Run until the query successfully returns a list."

# 7. Run agent and extract concepts
agent_response = agent.run(user_query)
concepts = [c.strip() for c in agent_response.split(",") if c.strip()]
print("🔍 Extracted Concepts:", concepts)

# 8. Search FAISS for each concept
print("\n🔗 Similar Concepts in Vector DB:")
for concept in concepts:
    vector = embedder.encode([concept], normalize_embeddings=True)
    D, I = index.search(vector, k=3)
    for i, score in zip(I[0], D[0]):
        matched_concept = skills[i]
        print(f"{concept} → {matched_concept} (score: {score:.4f})")
    print()
