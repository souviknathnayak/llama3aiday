from typing import Optional
from phi.assistant import Assistant
from phi.knowledge import AssistantKnowledge
from phi.llm.ollama import Ollama
from phi.embedder.ollama import OllamaEmbedder
from phi.embedder.openai import OpenAIEmbedder
from phi.llm.openai import OpenAIChat


def get_rag_assistant(
    llm_model: str = "llama3",
    embeddings_model: str = "nomic-embed-text",
    user_id: Optional[str] = None,
    run_id: Optional[str] = None,
    debug_mode: bool = True,
) -> Assistant:
    """Get a Local RAG Assistant."""
    
    # Define the embedder based on the embeddings model
    embedder = OllamaEmbedder(model=embeddings_model, dimensions=4096)
    embeddings_model_clean = embeddings_model.replace("-", "_")
    if embeddings_model == "nomic-embed-text":
        embedder = OllamaEmbedder(model=embeddings_model, dimensions=768)
    elif embeddings_model == "phi3":
        embedder = OllamaEmbedder(model=embeddings_model, dimensions=3072)
    elif embeddings_model == "text-embedding-ada-002":
        embedder = OpenAIEmbedder(api_key="sk-proj-aQQjmnqSnHKDpeulPt9iT3BlbkFJLPppAB9FH27joK5ZLrMn")  # Use an environment variable or secure method to store API keys
    
    # Define the knowledge base
    knowledge = AssistantKnowledge(
        embedder=embedder,
        num_documents=3,
    )

    if llm_model=="gpt4":
        # Instantiating the Assistant with pickle file storage
        assistant = Assistant(
            name="local_rag_assistant",
            run_id=run_id,
            user_id=user_id,
            llm=OpenAIChat(model="gpt-4-turbo", max_tokens=500, temperature=0.3),
            knowledge_base=knowledge,
            description="You are PPA Advisor, expert in analyzing power purchasing agreement and your task is to answer questions using the provided information",
            instructions=[
                "When a user asks a question, you will be provided with information about the question.",
                "Carefully read this information and provide a clear and concise answer to the user.",
                "Do not use phrases like 'based on my knowledge' or 'depending on the information'.",
            ],
            add_references_to_prompt=True,
            markdown=True,
            add_datetime_to_instructions=True,
            debug_mode=debug_mode,
        )
        
    # Instantiating the Assistant with pickle file storage
    assistant = Assistant(
        name="local_rag_assistant",
        run_id=run_id,
        user_id=user_id,
        llm=Ollama(model=llm_model),
        knowledge_base=knowledge,
        description="You are PPA Advisor, expert in analyzing power purchasing agreement and your task is to answer questions using the provided information",
        instructions=[
            "When a user asks a question, you will be provided with information about the question.",
            "Carefully read this information and provide a clear and concise answer to the user.",
            "Do not use phrases like 'based on my knowledge' or 'depending on the information'.",
        ],
        add_references_to_prompt=True,
        markdown=True,
        add_datetime_to_instructions=True,
        debug_mode=debug_mode,
    )

    #assistant.save_to_pickle()  # Save initial state to pickle
    return assistant

# Example usage
rag_assistant = get_rag_assistant()
response = rag_assistant.chat("What are the key considerations in a power purchasing agreement?")
print(response)
