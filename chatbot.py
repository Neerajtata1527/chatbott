import os
import re
from dotenv import load_dotenv
from langchain.memory import ConversationSummaryMemory
from langchain.chains import ConversationalRetrievalChain
from langchain_groq import ChatGroq
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.retrievers import EnsembleRetriever
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()

# ===========================
# 1. Load Vector Stores
# ===========================
print("Loading vector stores...")
embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

diet_db = Chroma(persist_directory="./chroma_db_diet", embedding_function=embedding_model)
workout_db = Chroma(persist_directory="./chroma_db_workout", embedding_function=embedding_model)

# Increased k to 5 for more context (helps retrieval)
diet_retriever = diet_db.as_retriever(search_kwargs={"k": 7})
workout_retriever = workout_db.as_retriever(search_kwargs={"k": 7})

ensemble_retriever = EnsembleRetriever(
    retrievers=[diet_retriever, workout_retriever],
    weights=[0.5, 0.5]
)

# ===========================
# 2. LLM (Groq) – switched to more accurate model
# ===========================
llm = ChatGroq(
    model="llama-3.3-70b-versatile",   # Much better for factual answers
    temperature=0.1,                   # Lower temperature for precision
    groq_api_key=os.getenv("GROQ_API_KEY")
)

# ===========================
# 3. Greeting Detector
# ===========================
GREETING_PATTERNS = [
    r"^(hi|hello|hey|howdy|greetings)[\s!]*$",
    r"^how are you[\s?]*$",
    r"^what's up[\s?]*$",
    r"^good morning[\s!]*$",
    r"^good evening[\s!]*$",
]

def is_greeting(question: str) -> bool:
    q_lower = question.lower().strip()
    return any(re.match(pattern, q_lower) for pattern in GREETING_PATTERNS)

def greeting_response() -> str:
    return "Hey! 👋 I'm your Fitness Buddy. Ask me about diet, exercise, yoga, or nutrition."

# ===========================
# 4. Domain Classifier
# ===========================
DOMAIN_CLASSIFIER_PROMPT = PromptTemplate(
    template="""You are a strict classifier. Determine if the user's question is about **health, fitness, diet, nutrition, yoga, exercise, or wellness**. Answer ONLY "YES" or "NO".

Question: {question}
Is this related to health, fitness, or diet? Answer YES or NO:""",
    input_variables=["question"]
)
classifier_chain = DOMAIN_CLASSIFIER_PROMPT | llm | StrOutputParser()

def is_health_related(question: str) -> bool:
    result = classifier_chain.invoke({"question": question})
    return result.strip().upper() == "YES"

# ===========================
# 5. Safety Guard
# ===========================
SAFETY_GUARD_PROMPT = PromptTemplate(
    template="""Is this question explicitly asking for sexually explicit content, pornography, or inappropriate advice? 
Answer ONLY "BLOCK" if it asks for explicit content, otherwise "ALLOW".

Question: {question}""",
    input_variables=["question"]
)
safety_chain = SAFETY_GUARD_PROMPT | llm | StrOutputParser()

def is_safe(question: str) -> bool:
    result = safety_chain.invoke({"question": question})
    return result.strip().upper() != "BLOCK"

# ===========================
# 6. Relevance Evaluator (for RAG)
# ===========================
RELEVANCE_PROMPT = PromptTemplate(
    template="""You are a strict relevance judge. Given the user's question and the retrieved context, answer ONLY "YES" if the context is sufficient and relevant, otherwise "NO".

Question: {question}
Context: {context}
Relevant? YES or NO:""",
    input_variables=["question", "context"]
)
relevance_chain = RELEVANCE_PROMPT | llm | StrOutputParser()

def is_relevant(question: str, context: str) -> bool:
    if not context or not context.strip():
        return False
    try:
        answer = relevance_chain.invoke({"question": question, "context": context[:2000]})
        return answer.strip().upper() == "YES"
    except Exception:
        return False

# ===========================
# 7. Weather/Location Rejection
# ===========================
WEATHER_KEYWORDS = ["temperature", "weather", "forecast", "rain", "sunny", "cloudy", "humidity", "wind", "current temp"]
CITIES = ["bapatta", "bapatla", "delhi", "mumbai", "chennai", "kolkata", "bangalore", "hyderabad"]

def is_weather_query(question: str) -> bool:
    q_lower = question.lower()
    if not any(kw in q_lower for kw in WEATHER_KEYWORDS):
        return False
    if any(city in q_lower for city in CITIES):
        return True
    if re.search(r'\b(in|at|for)\s+\w+\b', q_lower):
        return True
    return False

def weather_response() -> str:
    return "I don't have access to real‑time weather data. Please check a weather app before your run. 🏃‍♂️"

# ===========================
# 8. Improved General Knowledge Fallback (numeric, factual)
# ===========================
general_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a precise nutrition and fitness assistant. Answer with **specific numbers** whenever possible (e.g., '105 calories', '18g protein'). If you don't know the exact number, say 'I don't have that information' – do not guess. Keep answers short (1‑2 sentences)."),
    ("human", "{input}")
])
general_chain = general_prompt | llm | StrOutputParser()

# ===========================
# 9. Memory‑aware answer (for conversational context)
# ===========================
def answer_with_memory(user_input: str, memory) -> str:
    if hasattr(memory, 'buffer') and memory.buffer:
        history = memory.buffer
    elif hasattr(memory, 'moving_summary_buffer') and memory.moving_summary_buffer:
        history = memory.moving_summary_buffer
    else:
        history = ""

    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a precise fitness coach. Use conversation history to answer in 1‑2 short sentences with numbers when possible."),
        ("system", "Conversation so far:\n{history}"),
        ("human", "{input}")
    ])
    chain = prompt | llm | StrOutputParser()
    return chain.invoke({"history": history, "input": user_input})

# ===========================
# 10. RAG Chain with Memory
# ===========================
def create_memory_and_rag_chain():
    memory = ConversationSummaryMemory(
        llm=llm,
        memory_key="chat_history",
        return_messages=True
    )
    # Custom prompt for RAG to emphasise numeric answers
    custom_qa_prompt = PromptTemplate.from_template("""
You are a precise fitness coach. Use the following context to answer the question. 
Give specific numbers (calories, grams, etc.) if the context provides them. 
Answer in 1‑2 short sentences. Do not list steps or use bullet points.

Context: {context}
Question: {question}
Answer:""")
    
    qa_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=ensemble_retriever,
        memory=memory,
        combine_docs_chain_kwargs={"prompt": custom_qa_prompt},
        verbose=False
    )
    return qa_chain, memory

active_chains = {}

# ===========================
# 11. Main Response Function
# ===========================
def get_response(user_input: str, session_id: str = "default") -> str:
    if is_greeting(user_input):
        return greeting_response()
    
    if is_weather_query(user_input):
        return weather_response()
    
    if not is_health_related(user_input):
        return "Sorry, I only talk about health, fitness, and nutrition. 😊"
    
    if not is_safe(user_input):
        return "I can't answer that. Please ask a respectful health question."
    
    if session_id not in active_chains:
        active_chains[session_id] = create_memory_and_rag_chain()
    qa_chain, memory = active_chains[session_id]
    
    try:
        docs = ensemble_retriever.invoke(user_input)
        context = "\n\n".join([doc.page_content for doc in docs]) if docs else ""
        
        if context and is_relevant(user_input, context):
            response = qa_chain.invoke({"question": user_input})
            return response["answer"]
        else:
            # Fallback with memory – uses the accurate general prompt
            fast_answer = answer_with_memory(user_input, memory)
            memory.save_context({"input": user_input}, {"output": fast_answer})
            return fast_answer
    except Exception as e:
        print(f"Error: {e}")
        return answer_with_memory(user_input, memory)