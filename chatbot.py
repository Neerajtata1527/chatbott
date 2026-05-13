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

# Disable ChromaDB telemetry to save a tiny bit of overhead
os.environ["ANONYMIZED_TELEMETRY"] = "false"

# ==========================================
# 1. Lazy Load Embeddings + Vector Stores
# ==========================================
# CHANGE: Nothing loads at import time.
# Models and DBs are created once on first request, then cached.
# This prevents OOM kills during cold start on Render free tier.

_embedding_model = None
_ensemble_retriever = None


def get_embedding_model():
    global _embedding_model
    if _embedding_model is None:
        print("Loading embedding model...")
        _embedding_model = HuggingFaceEmbeddings(
            model_name="all-MiniLM-L6-v2",
            model_kwargs={"device": "cpu"},
            # CHANGE: encode_kwargs normalizes embeddings, slightly reduces
            # memory pressure from large float arrays
            encode_kwargs={"normalize_embeddings": True}
        )
        print("Embedding model loaded.")
    return _embedding_model


def get_ensemble_retriever():
    global _ensemble_retriever
    if _ensemble_retriever is None:
        print("Loading vector stores...")
        embeddings = get_embedding_model()

        diet_db = Chroma(
            persist_directory="./chroma_db_diet",
            embedding_function=embeddings
        )
        workout_db = Chroma(
            persist_directory="./chroma_db_workout",
            embedding_function=embeddings
        )

        # CHANGE: k=3 instead of k=5 — halves the number of doc chunks
        # held in memory per query; recall drop is minimal at this scale
        diet_retriever = diet_db.as_retriever(search_kwargs={"k": 3})
        workout_retriever = workout_db.as_retriever(search_kwargs={"k": 3})

        _ensemble_retriever = EnsembleRetriever(
            retrievers=[diet_retriever, workout_retriever],
            weights=[0.5, 0.5]
        )
        print("Vector stores ready.")
    return _ensemble_retriever


# ==========================================
# 2. LLM
# ==========================================
# CHANGE: LLM is also lazy — created once and reused.
# Groq client is very lightweight (HTTP only, no local model),
# so this is just avoiding repeated object creation.

_llm = None


def get_llm():
    global _llm
    if _llm is None:
        _llm = ChatGroq(
            model="llama-3.1-8b-instant",
            temperature=0.1,
            groq_api_key=os.getenv("GROQ_API_KEY")
        )
    return _llm


# ==========================================
# 3. Greeting Detector
# ==========================================
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
    return (
        "Hey! 👋 I'm your Fitness Buddy. "
        "Ask me about diet, nutrition, workouts, yoga, or fitness."
    )


# ==========================================
# 4. Better Domain Filter
# ==========================================
def is_health_related(question: str) -> bool:

    q = question.lower()

    blocked_topics = [
        "politics",
        "election",
        "movie",
        "bitcoin",
        "crypto",
        "stock market",
        "cricket score",
        "football score",
        "weather",
        "temperature",
        "rain",
        "porn",
        "xxx",
        "onlyfans",
        "pornhub"
    ]

    if any(word in q for word in blocked_topics):
        return False

    return True


# ==========================================
# 5. Safety Filter
# ==========================================
BLOCKED_KEYWORDS = [
    "porn",
    "xxx",
    "nude",
    "nsfw",
    "explicit",
    "adult video",
    "pornhub",
    "onlyfans",
    "sex video"
]


def is_safe(question: str) -> bool:

    q = question.lower()

    return not any(word in q for word in BLOCKED_KEYWORDS)


# ==========================================
# 6. Weather Query Filter
# ==========================================
WEATHER_KEYWORDS = [
    "temperature",
    "weather",
    "forecast",
    "rain",
    "sunny",
    "humidity",
    "wind"
]


def is_weather_query(question: str) -> bool:

    q = question.lower()

    return any(word in q for word in WEATHER_KEYWORDS)


def weather_response() -> str:
    return (
        "I don't have access to real-time weather data. "
        "Please check a weather app before your workout. 🏃"
    )


# ==========================================
# 7. Relevance Evaluator
# ==========================================
RELEVANCE_PROMPT = PromptTemplate(
    template="""
You are a strict relevance judge.

Given the question and retrieved context,
answer ONLY YES or NO.

Question:
{question}

Context:
{context}

Relevant?
""",
    input_variables=["question", "context"]
)


def is_relevant(question: str, context: str) -> bool:

    if not context.strip():
        return False

    try:
        relevance_chain = RELEVANCE_PROMPT | get_llm() | StrOutputParser()

        result = relevance_chain.invoke({
            "question": question,
            # CHANGE: context truncated to 1500 chars instead of 2000
            # to reduce token usage and Groq response time on free tier
            "context": context[:1500]
        })

        return result.strip().upper() == "YES"

    except Exception as e:

        print("Relevance check failed:", e)

        return False


# ==========================================
# 8. General Fallback Chain
# ==========================================
general_prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        "You are a precise fitness and nutrition assistant. "
        "Answer briefly in 1-2 sentences. "
        "Use numbers whenever possible. "
        "If unsure, say you don't know."
    ),
    ("human", "{input}")
])


# ==========================================
# 9. Memory-Aware Fallback
# ==========================================
def answer_with_memory(user_input: str, memory):

    if hasattr(memory, "buffer") and memory.buffer:

        history = memory.buffer

    elif (
        hasattr(memory, "moving_summary_buffer")
        and memory.moving_summary_buffer
    ):

        history = memory.moving_summary_buffer

    else:

        history = ""

    prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            "You are a helpful fitness coach. "
            "Use conversation history if useful. "
            "Keep answers short and factual."
        ),
        ("system", "Conversation:\n{history}"),
        ("human", "{input}")
    ])

    chain = prompt | get_llm() | StrOutputParser()

    return chain.invoke({
        "history": history,
        "input": user_input
    })


# ==========================================
# 10. Create Memory + RAG Chain
# ==========================================
def create_memory_and_rag_chain():

    memory = ConversationSummaryMemory(
        llm=get_llm(),
        memory_key="chat_history",
        return_messages=True
    )

    custom_prompt = PromptTemplate.from_template("""
You are a precise fitness assistant.

Use ONLY the provided context to answer.

Give exact values if available.
Keep answers short (1-2 sentences).

Context:
{context}

Question:
{question}

Answer:
""")

    qa_chain = ConversationalRetrievalChain.from_llm(
        llm=get_llm(),
        retriever=get_ensemble_retriever(),
        memory=memory,
        combine_docs_chain_kwargs={
            "prompt": custom_prompt
        },
        verbose=False
    )

    return qa_chain, memory


active_chains = {}

# ==========================================
# 11. Main Chat Function
# ==========================================
def get_response(user_input: str, session_id: str = "default") -> str:

    # Greeting — no model needed, return instantly
    if is_greeting(user_input):
        return greeting_response()

    # Weather — no model needed, return instantly
    if is_weather_query(user_input):
        return weather_response()

    # Domain Filter — no model needed, return instantly
    if not is_health_related(user_input):

        return (
            "Sorry, I only answer questions about "
            "fitness, diet, nutrition, and health 😊"
        )

    # Safety Filter — no model needed, return instantly
    if not is_safe(user_input):

        return (
            "I can't help with explicit or inappropriate content."
        )

    # CHANGE: retriever + session chain are only initialised here,
    # on the first real health question. Greetings/blocks never
    # trigger model loading, so Render's health-check ping (GET /)
    # won't load the 400 MB model either.
    if session_id not in active_chains:

        active_chains[session_id] = create_memory_and_rag_chain()

    qa_chain, memory = active_chains[session_id]

    try:

        # Retrieve docs
        docs = get_ensemble_retriever().invoke(user_input)

        context = "\n\n".join([
            doc.page_content for doc in docs
        ]) if docs else ""

        # Use RAG if context relevant
        if context and is_relevant(user_input, context):

            response = qa_chain.invoke({
                "question": user_input
            })

            return response["answer"]

        # Otherwise fallback
        fallback = answer_with_memory(user_input, memory)

        memory.save_context(
            {"input": user_input},
            {"output": fallback}
        )

        return fallback

    except Exception as e:

        print("Main chatbot error:", e)

        return (
            "Sorry, the server is busy right now. "
            "Please try again in a moment."
        )