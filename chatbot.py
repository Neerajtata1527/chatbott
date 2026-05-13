import os
import re
from dotenv import load_dotenv

from langchain.memory import ConversationSummaryMemory
from langchain.chains import ConversationalRetrievalChain
from langchain_groq import ChatGroq

from langchain_community.vectorstores import Chroma
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings

from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.retrievers import EnsembleRetriever
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()

os.environ["ANONYMIZED_TELEMETRY"] = "false"

# ==========================================
# 1. Lazy Load Embeddings + Vector Stores
# ==========================================
_embedding_model = None
_ensemble_retriever = None


def get_embedding_model():
    global _embedding_model
    if _embedding_model is None:
        print("Loading embedding model...")
        _embedding_model = FastEmbedEmbeddings(
            model_name="BAAI/bge-small-en-v1.5"
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
    r"^good afternoon[\s!]*$",
    r"^good night[\s!]*$",
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
# 4. Weather Query Filter
# ==========================================
WEATHER_KEYWORDS = [
    "weather", "forecast", "sunny", "humidity",
    "wind speed", "will it rain", "temperature today",
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
# 5. Safety Filter  (instant block, no LLM)
# ==========================================
BLOCKED_KEYWORDS = [
    "porn", "xxx", "nude", "nsfw", "explicit",
    "adult video", "pornhub", "onlyfans", "sex video",
    "hentai", "erotic",
]


def is_safe(question: str) -> bool:
    q = question.lower()
    return not any(word in q for word in BLOCKED_KEYWORDS)


# ==========================================
# 6. Domain Filter  — two-layer approach
#    Layer A: fast keyword allowlist  (no LLM cost)
#    Layer B: LLM judge only if Layer A uncertain
# ==========================================

# If ANY of these appear → definitely health/fitness → allow instantly
HEALTH_ALLOWLIST = [
    # nutrition values
    "calorie", "calories", "kcal", "protein", "carb", "carbs",
    "fat", "fats", "fiber", "fibre", "vitamin", "mineral",
    "calcium", "iron", "sodium", "potassium", "zinc", "folate",
    "omega", "macro", "micro", "nutrient", "nutrition",
    # food items
    "food", "eat", "eating", "meal", "diet", "recipe",
    "breakfast", "lunch", "dinner", "snack", "drink",
    "vegetable", "fruit", "dal", "rice", "roti", "chapati",
    "paneer", "chicken", "egg", "fish", "milk", "curd", "dahi",
    "ghee", "oil", "sugar", "salt", "spice", "supplement",
    "whey", "creatine", "bcaa", "protein powder", "probiotic",
    "mango", "banana", "apple", "orange", "spinach", "broccoli",
    "oats", "wheat", "millet", "jowar", "bajra", "ragi",
    # weight & body composition
    "weight", "bmi", "obesity", "overweight", "underweight",
    "weight loss", "weight gain", "lose weight", "gain weight",
    "bulk", "cut", "deficit", "surplus", "body fat", "lean",
    # exercise & fitness
    "workout", "exercise", "gym", "training", "fitness",
    "muscle", "strength", "cardio", "yoga", "meditation",
    "stretching", "flexibility", "run", "running", "jog",
    "jogging", "walk", "walking", "cycling", "swim", "swimming",
    "hiit", "crossfit", "pushup", "push-up", "pull up", "squat",
    "deadlift", "bench press", "rep", "sets", "rest day",
    "warm up", "cool down", "recovery", "soreness", "cramp",
    "plank", "lunge", "burpee", "abs", "core", "bicep", "tricep",
    "shoulder", "chest", "back workout", "leg day", "glutes",
    # health conditions
    "health", "healthy", "diabetes", "blood pressure", "bp",
    "cholesterol", "thyroid", "digestion", "gut", "stomach",
    "liver", "kidney", "heart", "blood sugar", "insulin",
    "sleep", "stress", "anxiety", "mental health", "immunity",
    "immune", "inflammation", "hydration", "water intake",
    "detox", "fasting", "intermittent fasting", "keto",
    "vegan", "vegetarian", "gluten", "lactose", "allergy",
    "injury", "pain", "physiotherapy", "rehab", "posture",
    "bone density", "joint", "ligament", "tendon",
]

# If ANY of these appear → definitely NOT health → block instantly
HARD_BLOCK_LIST = [
    "politics", "election", "politician", "minister", "cm ",
    "chief minister", "prime minister", "president of",
    "who is the", "who won", "ipl", "cricket score",
    "football score", "match result", "movie", "film", "actor",
    "actress", "celebrity", "bitcoin", "crypto", "stock market",
    "share price", "sensex", "nifty", "war", "army", "military",
    "news", "current affairs", "government", "law", "court",
    "judge", "murder", "crime", "robbery", "accident",
    "astrology", "horoscope", "zodiac", "lottery", "gambling",
    "coding", "programming", "python code", "javascript",
    "database", "sql", "machine learning", "artificial intelligence",
    "chatgpt", "chatbot", "llm", "geography", "history",
    "science project", "math", "equation", "solve this",
]

# LLM judge prompt — only called when neither list matches
DOMAIN_JUDGE_PROMPT = PromptTemplate(
    template="""You are a strict topic classifier for a fitness and nutrition chatbot.

Your job: decide if the question is related to health, fitness, nutrition, diet, exercise, yoga, body weight, medical conditions related to diet/exercise, or wellness.

Answer ONLY "YES" if it is health/fitness related.
Answer ONLY "NO" if it is about anything else (politics, entertainment, technology, current events, geography, history, general knowledge, etc.).

Question: {question}

Answer (YES or NO only):""",
    input_variables=["question"]
)


def is_health_related(question: str) -> bool:
    q = question.lower().strip()

    # Layer A1: hard block — instant reject
    if any(phrase in q for phrase in HARD_BLOCK_LIST):
        return False

    # Layer A2: allowlist — instant accept
    if any(keyword in q for keyword in HEALTH_ALLOWLIST):
        return True

    # Layer B: LLM judge for ambiguous questions
    # e.g. "is alcohol bad?" / "can stress affect my body?"
    try:
        domain_chain = DOMAIN_JUDGE_PROMPT | get_llm() | StrOutputParser()
        result = domain_chain.invoke({"question": question})
        return result.strip().upper().startswith("YES")
    except Exception as e:
        print("Domain check failed:", e)
        # Fail safe: block if uncertain
        return False


# ==========================================
# 7. Relevance Evaluator
# ==========================================
RELEVANCE_PROMPT = PromptTemplate(
    template="""You are a strict relevance judge.

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
            "context": context[:1500]
        })
        return result.strip().upper() == "YES"

    except Exception as e:
        print("Relevance check failed:", e)
        return False


# ==========================================
# 8. Memory-Aware Fallback
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
            "You are a helpful fitness and nutrition coach. "
            "ONLY answer questions about health, fitness, diet, nutrition, "
            "exercise, yoga, and wellness. "
            "If the question is about anything else, say: "
            "'I can only help with fitness, diet, and health topics 😊'. "
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
# 9. Create Memory + RAG Chain
# ==========================================
def create_memory_and_rag_chain():

    memory = ConversationSummaryMemory(
        llm=get_llm(),
        memory_key="chat_history",
        return_messages=True
    )

    custom_prompt = PromptTemplate.from_template("""
You are a precise fitness and nutrition assistant.

Use ONLY the provided context to answer.
ONLY answer questions about health, fitness, diet, nutrition, exercise, yoga, and wellness.
If asked about anything else, say: "I can only help with fitness, diet, and health topics 😊"

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
# 10. Main Chat Function
# ==========================================
def get_response(user_input: str, session_id: str = "default") -> str:

    # Instant responses — zero ML cost
    if is_greeting(user_input):
        return greeting_response()

    if is_weather_query(user_input):
        return weather_response()

    if not is_safe(user_input):
        return "I can't help with explicit or inappropriate content."

    # Two-layer domain filter
    if not is_health_related(user_input):
        return (
            "I'm your Fitness Buddy — I can only help with "
            "diet, nutrition, workouts, yoga, and health topics. "
            "Please ask me something related to fitness or health! 💪"
        )

    # Load models on first real health question
    if session_id not in active_chains:
        active_chains[session_id] = create_memory_and_rag_chain()

    qa_chain, memory = active_chains[session_id]

    try:

        docs = get_ensemble_retriever().invoke(user_input)

        context = "\n\n".join([
            doc.page_content for doc in docs
        ]) if docs else ""

        if context and is_relevant(user_input, context):
            response = qa_chain.invoke({"question": user_input})
            return response["answer"]

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