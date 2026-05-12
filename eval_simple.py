# custom_eval.py
import os
import pandas as pd
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

# Import your working components
from chatbot import ensemble_retriever, llm

load_dotenv()

# ===========================
# 1. Define evaluation prompts
# ===========================
faithfulness_prompt = PromptTemplate(
    template="""You are a strict judge. Given the question, answer, and retrieved context, determine if the answer is factually consistent with the context. Answer ONLY "YES" or "NO".

Question: {question}
Answer: {answer}
Context: {context}

Is the answer consistent with the context? (YES/NO):""",
    input_variables=["question", "answer", "context"]
)
faithfulness_chain = faithfulness_prompt | llm | StrOutputParser()

relevancy_prompt = PromptTemplate(
    template="""You are a strict judge. Given the question and answer, determine if the answer directly and completely addresses the question. Answer ONLY "YES" or "NO".

Question: {question}
Answer: {answer}

Does the answer directly answer the question? (YES/NO):""",
    input_variables=["question", "answer"]
)
relevancy_chain = relevancy_prompt | llm | StrOutputParser()

# ===========================
# 2. Test questions (20 Indian diet/fitness)
# ===========================
test_questions = [
    "How many calories in a banana?",
    "What is the protein content of paneer?",
    "How often should I run to lose weight?",
    "Benefits of drinking warm lemon water?",
    "Is intermittent fasting good for weight loss?",
    "How to perform a proper squat?",
    "What foods are rich in iron for vegetarians?",
    "How much water should I drink daily?",
    "Can I eat bananas on a keto diet?",
    "What is the glycemic index of brown rice?",
    "Best exercises for lower back pain?",
    "How many calories in a masala dosa?",
    "How to reduce bloating after meals?",
    "Is it bad to eat late at night?",
    "What should I eat before a morning workout?",
    "How to cure muscle soreness naturally?",
    "What are the side effects of too much protein?",
    "How to do a proper push-up?",
    "What is the best time to drink green tea?",
    "How many eggs can I eat per day?",
    "Role of fiber in weight loss?",
    "How to increase metabolism naturally?",
    "Symptoms of dehydration during exercise?",
    "How to choose the right running shoes?",
]

# ===========================
# 3. Helper to get answer and context
# ===========================
def get_answer_and_context(question):
    docs = ensemble_retriever.invoke(question)
    context = "\n\n".join([doc.page_content for doc in docs])
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a fitness assistant. Answer in 1-2 short sentences."),
        ("human", "{input}")
    ])
    chain = prompt | llm
    answer = chain.invoke({"input": question}).content
    return answer, context

# ===========================
# 4. Run evaluation
# ===========================
print("Running evaluation on 20 test questions...\n")
results = []

for i, q in enumerate(test_questions, 1):
    print(f"{i}/{len(test_questions)}: {q[:50]}...")
    answer, context = get_answer_and_context(q)
    
    faithful = faithfulness_chain.invoke({
        "question": q, "answer": answer, "context": context[:2000]
    }).strip().upper()
    relevant = relevancy_chain.invoke({
        "question": q, "answer": answer
    }).strip().upper()
    
    faithful_score = 1 if faithful == "YES" else 0
    relevant_score = 1 if relevant == "YES" else 0
    
    results.append({
        "question": q,
        "answer": answer,
        "faithfulness_score": faithful_score,
        "answer_relevancy_score": relevant_score,
    })
    
    print(f"  → Faithful: {faithful_score}, Relevancy: {relevant_score}")
    print(f"  Answer: {answer[:100]}...\n")

# ===========================
# 5. Calculate overall scores
# ===========================
df = pd.DataFrame(results)
overall_faithfulness = df["faithfulness_score"].mean()
overall_relevancy = df["answer_relevancy_score"].mean()

print("="*50)
print("FINAL EVALUATION METRICS")
print("="*50)
print(f"Faithfulness:      {overall_faithfulness:.3f} ({(overall_faithfulness*100):.1f}%)")
print(f"Answer Relevancy:  {overall_relevancy:.3f} ({(overall_relevancy*100):.1f}%)")
print("="*50)

# Save to CSV
df.to_csv("custom_evaluation_results.csv", index=False)
print("\nDetailed results saved to 'custom_evaluation_results.csv'")
