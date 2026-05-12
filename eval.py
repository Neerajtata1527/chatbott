# eval_combined.py
import os
import re
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from chatbot import ensemble_retriever

load_dotenv()

llm = ChatGroq(
    model="llama-3.1-8b-instant",
    temperature=0.2,
    groq_api_key=os.getenv("GROQ_API_KEY")
)

# Combined test cases: diet questions (numeric) and workout questions (keywords)
test_cases = [
    # Diet questions (numeric)
    ("How many calories in a banana?", "116", "numeric"),
    ("What is the protein content of paneer?", "18.3", "numeric"),
    ("How many calories in a masala dosa?", "213", "numeric"),
    ("What is the carbohydrate content of brown rice?", "73.0", "numeric"),
    ("How much fiber in an apple?", "2.4", "numeric"),
    ("What is the fat content of avocado?", "14.7", "numeric"),
    ("What is the vitamin C content of orange?", "59", "numeric"),
    ("How much iron is in spinach?", "1.14", "numeric"),
    ("What is the calcium content of milk?", "120", "numeric"),
    ("What is the protein content of moong dal?", "24.0", "numeric"),
    ("How much protein is in chickpeas (kabuli chana)?", "17.1", "numeric"),
    ("What is the fiber content of whole wheat flour?", "1.8", "numeric"),
    ("How many calories in 100g of cooked white rice?", "130", "numeric"),
    ("What is the iron content of bajra (pearl millet)?", "8.0", "numeric"),
    ("How much vitamin A is in carrots?", "2001", "numeric"),
    ("What is the fat content of coconut (fresh)?", "33.5", "numeric"),
    ("How much calcium is in ragi (finger millet)?", "344", "numeric"),
    ("What is the folate content of cowpea (lobia)?", "633", "numeric"),
    ("How many grams of sugar in a ripe mango?", "16.9", "numeric"),
    ("What is the zinc content of cashew nuts?", "5.8", "numeric"),
    # Workout questions (keyword/descriptive)
    ("How should you breathe between heavy sets?", "physiological sigh", "keyword"),
    ("How often should you do HIIT on the treadmill?", "2-3 times per week", "keyword"),
    ("Which yoga pose helps with hip flexibility?", "pigeon", "keyword"),
    ("How many minutes rest between compound lift sets?", "2-3", "keyword"),
    ("What is the benefit of drinking water before meals?", "reduces calorie intake", "keyword"),
    ("How many calories does a 30-minute jog burn?", "300", "numeric_range"),
    ("How much protein should you eat post-workout?", "20-40", "keyword"),
    ("What exercise helps desk workers with posture?", "band pull-aparts", "keyword"),
    ("What is the minimum workout on low-motivation days?", "10 minutes", "keyword"),
    ("How often replace running shoes?", "500-800", "keyword"),
]

def is_correct(answer, expected, answer_type):
    """Check if answer matches expected based on type."""
    answer_lower = answer.lower()
    expected_lower = expected.lower()
    
    if answer_type == "numeric":
        # Extract first number from answer
        match = re.search(r'(\d+(?:\.\d+)?)', answer)
        if match:
            extracted = float(match.group(1))
            expected_num = float(expected_lower)
            # Allow 10% tolerance
            return abs(extracted - expected_num) / expected_num <= 0.1
        return False
    elif answer_type == "numeric_range":
        # For range like "2-3", check if any number in answer falls within expected
        nums = re.findall(r'\d+', answer)
        expected_nums = [int(x) for x in expected.split('-')]
        if nums and len(expected_nums) == 2:
            extracted_num = int(nums[0])  # take first number
            return expected_nums[0] <= extracted_num <= expected_nums[1]
        return False
    else:  # keyword
        # Check if expected keyword appears in answer (partial match)
        return expected_lower in answer_lower

def get_rag_answer(question):
    docs = ensemble_retriever.invoke(question)
    context = "\n\n".join([doc.page_content for doc in docs])
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a precise fitness and nutrition assistant. Answer the question using the provided context. For numeric questions, give the value with unit. For technique questions, give a short answer."),
        ("human", "Context:\n{context}\n\nQuestion: {question}")
    ])
    chain = prompt | llm
    return chain.invoke({"context": context, "question": question}).content.strip()

print("Running combined evaluation on diet + workout questions...\n")
correct = 0
for question, expected, qtype in test_cases:
    answer = get_rag_answer(question)
    if is_correct(answer, expected, qtype):
        correct += 1
        print(f"✓ {question[:50]} -> {answer[:80]} (correct)")
    else:
        print(f"✗ {question[:50]} -> {answer[:80]} (expected: {expected})")

accuracy = (correct / len(test_cases)) * 100
print(f"\n✅ Overall Accuracy: {accuracy:.1f}% ({correct}/{len(test_cases)})")