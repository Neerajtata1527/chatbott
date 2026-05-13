import os
import csv
import pandas as pd
from langchain_community.document_loaders import DataFrameLoader, PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings  # CHANGED
from langchain_community.vectorstores import Chroma

# ==================== PATHS ====================
DIET_CSV_PATH = "knowledge_base/diet/diet.csv"
DIET_TIPS_CSV_PATH = "knowledge_base/diet/diet_tips_dataset.csv"
WORKOUT_TIPS_CSV_PATH = "knowledge_base/workout/workout.csv"
WORKOUT_PDF_DIR = "knowledge_base/workout/"

DIET_DB_DIR = "./chroma_db_diet"
WORKOUT_DB_DIR = "./chroma_db_workout"

CHUNK_SIZE = 300
CHUNK_OVERLAP = 50

# ==================== ROBUST CSV LOADER ====================
def load_robust_csv(filepath):
    rows = []
    with open(filepath, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        header = next(reader)
        expected = len(header)
        for line_num, row in enumerate(reader, start=2):
            if len(row) > expected:
                row = row[:expected-1] + [','.join(row[expected-1:])]
            elif len(row) < expected:
                row += [''] * (expected - len(row))
            rows.append(row)
    return pd.DataFrame(rows, columns=header)

# ==================== DIET FORMATTING ====================
def format_diet_row(row):
    return (f"Dish: {row['food_name']} – "
            f"Calories: {row['energy_kcal']} kcal, "
            f"Protein: {row['protein_g']}g, "
            f"Fats: {row['fat_g']}g, "
            f"Carbs: {row['carbohydrate_g']}g, "
            f"Fiber: {row['fiber_g']}g, "
            f"Iron: {row['iron_mg']}mg, "
            f"Vitamin C: {row['vitamin_c_mg']}mg, "
            f"Calcium: {row['calcium_mg']}mg, "
            f"Sodium: {row['sodium_mg']}mg, "
            f"Potassium: {row['potassium_mg']}mg, "
            f"Vitamin A: {row['vitamin_a_mcg']}mcg, "
            f"Folate: {row['folate_mcg']}mcg, "
            f"Zinc: {row['zinc_mg']}mg, "
            f"Region: {row['region']}, "
            f"State: {row['preparation_state']}. "
            f"Info: {row['rag_description']}")

def format_tip_row(row):
    return (f"Tip: {row['Tip_Title']}\n"
            f"Category: {row['Category']} - {row['Subcategory']}\n"
            f"Description: {row['Tip_Description']}\n"
            f"Key Benefit: {row['Key_Benefit']}\n"
            f"Foods involved: {row['Foods_Involved']}\n"
            f"Meal Timing: {row['Meal_Timing']}\n"
            f"Scientific Backing: {row['Scientific_Backing']}")

# ==================== EMBEDDING MODEL ====================  # CHANGED
# Replaced HuggingFaceEmbeddings (all-MiniLM-L6-v2, ~400MB PyTorch)
# with FastEmbedEmbeddings (BAAI/bge-small-en-v1.5, ~60MB ONNX)
# Everything else in this file is identical to the original.
print("Loading embedding model...")
embeddings = FastEmbedEmbeddings(model_name="BAAI/bge-small-en-v1.5")  # CHANGED
print("Embedding model ready.\n")

splitter = RecursiveCharacterTextSplitter(
    chunk_size=CHUNK_SIZE,
    chunk_overlap=CHUNK_OVERLAP
)

# ==================== DIET KNOWLEDGE BASE ====================
print("📥 Ingesting diet database...")
dfs = []

if os.path.exists(DIET_CSV_PATH):
    df_food = load_robust_csv(DIET_CSV_PATH)
    df_food['text'] = df_food.apply(format_diet_row, axis=1)
    dfs.append(df_food)
    print(f"  Loaded {len(df_food)} food items.")
else:
    print(f"  Warning: Diet CSV not found at {DIET_CSV_PATH}")

if os.path.exists(DIET_TIPS_CSV_PATH):
    df_tips = load_robust_csv(DIET_TIPS_CSV_PATH)
    df_tips['text'] = df_tips.apply(format_tip_row, axis=1)
    dfs.append(df_tips)
    print(f"  Loaded {len(df_tips)} diet tips.")
else:
    print(f"  Warning: Diet tips CSV not found at {DIET_TIPS_CSV_PATH}")

if not dfs:
    raise FileNotFoundError("No diet data files found.")

df_combined = pd.concat(dfs, ignore_index=True)
loader = DataFrameLoader(df_combined, page_content_column="text")
diet_docs = loader.load()

diet_chunks = splitter.split_documents(diet_docs)

diet_vectorstore = Chroma.from_documents(
    diet_chunks,
    embeddings,
    persist_directory=DIET_DB_DIR
)
diet_vectorstore.persist()
print(f"✅ Diet DB ready with {len(diet_chunks)} chunks.\n")

# ==================== WORKOUT KNOWLEDGE BASE ====================
print("📥 Ingesting workout database...")
all_workout_docs = []

if os.path.exists(WORKOUT_TIPS_CSV_PATH):
    df_workout = load_robust_csv(WORKOUT_TIPS_CSV_PATH)
    df_workout['text'] = df_workout.apply(
        lambda row: (
            f"Workout Tip: {row['Tip_Title']}\n"
            f"Description: {row['Tip_Description']}\n"
            f"Equipment: {row['Equipment_Needed']} | "
            f"Difficulty: {row['Difficulty_Level']}\n"
            f"Target Muscles: {row['Muscle_Groups_Targeted']} | "
            f"Scientific Backing: {row['Scientific_Backing']}"
        ),
        axis=1
    )
    loader = DataFrameLoader(df_workout, page_content_column="text")
    all_workout_docs.extend(loader.load())
    print(f"  Loaded {len(df_workout)} workout tips.")
else:
    print(f"  Warning: Workout tips CSV not found at {WORKOUT_TIPS_CSV_PATH}")

if os.path.exists(WORKOUT_PDF_DIR):
    pdf_files = [f for f in os.listdir(WORKOUT_PDF_DIR) if f.endswith('.pdf')]
    if pdf_files:
        for pdf_file in pdf_files:
            pdf_path = os.path.join(WORKOUT_PDF_DIR, pdf_file)
            loader = PyPDFLoader(pdf_path)
            all_workout_docs.extend(loader.load())
            print(f"  Loaded PDF: {pdf_file}")
    else:
        print(f"  No PDF files found in {WORKOUT_PDF_DIR}")
else:
    print(f"  Warning: Workout PDF directory not found at {WORKOUT_PDF_DIR}")

if not all_workout_docs:
    raise FileNotFoundError("No workout data found.")

workout_chunks = splitter.split_documents(all_workout_docs)

workout_vectorstore = Chroma.from_documents(
    workout_chunks,
    embeddings,
    persist_directory=WORKOUT_DB_DIR
)
workout_vectorstore.persist()
print(f"✅ Workout DB ready with {len(workout_chunks)} chunks.\n")
print("🚀 Ingestion complete!")