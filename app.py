from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
from chatbot import get_response
import os
from langchain_groq import ChatGroq  

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/chat", methods=["POST"])
def chat():
    try:
        data = request.get_json()
        user_input = data["message"]
        response = get_response(user_input)
        return jsonify({"response": response})
    except Exception as e:
        import traceback
        traceback.print_exc()  # This will print the full error in your terminal
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)
