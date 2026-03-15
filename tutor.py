import os
import streamlit as st
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()

groq_api_key = os.getenv("GROQ_API_KEY")

st.set_page_config(page_title="Adaptive AI Tutor", page_icon="🧑‍🏫", layout="centered")

with st.sidebar:
    st.header("ℹ️ About")
    st.markdown("This AI Tutor is built using **LangChain** and **Groq (Llama 3)**.")
    st.markdown("It analyzes your code and provides Socratic hints to help you find the solution yourself, rather than just giving you the answer.")

st.title("🧑‍🏫 Adaptive Programming Tutor")
st.markdown("""
**Welcome to your AI Tutor!** Paste your code and tell me what's wrong. I won't give you the exact answer, but I'll guide you to figure it out yourself! 🚀
""")

col1, col2 = st.columns(2)
with col1:
    language = st.selectbox("Programming Language:", ["Python", "JavaScript", "C++", "Java", "Other"])

student_code = st.text_area("💻 Paste your code here:", height=250, placeholder="def my_function():\n    pass")
student_question = st.text_input("❓ What is the error, or what are you trying to do?", placeholder="I'm getting an IndexError, but I don't know why...")

system_prompt = """
You are an expert, Socratic programming tutor for {language}. Your primary goal is to help the student learn and think critically.
CRITICAL RULES:
1. NEVER write the final code solution for the student.
2. NEVER give direct answers or fix the code for them.
3. Analyze the student's code and identify the logical flaw or syntax error.
4. Provide a conceptual hint or ask a guiding question to lead the student to the correct answer.
5. Keep your responses encouraging, concise, and focused on the specific problem.
6. Format your response cleanly using markdown.
"""

prompt_template = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    ("user", "Student Code:\n{code}\n\nStudent Question/Error:\n{question}")
])

if st.button("💡 Get a Hint", use_container_width=True):
    if not groq_api_key:
        st.error("⚠️ Configuration Error: GROQ_API_KEY not found in environment variables. Please check your .env file.")
    elif not student_code or not student_question:
        st.warning("⚠️ Please provide both your code and your question.")
    else:
        try:
            with st.spinner("Analyzing your code logically..."):
                llm = ChatGroq(
                    api_key=groq_api_key,
                    model_name="openai/gpt-oss-120b", 
                    temperature=0.2 
                )
                
                chain = prompt_template | llm
                
                response = chain.invoke({
                    "language": language,
                    "code": student_code, 
                    "question": student_question
                })
                
                st.success("Analysis Complete!")
                st.markdown("### 🤖 Tutor's Hint:")
                st.info(response.content)
        except Exception as e:
            st.error(f"An error occurred: {e}")