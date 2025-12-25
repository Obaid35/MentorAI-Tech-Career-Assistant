import gradio as gr
import os
import requests
import tempfile
from dotenv import load_dotenv

load_dotenv()

GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"
DEFAULT_MODEL = "llama-3.3-70b-versatile"


SYSTEM_PROMPT = """
You are MentorAI, a highly experienced and supportive AI Career Mentor specializing in tech careers.
Provide structured, practical and realistic guidance.
"""

def get_groq_models():
    url = "https://api.groq.com/openai/v1/models"
    headers = {
       "Authorization": f"Bearer {os.environ.get('GROQ_API_KEY')}",
       "Content-Type": "application/json"
    }
    r = requests.get(url, headers=headers)
    if r.status_code == 200:
       return [m["id"] for m in r.json().get("data", [])]
    return []

available_models = get_groq_models()
print("Available Models:", available_models)


def build_context(level, goal, features, tone):
    context = f"""
User Level: {level}
Career Goal: {goal}
Tone: {tone}

Requested Outputs:
"""
    for f in features:
        context += f"- {f}\n"

    context += """
If interview requested: give 3-6 Q/A.
If roadmap requested: give timeline phases.
If resume guidance requested: give practical improvements.
If portfolio guidance requested: suggest project ideas.
"""

    return context


def query_groq(message, history, level, goal, features, tone, creativity, model):
    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json"
    }

    context_block = build_context(level, goal, features, tone)

    messages = [{"role": "system", "content": SYSTEM_PROMPT + "\n" + context_block}]

    for msg in history:
        messages.append(msg)

    messages.append({"role": "user", "content": message})

    payload = {
        "model": model,
        "messages": messages,
        "temperature": creativity
    }

    response = requests.post(GROQ_API_URL, headers=headers, json=payload)

    if response.status_code == 200:
        return response.json()["choices"][0]["message"]["content"]
    else:
        return f"Error {response.status_code}: {response.text}"


def respond(message, history, level, goal, features, tone, creativity, model, status):
    if not features:
        bot_reply = "Please select at least one feature."
        history.append({"role": "user", "content": message})
        history.append({"role": "assistant", "content": bot_reply})
        return "", history, gr.update(visible=False)

    status = gr.update(value="⏳ MentorAI is typing...", visible=True)

    bot_reply = query_groq(message, history, level, goal, features, tone, creativity, model)

    history.append({"role": "user", "content": message})
    history.append({"role": "assistant", "content": bot_reply})

    return "", history, gr.update(visible=False)


def save_chat(history):
    if not history:
        return None

    text = "MentorAI Chat History\n\n"
    for msg in history:
        role = "User" if msg["role"] == "user" else "MentorAI"
        text += f"{role}: {msg['content']}\n\n"

    temp = tempfile.NamedTemporaryFile(delete=False, suffix=".txt")
    with open(temp.name, "w", encoding="utf-8") as f:
        f.write(text)

    return temp.name


with gr.Blocks(theme=gr.themes.Soft()) as demo:

    gr.Markdown("""
<center>
<h1>🚀 MentorAI</h1>
<h3>Your Smart Tech Career Mentor</h3>
<p>Internships • Jobs • Skills • Interviews • Portfolio</p>
</center>
""")

    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### ⚙️ Settings")

            level = gr.Dropdown(["Beginner", "Intermediate", "Advanced"], value="Beginner", label="Select Level")

            goal = gr.Dropdown(
                ["Internship", "Job", "Skill Learning", "Interview Prep"],
                value="Job",
                label="Career Goal"
            )

            features = gr.CheckboxGroup(
                ["Learning Roadmap", "Resume Tips", "Interview Q/A", "Portfolio Guidance"],
                label="Generate",
            )

            tone = gr.Dropdown(["Friendly", "Professional", "Mentor-like"], value="Professional", label="Tone Style")
            preferred_default = "llama-3.1-70b-versatile"

            model = gr.Dropdown(
                choices=available_models if available_models else [
                "llama-3.1-8b-instant",
                "llama-3.1-70b-versatile",
                "mixtral-8x7b-32768",
                "gemma-7b-it"
    ],
                value=(preferred_default if preferred_default in available_models else "llama-3.1-8b-instant"),
                label="Groq Model"
)


            creativity = gr.Slider(0.1, 1.0, value=0.7, step=0.1, label="Creativity")

            gr.Markdown("### ⚡ Quick Start Examples")

            example1 = gr.Button("📌 I am a beginner CS student, help me with a learning roadmap")
            example2 = gr.Button("📌 How can I prepare for internships?")
            example3 = gr.Button("📌 What projects should I add to my portfolio?")
            example4 = gr.Button("📌 Give me interview questions for a software developer role")

        with gr.Column(scale=2):
            gr.Markdown("### 💬 Chat")

            chatbot = gr.Chatbot()
            message = gr.Textbox(label="Ask something...")
            status_text = gr.Markdown(visible=False)

            state = gr.State([])

            with gr.Row():
                send = gr.Button("Send", variant="primary")
                clear = gr.Button("Clear")
                save = gr.Button("Download Chat")

            send.click(
                respond,
                [message, state, level, goal, features, tone, creativity, model, status_text],
                [message, chatbot, status_text]
)

            message.submit(
            respond,
            [message, state, level, goal, features, tone, creativity, model, status_text],
            [message, chatbot, status_text]
)

            clear.click(lambda: ([], []), None, [chatbot, state])


            saved = gr.File(label="Download File")
            save.click(save_chat, state, saved)

            # Example prompt functionality
            example1.click(lambda: "I am a beginner CS student, give me a roadmap", None, message)
            example2.click(lambda: "How can I prepare for internships?", None, message)
            example3.click(lambda: "Suggest portfolio projects based on industry demand", None, message)
            example4.click(lambda: "Give me interview questions and answers for software developer", None, message)


demo.launch()
