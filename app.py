import os, dotenv
import streamlit as st
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.utilities import ArxivAPIWrapper, WikipediaAPIWrapper
from langchain_community.tools import ArxivQueryRun, WikipediaQueryRun, DuckDuckGoSearchRun
from langchain.agents import create_react_agent
from langchain.agents import AgentExecutor
from langchain_community.callbacks.streamlit import StreamlitCallbackHandler
dotenv.load_dotenv()

## Wikipedia Tool
wiki_wrapper = WikipediaAPIWrapper(top_k_results=1, doc_content_chars_max=500)
wiki_tool = WikipediaQueryRun(api_wrapper=wiki_wrapper)
# Arxiv Tool
arxiv_wrapper = ArxivAPIWrapper(top_k_results=1, doc_content_chars_max=500)
arxiv_tool = ArxivQueryRun(api_wrapper=arxiv_wrapper)
# DuckDuckGo Search Tool
search = DuckDuckGoSearchRun(name="Internet Search")


prompt = ChatPromptTemplate.from_template("""
Answer the following user questions as best you can. Use the available tools to find the answer.
You have access to the following tools:\n
{tools}\n\n
To use a tool, please use the following format:
```
Thought: Do I need to use a tool? Yes
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
```
If one tool doesn't give the relavant information, use another tool.
When you have a response to say to the Human, or if you do not need to use a tool, you MUST use the format:
                                              
```
Thought: Do I need to use a tool? No
Final Answer: [your response here]
```
Begin!
                                              
Previous conversation history:
{chat_history}
New input: {input}

{agent_scratchpad}
""")

def create_groq_agent(llm, api_key, tools, question, chat_history):
    agent = create_react_agent(llm, tools, prompt)
    
    agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True, handle_parsing_errors=True, max_iterations=7)
    st_callback = StreamlitCallbackHandler(st.container(), expand_new_thoughts=True)
    
    response = agent_executor.invoke({"input":question, "chat_history":chat_history}, {"callbacks": [st_callback]})
    return response['output']
    
# Streamlit Code
st.set_page_config(page_icon=":mag:", page_title="Tools & Agent")
st.title(":green[Langchain] Search Agent")

with st.sidebar:
    with st.popover("Add Groq API Key", use_container_width=True):
        api_key = st.text_input("Get Your Groq API Key [Here](https://console.groq.com/keys)", type="password")
    st.divider()
    st.markdown("<h1 style='text-align: center; font-size: 30px;'>About the App✨</h1>", unsafe_allow_html=True)
    st.write("""Hi there! This is a langchain search agent app. First, you have to
             introduce your Groq API key. Then type your question and hit Enter, 
             the assistant will step by step retrieve the information relevant to
             your question from Wikipedia, Arxiv and DuckDuckGo Search and then it'll
             answer your question based on that information.""")

if "messages" not in st.session_state:
    st.session_state["messages"] = [
        {"role": "assistant", "content": "Hi there! How can I help you today?"}
    ]

for message in st.session_state.messages:
    if message["role"] == "user":
        st.chat_message("user", avatar="boss.png").write(message['content'])
    else:
        st.chat_message("assistant", avatar="robot.png").write(message['content'])
    
if api_key:
    if question := st.chat_input("What is Generative AI?"):
        st.session_state.messages.append({"role": "user", "content": question})
        st.chat_message("user", avatar="boss.png").write(question)

        llm = ChatGroq(model="llama-3.1-70b-versatile", api_key=api_key)
        tools = [wiki_tool, arxiv_tool, search]

        try:
            with st.chat_message("assistant", avatar="robot.png"):
                response = create_groq_agent(llm, api_key, tools, question, st.session_state.messages)
                st.markdown(response)
                
            st.session_state.messages.append({"role": "assistant", "content": response})
        except Exception as e:
            st.error(f"An error occurred: {e}")
else:
    st.info("Please enter your Groq API key in the sidebar to proceed.")
