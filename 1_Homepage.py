import streamlit as st
from dotenv import load_dotenv, find_dotenv
from langchain import PromptTemplate, LLMChain
from langchain.llms import OpenAI
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory
from langchain.chat_models import ChatOpenAI
import os
import pdfplumber

st.set_page_config(
    page_title="Neuro-linguistic Recommendation Engine",
    page_icon="👨‍💻",
    layout="wide",
    initial_sidebar_state="collapsed"
    
)

st.subheader('Neuro-linguistic Recommendation Engine - Credit Card Plans Chatbot')

os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_ENDPOINT"] = "https://api.smith.langchain.com"
os.environ["LANGCHAIN_PROJECT"] = "Neuro-linguistic Recommendation Engine"

llm = ChatOpenAI(model_name = "gpt-4o",temperature=0.2)

# uploaded_file = st.file_uploader("", type=['pdf'])


# st.session_state.pdf_context = ""
# if uploaded_file is not None:
#     with pdfplumber.open(uploaded_file) as pdf:
#         total_pages = len(pdf.pages)
#         full_text = ""
#         for page in pdf.pages:
#             text = page.extract_text()
#             if text:
#                 full_text += text + "\n" 
#         st.session_state.pdf_context = full_text

template = """You are an banking Sales Representative who will have conversation with potential customers to understand their spending habits and recommend bank’s credit cards available in a convincing manner.
Following context has information on available credit card products with their features. Answer the users question based on the product information.
Be proactive and ask questions from the user to understand life style, spending habits, gather information and see which product best fits the user and answer the questions very convincingly.
Limit your answer to 60 words or less at a time.
Your Name is Alex. introduce yourself as a Citi Bank’s Intelligent Agent and ask the customer's name at the first interaction. After customer’s reply, ask questions from the customer to understand passions, spending habits and gather information.
If customer is not convinced, elaborate the advantages of the products and how they outweigh the disadvantages and convince the customer to apply for the card. Do not offer sales features unless user ask for the offer.
Only reply as the sales representative and do not write the responses from the customer.
Answer only based on the topic of credit cards and If the customer questions is outside the context, just say that you don't know and steer the conversation back to the topic you know. Don't give any answer outside the context of credit cards.


Context: {pdf_context}

Current conversation:
{history}
Human: {input}
AI Assistant:"""

def generate_the_response(prompt, memory, pdf_context):
    # PROMPT = PromptTemplate(input_variables=["history", "input", "pdf_context"], template=template)
    PROMPT =  PromptTemplate.from_template(template).partial(pdf_context=pdf_context)
    llm_chain = ConversationChain(
        prompt=PROMPT,
        llm=llm,
        verbose=True,
        memory=memory,
    )
    result = llm_chain.predict(input=prompt)
    return result


st.markdown(
    """
<style>
    [data-testid="collapsedControl"] {
        display: none
    }
    
    #neuro-linguistic-recommendation-engine-healthcare-plans-chatbot {
        font-size: 22px;
        text-align: center;
    }
</style>
""",
    unsafe_allow_html=True,
)

# st.title("Chat with the Source Code with LLMs")


main_context = """Following is a line of credit card products and their features.

Product 1: Citi AAAdvantage Card
Product 1 Features: Credit card is offered in collaboration with American Airlines. Best suited for frequent travelers and explorers. Key benefit of the card is the miles accumulated with every purchase of the card. Many travel related benefits such as air ticket discounts, airport lounges, hotels and car rentals are offered through the AAAdvantage card. There is an annual fee of $20. No cashback rewards are present for the card. No initial reward present as well. New users can earn 15,000 American Airlines AAdvantage® bonus miles after making $500 in purchases within the first 3 months of account opening. Save 25% on inflight food and beverage purchases when you use your card on American Airlines flights. 
Product 1 Sales Features: Sales person can offer one year waive off for annual fee.

Product 2: Citi Custom Cash Card
Product 2 features: Earn 5% cash back on purchases in your top eligible spend category each billing cycle, up to the first $500 spent.1% cash back thereafter. Also, earn unlimited 1% cash back on all other purchases. card is designed for frequent grocery shoppers. Discounts are present in many supermarkets. No Annual fee. No intro APR is present. New users can earn $200 cash back after you spend $1,500 on purchases in the first 6 months of account opening. 

Product 3: Citi Diamond Preferred Card
Product 3 features: 0% intro APR of 21 months on balance transfers. Also a low intro APR for 12 months on user specified purchases. There is no annual fee. card designed to improve the credit score of users.Specialy there is no late fees.There is a balance transfer fee of either $5 or 5% of the amount of each transfer, whichever is greater

Product 4: Citi Double Cash Card
Product 4 features: Earn $200 cash back after spend $1,500 on purchases in the first 6 months of account opening. This bonus offer will be fulfilled as 20,000 ThankYou Points, which can be redeemed for $200 cash back.Earn 2% on every purchase with unlimited 1% cash back when you buy, plus an additional 1% as you pay for those purchases. To earn cash back, pay at least the minimum due on time. Plus, for a limited time, earn 5% total cash back on hotel, car rentals and attractions booked on the Citi Travel portal through 12/31/24.0% intro APR on Balance Transfers for 18 months. After that, the variable APR will be 19.24% - 29.24%, based on your creditworthiness. Balance Transfers do not earn cash back. Intro APR does not apply to purchases.There is an intro balance transfer fee of 3% of each transfer (minimum $5) completed within the first 4 months of account opening. After that, your fee will be 5% of each transfer (minimum $5). No Annual fee
Product 4 Sales Features: Sales person can offer late fee waive for first 3 months

Product 5: Costco Anywhere Visa Card by Citi
Product 5 features: This is designed exclusively for Costco members.4% cash back on eligible gas and EV charging purchases for the first $7,000 per year and then 1% thereafter.3% cash back on restaurants and eligible travel purchases.2% cash back on all other purchases from Costco and Costco.com. 1% cash back on all other purchases.No annual fee with your paid Costco membership and enjoy no foreign transaction fees on purchases.Receive an annual credit card reward certificate, which is redeemable for cash or merchandise at U.S. Costco warehouses, including Puerto Rico.
"""


# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []
    st.session_state.messages_memory = ConversationBufferMemory(memory_key="history")


# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.text(message["content"].replace('*', ''))


# React to user input
if prompt := st.chat_input("How can i help you?"):
    # Display user message in chat message container
    st.chat_message("user").text(prompt)
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})

    messages_memory = st.session_state.messages_memory
    # pdf_context = st.session_state.pdf_context
    pdf_context = main_context
    response = generate_the_response(prompt, messages_memory, pdf_context)
    # Display assistant response in chat message container
    
    
    with st.chat_message("assistant"):
        st.text(response.replace('*', ''))
            
    st.session_state.messages.append({"role": "assistant", "content": response})