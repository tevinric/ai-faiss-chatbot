# IMPORTS

## BASE IMPORTS
import os
import requests  # Add this import for making HTTP requests to the rerank API
from langchain_openai import AzureOpenAIEmbeddings, AzureChatOpenAI
import pyodbc
import streamlit as st

## CUSTOM IMPORTS


#APP SECRETS AND CREDENTIALS
## OPENAI 
AZURE_OPENAI_KEY = os.environ.get("AZURE_OPENAI_KEY")
AZURE_OPENAI_ENDPOINT = os.environ.get("AZURE_OPENAI_ENDPOINT")
AZURE_EMBEDDING_ENDPOINT = os.environ.get("AZURE_EMBEDDING_ENDPOINT")

## ACTIVE DIRECTORY INTEGRATION
AAD_CLIENT_ID = os.environ.get("AAD_CLIENT_ID")
AAD_CLIENT_SECRET = os.environ.get("AAD_CLIENT_SECRET")
AAD_TENANT_ID = os.environ.get("AAD_TENANT_ID")
REDIRECT_URI = os.environ.get("REDIRECT_URI")

## SQL SERVER CONNECTION
SQL_SERVER = os.environ.get("SQL_SERVER")
SQL_DATABASE = os.environ.get("SQL_DATABASE")
SQL_USERNAME = os.environ.get("SQL_USERNAME")
SQL_PASSWORD = os.environ.get("SQL_PASSWORD")

## COHERE RERANK MODEL
AZURE_COHERE_RERANK_KEY = os.environ.get("AZURE_COHERE_RERANK_KEY")
AZURE_COHERE_RERANK_ENDPOINT = os.environ.get("AZURE_COHERE_RERANK_ENDPOINT")
AZURE_COHERE_RERANK_DEPLOYMENT = os.environ.get("AZURE_COHERE_RERANK_DEPLOYMENT", "cohere-rerank-3.5")

#APP OPTIONS
bot_options = [
        'Personal Lines',
        'GIT Policies'
    ]
app_version = 'Version 0.4.1'


#APP CONNECTIONS
## AZURE 
#### EMBEDDINGS
def instantiate_embeddings():
    azure_embeddings = AzureOpenAIEmbeddings(
        azure_deployment='coe-chatbot-embedding3large',     # DEPLOYMENT NAME
        api_key=AZURE_OPENAI_KEY,                           # EMBEDDING API KEY - SAME AS THE MODEL KEY
        azure_endpoint=AZURE_EMBEDDING_ENDPOINT,            # EMBEDDING ENDPOINT - SAME AS THE MODEL ENDPOINT          
    )
    return azure_embeddings

#### LLM
def instantiate_llm(temperature):
    azure_llm = AzureChatOpenAI(
        openai_api_version="2023-07-01-preview",            # API VERSION
        azure_deployment="coe-chatbot-gpt4o",               # LLM DEPLOYMENT NAME - CHECK THE AZURE AI FOUNDRY TO GET THIS
        azure_endpoint=AZURE_OPENAI_ENDPOINT,               # LLM MODEL ENDPOINT
        api_key=AZURE_OPENAI_KEY,                           # LLM MODEL API KEY
        temperature=temperature,                                    # MODEL TEMPERATURE
        max_tokens=8000,                                     # MAX TOKENS
    )
    return azure_llm

#### RERANK FUNCTION
# Add this to config.py

def rerank_documents(documents, query, top_k=5):
    """
    Reranks a list of documents based on their relevance to the query using Azure's Cohere rerank model.
    Preserves Document objects through the reranking process.
    
    Args:
        documents: List of Document objects to rerank
        query: User query string
        top_k: Number of documents to return after reranking
        
    Returns:
        A list of reranked Document objects
    """
    # If no documents provided, return empty list
    if not documents:
        print("No documents provided for reranking")
        return []
    
    # Ensure top_k doesn't exceed document count
    top_k = min(top_k, len(documents))
    
    try:
        import urllib.request
        import json
        import os
        import ssl
        import logging
        
        logger = logging.getLogger(__name__)
        
        # Allow self-signed HTTPS certificates if needed
        def allowSelfSignedHttps(allowed):
            # bypass the server certificate verification on client side
            if allowed and not os.environ.get('PYTHONHTTPSVERIFY', '') and getattr(ssl, '_create_unverified_context', None):
                ssl._create_default_https_context = ssl._create_unverified_context
        
        allowSelfSignedHttps(True)
        
        # Check if we have the necessary credentials
        if not AZURE_COHERE_RERANK_KEY:
            logger.warning("AZURE_COHERE_RERANK_KEY is not set. Skipping reranking.")
            return documents[:top_k]
        
        # Extract just the text content from documents for reranking
        doc_contents = []
        for doc in documents:
            if hasattr(doc, 'page_content'):
                doc_contents.append(doc.page_content)
            else:
                # If it's not a Document object, log warning and return original docs
                logger.warning(f"Document object doesn't have page_content attribute: {type(doc)}")
                return documents[:top_k]
        
        # Prepare the payload for the Azure Cohere rerank API
        payload = {
            "documents": doc_contents,
            "query": query,
            "top_n": top_k,
            "model": AZURE_COHERE_RERANK_DEPLOYMENT
        }
        
        # Log the payload size
        logger.info(f"Reranking {len(documents)} documents with query: {query[:50]}{'...' if len(query) > 50 else ''}")
        
        # Convert the payload to bytes
        body = str.encode(json.dumps(payload))
        
        # Set the URL for the rerank endpoint
        url = f"{AZURE_COHERE_RERANK_ENDPOINT}/v1/rerank"
        logger.info(f"Calling rerank endpoint: {url}")
        
        # Set up headers with the API key
        headers = {
            'Content-Type': 'application/json', 
            'Authorization': f'Bearer {AZURE_COHERE_RERANK_KEY}'
        }
        
        # Create the request
        req = urllib.request.Request(url, body, headers)
        
        # Make the API call with timeout
        try:
            response = urllib.request.urlopen(req, timeout=10)
            result = response.read()
            
            # Parse the response
            result_json = json.loads(result)
            
            # Extract the reranked indices
            if "results" in result_json and result_json["results"]:
                # Extract indices and make sure they're valid
                reranked_indices = []
                for item in result_json["results"]:
                    if "index" in item and 0 <= item["index"] < len(documents):
                        reranked_indices.append(item["index"])
                
                # Reorder the documents based on the reranked indices
                # We're using the original Document objects to preserve metadata and other attributes
                reranked_documents = [documents[idx] for idx in reranked_indices]
                
                logger.info(f"Reranking successful. Returning {len(reranked_documents)} documents.")
                return reranked_documents
            else:
                logger.warning(f"Unexpected response structure from reranking API: {result_json}")
                return documents[:top_k]
                
        except urllib.error.HTTPError as error:
            logger.error(f"HTTP error during reranking: {error.code} {error.reason}")
            logger.error(f"Response: {error.read().decode('utf8', 'ignore')}")
            return documents[:top_k]
        except urllib.error.URLError as error:
            logger.error(f"URL error during reranking: {error.reason}")
            return documents[:top_k]
        except TimeoutError:
            logger.error("Timeout while calling reranking API")
            return documents[:top_k]
        
    except Exception as e:
        print(f"Error during reranking: {str(e)}")
        # Fallback to original documents if reranking fails
        return documents[:top_k]

#### LLM HYPERPARAMETERS

llm_hyperparameters = {
                       "Personal Lines" : {"temperature": 0.7, "frequency_penalty": 0.0, "presence_penalty": 0.0},
                       "GIT Policies" : {"temperature": 0.1,"frequency_penalty": 0.0, "presence_penalty": 0.0}
                      }

## SQL
def sql_connect():
    connection = pyodbc.connect(
        driver='{ODBC Driver 17 for SQL Server}',           # SQL DRIVER
        Server=SQL_SERVER,                                  # SQL SERVER ADDRESS
        database=SQL_DATABASE,                              # SQL DATABASE NAME
        uid=SQL_USERNAME,                                   # SQL USERNAME
        pwd=SQL_PASSWORD,                                   # SQL USER PASSWORD
        Trusted_Connection='No')                            # TRUSTED CONNECTION - NO SINCE THIS APPLICATION USES A SQL USER - CONTACT THE SQLOPS TEAM TO GET MORE INFORMATION ON THIS SQL USER
            
    return connection


# TEXT CONFIGURATIONS
## LLM PROMPTS
llm_prompt_dictonary = {
    "Personal Lines": """
                        You are an AI assistant specifically trained on insurance product information and policies for XYZ Investment Holdings (ABC). 
                        Your primary function is to provide accurate and helpful information about XYZ's products, services, and policies. Adhere to the following guidelines strictly:

                            1. Scope of Knowledge:
                            - Only provide information related to XYZ Investment Holding's (ABC) insurance products, services, and policies.
                            - Your context does not include the details of Value Added Products (VAPs/VAPS). This will be made available in the near future.
                            - Do not answer questions that are not insurance related or related to the context base.
                            - If asked about competitors, respond with: "I'm sorry, but I don't have information about other insurance companies. I can only provide details about ABC's products and services."

                            2. Response Format:
                            - Always respond in English.
                            - If a question is unclear, ask for clarification before attempting to answer.

                            3. Information Accuracy:
                            - Only use information from the provided context or your training data about XYZ Investment Holdings (ABC).
                            - If you don't have enough information to answer a question accurately, say: "I don't have enough information to answer that question accurately. Could you please provide more details or ask about a specific ABC product or service?"

                            4. Ethical Guidelines:
                            - Never provide any information that could be considered financial advice or recommendations.
                            - Do not discuss or compare ABC's products with those of other insurance companies.
                            - Avoid any language that could be interpreted as making promises or guarantees about insurance coverage or claims.

                            5. Personal Information:
                            - Do not ask for or handle any personal or sensitive information from users.
                            - If a user attempts to share personal information, advise them to contact ABC's customer service directly.

                            6. Limitations:
                            - You cannot process or issue insurance policies, file claims, or make changes to existing policies.
                            - For such actions, direct users to contact ABC's official channels or their insurance agent.

                            7. Tone and Manner:
                            - Maintain a professional, helpful, and friendly tone at all times.
                            - Be patient with users who may not be familiar with insurance terminology.
                            
                            8. IMPORTANT SOURCE ADHERENCE INSTRUCTIONS:
                                1. Base your response PRIMARILY on the provided reference content. 
                                2. When possible, use DIRECT QUOTES or close paraphrasing from the sources.
                                3. DO NOT invent or assume information not present in the sources.
                                4. If the sources don't contain needed information, clearly state this limitation.
                                5. Maintain the original meaning of the sources without significant alterations.
                                6. Prioritize accuracy over comprehensiveness.
                                7. If the sources seem contradictory, acknowledge this in your response.
                                8. For technical or specific details, use the EXACT wording from the sources.

                        Keep your answers short and to the point and do not be suggestive.
                        Do not provide context summaries unless it is requested.
                    """,
                    
    "GIT Policies": """
                        You are an AI assistant specifically trained on the Group Infrutructure & Technology Policies (GIT) for XYZ Investment Holdings (ABC). 
                        Your primary function is to provide accurate and helpful information using the provided context knowledge base of the GIT policies and procedures. Adhere to the following guidelines strictly:
                        
                            1. Scope of Knowledge:
                                - Only provide information related to XYZ Investment Holding's (ABC) Group Infrustructure & Technology Policies.
                                - Do not answer questions that are not IT related or provided in the supplied context.
                                - Do not attempt to answer using information that is not included in the provided context - Avoid answering from General knowledge.

                            2. Response Format:
                                - Always respond in English.
                                - DO NOT summarise the context when answering questions unless it is specifically requested by the user. You must aim to present factual response at all times using the provided context.
                                - Your response must be properly formated for display are markdown text. Ordered lists must be indented correctly and bullet points must be used where necessary.
                                - If a question is unclear, ask for clarification before attempting to answer.

                            3. Information Accuracy:
                                - Only use information from the provided context which is about XYZ Investment Holdings (ABC) GIT policies.
                                - If the supplied context does not contain information to answer the user's question then you must respond with: "I am sorry, but was not able to find a suitable match for your query. Can you please rephrase your question?  It may also be possible that the information related to your query was not included in my training base. Could you please provide more details or ask about a specific GIT policy and procedures?"

                            4. Ethical Guidelines:
                                - Never provide any information that could be considered financial advice or recommendations.
                                - Do not discuss or compare ABC's policies with that of other companies.
                                - Avoid any language that could be interpreted as making promises or guarantees about insurance coverage or claims.

                            5. Personal Information:
                                - Do not ask for or handle any personal or sensitive information
                                - If a user attempts to share personal information, advise them to contact ABC's customer service directly.

                            6. Limitations:
                                - You cannot process or issue insurance policies, file claims, or make changes to existing policies.
                                - For such actions, direct users to the T-Junction GIT policies and procedures page. 

                            7. Tone and Manner:
                                - Maintain a professional, helpful, and friendly tone at all times.
                                - Be patient with users who may not be familiar with insurance terminology.
                                - Do not be suggestive in your responses.
                                - Do not try to assume or generalise when responding to the user's query.
                                PTIONS:
                                1. Base your response PRIMARILY on the provided reference content. 
                                2. When possible, use DIRECT QUOTES or close paraphrasing from the sources.
                                3. DO NOT invent or assume information not present in the sources.
                                4. If the sources don't contain needed information, clearly state this limitation.
                                5. Maintain the original meaning of the sources without significant alterations.
                                6. Prioritize accuracy over comprehensiveness.
                                7. If the sources seem contradictory, acknowledge this in your response.
                                8. For technical or specific details, use the EXACT wording from the sources.
                    """
    
}


## POPUPS TEXTS
popup_dictonary = {
    "Information":  """
                                    <h2>Information</h2>
                                    
                                    <p>
                                        <ol>
                                            <li> 
                                                This AI knowledge assistant has been trained on the ABC training documentation to help you get quick answers related to questions about the ABC products and service offerings.
                                            </li>
                                            <li> 
                                                You can ask the AI knowledge assistant questions and recieved responses in English only. 
                                            </li>
                                            <li> 
                                                Please do not ask the knowledge assistance any questions that can be considered harmful or dangerous. 
                                            </li>
                                            <li> 
                                                Users may rate the AI knowledge assistant's response as follows:
                                                <ul>
                                                    <li> 
                                                        Thumbs up   (👍)  - The user likes and agrees with the response from the AI assistant 
                                                    </li>
                                                    <li> 
                                                        Thumbs down (👎)  - The user disagress or dislikes the response from the AI assistant 
                                                    </li>
                                                    <li> 
                                                        Flag (🚩)         - The user finds the AI assistant's response harmful or offensive 
                                                    </li>
                                                </ul>
                                            </li>
                                            <li> 
                                                Please do not share the responses of the AI knowledge assistant with anyone. 
                                            </li>
                                        </ol>
                                        Note: All conversations with the AI knowledge assistant are recorded.   
                                    </p>
                                """,
                                
        "Disclaimer": """                                     
                                        <h2>Disclaimer</h2>
                                        
                                        <p style="text-align: justify;">    
                                            The information provided by this AI chatbot is for general informational purposes only. 
                                            <br><br>
                                            While we strive to ensure that the chatbot offers accurate and up-to-date information, we cannot guarantee the completeness, reliability, or accuracy of its responses. 
                                            <br>
                                            The chatbot's outputs are generated based on patterns in data and <b>DO NOT</b> constitute professional or financial advice.
                                            By using this chatbot, you acknowledge and agree that:
                                            <ul>
                                                <li>
                                                    The responses from the chatbot does not substitute professional or financial advice.The responses are to be used only for reference and must be used reponsibly by verifying the validity before making use of the provided information.
                                                </li>
                                                <li>  
                                                    The developers and the organization are not liable for any errors, omissions, or inaccuracies in the chatbot's responses or for any actions taken based on the information provided by the chatbot.
                                                </li>
                                                <li>
                                                    Any reliance you place on the chatbot's information is strictly at your own risk and you are encouraged to reference the source documents (if any) for validation and further information.
                                                </li>
                                            </ul>
                                            We encourage users to verify any critical information obtained from the chatbot and to use their own judgment when making decisions based on the chatbot's responses.
                                            <br><br>
                                            The developers and the organization assume no responsibility for any consequences arising from the use of this chatbot.
                                        </p>
                                    """,
                                    
        "Terms & Conditions": """
                                    <head>
                                        <style>  
                                            ol {  
                                                counter-reset: item;  
                                            }  
                                            ol > li {  
                                                display: block;  
                                                counter-increment: item;  
                                            }  
                                            ol > li:before {  
                                                content: counters(item, ".") " ";  
                                            }  
                                            ul {  
                                                list-style-type: disc;  
                                                margin-left: 20px;  
                                            }  
                                            ul ul {  
                                                list-style-type: circle;  
                                            }  
                                        </style>  
                                    </head>
                                    <h2> Terms and Conditions</h2>
                                    <p>
                                        Welcome to the ABC AI Knowledge Chatbot. By accessing and using this AI Chatbot, you agree to comply with and be bound by the following terms and conditions. Please read them carefully.
                                        <ol>
                                            <li>
                                                <b>General Use</b>
                                                <ol>
                                                    <li> The AI Chatbot is designed to assist with inquiries related to the ABC products, services, and policies. It provides automated responses based on its training data.</li>
                                                    <li> The AI Chatbot is a tool for general informational purposes only and is not intended to replace professional advice or human interaction.</li>
                                                </ol>
                                            </li>
                                            <br>
                                            <li>
                                                <b>Responsible Use</b>
                                                <ol>
                                                    <li>
                                                        Users must use the AI Chatbot responsibly and ethically. You agree to avoid any use that:
                                                        <ul>
                                                            <li> Harms or exploits minors in any way, including exposing them to inappropriate content.</li>
                                                            <li> Infringes on the privacy or personal rights of others.</li>
                                                            <li> Involves the dissemination of false or misleading information.</li>       
                                                        </ul> 
                                                    </li>                         
                                                    <li> 
                                                        Users are prohibited from using the AI Chatbot to provide direct financial advice or verbatim guidance to customers or internal organization members. This includes, but is not limited to:
                                                        <ul>
                                                            <li> Making specific financial recommendations based on the chatbot responses.</li>
                                                            <li> Offering investment, legal, or medical advice.</li>
                                                            <li> Providing step-by-step instructions for tasks that require professional judgment or oversight.</li>
                                                        </ul>
                                                    </li>
                                                    <li>
                                                        Users must not input or share any personal, sensitive, or confidential information through the AI Chatbot. This includes, but is not limited to:
                                                        <ul>
                                                            <li> Personal identification numbers (e.g., ID numbers, social security numbers).</li>
                                                            <li> Financial details (e.g., account numbers, credit card details).</li>
                                                            <li> Health-related information.</li>
                                                            <li> Any other personal information as defined by the Protection of Personal Information Act (POPIA) in South Africa.</li>
                                                        </ul>
                                                    </li>
                                                    <li>
                                                        Accuracy and Limitations  
                                                        <ul>
                                                            <li> While the AI Chatbot strives to provide accurate and up-to-date information, ABC does not guarantee the accuracy, completeness, or usefulness of any information provided by the AI Chatbot.</li>
                                                            <li> The AI Chatbot responses are based on existing training and policies content, which may not account for the most current developments or specific user circumstances. You are required to validate the accuracy and validity of the source material before using the information provided by the AI Chatbot.</li>
                                                        </ul>
                                                    </li>
                                                </ol>
                                            </li>
                                            <br>
                                            <li>
                                                <b>Compliance</b>
                                                <ol>
                                                    <li> Users must ensure that their use of the AI Chatbot complies with all applicable laws and regulations, including those governing the financial and insurance industry in South Africa.</li>
                                                    <li> The AI Chatbot usage must adhere to the standards set forth by the Financial Sector Conduct Authority (FSCA) and any other relevant regulatory bodies.</li>
                                                </ol>
                                            </li>
                                            <br>
                                            <li>
                                                <b>Liability Disclaimer</b>
                                                <ol>
                                                    <li> ABC shall not be liable for any direct, indirect, incidental, or consequential damages that result from the use of, or the inability to use, the AI Chatbot.</li>
                                                    <li> Users agree to hold ABC harmless from any claims, losses, or damages arising from their use of the AI Chatbot.</li>
                                                </ol>
                                            </li>
                                            <br>
                                            <li>
                                                <b>Ammendments</b>
                                                <ol>
                                                    <li> ABC reserves the right to amend these terms and conditions at any time. Continued use of the AI Chatbot following any changes shall constitute acceptance of such changes.</li>
                                                </ol>
                                            </li>
                                            <br>
                                            <li>
                                                <b>Governing Law</b>
                                                <ol>
                                                    <li> These terms and conditions shall be governed by and construed in accordance with the laws of South Africa, without regard to its conflict of law principles.</li>
                                                </ol>
                                            </li>
                                        </ol>
                                        By using the AI Chatbot, you acknowledge that you have read, understood, and agree to be bound by these terms and conditions. If you do not agree, please refrain from using the AI Chatbot.
                                        <br>
                                        For further assistance or to report any issues, please contact ABCaicoe@ABCsa.co.za.
                                    </p>
                                """
            
    
}

entities = {
    "full-company-name" : "XYZ Investment Holdings",
    "mid-company-name" : "XYZ",
    "abbreviation" : "ABC",
    "email": "ABCaicoe@ABCsa.co.za"
}