import requests  
import streamlit as st  
from msal import ConfidentialClientApplication  
import os  
import time  
from streamlit_modal import Modal  
from config import AAD_CLIENT_ID, AAD_CLIENT_SECRET, AAD_TENANT_ID, REDIRECT_URI  
  
def initialize_app():  
    client_id = AAD_CLIENT_ID  
    tenant_id = AAD_TENANT_ID  
    client_secret = AAD_CLIENT_SECRET  
    authority_url = f"https://login.microsoftonline.com/{tenant_id}"  
    return ConfidentialClientApplication(client_id, authority=authority_url, client_credential=client_secret)  
  
def acquire_access_token(app, code, scopes, redirect_uri):  
    return app.acquire_token_by_authorization_code(code, scopes=scopes, redirect_uri=redirect_uri)  
  
def fetch_user_data(access_token):  
    headers = {"Authorization": f"Bearer {access_token}"}  
    graph_api_endpoint = "https://graph.microsoft.com/v1.0/me"  
    response = requests.get(graph_api_endpoint, headers=headers)  
    return response.json()  
  
def authentication_process(app):  
    scopes = ["User.Read"]  
    redirect_uri = REDIRECT_URI  
    current_url = str(st.query_params)
  
    auth_url = app.get_authorization_request_url(  
        scopes,  
        redirect_uri=redirect_uri,  
        state=current_url  
    )  
  
    modal = Modal("Disclaimer", key="disclaimer_modal")  
  
    if modal.is_open():  
        with modal.container():  
            # st.markdown("### Disclaimer")  
            st.write("Please read and accept the disclaimer to proceed with the authentication process.") 
            st.write("""
                     The information provided by this AI chatbot is for general informational purposes only. While we strive to ensure that the chatbot offers accurate and up-to-date information, we cannot guarantee the completeness, reliability, or accuracy of its responses. The chatbot's outputs are generated based on patterns in data and DO NOT constitute professional or financial advice.

                    By using this chatbot, you acknowledge and agree that:

                    The responses from the chatbot does not substitute professional or financial advice.The responses are to be used only for reference and must be used reponsibly by verifying the validity before making use of the provided information.
                    The developers and the organization are not liable for any errors, omissions, or inaccuracies in the chatbot's responses or for any actions taken based on the information provided by the chatbot.
                    Any reliance you place on the chatbot's information is strictly at your own risk and you are encouraged to reference the source documents (if any) for validation and further information.

                    We encourage users to verify any critical information obtained from the chatbot and to use their own judgment when making decisions based on the chatbot's responses. The developers and the organization assume no responsibility for any consequences arising from the use of this chatbot.
                     """) 
            if st.button("Accept"):  
                st.session_state["disclaimer_accepted"] = True  
                modal.close()  
                st.rerun()  
            if st.button("Decline"):  
                st.session_state["disclaimer_accepted"] = False  
                modal.close()  
                st.rerun()  
  
    if "disclaimer_accepted" in st.session_state:  
        if st.session_state["disclaimer_accepted"]:  
            st.markdown(  
                f"""  
                <div style='text-align: center; margin: 20px;'>  
                    <a href='{auth_url}' target='_self'>  
                        <button style='  
                            background-color: #0078D4;  
                            color: white;  
                            padding: 10px 20px;  
                            border: none;  
                            border-radius: 4px;  
                            cursor: pointer;  
                            font-size: 16px;'>  
                            Sign in with Microsoft  
                        </button>  
                    </a>  
                </div>  
                """,  
                unsafe_allow_html=True  
            )  
        else:  
            st.error("You must accept the disclaimer to proceed using the application")  
            time.sleep(3)  
            del st.session_state["disclaimer_accepted"]  
            st.rerun()  
    else:  
        
        col1, col2, col3, col4, col5 = st.columns(5)
        
        with col1:
            st.write(' ')
            
        with col2:
            st.write(' ')
            
        with col3:
            if st.button("Let's Go"):  
                modal.open()  
            
        with col4:
            st.write(' ')
            
        with col5:
            st.write(' ')
            
    if "code" in st.query_params:
        st.session_state["auth_code"] = st.query_params["code"] 
        token_result = acquire_access_token(app, st.session_state.auth_code, scopes, redirect_uri)  
        if "access_token" in token_result:  
            user_data = fetch_user_data(token_result["access_token"])  
            return user_data  
  
def login_ui():  
   
    st.write(" ")  
    st.write(" ")  
    st.write(" ")  
    st.write(" ")  
    col1, col2, col3 = st.columns(3)  
  
    with col1:  
        st.write(' ')  
    with col2:  
        st.image(os.path.join(os.getcwd(), "assets", "company-logo.png"), width=300)  
    with col3:  
        st.write(' ')  
  
    st.write(" ")  
    st.markdown("<h1 style='text-align: center; font-size: 20px;'>XYZ Knowledge Chatbot Authentication</h1>", unsafe_allow_html=True)  
    st.markdown("<p style='text-align: center;'>You must authenticate using your XYZ account to access the Chatbot</p>", unsafe_allow_html=True)  
  
    app = initialize_app()  
    user_data = authentication_process(app)  
    st.write(" ")  
    st.write(" ")  
    st.info("By accessing and using this chatbot, you acknowledge and consent that all conversations and messages exchanged will be recorded and stored. These records help improve our services and ensure quality support. Your continued use of this chatbot constitutes acceptance of this data collection practice.")  
      
    if user_data:  
        st.write("Welcome, ", user_data.get("displayName"))  
        st.session_state["authenticated"] = True  
        st.session_state["display_name"] = user_data.get("displayName")  
        st.session_state["user_email"] = user_data.get("mail")  
        st.rerun()  
  
# Uncomment the following lines to run the app  
# if __name__ == "__main__":  
#     login_ui()  