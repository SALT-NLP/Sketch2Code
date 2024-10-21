import streamlit as st
import QA
import UserFeedback
import Custom
import os


def clear_session_state():
    for state in st.session_state.keys():
        if state != 'playwright_installed':
            del st.session_state[state]


if "playwright_installed" not in st.session_state:
    st.session_state['playwright_installed'] = False

if not st.session_state.playwright_installed:
    os.system("playwright install")
    os.system("playwright install-deps")
    st.session_state.playwright_installed = True

if "last_selected_page" not in st.session_state:
    st.session_state["last_selected_page"] = "Custom Demo Agent"
    
# Sidebar for page selection
# st.sidebar.title("Navigation")
page = st.sidebar.selectbox("Select an agent:", ["Custom Demo Agent", "Question Asking Agent", "Feedback Following Agent"])

if st.session_state["last_selected_page"] != page:
    clear_session_state()
    st.session_state["last_selected_page"] = page
    st.rerun()

# Load the selected page
if page == "Question Asking Agent":
    QA.run()
elif page == "Feedback Following Agent":
    UserFeedback.run()
elif page == "Custom Demo Agent":
    Custom.run()