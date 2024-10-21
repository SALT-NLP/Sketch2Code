from openai import OpenAI
import streamlit as st
from PIL import Image
import base64
import time
import os
import uuid
from io import BytesIO

from prompt_utils import encode_image, get_agent_system_message, get_agent_user_augmented_user_message, get_direct_system_message, get_direct_text_augmented_user_message, get_user_augmented_message_with_qa_pairs
from utils import extract_html, extract_all_questions, remove_html_comments, extract_title, read_html, extract_text_from_html, rescale_image_loader, upload_to_s3
from screenshot import take_screenshot_from_html, take_screenshot


def run():
    with st.sidebar:
        openai_api_key = st.text_input("OpenAI API Key", key="chatbot_api_key", type="password")
        # user_id = st.text_input("Username", key="username", type="default")

    # Main title and caption
    st.title("💬 Sketch2Code Question Asking Agent")
    st.caption("🚀 An interactive agent that will proactively ask you questions regarding the design details and transform it to a webpage prototype!")
    model = 'gpt-4o'

    # Initialize the session state
    if "public_messages" not in st.session_state:
        st.session_state["public_messages"] = [{"role": "assistant", "type": "text", "content": "Upload a sketch and I will help you create a web prototype!"}]
    
    if "image_uploaded" not in st.session_state:
        st.session_state["image_uploaded"] = False
    
    if "agent_system_message" not in st.session_state:
        st.session_state["agent_system_message"] = get_agent_system_message()
    
    if "direct_system_message" not in st.session_state:
        st.session_state["direct_system_message"] = get_direct_system_message()
        
    if "questions" not in st.session_state:
        st.session_state["questions"] = []
    
    if "question_idx" not in st.session_state:
        st.session_state["question_idx"] = -1
    
    if "qa_pairs" not in st.session_state:
        st.session_state["qa_pairs"] = ''
    
    if 'description' not in st.session_state:
        st.session_state['description'] = ''
    
    if 'sketch_name' not in st.session_state:
        st.session_state['sketch_name'] = ""
    

    # Display the chat messages
    for msg in st.session_state.public_messages:
        if msg['type'] == 'text':
            st.chat_message(msg["role"]).write(msg["content"])
        elif msg['type'] == 'image':
            st.image(msg["content"], caption=msg["role"], use_column_width=True)
        elif msg['type'] == 'html':
            st.download_button(
                label="Download source code",
                key=str(uuid.uuid4()),
                data=msg["content"],
                file_name="webpage.html",
                mime="text/html",
            )

    if "sketch" not in st.session_state:
        # Sketch upload
        uploaded_image = st.file_uploader("Upload an sketch", type=["png", "jpg", "jpeg"])
        if uploaded_image is not None:
            # Display the uploaded image
            st.session_state.sketch_name = uploaded_image.name.split('.')[0]
            image = Image.open(uploaded_image)
            st.image(image, caption="Uploaded Sketch", use_column_width=True)
            st.session_state.public_messages.append({"role": "user", "type": "image", "content": image})
            image_format = uploaded_image.name.split('.')[-1].upper()  # Get the file extension (e.g., 'png', 'jpg')
            if image_format == "JPG":
                image_format = "JPEG"

            # Convert the image to base64 without changing its format
            buffered = BytesIO()
            image.save(buffered, format=image_format)
            base64_image = base64.b64encode(buffered.getvalue()).decode()
            st.session_state['sketch'] = base64_image
            st.session_state.image_uploaded = True     

    # Chat input and response
    placeholder = "Write a description of the intended webpage..." if st.session_state.question_idx == -1 else "Write your answers here..."
    if prompt := st.chat_input(placeholder=placeholder):
        # if not user_id:
        #     st.info("Please enter your Username to continue.")
        #     st.stop()
        
        if not openai_api_key:
            st.info("Please enter your OpenAI API Key to continue.")
            st.stop()
        
        if not st.session_state.image_uploaded:
            st.info("Please upload a sketch to continue.")
            st.stop()
        
        if st.session_state.question_idx >= len(st.session_state.questions):
            st.info("You have reached the maximum rounds of interactions")
            st.stop()
        
        client = OpenAI(api_key=openai_api_key)
        st.session_state.public_messages.append({"role": "user", "type": "text", "content": prompt})
        st.chat_message("user").write(prompt)
        st.chat_message("assistant").write("processing...")
        
        prompt_buffer = BytesIO()
        prompt_buffer.write(prompt.encode())
        prompt_buffer.seek(0)
        
        if st.session_state.question_idx == -1:
            agent_user_message = get_agent_user_augmented_user_message(st.session_state.sketch, prompt)
            st.session_state.description = prompt
            # upload_to_s3(f'QA/{user_id}_{st.session_state.sketch_name}_description.txt', prompt_buffer, 'text/plain')
            agent_messages = [
                {
                    "role": "system",
                    "content": st.session_state.agent_system_message
                },
                {
                    "role": "user",
                    "content": agent_user_message
                }
            ]

            response = client.chat.completions.create(
                model=model,
                messages=agent_messages,
                max_tokens=4096,
            )
            
            agent_resp = response.choices[0].message.content.strip()
            html_response = extract_html(agent_resp)
            
            if html_response:
                # upload_to_s3(f'QA/{user_id}_{st.session_state.sketch_name}_output_0.html', BytesIO(html_response.encode()), 'text/html')
                html_path = os.path.join(os.getcwd(), 'temp.html')
                with open(html_path, 'w') as f:
                        f.write(html_response)
                screenshot_image = take_screenshot(html_path)
                st.image(screenshot_image, caption="Generated Webpage Content", use_column_width=True)
                st.download_button(
                    label="Download source code",
                    key=str(uuid.uuid4()),
                    data=html_response,
                    file_name="webpage.html",
                    mime="text/html",
                )
                st.session_state.question_idx = 0
                st.session_state.public_messages.append({"role": "assistant", "type": "image", "content": screenshot_image})
                st.session_state.public_messages.append({"role": "assistant", "type": "html", "content": html_response})
                msg = "Here is the final generated webpage"
            else:
                st.session_state.questions = extract_all_questions(agent_resp)
                st.session_state.question_idx = 0
                # for i, question in enumerate(st.session_state.questions):
                    # upload_to_s3(f'QA/{user_id}_{st.session_state.sketch_name}_question_{i}.txt', BytesIO(question.encode()), 'text/plain')
                msg = st.session_state.questions[0]
        
        else:
            for _ in range(3):
                st.session_state.qa_pairs += f"\n{st.session_state.questions[st.session_state.question_idx]}\n{prompt}\n"
                # upload_to_s3(f'QA/{user_id}_{st.session_state.sketch_name}_answer_{st.session_state.question_idx}.txt', prompt_buffer, 'text/plain')
                qa_user_message = get_user_augmented_message_with_qa_pairs(st.session_state.sketch, st.session_state.description, st.session_state.qa_pairs)
                response = client.chat.completions.create(
                    model=model,
                    messages=[
                        {
                            "role": "system",
                            "content": st.session_state.direct_system_message,
                        },
                        {
                            "role": "user",
                            "content": qa_user_message,
                        }
                    ],
                    max_tokens=4096,
                    # temperature=0.0,
                )
                
                agent_resp = response.choices[0].message.content.strip()
                html_response = extract_html(agent_resp)
                if html_response:
                    break
                
                time.sleep(1)
            # upload_to_s3(f'QA/{user_id}_{st.session_state.sketch_name}_output_{st.session_state.question_idx}.html', BytesIO(html_response.encode()), 'text/html')
            html_path = os.path.join(os.getcwd(), 'temp.html')
            with open(html_path, 'w') as f:
                    f.write(html_response)
            screenshot_image = take_screenshot(html_path)
            st.image(screenshot_image, caption="Generated Webpage Content", use_column_width=True)
            st.download_button(
                label="Download source code",
                key=str(uuid.uuid4()),
                data=html_response,
                file_name="webpage.html",
                mime="text/html",
            )
            st.session_state.question_idx += 1
            st.session_state.public_messages.append({"role": "assistant", "type": "image", "content": screenshot_image})
            st.session_state.public_messages.append({"role": "assistant", "type": "html", "content": html_response})
            if st.session_state.question_idx < len(st.session_state.questions):
                msg = st.session_state.questions[st.session_state.question_idx]
            else:
                msg = "Here is the final generated webpage"
        
        # Append the chatbot's response to the session state
        st.session_state.public_messages.append({"role": "assistant", "type": "text", "content": msg})
        st.chat_message("assistant").write(msg)
        if st.session_state.question_idx == 0:
            st.rerun()