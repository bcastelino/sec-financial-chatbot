import streamlit as st
from datetime import datetime
from analyzer import SECChatbot
from chromadb.config import Settings
import chromadb
from config import Config
import base64
import re

def get_svg_base64(svg_path):
    with open(svg_path, "rb") as f:
        svg_bytes = f.read()
    return base64.b64encode(svg_bytes).decode()

def build_debug_context(chatbot, user_query):
    # ... (same as before, see previous message for full function) ...
    # Copy the context-building logic from your debug script here
    # Return the context string
    pass  # Replace with actual function body

# --- ChromaDB Connection Status ---
try:
    chroma_client = chromadb.PersistentClient(
        path=Config.CHROMA_PERSIST_DIRECTORY,
        settings=Settings(anonymized_telemetry=False)
    )
    collection = chroma_client.get_collection(Config.COLLECTION_NAME)
    doc_count = collection.count()
    companies = set()
    if doc_count > 0:
        all_metadata = collection.get(include=["metadatas"])
        for metadata in all_metadata["metadatas"]:
            if metadata and "company_name" in metadata:
                companies.add(metadata["company_name"])
    chroma_status = True
except Exception as e:
    chroma_status = False
    doc_count = 0
    companies = set()
    chroma_error = str(e)

# --- Sidebar ---
with st.sidebar:
    svg_base64 = get_svg_base64("icon.svg")
    st.markdown(
        f"""
        <div style="display: flex; align-items: center; gap: 0.7em; margin-bottom: 1.5em;">
            <img src="data:image/svg+xml;base64,{svg_base64}" width="38" style="margin-bottom:0;"/>
            <span style="font-size: 1.6em; font-weight: 700; color: #e0f11f;">SEC Data Assistant</span>
        </div>
        """,
        unsafe_allow_html=True
    )
    
    # New Conversation button
    if st.button("+ New Conversation"):
        if "history" in st.session_state:
            st.session_state.history = []
        if "last_error" in st.session_state:
            st.session_state.last_error = None
        st.rerun()
    
    st.markdown("### Dataset Coverage")
    st.markdown("<span style='color:#bdbdbd; font-weight:600;'>10K Filing Range: 2020 - 2024</span>", unsafe_allow_html=True)
    st.markdown("<span style='color:#bdbdbd; font-weight:600;'>Filings Types: Text & XBRL formats</span>", unsafe_allow_html=True)

    # Dynamically fetch companies from the vector store
    chatbot = SECChatbot()
    available_companies = chatbot.get_available_companies()
    st.markdown("### Companies")
    for company_name in available_companies:
        # Find company data from Config
        company_data = None
        for company in Config.TOP_COMPANIES:
            if company["name"] == company_name:
                company_data = company
                break

        if company_data:
            ticker = company_data["ticker"]
            name = company_data["name"]
            sector = company_data.get("sector", "Technology")
            icon = company_data.get("icon", "🏢")
            st.markdown(
                f"**{icon} {name}**  <span style='color:#bdbdbd;'>({ticker})</span><br>"
                f"<span style='color:#bdbdbd;font-size:0.98em;'>{sector}</span>",
                unsafe_allow_html=True
            )
        else:
            st.markdown(f"**🏢 {company_name}**<br><span style='color:#bdbdbd;font-size:0.98em;'>Technology</span>", unsafe_allow_html=True)

    # Show ingestion stats
    st.markdown("### Data Connection Status")
    if chroma_status:
        st.success(f"ChromaDB: Connected ({doc_count} documents)")
    else:
        st.error("ChromaDB: Disconnected")
        st.error(f"Error: {chroma_error}")
    st.markdown(f"Last checked: {datetime.now().strftime('%I:%M:%S %p')}")
    st.markdown("---")
    with st.expander("⚙️ Settings", expanded=False):
        st.write("Turn-on Debug mode to see content retrieved from Vector DB.")
        debug_mode = st.checkbox("Enable Debug Mode", value=False, help="Show context and retrieval debug info")
        st.session_state["debug_mode"] = debug_mode

# Inject custom CSS from styles.css
with open("styles.css") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# Centered logo
st.markdown(
    f'<div class="center-logo"><img src="data:image/svg+xml;base64,{get_svg_base64("icon.svg")}" width="70" style="border-radius:50%;"/></div>',
    unsafe_allow_html=True
)

# Title, subtitle, and description
st.markdown('<div class="main-title">Let\'s talk about SEC!</div>', unsafe_allow_html=True)
st.markdown('<div class="author-line">By <a href="https://github.com/bcastelino" target="_blank" style="color:#b0b0b0;text-decoration:underline;">Brian D. Castelino</a></div>', unsafe_allow_html=True)
st.markdown('<div class="description">AI-powered analysis of 10K filings (2020-24) for Apple, Microsoft, Nvidia, Google, Amazon & Meta.<br>Ask about revenue, risk factors, business strategy, and more.</div>', unsafe_allow_html=True)

# Quick starter buttons (row)
quick_starters = [
    "How much was Apple's revenue growth in 2022?",
    "How did Nvidia capitalize on the AI boom in 2024?",
    "Show me Amazon's risk factors in 2023."
]

# Render quick starters as Streamlit buttons in a single row
cols = st.columns(len(quick_starters))
for i, q in enumerate(quick_starters):
    if cols[i].button(q, key=f"quick_{i}"):
        st.session_state["quick_prompt"] = q

# If a quick starter was clicked, copy to clipboard and show feedback
if "quick_prompt" in st.session_state:
    st.markdown(f"""
        <script>
        navigator.clipboard.writeText({repr(st.session_state['quick_prompt'])});
        var toast = document.createElement('div');
        toast.innerText = 'Copied!';
        toast.style.position = 'fixed';
        toast.style.bottom = '40px';
        toast.style.left = '50%';
        toast.style.transform = 'translateX(-50%)';
        toast.style.background = '#232323';
        toast.style.color = '#e0f11f';
        toast.style.padding = '0.7em 1.5em';
        toast.style.borderRadius = '12px';
        toast.style.fontSize = '1.1em';
        toast.style.zIndex = 9999;
        document.body.appendChild(toast);
        setTimeout(function() {{ toast.remove(); }}, 1200);
        </script>
    """, unsafe_allow_html=True)

# JS to handle quick prompt button clicks (Streamlit workaround)
st.markdown(
    """
    <script>
    window.addEventListener('quickPrompt', function(e) {{
        const input = window.parent.document.querySelector('input[type="text"]');
        if (input) {{
            input.value = e.detail;
            input.dispatchEvent(new Event('input', {{ bubbles: true }}));
        }}
    }});
    </script>
    """,
    unsafe_allow_html=True
)

# --- Chat History Display ---
if "history" in st.session_state and st.session_state.history:
    st.markdown("<span style='color:#bdbdbd; font-weight:900; font-size: 2.0em;'>Financial Chat Room</span>", unsafe_allow_html=True)
    chat_container = st.container()
    with chat_container:
        for i, message in enumerate(st.session_state.history):
            if message["is_user"]:
                with st.chat_message("user"):
                    st.write(message['message'])
                    st.caption(message['timestamp'])
            else:
                with st.chat_message("assistant"):
                    st.markdown(message['message'])
                    st.caption(message['timestamp'])

# --- Chat input at the bottom ---
prompt = st.chat_input("Ask about 10K filings (2020-2024) for Apple, Microsoft, Nvidia, Google, Amazon, Meta...")

# Handle quick prompts
if "quick_prompt" in st.session_state:
    prompt = st.session_state.quick_prompt
    del st.session_state.quick_prompt

if "chatbot" not in st.session_state:
    st.session_state.chatbot = SECChatbot()

if prompt:
    if "history" not in st.session_state:
        st.session_state.history = []
    st.session_state.history.append({
        "message": prompt,
        "is_user": True,
        "timestamp": datetime.now().strftime("%H:%M")
    })
    debug_info = None
    with st.spinner("Thinking..."):
        try:
            if st.session_state.get("debug_mode"):
                import re
                import types
                def debug_generate_response(self, user_query):
                    debug_lines = []
                    context = self.build_context(user_query, debug=True, debug_lines=debug_lines)
                    # Debug the LLM call
                    try:
                        full_prompt = f"""You are a financial analyst assistant. Use the following SEC filing information to answer the user's question.\n\nContext from SEC filings:\n{context}\n\nUser Question: {user_query}\n\nPlease provide a comprehensive answer based on the SEC filing data above. If the information is not available in the provided context, say so clearly."""
                        debug_lines.append(f"[DEBUG] Full prompt length: {len(full_prompt)}")
                        debug_lines.append(f"[DEBUG] Full prompt preview:\n{full_prompt[:2000]}\n{'...' if len(full_prompt) > 2000 else ''}")
                        import requests
                        headers = {
                            "Authorization": f"Bearer {self.openrouter_api_key}",
                            "Content-Type": "application/json"
                        }
                        data = {
                            "model": Config.DEFAULT_MODEL,
                            "messages": [
                                {"role": "user", "content": full_prompt}
                            ],
                            "temperature": 0.1,
                            "max_tokens": 2000
                        }
                        debug_lines.append(f"[DEBUG] Making API call to OpenRouter...")
                        response = requests.post("https://openrouter.ai/api/v1/chat/completions", headers=headers, json=data, timeout=30)
                        debug_lines.append(f"[DEBUG] API Response Status: {response.status_code}")
                        if response.status_code == 200:
                            response_data = response.json()
                            debug_lines.append(f"[DEBUG] API Response structure: {list(response_data.keys())}")
                            if 'choices' in response_data and response_data['choices']:
                                llm_response = response_data['choices'][0]['message']['content']
                                debug_lines.append(f"[DEBUG] LLM Response length: {len(llm_response)}")
                                debug_lines.append(f"[DEBUG] LLM Response preview:\n{llm_response[:1000]}\n{'...' if len(llm_response) > 1000 else ''}")
                                st.session_state["last_debug_info"] = "\n".join(debug_lines)
                                return llm_response
                            else:
                                debug_lines.append(f"[DEBUG] No choices in response: {response_data}")
                                st.session_state["last_debug_info"] = "\n".join(debug_lines)
                                return f"OpenRouter API error: Unexpected response format - {response_data}"
                        else:
                            debug_lines.append(f"[DEBUG] API Error: {response.status_code} - {response.text}")
                            st.session_state["last_debug_info"] = "\n".join(debug_lines)
                            return f"OpenRouter API error: {response.status_code} - {response.text}"
                    except Exception as e:
                        debug_lines.append(f"[DEBUG] Exception during LLM call: {str(e)}")
                        st.session_state["last_debug_info"] = "\n".join(debug_lines)
                        return f"API call failed: {str(e)}"
                    st.session_state["last_debug_info"] = "\n".join(debug_lines)
                    return context
                # Patch the method
                st.session_state.chatbot.generate_response = types.MethodType(debug_generate_response, st.session_state.chatbot)
            else:
                st.session_state["last_debug_info"] = None
            response = st.session_state.chatbot.generate_response(prompt)
            # Remove strict context check: always call LLM, even if context is empty
            # Only handle API errors or empty response gracefully
            if not response:
                st.session_state.last_error = "No response generated. Check your backend or API keys."
                response = "Sorry, I couldn't generate a response. Please try again."
            elif response.startswith("OpenRouter API error:") or response.startswith("API call failed:") or response.startswith("Unexpected API response"):
                st.session_state.last_error = response
                response = "Sorry, there was an API error. Please check your configuration."
            else:
                st.session_state.last_error = None
        except Exception as e:
            st.session_state.last_error = f"Error generating response: {e}"
            response = "Sorry, there was an error processing your request."
    st.session_state.history.append({
        "message": response,
        "is_user": False,
        "timestamp": datetime.now().strftime("%H:%M")
    })
    st.rerun()

# After chat input, display any error
if st.session_state.get("last_error"):
    st.error(st.session_state.last_error)

# Show debug info if enabled and available
if st.session_state.get("debug_mode") and st.session_state.get("last_debug_info"):
    with st.expander("🔍 Debug Info (Context & Retrieval)", expanded=False):
        st.text(st.session_state["last_debug_info"])