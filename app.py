


import os
import streamlit as st
import numpy as np
import fitz  # PyMuPDF
from ultralytics import YOLO
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
from langchain_core.output_parsers import StrOutputParser
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from sklearn.decomposition import PCA
from langchain_openai import ChatOpenAI
import string
import re

os.system("curl -L -o best.pt https://huggingface.co/spaces/zliang/PDFReadingAssistant/resolve/main/best.pt")
# Load the trained model
model = YOLO("best.pt")
openai_api_key = os.environ.get("openai_api_key")

# Define the class indices for figures, tables, and text
figure_class_index = 4  # class index for figures
table_class_index = 3   # class index for tables

# Global variables to store embeddings and contents
global_embeddings = None
global_split_contents = None

def clean_text(text):
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def remove_references(text):
    reference_patterns = [
        r'\bReferences\b', r'\breferences\b', r'\bBibliography\b', r'\bCitations\b',
        r'\bWorks Cited\b', r'\bReference\b', r'\breference\b'
    ]
    lines = text.split('\n')
    for i, line in enumerate(lines):
        if any(re.search(pattern, line, re.IGNORECASE) for pattern in reference_patterns):
            return '\n'.join(lines[:i])
    return text

def save_uploaded_file(uploaded_file):
    with open(uploaded_file.name, 'wb') as f:
        f.write(uploaded_file.getbuffer())
    return uploaded_file.name

def summarize_pdf(pdf_file_path, num_clusters=10):
    embeddings_model = OpenAIEmbeddings(model="text-embedding-3-small", api_key=openai_api_key)
    llm = ChatOpenAI(model="gpt-3.5-turbo", api_key=openai_api_key, temperature=0.3)
    prompt = ChatPromptTemplate.from_template(
        """Could you please provide a concise and comprehensive summary of the given Contexts? 
        The summary should capture the main points and key details of the text while conveying the author's intended meaning accurately. 
        Please ensure that the summary is well-organized and easy to read, with clear headings and subheadings to guide the reader through each section. 
        The length of the summary should be appropriate to capture the main points and key details of the text, without including unnecessary information or becoming overly long. 
        example of summary:
        ## Summary:
        ## Key points:
        Contexts: {topic}"""
    )
    output_parser = StrOutputParser()
    chain = prompt | llm | output_parser

    loader = PyMuPDFLoader(pdf_file_path)
    docs = loader.load()
    full_text = "\n".join(doc.page_content for doc in docs)
    cleaned_full_text = remove_references(full_text)
    cleaned_full_text = clean_text(cleaned_full_text)

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=0,separators=["\n\n", "\n",".", " "])
    split_contents = text_splitter.split_text(cleaned_full_text)
    embeddings = embeddings_model.embed_documents(split_contents)

    X = np.array(embeddings)
    kmeans = KMeans(n_clusters=num_clusters, init='k-means++', random_state=0).fit(embeddings)
    cluster_centers = kmeans.cluster_centers_

    closest_point_indices = []
    for center in cluster_centers:
        distances = np.linalg.norm(embeddings - center, axis=1)
        closest_point_indices.append(np.argmin(distances))
    
    extracted_contents = [split_contents[idx] for idx in closest_point_indices]
    results = chain.invoke({"topic": ' '.join(extracted_contents)})

    summary_sentences = re.split(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?)\s', results)
    summary_embeddings = embeddings_model.embed_documents(summary_sentences)
    extracted_embeddings = embeddings_model.embed_documents(extracted_contents)
    similarity_matrix = cosine_similarity(summary_embeddings, extracted_embeddings)

    cited_results = results
    relevant_sources = []
    source_mapping = {}
    sentence_to_source = {}
    similarity_threshold = 0.6

    for i, sentence in enumerate(summary_sentences):
        if sentence in sentence_to_source:
            continue
        max_similarity = max(similarity_matrix[i])
        if max_similarity >= similarity_threshold:
            most_similar_idx = np.argmax(similarity_matrix[i])
            if most_similar_idx not in source_mapping:
                source_mapping[most_similar_idx] = len(relevant_sources) + 1
                relevant_sources.append((most_similar_idx, extracted_contents[most_similar_idx]))
            citation_idx = source_mapping[most_similar_idx]
            citation = f"([Source {citation_idx}](#source-{citation_idx}))"
            cited_sentence = re.sub(r'([.!?])$', f" {citation}\\1", sentence)
            sentence_to_source[sentence] = citation_idx
            cited_results = cited_results.replace(sentence, cited_sentence)
    
    sources_list = "\n\n## Sources:\n"
    for idx, (original_idx, content) in enumerate(relevant_sources):
        sources_list += f"""
<details style="margin: 10px 0; padding: 10px; border: 1px solid #ccc; border-radius: 5px; background-color: #f9f9f9;">
<summary style="font-weight: bold; cursor: pointer;">Source {idx + 1}</summary>
<pre style="white-space: pre-wrap; word-wrap: break-word; margin-top: 10px;">{content}</pre>
</details>
"""
    cited_results += sources_list
    return cited_results

def qa_pdf(pdf_file_path, query, num_clusters=5, similarity_threshold=0.6):
    global global_embeddings, global_split_contents

    # Initialize models and embeddings
    embeddings_model = OpenAIEmbeddings(model="text-embedding-3-small", api_key=openai_api_key)
    llm = ChatOpenAI(model="gpt-3.5-turbo", api_key=openai_api_key, temperature=0.3)
    prompt = ChatPromptTemplate.from_template(
        """Please provide a detailed and accurate answer to the given question based on the provided contexts. 
        Ensure that the answer is comprehensive and directly addresses the query. 
        If necessary, include relevant examples or details from the text.
        Question: {question}
        Contexts: {contexts}"""
    )
    output_parser = StrOutputParser()
    chain = prompt | llm | output_parser

    # Load and process the PDF if not already loaded
    if global_embeddings is None or global_split_contents is None:
        loader = PyMuPDFLoader(pdf_file_path)
        docs = loader.load()
        full_text = "\n".join(doc.page_content for doc in docs)
        cleaned_full_text = remove_references(full_text)
        cleaned_full_text = clean_text(cleaned_full_text)

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=0, separators=["\n\n", "\n", ".", " "])
        global_split_contents = text_splitter.split_text(cleaned_full_text)
        global_embeddings = embeddings_model.embed_documents(global_split_contents)

    # Embed the query and find the most relevant contexts
    query_embedding = embeddings_model.embed_query(query)
    similarity_scores = cosine_similarity([query_embedding], global_embeddings)[0]
    top_indices = np.argsort(similarity_scores)[-num_clusters:]
    relevant_contents = [global_split_contents[i] for i in top_indices]

    # Generate the answer using the LLM chain
    results = chain.invoke({"question": query, "contexts": ' '.join(relevant_contents)})

    # Split the answer into sentences and embed them
    answer_sentences = re.split(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?)\s', results)
    answer_embeddings = embeddings_model.embed_documents(answer_sentences)
    relevant_embeddings = embeddings_model.embed_documents(relevant_contents)
    similarity_matrix = cosine_similarity(answer_embeddings, relevant_embeddings)

    # Map sentences to sources and create citations
    cited_results = results
    relevant_sources = []
    source_mapping = {}
    sentence_to_source = {}

    for i, sentence in enumerate(answer_sentences):
        if sentence in sentence_to_source:
            continue
        max_similarity = max(similarity_matrix[i])
        if max_similarity >= similarity_threshold:
            most_similar_idx = np.argmax(similarity_matrix[i])
            if most_similar_idx not in source_mapping:
                source_mapping[most_similar_idx] = len(relevant_sources) + 1
                relevant_sources.append((most_similar_idx, relevant_contents[most_similar_idx]))
            citation_idx = source_mapping[most_similar_idx]
            citation = f"<strong style='color:blue;'>[Source {citation_idx}]</strong>"            
            cited_sentence = re.sub(r'([.!?])$', f" {citation}\\1", sentence)
            sentence_to_source[sentence] = citation_idx
            cited_results = cited_results.replace(sentence, cited_sentence)

    # Format the sources for markdown rendering
    sources_list = "\n\n## Sources:\n"
    for idx, (original_idx, content) in enumerate(relevant_sources):
        sources_list += f"""
<details style="margin: 10px 0; padding: 10px; border: 1px solid #ccc; border-radius: 5px; background-color: #f9f9f9;">
<summary style="font-weight: bold; cursor: pointer;">Source {idx + 1}</summary>
<pre style="white-space: pre-wrap; word-wrap: break-word; margin-top: 10px;">{content}</pre>
</details>
"""
    cited_results += sources_list
    return cited_results


def infer_image_and_get_boxes(image, confidence_threshold=0.6):
    results = model.predict(image)
    boxes = [
        (int(box.xyxy[0][0]), int(box.xyxy[0][1]), int(box.xyxy[0][2]), int(box.xyxy[0][3]), int(box.cls[0]))
        for result in results for box in result.boxes
        if int(box.cls[0]) in {figure_class_index, table_class_index} and box.conf[0] > confidence_threshold
    ]
    return boxes

def crop_images_from_boxes(image, boxes, scale_factor):
    figures = []
    tables = []
    for (x1, y1, x2, y2, cls) in boxes:
        cropped_img = image[int(y1 * scale_factor):int(y2 * scale_factor), int(x1 * scale_factor):int(x2 * scale_factor)]
        if cls == figure_class_index:
            figures.append(cropped_img)
        elif cls == table_class_index:
            tables.append(cropped_img)
    return figures, tables


def process_pdf(pdf_file_path):
    doc = fitz.open(pdf_file_path)
    all_figures = []
    all_tables = []
    low_dpi = 50
    high_dpi = 300
    scale_factor = high_dpi / low_dpi
    low_res_pixmaps = [page.get_pixmap(dpi=low_dpi) for page in doc]
    
    for page_num, low_res_pix in enumerate(low_res_pixmaps):
        low_res_img = np.frombuffer(low_res_pix.samples, dtype=np.uint8).reshape(low_res_pix.height, low_res_pix.width, 3)
        boxes = infer_image_and_get_boxes(low_res_img)
        
        if boxes:
            high_res_pix = doc[page_num].get_pixmap(dpi=high_dpi)
            high_res_img = np.frombuffer(high_res_pix.samples, dtype=np.uint8).reshape(high_res_pix.height, high_res_pix.width, 3)
            figures, tables = crop_images_from_boxes(high_res_img, boxes, scale_factor)
            all_figures.extend(figures)
            all_tables.extend(tables)
    
    return all_figures, all_tables

# Set the page configuration for a modern look

# Set the page configuration for a modern look
# Set the page configuration for a modern look
st.set_page_config(page_title="PDF Reading Assistant", page_icon="📄", layout="wide")

# Add some custom CSS for a modern look
st.markdown("""
    <style>
        /* Main background and padding */
        .main {
            background-color: #f8f9fa;
            padding: 2rem;
            font-family: 'Arial', sans-serif;
        }
        
        /* Section headers */
        .section-header {
            font-size: 2rem;
            font-weight: bold;
            color: #343a40;
            margin-top: 2rem;
            margin-bottom: 1rem;
            text-align: center;
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        }

        /* Containers */
        .uploaded-file-container, .chat-container, .summary-container, .extract-container {
            padding: 2rem;
            background-color: #ffffff;
            border-radius: 10px;
            margin-bottom: 2rem;
            box-shadow: 0 4px 12px rgba(0, 0, 0, 0.1);
        }

        /* Buttons */
        .stButton>button {
            background-color: #007bff;
            color: white;
            padding: 0.6rem 1.2rem;
            border-radius: 5px;
            border: none;
            cursor: pointer;
            font-size: 1rem;
            transition: background-color 0.3s ease, transform 0.3s ease;
        }
        .stButton>button:hover {
            background-color: #0056b3;
            transform: translateY(-2px);
        }

        /* Chat messages */
        .chat-message {
            padding: 1rem;
            border-radius: 10px;
            margin-bottom: 1rem;
            font-size: 1rem;
            transition: all 0.3s ease;
            box-shadow: 0 2px 8px rgba(0, 0, 0, 0.1);
        }
        .chat-message.user {
            background-color: #e6f7ff;
            border-left: 5px solid #007bff;
            text-align: left;
        }
        .chat-message.bot {
            background-color: #fff0f1;
            border-left: 5px solid #dc3545;
            text-align: left;
        }

        /* Input area */
        .input-container {
            display: flex;
            align-items: center;
            gap: 10px;
            margin-top: 1rem;
        }
        .input-container textarea {
            border: 2px solid #ccc;
            border-radius: 10px;
            padding: 10px;
            width: 100%;
            background-color: #fff;
            transition: border-color 0.3s ease;
            margin: 0;
            font-size: 1rem;
        }
        .input-container textarea:focus {
            border-color: #007bff;
            outline: none;
        }
        .input-container button {
            background-color: #007bff;
            color: white;
            padding: 0.6rem 1.2rem;
            border-radius: 5px;
            border: none;
            cursor: pointer;
            font-size: 1rem;
            transition: background-color 0.3s ease, transform 0.3s ease;
        }
        .input-container button:hover {
            background-color: #0056b3;
            transform: translateY(-2px);
        }

        /* Expander */
        .st-expander {
            border: none;
            box-shadow: 0 4px 12px rgba(0, 0, 0, 0.1);
            margin-bottom: 2rem;
        }

        /* Markdown elements */
        .stMarkdown {
            font-size: 1rem;
            color: #343a40;
            line-height: 1.6;
        }
        
        /* Titles and subtitles */
        .stTitle {
            color: #343a40;
            text-align: center;
            margin-bottom: 1rem;
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        }
        .stSubtitle {
            color: #6c757d;
            text-align: center;
            margin-bottom: 1rem;
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        }
    </style>
""", unsafe_allow_html=True)

# Streamlit interface
# Streamlit interface
st.title("📄 PDF Reading Assistant")
st.markdown("### Extract tables, figures, summaries, and answers from your PDF files easily.")

uploaded_file = st.file_uploader("Upload a PDF", type="pdf")
if uploaded_file:
    file_path = save_uploaded_file(uploaded_file)

    if 'figures' not in st.session_state:
        st.session_state['figures'] = None
    if 'tables' not in st.session_state:
        st.session_state['tables'] = None
    if 'summary' not in st.session_state:
        st.session_state['summary'] = None

    with st.container():
        st.markdown("<div class='section-header'>Extract Tables and Figures</div>", unsafe_allow_html=True)
        with st.expander("Click to Extract Tables and Figures", expanded=True):
            with st.container():
                extract_button = st.button("Extract")
                if extract_button:
                    figures, tables = process_pdf(file_path)
                    st.session_state['figures'] = figures
                    st.session_state['tables'] = tables

                if st.session_state['figures']:
                    col1, col2 = st.columns(2)
                    with col1:
                        st.write("### Figures")
                        for figure in st.session_state['figures']:
                            st.image(figure, use_column_width=True)
                    with col2:
                        st.write("### Tables")
                        for table in st.session_state['tables']:
                            st.image(table, use_column_width=True)
                else:
                    st.write("No figures or tables found.")
    
    with st.container():
        st.markdown("<div class='section-header'>Get Summary</div>", unsafe_allow_html=True)
        with st.expander("Click to Generate Summary", expanded=True):
            with st.container():
                summary_button = st.button("Generate Summary")
                if summary_button:
                    summary = summarize_pdf(file_path)
                    st.session_state['summary'] = summary

                if st.session_state['summary']:
                    st.markdown(st.session_state['summary'], unsafe_allow_html=True)

    with st.container():
        st.markdown("<div class='section-header'>Chat with your PDF</div>", unsafe_allow_html=True)
        st.write("### Chat with your PDF")
        if 'chat_history' not in st.session_state:
            st.session_state['chat_history'] = []

        for chat in st.session_state['chat_history']:
            chat_user_class = "user" if chat["user"] else ""
            chat_bot_class = "bot" if chat["bot"] else ""
            st.markdown(f"<div class='chat-message {chat_user_class}'>{chat['user']}</div>", unsafe_allow_html=True)
            st.markdown(f"<div class='chat-message {chat_bot_class}'>{chat['bot']}</div>", unsafe_allow_html=True)

        with st.form(key="chat_form", clear_on_submit=True):
            user_input = st.text_area("Ask a question about the PDF:", key="user_input")
            submit_button = st.form_submit_button(label="Send")

            if submit_button and user_input:
                st.session_state['chat_history'].append({"user": user_input, "bot": None})
                answer = qa_pdf(file_path, user_input)
                st.session_state['chat_history'][-1]["bot"] = answer
                st.experimental_rerun()
