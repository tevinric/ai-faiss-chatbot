# IMPORTS
## BASE IMPORTS
from datetime import datetime, timedelta, timezone
import os
import re
import streamlit as st
import uuid
from collections import Counter

## CUSTOM IMPORTS
import config
from login_ui import login_ui

# DATA PROCESSING FUNCTIONS
### PREPROCESS QUERY TO NORMALISE THE TEXT WHICH WILL HELP WITH VECTOR SEARCHING
def preprocess_query(query: str) -> str:
    """Preprocess the query for better matching."""
    # Convert to lowercase
    query = query.lower()
    # Remove extra whitespace
    query = ' '.join(query.split())
    # Remove punctuation that might affect matching
    query = re.sub(r'[^\w\s]', ' ', query)
    return query

### PREPARE SOURCE DOCUMENTS WITH HIGHLIGHTED CONTENT
def prepare_source_documents(source_documents, assistant_response=""):
    """
    Prepare source documents with highlighted content for display with Streamlit expanders.
    Returns a list of source dictionaries with title and content.
    """
    if not source_documents:
        return []
    
    # Create a list to hold formatted sources
    prepared_sources = []
    
    # Keep track of sources already processed
    seen_sources = set()
    
    # Extract key phrases from the assistant response for better matching
    response_key_phrases = extract_key_phrases(assistant_response)
    
    for doc in source_documents:
        source = doc.metadata.get('source', 'Unknown')
        page = doc.metadata.get('page', 'N/A')
        
        # Create a unique identifier for the source
        source_key = f"{source}_{page}"
        
        if source_key not in seen_sources:
            seen_sources.add(source_key)
            
            # Create the source title
            title = f"{source}"
            if page != 'N/A':
                title += f" (Page {page})"
            
            # Highlight matching content with enhanced algorithm
            highlighted_content = highlight_content(doc.page_content, assistant_response, response_key_phrases)
            
            # Add to list of prepared sources
            prepared_sources.append({
                "title": title,
                "content": highlighted_content,
                "match_score": highlighted_content.get("match_score", 0)
            })
    
    # Sort sources by match score (highest first)
    prepared_sources.sort(key=lambda x: x["match_score"], reverse=True)
    
    return prepared_sources

def extract_key_phrases(text):
    """
    Extract important phrases from text to use for better matching.
    Returns a list of phrases with their importance score.
    """
    if not text:
        return []
    
    # Convert to lowercase and clean up
    text = text.lower()
    
    # Remove common punctuation
    text = re.sub(r'[,.;:!?()"\']', ' ', text)
    
    # Get all words
    words = text.split()
    
    # Filter out very common words (simple stopwords)
    stopwords = {'the', 'a', 'an', 'and', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
                'to', 'of', 'for', 'in', 'on', 'at', 'by', 'with', 'about', 'from', 'as', 'this', 'that'}
    filtered_words = [word for word in words if word not in stopwords and len(word) > 2]
    
    # Count word frequencies to identify important terms
    word_counts = Counter(filtered_words)
    
    # Create n-grams (2-4 word phrases)
    phrases = []
    
    # Add important single words
    for word, count in word_counts.items():
        if count > 1 or len(word) > 5:  # Important if repeated or long word
            phrases.append((word, count * len(word) / 5))  # Score: frequency × length factor
    
    # Add 2-grams
    for i in range(len(words) - 1):
        bigram = f"{words[i]} {words[i+1]}".lower()
        # Calculate importance based on component words
        score = sum(word_counts.get(w, 0) for w in bigram.split() if w not in stopwords)
        if score > 0:
            phrases.append((bigram, score))
    
    # Add 3-grams
    for i in range(len(words) - 2):
        trigram = f"{words[i]} {words[i+1]} {words[i+2]}".lower()
        # Calculate importance based on component words
        score = sum(word_counts.get(w, 0) for w in trigram.split() if w not in stopwords)
        if score > 0:
            phrases.append((trigram, score * 1.2))  # Boost 3-grams slightly
    
    # Add 4-grams
    for i in range(len(words) - 3):
        fourgram = f"{words[i]} {words[i+1]} {words[i+2]} {words[i+3]}".lower()
        # Calculate importance based on component words
        score = sum(word_counts.get(w, 0) for w in fourgram.split() if w not in stopwords)
        if score > 0:
            phrases.append((fourgram, score * 1.5))  # Boost 4-grams more
    
    # Sort by importance score
    phrases.sort(key=lambda x: x[1], reverse=True)
    
    # Return top phrases (limit to most important ones)
    return phrases[:50]  # Adjust number based on performance testing

def highlight_content(source_text, assistant_response, key_phrases=None):
    """
    Enhanced algorithm to identify portions of source text that appear in the assistant's response.
    Uses a more flexible matching approach with varying phrase lengths and importance weighting.
    Returns the source text with highlighted spans marked up for later use.
    """
    if not assistant_response:
        return {"text": source_text, "spans": [], "match_score": 0}
    
    # Convert texts to lowercase for case-insensitive matching
    source_lower = source_text.lower()
    response_lower = assistant_response.lower()
    
    # Words to highlight (stored as start/end positions)
    highlight_spans = []
    
    # Track overall match score for this source
    total_match_score = 0
    
    # First approach: Find direct matches of varying lengths
    # Dynamically adjust minimum phrase length based on text length
    text_length = len(source_lower.split())
    min_phrase_length = max(2, min(4, text_length // 25))  # Shorter for shorter texts
    
    # Create word tokens from source text with position tracking
    source_words = source_lower.split()
    
    # Phase 1: Look for longer exact phrase matches (more reliable)
    for phrase_length in range(min(8, len(source_words)), min_phrase_length - 1, -1):
        # Generate all phrases of current length from source
        for i in range(len(source_words) - phrase_length + 1):
            # Generate the phrase at the current position
            phrase = ' '.join(source_words[i:i+phrase_length])
            
            # If the phrase is substantive (not just stopwords) and appears in the response
            if len(phrase) > 10 and phrase in response_lower:
                # Find the exact position in the original text
                start_pos = source_lower.find(phrase)
                if start_pos != -1:
                    end_pos = start_pos + len(phrase)
                    
                    # Calculate match score based on phrase length and quality
                    match_score = len(phrase) * (phrase_length / 2)
                    total_match_score += match_score
                    
                    # Store the span to highlight
                    highlight_spans.append((start_pos, end_pos, match_score))
    
    # Phase 2: Look for key phrases and important terms if provided
    if key_phrases:
        for phrase, importance in key_phrases:
            if len(phrase) > 3:  # Only consider substantive phrases
                # Look for each key phrase in the source
                phrase_pos = 0
                while True:
                    phrase_pos = source_lower.find(phrase, phrase_pos)
                    if phrase_pos == -1:
                        break
                    
                    # Found a match
                    start_pos = phrase_pos
                    end_pos = start_pos + len(phrase)
                    
                    # Calculate score based on phrase importance
                    match_score = importance * len(phrase) / 4
                    total_match_score += match_score
                    
                    # Store the span
                    highlight_spans.append((start_pos, end_pos, match_score))
                    
                    # Move to next potential match
                    phrase_pos = end_pos
    
    # Phase 3: Special case for shorter documents - look for technical terms or proper nouns
    if text_length < 100:
        # Simple pattern for potential technical terms or proper nouns
        tech_terms_pattern = r'\b[A-Z][a-z]+\b|\b[A-Z]{2,}\b|\b\w+(?:-\w+)+\b'
        tech_terms = re.finditer(tech_terms_pattern, source_text)
        
        for term in tech_terms:
            term_text = term.group(0)
            if term_text.lower() in response_lower:
                start_pos = term.start()
                end_pos = term.end()
                match_score = len(term_text) * 0.5  # Lower weight for these matches
                total_match_score += match_score
                highlight_spans.append((start_pos, end_pos, match_score))
    
    # Filter out low-quality spans if we have enough good matches
    if len(highlight_spans) > 10:
        # Sort by score
        highlight_spans.sort(key=lambda x: x[2], reverse=True)
        # Keep only the better matches
        highlight_spans = highlight_spans[:max(5, len(highlight_spans) // 2)]
    
    # Convert to tuples with just position info for rendering
    position_spans = [(start, end) for start, end, _ in highlight_spans]
    
    # Merge overlapping spans
    position_spans.sort(key=lambda x: x[0])
    merged_spans = []
    for span in position_spans:
        if not merged_spans or span[0] > merged_spans[-1][1] + 3:  # Allow small gaps (3 chars)
            merged_spans.append(span)
        else:
            merged_spans[-1] = (merged_spans[-1][0], max(merged_spans[-1][1], span[1]))
    
    # Return structured data with the text, spans to highlight, and match score
    return {
        "text": source_text,
        "spans": merged_spans,
        "match_score": total_match_score
    }

def render_highlighted_text(text_data):
    """
    Renders text with highlighted spans using Streamlit's markdown component.
    Enhanced to provide better visual highlighting and context.
    Always shows the complete text without truncation.
    """
    # If text_data is just a string (for backward compatibility)
    if isinstance(text_data, str):
        return text_data
        
    text = text_data["text"]
    spans = text_data["spans"]
    
    # If no spans to highlight, just return the text
    if not spans:
        return text
    
    # Create a list of chunks with highlighting
    chunks = []
    last_end = 0
    
    for start, end in spans:
        # Add text before highlight - always include the full text
        if start > last_end:
            chunks.append(text[last_end:start])
        
        # Add highlighted text with enhanced styling
        highlighted_text = text[start:end]
        chunks.append(f'<mark class="highlight-match">{highlighted_text}</mark>')
        
        last_end = end
    
    # Add any remaining text - always include the full text
    if last_end < len(text):
        chunks.append(text[last_end:])
    
    # Join and return
    return "".join(chunks)

# DISPLAY FUNCTONS
### CUSTOM CSS
def add_custom_css():
    st.markdown("""
        <style>
        .thinking-container {
            display: flex;
            align-items: center;
            gap: 8px;
            padding: 10px;
        }
        
        .thinking-dot {
            width: 8px;
            height: 8px;
            border-radius: 50%;
            background-color: #ffffff;
            animation: thinking 1.5s infinite;
            opacity: 0.3;
        }
        
        .thinking-dot:nth-child(1) { animation-delay: 0s; }
        .thinking-dot:nth-child(2) { animation-delay: 0.2s; }
        .thinking-dot:nth-child(3) { animation-delay: 0.4s; }
        
        @keyframes thinking {
            0%, 100% { opacity: 0.3; transform: scale(1); }
            50% { opacity: 1; transform: scale(1.1); }
        }
        
        /* Feedback button styling */
        .feedback-button button {
            min-width: 45px;
            height: 38px;
        }
        
        /* Style for feedback popover */
        .feedback-popover textarea {
            min-height: 100px;
            margin-bottom: 15px;
        }
        
        /* Prevent enter key from submitting form */
        textarea {
            resize: vertical;
        }
        
        /* Enhanced source highlighting styles */
        mark.source-match {
            background-color: rgba(255, 200, 0, 0.4);
            padding: 0 2px;
            border-radius: 3px;
            font-weight: 500;
            border-bottom: 1px solid rgba(255, 165, 0, 0.7);
            box-shadow: 0 1px 2px rgba(0,0,0,0.1);
        }
        
        /* Source content container */
        .source-content-container {
            white-space: pre-wrap;
            line-height: 1.5;
            padding: 12px;
            background-color: rgba(0, 0, 0, 0.02);
            border-radius: 5px;
            border: 1px solid rgba(0, 0, 0, 0.05);
            margin: 8px 0;
            font-family: 'Roboto Mono', monospace;
            font-size: 0.9em;
        }
        
        /* Source header styling */
        .source-header {
            display: flex;
            justify-content: space-between;
            padding: 6px 0;
            border-bottom: 1px solid rgba(0, 0, 0, 0.05);
            margin-bottom: 8px;
            font-weight: 500;
        }
        
        /* Match statistics */
        .match-stats {
            font-size: 0.9em;
            padding: 3px 6px;
            background-color: #f1f8ff;
            border-radius: 4px;
            color: #0366d6;
            font-weight: normal;
        }
        
        /* Highlight legend */
        .highlight-legend {
            margin-top: 10px;
            padding: 8px;
            background-color: #f8f9fa;
            border-radius: 4px;
            font-size: 0.85em;
            color: #555;
        }
        
        .legend-item {
            display: inline-block;
            margin-right: 12px;
        }
        
        .legend-color {
            display: inline-block;
            width: 12px;
            height: 12px;
            margin-right: 4px;
            border-radius: 2px;
            vertical-align: middle;
        }
        </style>
    """, unsafe_allow_html=True)

def render_app_header(base_directory): 
    col1_im, col2_im, col3_im = st.columns(3)
    with col1_im:
        st.write(' ')
    with col2_im:
        st.image(os.path.join(base_directory, 'assets', "company-logo.png"), use_column_width=True)
    with col3_im:
        st.write(' ')
                                 
    st.markdown(f"<h2 style='text-align: center; color: auto;'>ABC Knowledge Assistant</h2>", unsafe_allow_html=True)

def render_page_config(base_directory):
    st.set_page_config(
        page_title=f"ABC Knowledge Assistant",
        page_icon=os.path.join(base_directory, 'assets', "company-logo.png"),
        layout="centered",
    )


def render_popup_menu():
    column_button1, column_button2, column_button3 = st.columns(3)
    
    with column_button1:
        insert_popover("Information")
    
    with column_button2:
        insert_popover("Disclaimer")      
            
    with column_button3:
        insert_popover("Terms & Conditions")   
        
        
def render_sidebar_menu(base_directory,app_version,bot_options):
    st.sidebar.image(os.path.join(base_directory, 'assets', 'company-logo.png'), width=100)
    st.sidebar.markdown(' ')
    st.sidebar.markdown(f"## Use AI to better understand the ABC products, services and policies!")
    st.sidebar.markdown(' ')
    st.sidebar.markdown(' ')
    st.sidebar.write('Logged in with:', st.session_state.user_email)

    st.sidebar.markdown(' ')
    bot_type_selected = st.sidebar.selectbox('Choose the type of AI Assistant', bot_options)
    st.sidebar.markdown(' ')       
    
    ## Include file option to download files (if any)
    if bot_type_selected == 'Personal Lines':
        # Read the PDF file  
        with open(os.path.join(base_directory, 'downloadables', 'Personal_Lines_bot', 'Personal Lines Training Doc - R003.pdf'), "rb") as file:  
            pdf_content = file.read()  
        
        # Create a download button for the PDF  
        st.sidebar.download_button(  
            label="Download Reference Doc",  
            data=pdf_content,  
            file_name="Personal Lines Training Doc - R003.pdf",  
            mime="application/pdf"  
        )  
        
    st.sidebar.write(' ')
    st.sidebar.markdown(f"""
    Please note that the AI knowledge assistant can make mistakes when responding.
    
    If you encounter any challenges, please send an email explaining the issue to ABCaicoe@ABCsa.co.za.
    """)
    st.sidebar.write(' ')
    
    # Add spacing in sidebar
    for _ in range(5):
        st.sidebar.markdown(' ')

    st.sidebar.markdown(f"Powered by the ABC AI Center of Excellence")
    st.sidebar.markdown(str(app_version))
    
    # Add spacing in sidebar
    for _ in range(3):
        st.sidebar.markdown(' ')

    if st.sidebar.button("Logout"):
        # Clear all session state
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        
        # Set authenticated to False
        st.session_state.authenticated = False
        
        # Rerun the application to redirect to the login page
        st.rerun()
        
    return bot_type_selected 


def thinking_animation():
    return st.markdown("""
        <div class="thinking-container">
            <div class="thinking-dot"></div>
            <div class="thinking-dot"></div>
            <div class="thinking-dot"></div>
        </div>
    """, unsafe_allow_html=True)
    
### POPOPS
def insert_popover(name):
    with st.popover(str(name)):
        st.markdown(config.popup_dictonary[str(name)], unsafe_allow_html=True)

# Direct feedback update function with improved record handling
def direct_feedback_update(sql_database, cursor, logger, conversation_id, user_message, feedback_type, feedback_comment):
    """
    Directly updates the database with feedback - handles case where record might not exist yet
    """
    try:
        # Log the parameters to ensure we have them
        logger.info(f"DIRECT UPDATE: Feedback type={feedback_type}, comment={feedback_comment}")
        logger.info(f"DIRECT UPDATE: Conversation ID={conversation_id}, Message={user_message[:50]}...")
        
        # Find the record to update
        query = f"SELECT * FROM [{sql_database}].[dbo].[conversations] WHERE conversation_id = ? AND CAST(user_message AS NVARCHAR(MAX)) = ?"
        cursor.execute(query, (conversation_id, user_message))
        items = cursor.fetchall()
        
        primary_key_id = str(uuid.uuid4())
        timestamp = (datetime.now(timezone.utc) + timedelta(hours=2)).strftime('%Y-%m-%d %H:%M:%S')
        
        if items:
            # Update existing record
            logger.info(f"DIRECT UPDATE: Found existing record, updating")
            update_query = f"UPDATE [{sql_database}].[dbo].[conversations] SET feedback_response = ?, feedback_comment = ? WHERE conversation_id = ? AND CAST(user_message AS NVARCHAR(MAX)) = ?"
            cursor.execute(update_query, (feedback_type, feedback_comment, conversation_id, user_message))
            logger.info(f"DIRECT UPDATE: Successfully updated feedback in database")
            cursor.commit()
            return True
        else:
            # Record not found - this could happen if the background logging didn't complete
            # We'll create a new record with just the feedback information
            logger.info(f"DIRECT UPDATE: Record not found, creating new feedback-only record")
            try:
                # First try to find any record with this conversation_id to get associated data
                find_conversation_query = f"SELECT TOP 1 * FROM [{sql_database}].[dbo].[conversations] WHERE conversation_id = ?"
                cursor.execute(find_conversation_query, (conversation_id,))
                conversation_record = cursor.fetchone()
                
                if conversation_record:
                    # We found a record with this conversation ID, let's use its metadata
                    logger.info(f"DIRECT UPDATE: Found conversation with same ID, using its metadata")
                    
                    # Extract the column names
                    columns = [column[0] for column in cursor.description]
                    
                    # Create a dictionary from the row data
                    record_dict = {columns[i]: conversation_record[i] for i in range(len(columns))}
                    
                    # Now insert a new record using this metadata plus our feedback
                    insert_query = f"""
                    INSERT INTO [{sql_database}].[dbo].[conversations] (id, conversation_id, user_message, assistant_message, timestamp, feedback_response, [user_name], [user_email], [bot_type], feedback_comment)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """
                    
                    cursor.execute(insert_query, (
                        primary_key_id, 
                        conversation_id,
                        user_message, 
                        record_dict.get('assistant_message', ""),
                        timestamp,
                        feedback_type,
                        record_dict.get('user_name', ""),
                        record_dict.get('user_email', ""),
                        record_dict.get('bot_type', ""),
                        feedback_comment
                    ))
                    
                else:
                    # No record found at all for this conversation_id
                    # This is a fallback - we'll create a minimal record with just feedback
                    logger.info(f"DIRECT UPDATE: No conversation found, creating minimal record")
                    
                    # Create a basic insert with the information we have
                    basic_insert_query = f"""
                    INSERT INTO [{sql_database}].[dbo].[conversations] (id, conversation_id, user_message, feedback_response, timestamp, feedback_comment)
                    VALUES (?, ?, ?, ?, ?, ?)
                    """
                    
                    cursor.execute(basic_insert_query, (
                        primary_key_id,
                        conversation_id,
                        user_message,
                        feedback_type,
                        timestamp,
                        feedback_comment
                    ))
                
                cursor.commit()
                logger.info(f"DIRECT UPDATE: Successfully created new record with feedback")
                return True
                
            except Exception as insert_error:
                logger.error(f"DIRECT UPDATE: Error creating new record: {str(insert_error)}")
                return False
    except Exception as e:
        logger.error(f"DIRECT UPDATE: Error updating feedback: {str(e)}")
        return False

# LOGGING FUNCTIONS
### LOG RECORD TO SQL DATABASE
def log_conversation(sql_database, cursor, logger, conversation_id, user_message, assistant_message, feedback_response, display_name, user_email, bot_type_selected, feedback_comment=None):
    primary_key_id = str(uuid.uuid4())
    timestamp = (datetime.now(timezone.utc) + timedelta(hours=2)).strftime('%Y-%m-%d %H:%M:%S')  
    
    try:
        logger.info(f"LOG: Starting conversation logging for conversation_id={conversation_id}")
        
        query = f"SELECT * FROM [{sql_database}].[dbo].[conversations] WHERE conversation_id = ? AND CAST(user_message AS NVARCHAR(MAX)) = ?"
        cursor.execute(query, (conversation_id, user_message))
        items = cursor.fetchall()
        
        if items:
            logger.info(f"LOG: Found existing record, updating")
            if feedback_comment is not None:
                update_query = f"UPDATE [{sql_database}].[dbo].[conversations] SET feedback_response = ?, feedback_comment = ? WHERE conversation_id = ? AND CAST(user_message AS NVARCHAR(MAX)) = ?"
                cursor.execute(update_query, (feedback_response, feedback_comment, conversation_id, user_message))
            else:
                update_query = f"UPDATE [{sql_database}].[dbo].[conversations] SET feedback_response = ? WHERE conversation_id = ? AND CAST(user_message AS NVARCHAR(MAX)) = ?"
                cursor.execute(update_query, (feedback_response, conversation_id, user_message))
        else:
            logger.info(f"LOG: No existing record found, inserting new record")
            if feedback_comment is not None:
                insert_query = f"""
                INSERT INTO [{sql_database}].[dbo].[conversations] (id, conversation_id, user_message, assistant_message, timestamp, feedback_response, [user_name], [user_email], [bot_type], feedback_comment)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """
                cursor.execute(insert_query, (primary_key_id, conversation_id, user_message, assistant_message, timestamp, feedback_response, display_name, user_email, bot_type_selected, feedback_comment))
            else:
                insert_query = f"""
                INSERT INTO [{sql_database}].[dbo].[conversations] (id, conversation_id, user_message, assistant_message, timestamp, feedback_response, [user_name], [user_email], [bot_type])
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """
                cursor.execute(insert_query, (primary_key_id, conversation_id, user_message, assistant_message, timestamp, feedback_response, display_name, user_email, bot_type_selected))
        
        cursor.commit()
        logger.info(f"LOG: Successfully committed conversation to database")
        return True
    except Exception as e:
        logger.error(f"LOG: Error logging conversation: {str(e)}")
        return False
        
# Initialize feedback state variables
def init_feedback_state():
    """
    Initialize any session state variables needed for feedback
    """
    pass  # All state is now handled locally with popovers

# Add these improved functions to functions.py

import re
from difflib import SequenceMatcher
from collections import defaultdict

def find_text_overlaps(source_text, response_text, min_length=10):
    """
    Find overlapping text between source document and LLM response.
    Uses an advanced algorithm to detect even paraphrased content.
    
    Args:
        source_text: Text from the source document
        response_text: Text from the LLM response
        min_length: Minimum length of overlap to consider
        
    Returns:
        List of (start, end) positions in the source text that match the response
    """
    # Prepare the texts - lowercase for better matching
    source_lower = source_text.lower()
    response_lower = response_text.lower()
    
    # Initialize list for overlaps
    overlaps = []
    
    # Method 1: Find exact phrase matches of varying lengths
    # Start with longer phrases for more accurate matching
    source_words = source_lower.split()
    response_words = response_lower.split()
    
    # Try different phrase lengths, starting with longer ones
    for phrase_length in range(min(8, len(response_words)), 2, -1):
        # Check all phrases of this length in the response
        for i in range(len(response_words) - phrase_length + 1):
            phrase = ' '.join(response_words[i:i+phrase_length])
            
            # Skip very short phrases
            if len(phrase) < min_length:
                continue
                
            # Find all occurrences of this phrase in the source
            start_pos = 0
            while True:
                start_pos = source_lower.find(phrase, start_pos)
                if start_pos == -1:
                    break
                    
                end_pos = start_pos + len(phrase)
                # Add this overlap to our list
                overlaps.append((start_pos, end_pos))
                
                # Move past this match
                start_pos = end_pos
    
    # Method 2: Use SequenceMatcher for fuzzy matching
    # This catches paraphrased content or sections with minor changes
    s = SequenceMatcher(None, source_lower, response_lower)
    for block in s.get_matching_blocks():
        if block.size >= min_length:
            # Only consider significant matches
            overlaps.append((block.a, block.a + block.size))
    
    # Method 3: Check for shared unique phrases
    # Extract unique phrases from response (3-4 word groups)
    unique_phrases = []
    for i in range(len(response_words) - 3):
        phrase = ' '.join(response_words[i:i+4])
        if len(phrase) >= min_length:
            unique_phrases.append(phrase)
    
    # Find these phrases in the source
    for phrase in unique_phrases:
        start_pos = 0
        while True:
            start_pos = source_lower.find(phrase, start_pos)
            if start_pos == -1:
                break
                
            end_pos = start_pos + len(phrase)
            overlaps.append((start_pos, end_pos))
            start_pos = end_pos
    
    # Merge overlapping spans
    if not overlaps:
        return []
        
    overlaps.sort(key=lambda x: x[0])
    merged = [overlaps[0]]
    
    for current in overlaps[1:]:
        previous = merged[-1]
        # If current span overlaps with previous span
        if current[0] <= previous[1] + 3:  # Allow small gaps (3 chars)
            # Merge them
            merged[-1] = (previous[0], max(previous[1], current[1]))
        else:
            # Add as a new span
            merged.append(current)
    
    return merged

def prepare_source_documents(source_documents, assistant_response=""):
    """
    Prepare source documents with highlighted content for display with Streamlit expanders.
    Returns a list of source dictionaries with title and content.
    """
    if not source_documents:
        return []
    
    # Create a list to hold formatted sources
    prepared_sources = []
    
    # Keep track of sources already processed
    seen_sources = set()
    
    for doc in source_documents:
        # Skip if we don't have page_content
        if not hasattr(doc, 'page_content'):
            continue
            
        source = doc.metadata.get('source', 'Unknown')
        page = doc.metadata.get('page', 'N/A')
        
        # Create a unique identifier for the source
        source_key = f"{source}_{page}"
        
        if source_key not in seen_sources:
            seen_sources.add(source_key)
            
            # Create the source title
            title = f"{source}"
            if page != 'N/A':
                title += f" (Page {page})"
            
            # Find text overlaps between this source and the response
            overlaps = find_text_overlaps(doc.page_content, assistant_response)
            
            # Calculate a match score based on total overlap length
            total_overlap = sum(end-start for start, end in overlaps)
            match_percentage = (total_overlap / len(doc.page_content)) * 100 if doc.page_content else 0
            
            # Add to list of prepared sources
            prepared_sources.append({
                "title": title,
                "content": {
                    "text": doc.page_content,
                    "spans": overlaps,
                },
                "overlap_percentage": match_percentage,
                "total_overlap": total_overlap
            })
    
    # Sort sources by match score (highest first)
    prepared_sources.sort(key=lambda x: x["total_overlap"], reverse=True)
    
    return prepared_sources

def render_highlighted_text(content_data):
    """
    Renders text with highlighted spans that overlap with the assistant's response.
    Provides visual cues about which parts of the source were used.
    """
    if isinstance(content_data, str):
        return content_data
        
    text = content_data["text"]
    spans = content_data["spans"]
    
    # If no spans to highlight, just return the text
    if not spans:
        return text
    
    # Create a list of chunks with highlighting
    chunks = []
    last_end = 0
    
    for start, end in spans:
        # Add text before highlight
        if start > last_end:
            chunks.append(text[last_end:start])
        
        # Add highlighted text with enhanced styling
        highlighted_text = text[start:end]
        chunks.append(f'<mark style="background-color: rgba(255, 200, 0, 0.5); padding: 0 1px; border-radius: 2px; font-weight: 500;">{highlighted_text}</mark>')
        
        last_end = end
    
    # Add any remaining text
    if last_end < len(text):
        chunks.append(text[last_end:])
    
    # Join the chunks and return
    return "".join(chunks)

def display_sources_with_highlights(source_documents, answer_text=""):
    """
    Enhanced display of source documents with highlighted text that matches the assistant's response.
    Also provides a percentage of overlap for each source.
    """
    if not source_documents:
        return
    
    # Use the function to prepare source documents with highlighting
    prepared_sources = prepare_source_documents(source_documents, answer_text)
    
    if prepared_sources:
        st.markdown("### Source References")
        st.markdown("*Information drawn from these reference documents:*")
        
        # Create an expander for each source
        for source in prepared_sources:
            with st.expander(f"{source['title']} - {source['overlap_percentage']:.1f}% content overlap", expanded=False):
                # Add a note about overlap percentage
                overlap_color = "green" if source["overlap_percentage"] > 20 else "orange"
                st.markdown(f"<div style='color: {overlap_color}; margin-bottom: 8px;'>This source has a {source['overlap_percentage']:.1f}% content overlap with the response</div>", unsafe_allow_html=True)
                
                # Get the highlighted text
                highlighted_text = render_highlighted_text(source["content"])
                
                # Display the highlighted content
                st.markdown(f'<div style="white-space: pre-wrap; line-height: 1.5; padding: 10px; background-color: rgba(0, 0, 0, 0.02); border-radius: 5px;">{highlighted_text}</div>', unsafe_allow_html=True)