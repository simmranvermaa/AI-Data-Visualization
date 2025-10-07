import os
import json
import re
import sys
import io
import contextlib
import warnings
from typing import Optional, List, Any, Tuple
from PIL import Image
from pandas.core import resample
import streamlit as st
import pandas as pd
import base64
from io import BytesIO
import google.generativeai as genai
from e2b_code_interpreter import Sandbox
from streamlit_mic_recorder import speech_to_text

warnings.filterwarnings("ignore", category=UserWarning, module="pydantic")

# Match python code blocks like ```python ... ```, ```py ... ```, or bare ``` ... ```
pattern = re.compile(r"```(?:python|py)?\s*(.*?)```", re.DOTALL)

def _resolve_model_candidates(selected: str) -> List[str]:
    """Return a list of model name candidates to try with google-generativeai."""
    candidates = []
    if selected:
        candidates.append(selected)
    # Prefer latest aliases
    mapping = {
        "gemini-1.5": "gemini-2.5-flash",
        "gemini-1.5-pro": "gemini-2.5-pro",
        "gemini-1.5-flash-002": "gemini-2.0-flash",
        "gemini-1.5-pro-002": "gemini-2.0-pro",
    }
    if selected in mapping:
        candidates.insert(0, mapping[selected])
    # Deduplicate while preserving order
    seen = set()
    ordered = []
    for c in candidates:
        if c and c not in seen:
            seen.add(c)
            ordered.append(c)
    return ordered or ["gemini-1.5-flash-latest"]

def code_interpret(e2b_code_interpreter: Sandbox, code: str, dataset_path: str) -> Tuple[Optional[List[Any]], str]:
    with st.spinner('Executing code in E2B sandbox...'):
        stdout_capture = io.StringIO()
        stderr_capture = io.StringIO()

        with contextlib.redirect_stdout(stdout_capture), contextlib.redirect_stderr(stderr_capture):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                # Inject dataset_path variable so generated code can use it directly
                injected_code = f"dataset_path = r'''{dataset_path}'''\n" + code
                exec = e2b_code_interpreter.run_code(injected_code)

        if stderr_capture.getvalue():
            print("[Code Interpreter Warnings/Errors]", file=sys.stderr)
            print(stderr_capture.getvalue(), file=sys.stderr)

        if stdout_capture.getvalue():
            print("[Code Interpreter Output]", file=sys.stdout)
            print(stdout_capture.getvalue(), file=sys.stdout)

        if exec.error:
            print(f"[Code Interpreter ERROR] {exec.error}", file=sys.stderr)
            return None, stdout_capture.getvalue()
        return exec.results, stdout_capture.getvalue()

def match_code_blocks(llm_response: str) -> str:
    match = pattern.search(llm_response)
    if match:
        code = match.group(1)
        return code
    return ""

def chat_with_llm(e2b_code_interpreter: Sandbox, user_message: str, dataset_path: str) -> Tuple[Optional[List[Any]], str]:
    # Update system prompt to include dataset path information
    system_prompt = f"""You're a Python data analyst and data visualization expert. You are given a dataset at path '{dataset_path}' and also the user's query.
Understand the user's query and analyze the dataset and answer the user's query, select the type of chart most suited for the query if not mentioned in the query. answer the users query with a response by writing and running Python code to solve it.
STRICT FORMAT: Return ONLY one Python code block fenced by triple backticks. No explanations before or after. The code must:
- Load the dataset using the path variable '{dataset_path}' (use pandas.read_csv for .csv; pandas.read_excel for .xlsx/.xls)
- Perform the requested analysis
- Create plots using matplotlib or plotly
- Do NOT use plt.show()
- Instead, assign the figure to a variable (e.g., fig) and return it at the end of the script.
- If using matplotlib, call `plt.gcf()` to get the current figure.
- Ensure the last line of the code returns `fig` or a list of figures.Do not include any prose or 'enhanced code' commentary outside the code block."""

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_message},
    ]

    with st.spinner('Getting response from Google Gemini model...'):
        # Configure Gemini client
        genai.configure(api_key=st.session_state.gemini_api_key)
        model = None
        last_error = None
        for name in _resolve_model_candidates(st.session_state.model_name):
            try:
                model = genai.GenerativeModel(name)
                break
            except Exception as e:
                last_error = e
        if not model:
            raise last_error or RuntimeError("Failed to initialize Gemini model")
        # Provide system and user content
        response = model.generate_content([
            {"text": messages[0]["content"]},
            {"text": messages[1]["content"]},
        ])

        response_text = response.text if hasattr(response, 'text') else str(response)
        python_code = match_code_blocks(response_text)
        
        # Retry once with stricter instruction if no code block found
        if not python_code:
            strict_system = (
                messages[0]["content"]
                + "\n\nSTRICT FORMAT: Return ONLY one Python code block fenced by triple backticks. No explanations."
            )
            retry_response = model.generate_content([
                {"text": strict_system},
                {"text": messages[1]["content"]},
            ])
            response_text = retry_response.text if hasattr(retry_response, 'text') else str(retry_response)
            python_code = match_code_blocks(response_text)
        
        if python_code:
            code_interpreter_results, stdout_text = code_interpret(e2b_code_interpreter, python_code, dataset_path)
            return code_interpreter_results, response_text, stdout_text
        else:
            st.warning(f"Failed to match any Python code in model's response")
            return None, response_text, ""


def enhance_query_to_structured_prompt(user_message: str) -> Tuple[str, dict]:
    """Step 1: Ask Gemini to convert freeform query into a structured prompt with chart choice.
    Returns the raw text and the parsed dict (best-effort)."""
    genai.configure(api_key=st.session_state.gemini_api_key)
    model = None
    last_error = None
    for name in _resolve_model_candidates(st.session_state.model_name):
        try:
            model = genai.GenerativeModel(name)
            break
        except Exception as e:
            last_error = e
    if not model:
        raise last_error or RuntimeError("Failed to initialize Gemini model")
    instruction = (
        "You are a data visualization planner. Convert the given natural-language analytics query into a STRICT JSON object with fields: "
        "{objective: string, chart_type: string, x: string|null, y: string|null, group_by: string|null, aggregation: string|null, filters: string[]|null, notes: string|null}. "
        "Pick one chart_type best suited for the query (e.g., bar, line, scatter, histogram, box, pie). "
        "Only return the JSON. No markdown, no explanation."
    )
    response = model.generate_content([
        {"text": instruction},
        {"text": f"QUERY: {user_message}"},
    ])
    text = response.text if hasattr(response, 'text') else str(response)
    # Try to parse JSON from response
    try:
        # In case model wraps in code fences
        cleaned = text.strip()
        if cleaned.startswith("```)" ):
            pass
        if cleaned.startswith("```"):
            cleaned = cleaned.strip('`')
            cleaned = cleaned.replace('json', '', 1).strip()
        structured = json.loads(cleaned)
    except Exception:
        structured = {}
    return text, structured


def generate_and_run_code(e2b_code_interpreter: Sandbox, structured: dict, dataset_path: str) -> Tuple[Optional[List[Any]], str, str]:
    """Step 2: Ask Gemini to produce code using the structured prompt; execute and return results and stdout."""
    # Build a focused system prompt using the structured info
    objective = structured.get('objective') if isinstance(structured, dict) else None
    chart_type = structured.get('chart_type') if isinstance(structured, dict) else None
    guidance = f"Objective: {objective}\nChart type: {chart_type}\nUse dataset_path: {dataset_path}"

    system_prompt = f"""You are a Python data analyst. Generate ONLY one Python code block to satisfy the objective below and render the specified chart.
- Use the pre-defined variable dataset_path to load the dataset (pandas.read_csv for .csv; pandas.read_excel for .xlsx/.xls). Do NOT hardcode any file path.
- Create the chart using matplotlib or plotly matching the requested chart type when possible.
- If grouping/aggregation is implied, compute it.
- Ensure the code either returns figure objects (matplotlib/plotly) or prints a base64 PNG string accessible via .png; avoid only calling plt.show().
- """

    genai.configure(api_key=st.session_state.gemini_api_key)
    model = None
    last_error = None
    for name in _resolve_model_candidates(st.session_state.model_name):
        try:
            model = genai.GenerativeModel(name)
            break
        except Exception as e:
            last_error = e
    if not model:
        raise last_error or RuntimeError("Failed to initialize Gemini model")
    response = model.generate_content([
        {"text": system_prompt},
        {"text": guidance},
    ])
    response_text = response.text if hasattr(response, 'text') else str(response)
    python_code = match_code_blocks(response_text)

    if not python_code:
        # Retry with tighter instruction
        retry = model.generate_content([
            {"text": system_prompt + "\nReturn only one Python code block."},
            {"text": guidance},
        ])
        response_text = retry.text if hasattr(retry, 'text') else str(retry)
        python_code = match_code_blocks(response_text)

    if python_code:
        code_interpreter_results, stdout_text = code_interpret(e2b_code_interpreter, python_code, dataset_path)
        return code_interpreter_results, response_text, stdout_text
    else:
        return None, response_text, ""


def generate_two_line_insight(user_message: str, structured_text: str, stdout_text: str) -> str:
    """Step 3: Ask Gemini to produce a 2-line insight based on the objective and what the code printed to stdout."""
    genai.configure(api_key=st.session_state.gemini_api_key)
    model = genai.GenerativeModel(st.session_state.model_name)
    prompt = (
        "You are a concise data analyst. Based on the user's query, the structured prompt and the code's textual outputs, "
        "write exactly TWO short and specific sentences that answer the users query even if they do not look at the chart."
    )
    context = (
        f"USER QUERY:\n{user_message}\n\nSTRUCTURED PROMPT (raw):\n{structured_text}\n\nCODE STDOUT (truncated):\n{stdout_text[-2000:]}"
    )
    response = model.generate_content([
        {"text": prompt},
        {"text": context},
    ])
    return response.text.strip() if hasattr(response, 'text') else str(response)

def upload_dataset(code_interpreter: Sandbox, uploaded_file) -> str:
    dataset_path = f"./{uploaded_file.name}"
    
    try:
        code_interpreter.files.write(dataset_path, uploaded_file)
        return dataset_path
    except Exception as error:
        st.error(f"Error during file upload: {error}")
        raise error


def main():
    """Main Streamlit application."""
    st.title("📊 AI Data Visualization Agent")
    st.write("Upload your dataset and ask questions about it!")

    # Initialize session state variables
    if 'gemini_api_key' not in st.session_state:
        st.session_state.gemini_api_key = ''
    if 'e2b_api_key' not in st.session_state:
        st.session_state.e2b_api_key = ''
    if 'model_name' not in st.session_state:
        st.session_state.model_name = ''
    if 'query_text' not in st.session_state:
        st.session_state.query_text = 'Your Query Here'
    if 'voice_capture' not in st.session_state:
        st.session_state.voice_capture = False

    with st.sidebar:
        st.header("API Keys and Model Configuration")
        st.session_state.gemini_api_key = st.sidebar.text_input("Google Gemini API Key", type="password")
        st.sidebar.info("💡 Create a free-tier Gemini API key in Google AI Studio")
        st.sidebar.markdown("[Get Gemini API Key](https://aistudio.google.com/app/apikey)")
        
        st.session_state.e2b_api_key = st.sidebar.text_input("Enter E2B API Key", type="password")
        st.sidebar.markdown("[Get E2B API Key](https://e2b.dev/docs/legacy/getting-started/api-key)")
        
        # Add model selection dropdown (Gemini - requested options)
        model_options = {
            "Gemini 2.5 Pro": "gemini-2.5-pro",
            "Gemini 2.5 Flash": "gemini-2.5-flash",
            "Gemini 2.5 Flash Lite": "gemini-2.5-flash-lite",
            "Gemini 2.0 Flash 001": "gemini-2.0-flash-001",
            "Gemini 2.0 Flash Lite 001": "gemini-2.0-flash-lite-001",
        }
        st.session_state.model_name = st.selectbox(
            "Select Model",
            options=list(model_options.keys()),
            index=0  # Default to first option
        )
        st.session_state.model_name = model_options[st.session_state.model_name]

    uploaded_file = st.file_uploader("Choose a dataset file (CSV or Excel)", type=["csv", "xlsx", "xls"])
    
    if uploaded_file is not None:
        # Display dataset with toggle
        file_ext = os.path.splitext(uploaded_file.name)[1].lower()
        try:
            if file_ext == ".csv":
                df = pd.read_csv(uploaded_file)
            elif file_ext in (".xlsx", ".xls"):
                df = pd.read_excel(uploaded_file)
            else:
                st.error("Unsupported file type. Please upload a CSV or Excel file.")
                return
        except Exception as read_error:
            st.error(f"Failed to read file: {read_error}")
            return
        st.write("Dataset:")
        show_full = st.checkbox("Show full dataset")
        if show_full:
            st.dataframe(df)
        else:
            st.write("Preview (first 5 rows):")
            st.dataframe(df.head())
        # Query input with inline mic (placed adjacent, ChatGPT-style)
        st.markdown(
            """
            <style>
            /* Try to vertically center the mic component with the textarea */
            .mic-wrapper { display: flex; height: 100%; align-items: center; justify-content: center; }
            /* Reduce extra label spacing on the textarea for better alignment */
            div[data-testid="stTextArea"] > label { margin-bottom: 4px; }
            </style>
            """,
            unsafe_allow_html=True,
        )
        query_col, mic_col = st.columns([12, 1])
        with query_col:
            query = st.text_area(
                "What would you like to know about your data?",
                value=st.session_state.query_text,
                key="query_text_area"
            )
        with mic_col:
            st.markdown('<div class="mic-wrapper">', unsafe_allow_html=True)
            transcript = speech_to_text(
                language='en-US',
                just_once=True,
                start_prompt='🎙️',
                stop_prompt='⏹️',
                use_container_width=True,
                key='stt_inline'
            )
            st.markdown('</div>', unsafe_allow_html=True)
        if transcript and isinstance(transcript, str):
            st.session_state.query_text = transcript
            # also update the local variable so this run uses it immediately
            query = transcript
        # Keep session state synced if user types
        if query != st.session_state.query_text:
            st.session_state.query_text = query
        
        if st.button("Analyze"):
            if not st.session_state.gemini_api_key or not st.session_state.e2b_api_key:
                st.error("Please enter both API keys in the sidebar.")
            else:
                with Sandbox(api_key=st.session_state.e2b_api_key) as code_interpreter:
                    # Upload the dataset
                    dataset_path = upload_dataset(code_interpreter, uploaded_file)

                    # Step 1: Enhance query to structured prompt
                    with st.spinner('Step 1: Enhancing query into a structured prompt...'):
                        structured_text, structured = enhance_query_to_structured_prompt(query)
                    with st.expander("Step 1: Structured prompt", expanded=False):
                        st.code(structured_text or json.dumps(structured, indent=2), language="json")

                    # Step 2: Generate and run code
                    with st.spinner('Step 2: Generating and executing code...'):
                        code_results, code_llm_response, stdout_text = chat_with_llm(code_interpreter, query, dataset_path)
                        
                        #results, response_text, stdout_text = generate_and_run_code(code_interpreter, structured, dataset_path)
                    with st.expander("Step 2: Generated code", expanded=False):
                        st.code(code_llm_response, language="python")
                    # Step 3: Generate two-line insight
                    #with st.spinner('Step 3: Summarizing insight...'):
                     #   two_line_insight = generate_two_line_insight(query, structured_text, stdout_text or "")
                    #with st.expander("Step 3: Two-line insight", expanded=False):
                     #   st.code(two_line_insight, language="text")
                    #if two_line_insight:
                     #   st.subheader("Insight")
                      #  st.write(two_line_insight)

                    # Optional: Show any SQL lines emitted by the executed code
                    if stdout_text:
                        sql_lines = [line for line in stdout_text.splitlines() if line.strip().upper().startswith("SQL:")]
                        if sql_lines:
                            with st.expander("SQL queries executed", expanded=False):
                                for sql in sql_lines:
                                    st.code(sql.split(":",1)[1].strip(), language="sql")

                    # Display results/visualizations
                    #if code_results:
                        '''
                        for result in code_results:
                            if hasattr(result, 'png') and result.png:  # Check if PNG data is available
                                # Decode the base64-encoded PNG data
                                png_data = base64.b64decode(result.png)
                                
                                # Convert PNG data to an image and display it
                                image = Image.open(BytesIO(png_data))
                                st.image(image, caption="Generated Visualization", use_container_width=False)
                            elif isinstance(result, Image.Image):
                                st.image(result, caption="Generated Visualization", use_container_width=False)
                            elif isinstance(result, (bytes, bytearray)):
                                try:
                                    st.image(BytesIO(result), caption="Generated Visualization", use_container_width=False)
                                except Exception:
                                    st.write(result)
                            elif hasattr(result, 'figure'):  # For matplotlib figures
                                fig = result.figure  # Extract the matplotlib figure
                                st.pyplot(fig)  # Display using st.pyplot
                            elif hasattr(result, 'show'):  # For plotly figures
                                st.plotly_chart(result)
                            elif isinstance(result, (pd.DataFrame, pd.Series)):
                                st.dataframe(result)
                            else:
                                st.write(result)
                                '''
                    if code_results:
    # If it's a single object, make it iterable
                        result = code_results if isinstance(code_results, list) else [code_results]
    
                        for res in result:
                        # Case 1: Matplotlib Figure
                            if hasattr(res, "savefig"):
                                st.pyplot(res)
                        
                        # Case 2: Plotly Figure
                            elif hasattr(res, "to_plotly_json"):
                                st.plotly_chart(res)
                        
                        # Case 3: E2B Encoded Image
                            elif hasattr(res, "png") and res.png:
                                png_data = base64.b64decode(res.png)
                                image = Image.open(BytesIO(png_data))
                                st.image(image, caption="Generated Visualization", use_container_width=True)
                        
                        # Case 4: Pandas DataFrame or Series
                            elif isinstance(res, (pd.DataFrame, pd.Series)):
                                st.dataframe(resample)
                        
                        # Case 5: Raw base64 PNG string (some AI code might print that)
                            elif isinstance(res, str) and res.startswith("iVBOR"):
                                image = Image.open(BytesIO(base64.b64decode(res)))
                                st.image(image, caption="Generated Visualization", use_container_width=True)
                        
                        # Case 6: Fallback
                            else:
                                st.write(res)
                        else:
                            st.warning("No visualization or data returned by the AI-generated code.")

if __name__ == "__main__":
    main()