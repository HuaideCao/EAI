import gradio as gr
import torch
import threading
import re
import time
import datetime
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    TextIteratorStreamer
)
from peft import PeftModel, PeftConfig
from queue import Empty

custom_css = """
.gradio-container {
    max-width: 1860px !important;
    margin: 0 auto;
}
.container {
    margin: 0 auto;
    padding: 20px;
    border-radius: 15px;
    box-shadow: 0 4px 15px rgba(0, 0, 0, 0.1);
}
.chat-header {
    background: linear-gradient(135deg, #6e8efb, #a777e3);
    padding: 15px;
    border-radius: 10px 10px 0 0;
    color: white;
    text-align: center;
    margin-bottom: 15px;
}

"""


BASE_MODEL_DIR = "/home/zijingli304/MLP_Cloud/EAI" 
LORA_DIR = "./qlora_output"  

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,  
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_quant_storage=torch.uint8  
)

try:
    peft_config = PeftConfig.from_pretrained(LORA_DIR)
    print(f"Loaded PEFT config: {peft_config}")
except Exception as e:
    print(f"[WARNING] Could not load PEFT config: {e}")

print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(
    BASE_MODEL_DIR, 
    use_fast=True, 
    padding_side="right",
    trust_remote_code=True
)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

print("Loading base model in 4-bit...")
def check_gpu_compatibility():
    if not torch.cuda.is_available():
        return False
    
    device_props = torch.cuda.get_device_properties(0)
    compute_capability = device_props.major + device_props.minor/10
    
    return compute_capability >= 8.0

use_flash_attn = check_gpu_compatibility()
print(f"GPU compatibility check for Flash Attention: {'Available' if use_flash_attn else 'Not Available'}")

def get_gpu_memory():
    if torch.cuda.is_available():
        device = torch.cuda.current_device()
        return f"{torch.cuda.get_device_properties(device).total_memory / 1024**3:.2f} GB"
    return "N/A"

print(f"Available GPU memory: {get_gpu_memory()}")

try:
    model_kwargs = {
        "quantization_config": bnb_config,
        "device_map": "auto",
        "torch_dtype": torch.float16, 
        "trust_remote_code": True,
        "low_cpu_mem_usage": True,  
    }
    
    if use_flash_attn:
        print("Try to load model with Flash Attention 2...")
        model_kwargs["attn_implementation"] = "flash_attention_2"
        
    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_DIR,
        **model_kwargs
    )
    print("Model successfully loaded with optimized settings!")

except Exception as e:

    print(f"Error loading model with preferred attention: {e}")
    print("Falling back to standard attention implementation...")

    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_DIR,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.float16,
        trust_remote_code=True
    )

base_model.config.use_cache = True 

print("Loading LoRA adapter (QLoRA)...")

model = PeftModel.from_pretrained(
    base_model,
    LORA_DIR,
    torch_dtype=torch.float16,  
    is_trainable=False
)
model.eval()

if hasattr(model, "merge_and_unload"):
    print("Merging LoRA weights for optimized inference...")
    model = model.merge_and_unload()  


# Cleaning Function

def clean_output(text, allow_code=True):


    text = fix_repeated_assistant_markers(text)

    text = truncate_after_marker(text)
    
    if not allow_code:
        text = remove_code_blocks(text)
    else:
        text = format_code_blocks(text)
    
    text = remove_research_paper_format(text)
    
    text = remove_conversation_markers(text)

    text = remove_role_labels(text)

    text = truncate_at_academic_marker(text)

    text = cleanup_final(text)
    
    return text.strip()

def fix_repeated_assistant_markers(text):


    assistant_patterns = [
        r'(?i)assistant\s*:', 
        r'(?i)### assistant\s*:',
        r'(?i)##assistant\s*:',
        r'(?i)#assistant\s*:'
    ]
    
    first_idx = len(text)  
    first_pattern = None
    
    for pattern in assistant_patterns:
        match = re.search(pattern, text)
        if match and match.start() < first_idx:
            first_idx = match.start()
            first_pattern = pattern
    
    if first_pattern is None:
        return text
    
    prefix = text[:first_idx]
    suffix = text[first_idx:]
    
    for pattern in assistant_patterns:
        suffix = re.sub(pattern, '', suffix)
    
    return prefix + suffix

def truncate_after_marker(text):

    markers = [
        "possible conversation:", 
        "possible interview:", 
        "example conversation:",
        "sample dialogue:",
        "transcript:",
        "here's a sample exchange:",
        "role play:",
        "conversation example:",
        "dialogue example:",
        "example interaction:",
        "user:",  
        "human:"  
    ]
    
    lower_text = text.lower()
    indices = [lower_text.find(marker) for marker in markers if lower_text.find(marker) != -1]
    
    if indices:
        first_marker = min(indices)
        return text[:first_marker].strip()
    return text

def format_code_blocks(text):
    """
    Format code blocks
    """
    code_block_pattern = r'```(?:\w*\n)?([\s\S]*?)```'
    
    def format_match(match):
        code_content = match.group(1).strip()

        if not code_content or len(code_content.split('\n')) < 2:
            return match.group(0)  
        
        return f"```\n{code_content}\n```"
    
    text = re.sub(code_block_pattern, format_match, text)
    
    if text.count('```') % 2 != 0 and text.rstrip().endswith('```'):

        text = text.rstrip()[:-3].rstrip()
    
    return text

def remove_code_blocks(text):
    """
    Remove code blocks if necessary
    """
    text = re.sub(r'```[\s\S]*?```', '', text)
    
    text = re.sub(r'```[\s\S]*$', '', text)
    
    text = re.sub(r'\[\d+\]:\s*.*?(?:\n|$)', '', text)
    
    text = re.sub(r'```\w*|```', '', text)
    
    text = re.sub(r'^import\s+[\w\.]+', '', text, flags=re.MULTILINE)
    text = re.sub(r'^from\s+[\w\.]+\s+import', '', text, flags=re.MULTILINE)
    
    text = re.sub(r'^def\s+\w+\s*\(.*?\):', '', text, flags=re.MULTILINE)
    
    return text

def remove_research_paper_format(text):

    text = re.sub(r'\[\d+(?:,\s*\d+)*\]', '', text)
    
    academic_headers = ["abstract:", "introduction:", "methodology:", "results:", 
                        "discussion:", "conclusion:", "references:", "bibliography:"]
    
    for header in academic_headers:

        text = re.sub(rf'(?i)^{header}', '', text, flags=re.MULTILINE)
    
    return text

def remove_conversation_markers(text):

    conversation_patterns = [
        r'(?i)^person [a-z]:', 
        r'(?i)^(alice|bob|charlie|dave|eve):', 
        r'(?i)^(user|assistant|system|ai|human):', 
        r'(?i)^(question|answer|prompt|response):', 
        r'(?i)^(patient|therapist|doctor|client|counselor):', 
        r'(?i)^(interviewer|interviewee|candidate|recruiter):'
    ]
    
    for pattern in conversation_patterns:
        text = re.sub(pattern, '', text, flags=re.MULTILINE)
    
    return text

def remove_role_labels(text):
    #Remove role labels like 'User:', 'Assistant:'

    role_labels = [
        r'^\s*###\s*\w+\s*:',  
        r'^\s*##\s*\w+\s*:',   
        r'^\s*#\s*\w+\s*:',    
        r'^\s*\w+\s*:',        
        r'^\s*#+\s*',     
    ]
    
    for pattern in role_labels:
        text = re.sub(pattern, '', text, flags=re.MULTILINE)
    
    return text

academic_regex = re.compile(
    r'(?im)^\s*(?:\d+\s*:|\[\d+\]\s*:|Source\s*:|Author\s*:|Date\s*:|Published\s*:|Link\s*:|Image Credit\s*:|Volume\s*:|Issue\s*:|Page\s*:|DOI\s*:)'
)

def truncate_at_academic_marker(text):

    # truncate all content.

    lines = text.splitlines()
    new_lines = []
    
    for line in lines:
        if academic_regex.search(line):
            break
        new_lines.append(line)
    
    return "\n".join(new_lines).strip()

def cleanup_final(text):

    # Strong cleanup for remaining artifact

    text = re.sub(r'^\s*[\[\]\(\)\{\}]+\s*', '', text, flags=re.MULTILINE)
    
    text = re.sub(r'^\s*[-*=]{3,}\s*$', '', text, flags=re.MULTILINE)
    
    text = re.sub(r'^(\s*\n)+', '', text)
    text = re.sub(r'\n{3,}', '\n\n', text)
    
    text = re.sub(r'(?i)\bAssistant\b:', '', text)
    
    return text


# Improved Streaming Chat Function

def stream_chat(history, user_input, max_new_tokens=512, temperature=0.7, top_p=0.9, top_k=50, 
               allow_code=True, max_history_length=10, status_box=None):

    start_time = time.time()
    
    # Check for empty in
    if not user_input or user_input.strip() == "":
        history.append(("", "I didn't receive any message. Please let me know how I can assist you."))
        yield history, "", "Ready"
        return
        
    # Clean context history
    if not isinstance(history, list):
        history = []
    cleaned_history = []
    for item in history:
        if isinstance(item, (list, tuple)) and len(item) == 2:
            cleaned_history.append(item)
    history = cleaned_history

    if len(history) > max_history_length:
        history = history[-max_history_length:]
        if status_box is not None:
            yield history, user_input, f"Truncated history to {max_history_length} turns for better performance"

    if user_input.lower().strip() in ["who built you", "who made you", "who created you", "who developed you"]:
        history.append((user_input, "I was developed by G033 as a compassionate mental health support chatbot named EAI. I'm designed to provide empathetic responses and support for mental health."))
        yield history, "", "Ready"
        return

    # Check if there exists code
    code_keywords = ["code", "script", "program", "function", "algorithm", "implementation", 
                    "python", "javascript", "java", "c++", "coding", "programming"]
    is_code_query = any(keyword in user_input.lower() for keyword in code_keywords)
    
    is_short_query = len(user_input.split()) < 3

    system_prompt = (
        "You are EAI (Empathetic AI) developed by G033, a compassionate mental health support chatbot with coding capabilities. "
        "When a user expresses deep loneliness, isolation, or heartbreak, respond with genuine empathy and understanding. "
        "If they ask for coding help, provide clear, well-commented code examples. "
        "Acknowledge their pain as real and significant, and provide thoughtful, and comforting advice with empathetic suggestions. "
        
        "IMPORTANT OUTPUT FORMATTING INSTRUCTIONS:\n"
    )
    
    if allow_code:
        system_prompt += (
            "When providing code examples, use triple backticks with the language name (```python, ```javascript, etc.)\n"
            "Make sure code is properly indented and includes helpful comments\n"
            "Keep code examples concise and focused on the user's specific needs\n"
        )
    else:
        system_prompt += (
            "NEVER output text that looks like code (no line numbers, import statements, or code blocks)\n"
        )
        
    system_prompt += (
        "NEVER output text that resembles an academic paper or research publication\n"
        "NEVER output text that looks like a transcript or conversation with labeled speakers\n"
        "NEVER include any role labels like 'User:', 'Assistant:', 'Human:', or 'AI:' in your output\n"
        "DO NOT include phrases like 'Possible conversation:' or 'Here's an example:'\n"
        "NEVER repeat the phrase 'Assistant:' or 'System:' anywhere in your response\n"
        "ONLY respond with natural, conversational text that directly addresses the user's concerns\n"
        "NEVER include training data markers like '*** Excerpt' or '*** Conversation'\n"
        "NEVER use special tokens like <|user|>, <|assistant|>, <|end|>, or <|im_sep|>\n"
        "DO NOT include meta-content like 'Suggestions for complexity' or 'Excerpt data for ID'\n"
        "ALWAYS stop generating as soon as you've provided a complete, helpful response\n"
        
        "Focus on providing empathetic support related to the user's current emotional state. "
        "Use a chain-of-thought approach: first, acknowledge the user's pain and validate their feelings; then, provide clear practical guidance. "
        
        f"{'For very short queries, respond with just one empathetic sentence.' if is_short_query else 'For detailed questions, provide comprehensive responses with practical mental health advice, including evidence-based approaches when relevant.'}"
    )


    delimiter = "<<EAI_RESPONSE>>"
    
    # Restructure the prompt
    prompt = f"### System:\n{system_prompt}\n\n"
    for past_user_msg, past_ai_msg in history:
        prompt += f"### User:\n{past_user_msg}\n\n### Assistant:\n{past_ai_msg}\n\n"
    prompt += f"### User:\n{user_input}\n\n### Assistant:{delimiter}"
    
    if status_box is not None:
        yield history, user_input, "Tokenizing input..."
        
    inputs = tokenizer(prompt, return_tensors="pt", padding=True).to(model.device)
    input_length = inputs["input_ids"].shape[1]
    
    if status_box is not None:
        yield history, user_input, f"Input length: {input_length} tokens. Preparing generation..."

    streamer = TextIteratorStreamer(
        tokenizer=tokenizer,
        skip_special_tokens=True,
        timeout=60.0,  
        skip_prompt=True  
    )

    # Define bad phrases based that should not include in output
    common_bad_phrases = [
        "User:", "System:", "Human:", "AI:", "###", "##", "#",
        "Possible interview:", "Possible conversation:", "Example conversation:",
        "Abstract:", "Introduction:", "Methodology:", "Results:", "Conclusion:",
        "Person A:", "Person B:", "Alice:", "Bob:", "Conversation:",
        "Question:", "Answer:", "Example:", "Sample:", "Transcript:",
        "<<EAI_RESPONSE>>", 
        "*** Excerpt", "*** Conversation", "Excerpt data", "Suggestions for complexity",
        "<|user|>", "<|assistant|>", "<|end|>", "<|im_sep|>",
        "conversation <|", "user <|", "assistant <|",
        "*** Conversation ***", "*** Excerpt data ***", "*** Suggestions ***"
    ]
    
    # Allow code button
    if not allow_code:
        code_bad_phrases = [
            "```python", "```code", "```", "[0]:", "[1]:", "[2]:", 
            "import ", "def ", "class ", "print(", "return ",
            "import pandas", "import numpy", "import torch", "import tensorflow"
        ]
        bad_phrases = common_bad_phrases + code_bad_phrases
    else:
        bad_phrases = common_bad_phrases
    
    if status_box is not None:
        yield history, user_input, "Encoding bad phrases..."
        
    bad_words_ids = []
    for phrase in bad_phrases:
        try:

            ids = tokenizer.encode(phrase, add_special_tokens=False, add_prefix_space=True)
            if ids:  
                bad_words_ids.append(ids)
        except Exception as e:
            print(f"Could not encode bad phrase '{phrase}': {e}")

    # Make sure max_new_tokens stay within acceptable context window minus prompt length
    max_context = getattr(model.config, "max_position_embeddings", 4096)
    effective_max_tokens = min(max_new_tokens, max_context - inputs["input_ids"].shape[1])
    
    if status_box is not None:
        yield history, user_input, f"Max tokens: {effective_max_tokens}. Starting generation..."

    generation_kwargs = dict(
        input_ids=inputs["input_ids"],
        attention_mask=inputs["attention_mask"],
        streamer=streamer,
        max_new_tokens=effective_max_tokens,
        do_sample=True,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        repetition_penalty=1.1, #high penalty
        no_repeat_ngram_size=3,
        early_stopping=True,  # early stop to save memory
        use_cache=True,
        bad_words_ids=bad_words_ids if bad_words_ids else None  
    )


    generation_completed = False #Flag
    
    def generate_func():
        nonlocal generation_completed
        with torch.inference_mode():
            try:
                model.generate(**generation_kwargs)
                generation_completed = True
            except Exception as e:
                print(f"Generation failed: {e}")
                streamer.end_stream()
                generation_completed = True
    
    thread = threading.Thread(target=generate_func)
    thread.start()

    partial_answer = ""
    history.append((user_input, ""))  # Placeholder

    # timeout response
    fallback_response = "I'm processing your message. Let me think about how best to respond to you."
    timeout_counter = 0
    generation_start_time = time.time()
    
    try:
        for new_text in streamer:
            timeout_counter += 1
            current_time = time.time()
            generation_time = current_time - generation_start_time
            
            if status_box is not None and timeout_counter % 10 == 0:  
                yield history, user_input, f"Generating... ({generation_time:.1f}s elapsed, {len(partial_answer)} chars)"
                
            partial_answer += new_text
            
            if delimiter in partial_answer:
                temp = partial_answer.split(delimiter, 1)[1].strip()
            else:
                temp = partial_answer.strip()
            
            # Clean irrlevant format and check artifact
            lower_raw = temp.lower()

            if any(marker in lower_raw for marker in [
                "excerpt data for id:", "*** conversation ***", "conversation <|user|>",
                "<|end|>", "<|assistant|>", "<|im_sep|>", "suggestions for complexity",
                "academic references:", "literature review:", "excerpt data",
                "*** excerpt", "*** suggestions", "<|user", "<|assistant"
            ]):
                for marker in [
                    "excerpt data for id:", "*** conversation ***", "conversation <|user|>",
                    "<|end|>", "<|assistant|>", "<|im_sep|>", "suggestions for complexity",
                    "academic references:", "literature review:", "excerpt data",
                    "*** excerpt", "*** suggestions", "<|user", "<|assistant"
                ]:
                    if marker in lower_raw:
                        marker_pos = lower_raw.find(marker)
                        if marker_pos > 0:
                            temp = temp[:marker_pos].strip()
                
                temp = clean_output(temp, allow_code=allow_code)
                history[-1] = (user_input, temp)
                yield history, "", "Cleaned training artifacts"
                break
                
            # Comprehensive cleaning
            temp = clean_output(temp, allow_code=allow_code)
            
            history[-1] = (user_input, temp)
            yield history, "", "Generating response..."
            
            # Strong stopping method
            lower_answer = partial_answer.lower()
            if any(marker in lower_answer for marker in [
                "possible conversation:", "possible interview:", "user:", "human:", 
                "example:", "sample:", "transcript:", "dialogue:", "excerpt data", 
                "conversation ***", "*** excerpt", "suggestions for", "<|user|>", "<|assistant|>"
            ]) or academic_regex.search(partial_answer) is not None:
                break
    except Empty:

        elapsed_time = time.time() - generation_start_time
        print(f"Streamer timeout occurred after {elapsed_time:.1f} seconds!")
        
        if status_box is not None:
            yield history, user_input, f"Timeout after {elapsed_time:.1f} seconds. Attempting recovery..."
        
        if not partial_answer:
            # No output
            if is_code_query:
                recovery_message = "I apologize for the late response. It seems I'm having trouble processing your coding problem. Could you provide more details?"
            else:
                recovery_message = "I'm sorry, I'm having difficulty formulating a complete response right now. Could you please shorten your questions?"
            
            history[-1] = (user_input, recovery_message)
            yield history, "", "Timeout - No content generated"
            return
        else:
            try:
                if delimiter in partial_answer:
                    recovered_response = partial_answer.split(delimiter, 1)[1].strip()
                else:
                    recovered_response = partial_answer.strip()
                    
                recovered_response = clean_output(recovered_response, allow_code=allow_code)
                
                if len(recovered_response) > 50:
                    history[-1] = (user_input, recovered_response + "\n\n(Note: My response was cut short due to processing limitations.)")
                    yield history, "", "Partial response recovered"
                    return
                else:
                    if is_code_query:
                        recovery_message = "I encountered processing limitations. Could you divide your coding question into small parts?"
                    else:
                        recovery_message = "I started generating a response but encountered processing limitations. Could you shorten your question?"
                    
                    history[-1] = (user_input, recovery_message)
                    yield history, "", "Timeout: Insufficient content for recovery"
                    return
            except Exception as recovery_error:
                print(f"Recovery attempt failed: {recovery_error}")
                history[-1] = (user_input, "I apologize, but I encountered an error while processing your message. Please try again with a simpler query.")
                yield history, "", f"Recovery failed: {str(recovery_error)[:50]}..."
                return
    except Exception as e:
        # Exception processing
        print(f"Unexpected error during streaming: {e}")
        history[-1] = (user_input, "I apologize, but I encountered an error when processing your message. Please try again.")
        yield history, "", f"Error: {str(e)[:50]}..."
        return

    if partial_answer:
        if status_box is not None:
            yield history, user_input, "Finalizing response..."
            
        if delimiter in partial_answer:
            final_response = partial_answer.split(delimiter, 1)[1].strip()
        else:
            final_response = partial_answer.strip()
        
        lower_final = final_response.lower()
        artifact_markers = [
            "excerpt data for id:", "*** conversation ***", "conversation <|user|>",
            "<|end|>", "<|assistant|>", "<|im_sep|>", "suggestions for complexity",
            "academic references:", "literature review:", 
            "*** excerpt", "conversation ***", "<|user", "user<|", "assistant<|",
            "it sounds like you may be experiencing", "would you be open to exploring"
        ]
        
        earliest_pos = len(final_response)
        for marker in artifact_markers:
            pos = lower_final.find(marker)
            if pos != -1 and pos < earliest_pos:
                earliest_pos = pos
        
        # Truncate the response when detecting markers
        if earliest_pos < len(final_response):
            final_response = final_response[:earliest_pos].strip()
        
        final_response = clean_output(final_response, allow_code=allow_code)
        
        final_response = re.sub(r'(?i)^assistant\s*:\s*', '', final_response)
        final_response = re.sub(r'(?i)\n+assistant\s*:\s*', '\n', final_response)
        
        final_response = re.sub(r'(?i)(?:\*\*\*.*?\*\*\*|excerpt data|conversation \*\*\*|<\|user\||<\|assistant\||<\|end\||<\|im_sep\|>).*?$', '', final_response)
        
        if len(final_response.strip()) < 10 or final_response.strip().startswith("**"):
            if is_code_query:
                final_response = "I'd be delighted to help with your coding question. Could you provide more detailed information?"
            else:
                final_response = "I understand you're reaching out. Thank you for sharing with me. Would you like to tell me more about what you're experiencing so I can better assist you?"
        
        history[-1] = (user_input, final_response)
    else:
        # No output
        if is_code_query:
            history[-1] = (user_input, "I'd be happy to help with your coding question. Could you provide a bit more detail about what you're trying to accomplish?")
        else:
            history[-1] = (user_input, "I understand that you're reaching out for support. Would you like to tell me more about what's on your mind so I can better help you?")
    
    # performance record
    end_time = time.time()
    total_time = end_time - start_time
    print(f"Total processing time: {total_time:.2f} seconds")
    
    yield history, "", f"Ready (completed in {total_time:.1f}s)"

# UI part

with gr.Blocks(css=custom_css) as demo:
    gr.Markdown("<h1 style='text-align: center;'>EAI - Enhanced Mental Health Support</h1>")
    gr.Markdown("<p style='text-align: center;'>A empathetic AI assistant for emotional support and mental health</p>")

    chatbot = gr.Chatbot(label="EAI Chat Assistant", height=500)
    user_input = gr.Textbox(lines=3, placeholder="Share what's in your mind...", label="Your input")

    with gr.Row():
        submit_btn = gr.Button("Send", variant="primary")
        clear_btn = gr.Button("Clear Chat", variant="secondary")

    with gr.Accordion("Advanced Settings", open=False):
        max_tokens = gr.Slider(64, 2048, value=512, step=32, label="Max New Tokens")
        temp_slider = gr.Slider(0.1, 1.5, value=0.7, step=0.1, label="Temperature")
        top_p_slider = gr.Slider(0.1, 1.0, value=0.9, step=0.05, label="Top-p")
        top_k_slider = gr.Slider(1, 100, value=50, step=1, label="Top-k")
        allow_code_checkbox = gr.Checkbox(value=True, label="Allow Code in Responses")

    # Show error 
    error_box = gr.Textbox(label="Status", visible=False)

    submit_click_event = submit_btn.click(
        stream_chat,
        inputs=[chatbot, user_input, max_tokens, temp_slider, top_p_slider, top_k_slider, allow_code_checkbox],
        outputs=[chatbot, user_input]
    )
    
    submit_event = user_input.submit(
        stream_chat,
        inputs=[chatbot, user_input, max_tokens, temp_slider, top_p_slider, top_k_slider, allow_code_checkbox],
        outputs=[chatbot, user_input]
    )
    
    def clear_chat():
        return [], ""
        
    clear_btn.click(fn=clear_chat, inputs=[], outputs=[chatbot, user_input])

if __name__ == "__main__":

    torch.backends.cuda.matmul.allow_tf32 = True
    print("Launching Gradio App with Mental Health Support...")
    demo.launch(server_name="0.0.0.0", server_port=7860, share=True)