from mlx_lm import load, stream_generate

def summarize_text(text, model_path="mlx-community/Llama-3.2-1B-Instruct-4bit"):
    model, tokenizer = load(model_path)
    prompt = f"""
    Please summarize the following text in bullet lists.
    Here is the text: {text}
    """
    if tokenizer.chat_template is not None:
        messages = [{"role": "user", "content": prompt}]
        prompt = tokenizer.apply_chat_template(messages, add_generation_prompt=True)

    for response in stream_generate(model, tokenizer, prompt, max_tokens=2048):
        print(response.text, end="", flush=True)