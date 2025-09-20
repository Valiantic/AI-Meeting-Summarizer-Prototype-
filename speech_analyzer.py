import torch
import os
import gradio as gr
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM

#######------------- LLM-------------####
# Initialize a free, small summarization model from Hugging Face
# Using T5-small which is specifically trained for summarization
summarizer = pipeline(
    "summarization",
    model="t5-small",
    tokenizer="t5-small",
    device=0 if torch.cuda.is_available() else -1  # Use GPU if available, else CPU
)

# Alternative smaller models you can try:
# model="t5-small" - Very small and fast
# model="sshleifer/distilbart-cnn-12-6" - Smaller version of BART
# model="google/flan-t5-small" - Instruction-tuned T5

#######------------- Speech2text-------------####
def transcript_audio(audio_file):
    # Initialize the speech recognition pipeline
    
    pipe = pipeline(
        "automatic-speech-recognition",
        model="openai/whisper-tiny.en",
        chunk_length_s=30,
    )
    
    # Transcribe the audio file and return the result
    transcript_txt = pipe(audio_file, batch_size=8)["text"]
    
    # Summarize the transcript using the free BART model
    # Split long text into chunks if needed (BART has a max input length)
    max_chunk_length = 1024  # BART's max input length
    
    if len(transcript_txt.split()) > max_chunk_length:
        # Split into chunks
        words = transcript_txt.split()
        chunks = [' '.join(words[i:i+max_chunk_length]) for i in range(0, len(words), max_chunk_length)]
        
        # Summarize each chunk
        chunk_summaries = []
        for chunk in chunks:
            summary = summarizer(chunk, max_length=150, min_length=30, do_sample=False)
            chunk_summaries.append(summary[0]['summary_text'])
        
        # If multiple chunks, summarize the combined summaries
        if len(chunk_summaries) > 1:
            combined_summaries = ' '.join(chunk_summaries)
            final_summary = summarizer(combined_summaries, max_length=200, min_length=50, do_sample=False)
            result = final_summary[0]['summary_text']
        else:
            result = chunk_summaries[0]
    else:
        # Directly summarize if text is short enough
        summary = summarizer(transcript_txt, max_length=150, min_length=30, do_sample=False)
        result = summary[0]['summary_text']
    
    # Format the output with both transcript and summary
    formatted_result = f"**TRANSCRIPT:**\n{transcript_txt}\n\n**SUMMARY:**\n{result}"
    
    return formatted_result

#######------------- Gradio-------------####
audio_input = gr.Audio(sources="upload", type="filepath")
output_text = gr.Textbox()

iface = gr.Interface(fn=transcript_audio, title="Ai Meeting Summarizer", inputs=audio_input, outputs=output_text)
iface.launch(server_name="127.0.0.1", server_port= 7860)