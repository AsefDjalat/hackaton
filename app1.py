from flask import Flask, request, jsonify, render_template
import openai
import PyPDF2
from tenacity import retry, wait_fixed, stop_after_attempt
import os

app = Flask(__name__)

# Set your OpenAI API key



def extract_text_from_pdf(pdf_path):
    print("extracting text")
    reader = PyPDF2.PdfReader(pdf_path)
    text = ""
    for page in reader.pages:
        text += page.extract_text()
    return text


def split_text_into_chunks(text, max_tokens=2000):
    print("splitting text")
    words = text.split()
    chunks = []
    current_chunk = []
    current_length = 0

    for word in words:
        current_length += len(word) + 1  # adding 1 for the space
        if current_length <= max_tokens:
            current_chunk.append(word)
        else:
            chunks.append(" ".join(current_chunk))
            current_chunk = [word]
            current_length = len(word) + 1

    # Add the last chunk
    if current_chunk:
        chunks.append(" ".join(current_chunk))

    return chunks


@retry(wait=wait_fixed(2), stop=stop_after_attempt(5))
def summarize_text_chunk(chunk):
    print("summerizing text chunks")
    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": "You are a very efficient and helpful assistant."},
            {"role": "user", "content": f"kan je het volgende text samenvatten zonder te vertalen: {chunk}"}
        ]
    )
    return response.choices[0].message['content']


def summarize_text_chunks(chunks):
    print("summarizing text chunks")
    summaries = [summarize_text_chunk(chunk) for chunk in chunks]
    return "\n".join(summaries)


def split_summary_into_chunks(summary, max_tokens=2000):
    print("splitting summary in to chunks")
    words = summary.split()
    chunks = []
    current_chunk = []
    current_length = 0

    for word in words:
        current_length += len(word) + 1  # adding 1 for the space
        if current_length <= max_tokens:
            current_chunk.append(word)
        else:
            chunks.append(" ".join(current_chunk))
            current_chunk = [word]
            current_length = len(word) + 1

    # Add the last chunk
    if current_chunk:
        chunks.append(" ".join(current_chunk))

    return chunks


def summarize_pdf(pdf_path):
    # Step 1: Extract text from PDF
    text = extract_text_from_pdf(pdf_path)

    # Step 2: Split text into chunks
    chunks = split_text_into_chunks(text)

    # Step 3: Summarize the chunks
    chunk_summaries = summarize_text_chunks(chunks)

    # Step 4: Merge summaries
    final_chunks = split_summary_into_chunks(chunk_summaries)
    final_summary = summarize_text_chunks(final_chunks)

    return final_summary


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    pdf_path = os.path.join('uploads', file.filename)
    file.save(pdf_path)

    summary = summarize_pdf(pdf_path)

    return jsonify({"summary": summary})


if __name__ == '__main__':
    if not os.path.exists('uploads'):
        os.makedirs('uploads')
    app.run(debug=True)
