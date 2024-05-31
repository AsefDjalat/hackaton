from flask import Flask, request, jsonify, make_response, render_template
from markupsafe import Markup
import openai
import PyPDF2
from tenacity import retry, wait_fixed, stop_after_attempt
import os
import apikeyforopenai

app = Flask(__name__)

# Set your OpenAI API key0



def extract_text_from_pdf(pdf_path):
    print("extract text from pdf")
    reader = PyPDF2.PdfReader(pdf_path)
    text = ""
    for page in reader.pages:
        text += page.extract_text()
    return text


def save_text_to_file(text, output_txt_path):
    print("save text to file")
    with open(output_txt_path, 'w', encoding='utf-8') as f:
        f.write(text)


def split_text_into_chunks(text, max_tokens=2000):
    print("split text in chunks")
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
    print("summarize text chunk")
    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": "Je maakt een samenvatting van de tekst die je ontvangt in het Nederlands. Geef de samenvatting met html tekstopmaak tags."},
            {"role": "user", "content": f"{chunk}"}
        ]
    )
    return response.choices[0].message['content']


def summarize_text_chunks(chunks):
    summaries = [summarize_text_chunk(chunk) for chunk in chunks]
    return "\n".join(summaries)


def split_summary_into_chunks(summary, max_tokens=2000):
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


def summarize_pdf(pdf_path, intermediate_txt_path, summary_txt_path):
    # Step 1: Extract text from PDF
    text = extract_text_from_pdf(pdf_path)

    # Step 2: Save extracted text to a text file
    save_text_to_file(text, intermediate_txt_path)

    # Step 3: Split text into chunks
    chunks = split_text_into_chunks(text)

    # Step 4: Summarize the chunks
    chunk_summaries = summarize_text_chunks(chunks)

    # Step 5: Merge summaries and write intermediate summary to a text file
    save_text_to_file(chunk_summaries, summary_txt_path)

    # Step 6: Split the merged summary if needed and summarize again
    final_chunks = split_summary_into_chunks(chunk_summaries)
    final_summary = summarize_text_chunks(final_chunks)

    # Step 7: Save the final summary to a text file
    save_text_to_file(final_summary, summary_txt_path)


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
    intermediate_txt_path = os.path.join('uploads', 'intermediate_text.txt')
    summary_txt_path = os.path.join('uploads', 'summary.txt')

    file.save(pdf_path)

    summarize_pdf(pdf_path, intermediate_txt_path, summary_txt_path)

    response = make_response(jsonify({"message": "Success"}), 200)
    return response

@app.route('/summary')
def summary():
    # filename = request.args.get('file')
    # if not filename:
    #     return "No file specified", 400

    summary_txt_path = os.path.join('uploads', 'summary.txt')
    with open(summary_txt_path, 'r') as f:
        summary = f.read()

    return render_template('summary.html', summary=Markup(summary))

if __name__ == '__main__':
    if not os.path.exists('uploads'):
        os.makedirs('uploads')
    app.run(debug=True)
