import openai
import PyPDF2
from tenacity import retry, wait_fixed, stop_after_attempt

# Set your OpenAI API key




def extract_text_from_pdf(pdf_path):
    reader = PyPDF2.PdfReader(pdf_path)
    text = ""
    for page in reader.pages:
        text += page.extract_text()
    return text


def save_text_to_file(text, output_txt_path):
    with open(output_txt_path, 'w') as f:
        f.write(text)


def split_text_into_chunks(text, max_tokens=2000):
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
    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": "You are a very efficient and helpful assistant."},
            {"role": "user", "content": f"Summarize the following text: {chunk}"}
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


# Example usage
pdf_path = 'ISTQB-Syllabus_2018.pdf'
intermediate_txt_path = 'intermediate_text.txt'
summary_txt_path = 'summary.txt'

summarize_pdf(pdf_path, intermediate_txt_path, summary_txt_path)
print(f"Summary saved to {summary_txt_path}")
