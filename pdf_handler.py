from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema.document import Document
from llm_chains import load_vectordb, create_embeddings
import pypdfium2

def get_pdf_texts(pdfs_bytes_list):
    if not pdfs_bytes_list:
        print("Warning: Empty PDF bytes list")
        return []
    
    texts = []
    for i, pdf_bytes in enumerate(pdfs_bytes_list):
        try:
            text = extract_text_from_pdf(pdf_bytes.getvalue())
            if text.strip():  # Check if extracted text is not empty
                texts.append(text)
            else:
                print(f"Warning: No text extracted from PDF {i+1}")
        except Exception as e:
            print(f"Error processing PDF {i+1}: {str(e)}")
    
    print(f"Extracted text from {len(texts)} PDFs")
    return texts

def extract_text_from_pdf(pdf_bytes):
    try:
        pdf_file = pypdfium2.PdfDocument(pdf_bytes)
        text = "\n".join(
            pdf_file.get_page(page_number).get_textpage().get_text_range() 
            for page_number in range(len(pdf_file))
        )
        return text
    except Exception as e:
        print(f"Error in PDF extraction: {str(e)}")
        raise

def get_text_chunks(text):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=2000,
        chunk_overlap=50,
        separators=["\n", "\n\n"]
    )
    chunks = splitter.split_text(text)
    print(f"Created {len(chunks)} chunks from text")
    return chunks

def get_document_chunks(text_list):
    if not text_list:
        print("Warning: Empty text list")
        return []
    
    documents = []
    for i, text in enumerate(text_list):
        chunks = get_text_chunks(text)
        for chunk in chunks:
            if chunk.strip():  # Only add non-empty chunks
                documents.append(Document(page_content=chunk))
    
    print(f"Created {len(documents)} document chunks")
    return documents

def add_documents_to_db(pdfs_bytes):
    if not pdfs_bytes:
        print("Error: No PDFs provided")
        return
    
    print(f"Processing {len(pdfs_bytes)} PDFs")
    
    # Get texts from PDFs
    texts = get_pdf_texts(pdfs_bytes)
    if not texts:
        print("Error: No text extracted from PDFs")
        return
    
    # Create document chunks
    documents = get_document_chunks(texts)
    if not documents:
        print("Error: No document chunks created")
        return
    
    # Load vector database
    try:
        vector_db = load_vectordb(create_embeddings())
        vector_db.add_documents(documents)
        print(f"Successfully added {len(documents)} documents to db.")
    except Exception as e:
        print(f"Error adding documents to vector db: {str(e)}")
        raise