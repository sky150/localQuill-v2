from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from dotenv import load_dotenv
import os


# Split documents into smaller chunks
def split_documents(documents: list[Document]):
    load_dotenv()
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=int(os.getenv("CHUNK_SIZE", 1000)),
        chunk_overlap=int(os.getenv("CHUNK_OVERLAP", 200)),
        length_function=len,
        is_separator_regex=False,
    )
    return text_splitter.split_documents(documents)


# Id Format: "source:page:chunk_index"; page is omitted when the loader provides none
def calculate_chunk_ids(chunks):
    last_page_id = None
    current_chunk_index = 0

    for chunk in chunks:
        source = chunk.metadata["source"] # raises error if its missing
        page = chunk.metadata.get("page")
        current_page_id = f"{source}:{page}" if page is not None else source

        # If the page ID is the same as the last one, increment the index.
        if current_page_id == last_page_id:
            current_chunk_index += 1
        else:
            current_chunk_index = 0

        chunk_id = f"{current_page_id}:{current_chunk_index}"
        last_page_id = current_page_id

        # Add it to the page meta-data.
        chunk.metadata["id"] = chunk_id

    return chunks

