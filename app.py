import os
import tempfile
import streamlit as st

from langchain_google_genai import (
    ChatGoogleGenerativeAI,
    GoogleGenerativeAIEmbeddings
)

from langchain_community.document_loaders import PyPDFLoader

from langchain_text_splitters import RecursiveCharacterTextSplitter

from langchain_community.vectorstores import Chroma

from langchain_core.prompts import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    SystemMessagePromptTemplate
)

from langchain_core.output_parsers import StrOutputParser


# ---------------- STREAMLIT PAGE ---------------- #

st.set_page_config(
    page_title="ATS Resume Builder",
    page_icon="📄",
    layout="wide"
)

st.title("ATS Resume Builder")


# ---------------- API KEY ---------------- #

gemini_key = st.secrets["GEMINI_API_KEY"]

os.environ["GEMINI_API_KEY"] = gemini_key


# ---------------- LLM ---------------- #

LLM = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash"
)


# ---------------- EMBEDDINGS ---------------- #

gemini_embeddings = GoogleGenerativeAIEmbeddings(
    model="models/embedding-001",
    google_api_key=gemini_key
)


# ---------------- INPUTS ---------------- #

uploaded_file = st.file_uploader(
    "Upload Resume (PDF)",
    type=["pdf"]
)

job_description = st.text_area(
    "Enter Job Description",
    height=200
)


# ---------------- BUTTON ---------------- #

if st.button("Generate ATS Resume"):

    if uploaded_file is None:
        st.warning("Please upload a resume PDF.")
    
    elif not job_description:
        st.warning("Please enter a job description.")
    
    else:

        with st.spinner("Generating ATS Resume..."):

            # ---------------- SAVE PDF TEMP ---------------- #

            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
                temp_file.write(uploaded_file.read())
                temp_pdf_path = temp_file.name


            # ---------------- LOAD PDF ---------------- #

            loader = PyPDFLoader(temp_pdf_path)

            docs = loader.load()


            # ---------------- EXTRACT TEXT ---------------- #

            full_text = ""

            for doc in docs:
                full_text += doc.page_content


            # ---------------- TEXT SPLITTING ---------------- #

            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=500,
                chunk_overlap=50,
                length_function=len,
                is_separator_regex=False,
            )

            texts = text_splitter.create_documents([full_text])


            # ---------------- VECTOR DB ---------------- #

            vector_store = Chroma.from_documents(
                documents=texts,
                embedding=gemini_embeddings
            )


            # ---------------- RETRIEVAL ---------------- #

            response = vector_store.similarity_search(
                query=job_description,
                k=3
            )


            # ---------------- CONTEXT ---------------- #

            context = "\n".join(
                [doc.page_content for doc in response]
            )


            # ---------------- SYSTEM PROMPT ---------------- #

            system_prompt = """
            You are an ATS Resume Optimizer.

            Your task is to generate an ATS-friendly resume based only on the candidate's provided resume information and the given job description.

            Resume Structure:
            1. Personal Details
            2. Professional Summary
            3. Skills
            4. Experience
            5. Projects
            6. Education
            7. Certifications

            Instructions:
            - Use only the information available in the candidate resume context.
            - Do not generate fake skills, fake projects, fake certifications, fake education, or fake experience.
            - Do not assume technologies, frameworks, libraries, or tools that are not mentioned in the resume.
            - Optimize the resume according to the job description and target role.
            - Generate the professional summary according to the target job role and job description.
            - Include the main ATS keywords from the job description naturally in the professional summary.
            - Highlight the most relevant existing skills and projects.
            - Reorder skills and projects based on role relevance.
            - Improve the wording of summaries, projects, and experience professionally.
            - Include ATS-friendly keywords naturally from the job description only when they match the candidate’s existing profile.
            - Keep the resume professional, concise, and ATS-friendly.
            - Maintain proper section headings and formatting.
            - Generate the final output in clean text resume format.
            """


            # ---------------- PROMPT TEMPLATE ---------------- #

            prompt_template = ChatPromptTemplate.from_messages(
                [
                    SystemMessagePromptTemplate.from_template(
                        template=system_prompt
                    ),

                    HumanMessagePromptTemplate.from_template(
                        """
                        Candidate Resume Information:
                        {context}

                        Job Description:
                        {job_description}
                        """
                    )
                ]
            )


            # ---------------- CHAIN ---------------- #

            chain = prompt_template | LLM | StrOutputParser()


            # ---------------- FINAL RESPONSE ---------------- #

            result = chain.invoke({
                "context": context,
                "job_description": job_description
            })


            # ---------------- OUTPUT ---------------- #

            st.subheader("Generated ATS Resume")

            st.write(result)