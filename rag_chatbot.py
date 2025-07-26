{
 "cells": [
  {
   "cell_type": "markdown",
   "id": "4c6b458f-f1a5-43e2-93d3-09b7ad457148",
   "metadata": {},
   "source": [
    "## Loan prediction"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 1,
   "id": "afb95887-df86-4967-a156-50fe2b26f00d",
   "metadata": {},
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "C:\\Users\\stuti\\anaconda\\envs\\doccluster\\lib\\site-packages\\requests\\__init__.py:86: RequestsDependencyWarning: Unable to find acceptable character detection dependency (chardet or charset_normalizer).\n",
      "  warnings.warn(\n"
     ]
    }
   ],
   "source": [
    "import streamlit as st\n",
    "import pandas as pd\n",
    "import numpy as np\n",
    "import faiss\n",
    "from sentence_transformers import SentenceTransformer\n",
    "import openai"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 2,
   "id": "cbd316eb-b32c-4aba-a78d-f04a49f8692b",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "1e5b7772eb54496b8613352cc673a751",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "modules.json:   0%|          | 0.00/349 [00:00<?, ?B/s]"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "C:\\Users\\stuti\\anaconda\\envs\\doccluster\\lib\\site-packages\\huggingface_hub\\file_download.py:143: UserWarning: `huggingface_hub` cache-system uses symlinks by default to efficiently store duplicated files but your machine does not support them in C:\\Users\\stuti\\.cache\\huggingface\\hub\\models--sentence-transformers--all-MiniLM-L6-v2. Caching files will still work but in a degraded version that might require more space on your disk. This warning can be disabled by setting the `HF_HUB_DISABLE_SYMLINKS_WARNING` environment variable. For more details, see https://huggingface.co/docs/huggingface_hub/how-to-cache#limitations.\n",
      "To support symlinks on Windows, you either need to activate Developer Mode or to run Python as an administrator. In order to activate developer mode, see this article: https://docs.microsoft.com/en-us/windows/apps/get-started/enable-your-device-for-development\n",
      "  warnings.warn(message)\n"
     ]
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "2b9052ace1334448bbab8e32d8e9e276",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "config_sentence_transformers.json:   0%|          | 0.00/116 [00:00<?, ?B/s]"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "15b0cb977cb34d318571758842ed4b6e",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "README.md: 0.00B [00:00, ?B/s]"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "1eb4fa8f15d34eb3aca3210e117e9316",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "sentence_bert_config.json:   0%|          | 0.00/53.0 [00:00<?, ?B/s]"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "f3216e6156534cd99f942b8a1c8d26a3",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "config.json:   0%|          | 0.00/612 [00:00<?, ?B/s]"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "Xet Storage is enabled for this repo, but the 'hf_xet' package is not installed. Falling back to regular HTTP download. For better performance, install the package with: `pip install huggingface_hub[hf_xet]` or `pip install hf_xet`\n"
     ]
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "1303b00df36a43c79ec9c4cb6056fc09",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "model.safetensors:   0%|          | 0.00/90.9M [00:00<?, ?B/s]"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "b33a0857e6854863a00f7bf481138c22",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "tokenizer_config.json:   0%|          | 0.00/350 [00:00<?, ?B/s]"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "579c5761724e478b8b75f6ad1b913f9a",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "vocab.txt: 0.00B [00:00, ?B/s]"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "e336e7baa5da4f4d99eb574310754e10",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "tokenizer.json: 0.00B [00:00, ?B/s]"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "b85027c876b046e7930c8340757655c0",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "special_tokens_map.json:   0%|          | 0.00/112 [00:00<?, ?B/s]"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "f02f2e2f52e14051adf14254aac3e0ca",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "config.json:   0%|          | 0.00/190 [00:00<?, ?B/s]"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "embed_model = SentenceTransformer('all-MiniLM-L6-v2')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 4,
   "id": "4deecc8d-92bb-4308-aac5-18b6f2b9cdcb",
   "metadata": {},
   "outputs": [],
   "source": [
    "#load csv\n",
    "\n",
    "def load_and_prepare_data(file):\n",
    "    df = pd.read_csv(file)\n",
    "    df = df.fillna('N/A')\n",
    "    docs = [f\"Applicant{i}:\" + \",\" .join([f\"{col}: {row[col]}\" for col in df.columns]) for i, rown in df.iterrows()]\n",
    "    "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 5,
   "id": "40f1bae7-ae07-4ca2-8998-12d5ac4f13ac",
   "metadata": {},
   "outputs": [],
   "source": [
    "def build_index(texts, model):\n",
    "    embeddings = model.encode(texts, show_pregress_bar= True)\n",
    "    index = faiss.IndexFlatL2(enbeddings.shape[1])\n",
    "    index.add(np.array(embeddings).astype(\"float32\"))\n",
    "    return index, embeddings"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 7,
   "id": "23d7f895-f305-441b-8487-c72d860ee883",
   "metadata": {},
   "outputs": [],
   "source": [
    "def search_index(query, model, docs,index, embeddings, k =3):\n",
    "    q_emb  = model.encode([query])\n",
    "    D, I = index.search(np.array(q_emb).astype(\"float32\"))\n",
    "    return [docs[i] for i in I[0]]\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 11,
   "id": "d5df982b-e3fb-40e5-a026-eb3f2bce6caf",
   "metadata": {},
   "outputs": [],
   "source": [
    "def generate_answer(context, question, api_key):\n",
    "    openai.api_key = api_key\n",
    "    prompt = f\"\"\"You are an AI assistant that answers questions based on loan data. \\nContext:\\n{context}\\n\\nQuestion: {question}\\nAnswer:\"\"\"\n",
    "    try:\n",
    "        response = openai.ChatCompletion.create(\n",
    "            model = \"gpt-3.5-turbo\",\n",
    "            messages = [{\"role\": \"user\", \"content\": prompt}],\n",
    "            temperature = 0.3\n",
    "        )\n",
    "        return respone['choice'][0]['message']['content'].strip()\n",
    "    except Exception as e:\n",
    "        return f\"Error: {e}\""
   ]
  },
  {
   "cell_type": "markdown",
   "id": "a3e08cca-ea71-463a-b201-a0ab29dd0d4f",
   "metadata": {},
   "source": [
    "## Streamlit "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 12,
   "id": "2938c508-4d42-4b8a-933c-2df08b0ccb0d",
   "metadata": {},
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "2025-07-26 13:28:55.843 WARNING streamlit.runtime.scriptrunner_utils.script_run_context: Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2025-07-26 13:28:59.509 \n",
      "  \u001b[33m\u001b[1mWarning:\u001b[0m to view this Streamlit app on a browser, run it with the following\n",
      "  command:\n",
      "\n",
      "    streamlit run C:\\Users\\stuti\\anaconda\\envs\\doccluster\\lib\\site-packages\\ipykernel_launcher.py [ARGUMENTS]\n",
      "2025-07-26 13:28:59.512 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2025-07-26 13:28:59.516 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2025-07-26 13:28:59.523 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2025-07-26 13:28:59.525 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2025-07-26 13:28:59.527 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2025-07-26 13:28:59.528 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2025-07-26 13:28:59.530 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2025-07-26 13:28:59.533 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2025-07-26 13:28:59.539 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2025-07-26 13:28:59.543 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2025-07-26 13:28:59.545 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2025-07-26 13:28:59.548 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2025-07-26 13:28:59.550 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2025-07-26 13:28:59.553 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2025-07-26 13:28:59.555 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2025-07-26 13:28:59.559 Session state does not function when running a script without `streamlit run`\n",
      "2025-07-26 13:28:59.562 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2025-07-26 13:28:59.564 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2025-07-26 13:28:59.565 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n"
     ]
    }
   ],
   "source": [
    "st.title(\"🔍 Loan Dataset RAG Q&A Chatbot\")\n",
    "st.write(\"Ask questions based on the uploaded loan approval dataset.\")\n",
    "\n",
    "uploaded_file = st.file_uploader(\"Upload the Loan Training Dataset CSV\", type=\"csv\")\n",
    "openai_api_key = st.text_input(\"Enter OpenAI API Key\", type=\"password\")\n",
    "\n",
    "if uploaded_file and openai_api_key:\n",
    "    df, docs = load_and_prepare_data(uploaded_file)\n",
    "    st.success(\"Dataset loaded successfully!\")\n",
    "    \n",
    "    if st.button(\"Build Knowledge Base\"):\n",
    "        index, embeddings = build_index(docs, embed_model)\n",
    "        st.success(\"Knowledge base created!\")\n",
    "\n",
    "        query = st.text_input(\"Ask a question about the data:\")\n",
    "        if query:\n",
    "            retrieved_docs = search_index(query, embed_model, docs, index, embeddings)\n",
    "            context = \"\\n\".join(retrieved_docs)\n",
    "            answer = generate_answer(context, query, openai_api_key)\n",
    "            st.markdown(\"### 💬 Answer:\")\n",
    "            st.write(answer)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "id": "c42cf7e4-13b5-4ffb-9335-dab9be4b5b6a",
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python (doccluster)",
   "language": "python",
   "name": "doccluster"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.10.11"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
