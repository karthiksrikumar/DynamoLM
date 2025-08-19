import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import LlamaModel, LlamaTokenizer, DPRContextEncoder, DPRQuestionEncoder, DPRContextEncoderTokenizer, DPRQuestionEncoderTokenizer
from typing import List, Dict, Tuple, Optional
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss

class DocumentRetriever:
    def __init__(self, context_encoder_name: str = 'facebook/dpr-ctx_encoder-single-nq-base'):
        self.context_encoder = DPRContextEncoder.from_pretrained(context_encoder_name)
        self.context_tokenizer = DPRContextEncoderTokenizer.from_pretrained(context_encoder_name)
        self.question_encoder = DPRQuestionEncoder.from_pretrained('facebook/dpr-question_encoder-single-nq-base')
        self.question_tokenizer = DPRQuestionEncoderTokenizer.from_pretrained('facebook/dpr-question_encoder-single-nq-base')
        
        self.document_embeddings = None
        self.document_texts = None
        self.faiss_index = None
        
    def build_index(self, documents: List[str], batch_size: int = 16):
        embeddings = []
        self.context_encoder.eval()
        
        with torch.no_grad():
            for i in range(0, len(documents), batch_size):
                batch_docs = documents[i:i + batch_size]
                inputs = self.context_tokenizer(
                    batch_docs,
                    return_tensors='pt',
                    padding=True,
                    truncation=True,
                    max_length=512
                )
                outputs = self.context_encoder(**inputs)
                embeddings.append(outputs.pooler_output.cpu().numpy())
        
        self.document_embeddings = np.vstack(embeddings)
        self.document_texts = documents
        
        embedding_dim = self.document_embeddings.shape[1]
        self.faiss_index = faiss.IndexFlatIP(embedding_dim)
        self.faiss_index.add(self.document_embeddings)
    
    def retrieve(self, queries: List[str], top_k: int = 5) -> List[List[Dict[str, any]]]:
        if self.faiss_index is None:
            raise ValueError("Index not built. Call build_index first.")
        
        self.question_encoder.eval()
        results = []
        
        with torch.no_grad():
            inputs = self.question_tokenizer(
                queries,
                return_tensors='pt',
                padding=True,
                truncation=True,
                max_length=512
            )
            query_embeddings = self.question_encoder(**inputs).pooler_output.cpu().numpy()
            
            scores, indices = self.faiss_index.search(query_embeddings, top_k)
            
            for i, (query_scores, query_indices) in enumerate(zip(scores, indices)):
                query_results = []
                for score, idx in zip(query_scores, query_indices):
                    if idx < len(self.document_texts):
                        query_results.append({
                            'text': self.document_texts[idx],
                            'score': float(score),
                            'index': int(idx)
                        })
                results.append(query_results)
        
        return results

class RAGModel(nn.Module):
    """LLaMA-7B with Retrieval-Augmented Generation using DPR for document retrieval."""
    
    def __init__(self, transformer_name: str, num_classes: int = 10, top_k_docs: int = 3):
        super().__init__()
        self.transformer = LlamaModel.from_pretrained(transformer_name)
        self.tokenizer = LlamaTokenizer.from_pretrained(transformer_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
        self.hidden_dim = self.transformer.config.hidden_size
        self.top_k_docs = top_k_docs
        
        self.retriever = DocumentRetriever()
        
        self.context_projection = nn.Linear(768, self.hidden_dim)
        
        self.fusion_layer = nn.MultiheadAttention(
            embed_dim=self.hidden_dim,
            num_heads=8,
            dropout=0.1,
            batch_first=True
        )
        
        self.output_head = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(self.hidden_dim // 2, num_classes)
        )
        
    def setup_retrieval_corpus(self, documents: List[str]):
        self.retriever.build_index(documents)
    
    def forward(self, 
                input_ids: torch.Tensor, 
                attention_mask: torch.Tensor,
                queries: Optional[List[str]] = None) -> torch.Tensor:
        
        batch_size = input_ids.size(0)
        
        text_outputs = self.transformer(input_ids=input_ids, attention_mask=attention_mask)
        text_embeddings = text_outputs.last_hidden_state
        
        if queries is not None and self.retriever.faiss_index is not None:
            retrieved_docs = self.retriever.retrieve(queries, self.top_k_docs)
            
            context_embeddings = []
            for batch_idx, docs in enumerate(retrieved_docs):
                if docs:
                    doc_texts = [doc['text'] for doc in docs[:self.top_k_docs]]
                    
                    context_inputs = self.retriever.context_tokenizer(
                        doc_texts,
                        return_tensors='pt',
                        padding=True,
                        truncation=True,
                        max_length=256
                    )
                    
                    with torch.no_grad():
                        context_outputs = self.retriever.context_encoder(**context_inputs)
                        context_emb = context_outputs.pooler_output.mean(dim=0, keepdim=True)
                else:
                    context_emb = torch.zeros(1, 768, device=input_ids.device)
                
                context_embeddings.append(context_emb)
            
            context_embeddings = torch.cat(context_embeddings, dim=0)
            context_embeddings = self.context_projection(context_embeddings)
            context_embeddings = context_embeddings.unsqueeze(1).expand(-1, text_embeddings.size(1), -1)
            
            fused_embeddings, _ = self.fusion_layer(
                text_embeddings,
                context_embeddings,
                context_embeddings
            )
            
            final_embeddings = fused_embeddings.mean(dim=1)
        else:
            final_embeddings = text_embeddings.mean(dim=1)
        
        logits = self.output_head(final_embeddings)
        return logits

def create_rag_model(transformer_name: str, documents: List[str], num_classes: int = 10) -> RAGModel:
    model = RAGModel(transformer_name, num_classes)
    model.setup_retrieval_corpus(documents)
    return model

if __name__ == "__main__":
    # Documents would be pulled from Wikipedia data here which uses Wikipedia passages, making it ideal for general knowledge retrieval tests
    documents = []  # Wikipedia articles are  loaded here (use your own data)
    
    model = create_rag_model("meta-llama/Llama-2-7b-hf", documents)
    
    tokenizer = model.tokenizer
    queries = [#added from the json file]
    inputs = tokenizer(queries, return_tensors='pt', padding=True, truncation=True)
    
    with torch.no_grad():
        outputs = model(inputs['input_ids'], inputs['attention_mask'], queries)
        print(f"Output shape: {outputs.shape}")
