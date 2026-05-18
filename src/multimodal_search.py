import numpy as np
from PIL import Image
import torch
from transformers import CLIPProcessor, CLIPModel
import requests
from io import BytesIO
import pickle
from sklearn.metrics.pairwise import cosine_similarity
import os
from typing import List, Dict, Tuple

class MultiModalSearchEngine:
    """Multi-modal search engine for book covers and text using CLIP"""
    
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"🖥️ Using device: {self.device}")
        
        # Load CLIP model
        print("📦 Loading CLIP model...")
        self.model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        self.model.to(self.device)
        self.model.eval()
        
        # Storage for embeddings
        self.image_embeddings = None
        self.text_embeddings = None
        self.documents = []
        self.image_paths = []
        
    def generate_synthetic_book_covers(self, documents, output_dir='../data/book_covers'):
        """Generate synthetic book covers for demonstration (using placeholders)"""
        
        os.makedirs(output_dir, exist_ok=True)
        image_paths = []
        
        print(f"🎨 Generating synthetic book covers in {output_dir}...")
        
        for i, doc in enumerate(documents[:100]):  # Limit to 100 for demo
            # Create a simple colored image based on genre/content
            # In real implementation, you'd have actual book covers
            
            # Determine color based on content keywords
            content = (doc['title'] + ' ' + doc['content']).lower()
            
            if 'mystery' in content or 'detective' in content:
                color = (25, 25, 112)  # Midnight blue
            elif 'romance' in content or 'love' in content:
                color = (255, 182, 193)  # Light pink
            elif 'science' in content or 'space' in content:
                color = (70, 130, 180)  # Steel blue
            elif 'horror' in content or 'scary' in content:
                color = (139, 0, 0)  # Dark red
            elif 'fantasy' in content or 'magic' in content:
                color = (138, 43, 226)  # Blue violet
            else:
                color = (128, 128, 128)  # Gray
            
            # Create image with color and text
            img = Image.new('RGB', (200, 300), color=color)
            
            # Save image
            path = os.path.join(output_dir, f'book_cover_{i}.png')
            img.save(path)
            image_paths.append(path)
            
            if (i + 1) % 20 == 0:
                print(f"  Generated {i + 1} covers...")
        
        print(f"✅ Generated {len(image_paths)} synthetic book covers")
        return image_paths
    
    def encode_images(self, image_paths: List[str], batch_size: int = 32):
        """Encode book cover images using CLIP"""
        
        print(f"🖼️ Encoding {len(image_paths)} images...")
        embeddings = []
        
        for i in range(0, len(image_paths), batch_size):
            batch_paths = image_paths[i:i + batch_size]
            batch_images = []
            
            for path in batch_paths:
                try:
                    image = Image.open(path).convert('RGB')
                    batch_images.append(image)
                except Exception as e:
                    print(f"  ⚠️ Error loading {path}: {e}")
                    # Create placeholder
                    batch_images.append(Image.new('RGB', (200, 300), color=(128, 128, 128)))
            
            # Process batch
            with torch.no_grad():
                inputs = self.processor(images=batch_images, return_tensors="pt", padding=True)
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                image_features = self.model.get_image_features(**inputs)
                
                # Normalize embeddings
                image_features = image_features / image_features.norm(dim=-1, keepdim=True)
                embeddings.append(image_features.cpu().numpy())
            
            if (i + batch_size) % 100 == 0:
                print(f"  Processed {min(i + batch_size, len(image_paths))} images...")
        
        self.image_embeddings = np.vstack(embeddings)
        self.image_paths = image_paths
        print(f"✅ Encoded images to shape: {self.image_embeddings.shape}")
        
        return self.image_embeddings
    
    def encode_texts(self, documents: List[Dict], batch_size: int = 32):
        """Encode book descriptions using CLIP"""
        
        print(f"📝 Encoding {len(documents)} text descriptions...")
        embeddings = []
        
        texts = [f"{doc['title']}. {doc['content'][:200]}" for doc in documents]
        
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            
            with torch.no_grad():
                inputs = self.processor(text=batch_texts, return_tensors="pt",
                                       padding=True, truncation=True, max_length=77)
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                text_features = self.model.get_text_features(**inputs)
                
                # Normalize embeddings
                text_features = text_features / text_features.norm(dim=-1, keepdim=True)
                embeddings.append(text_features.cpu().numpy())
            
            if (i + batch_size) % 100 == 0:
                print(f"  Processed {min(i + batch_size, len(texts))} texts...")
        
        self.text_embeddings = np.vstack(embeddings)
        self.documents = documents
        print(f"✅ Encoded texts to shape: {self.text_embeddings.shape}")
        
        return self.text_embeddings
    
    def text_to_image_search(self, query: str, top_k: int = 5):
        """Search for book covers using text query"""
        
        if self.image_embeddings is None:
            raise ValueError("No image embeddings found. Please encode images first.")
        
        # Encode query
        with torch.no_grad():
            inputs = self.processor(text=[query], return_tensors="pt",
                                   padding=True, truncation=True, max_length=77)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            query_features = self.model.get_text_features(**inputs)
            query_features = query_features / query_features.norm(dim=-1, keepdim=True)
            query_embedding = query_features.cpu().numpy()
        
        # Calculate similarities
        similarities = cosine_similarity(query_embedding, self.image_embeddings).flatten()
        top_indices = similarities.argsort()[-top_k:][::-1]
        
        results = []
        for idx in top_indices:
            results.append({
                'image_path': self.image_paths[idx],
                'document': self.documents[idx] if idx < len(self.documents) else None,
                'similarity_score': float(similarities[idx]),
                'rank': len(results) + 1
            })
        
        return results
    
    def image_to_text_search(self, image_path: str, top_k: int = 5):
        """Search for books using an image (reverse image search)"""
        
        if self.text_embeddings is None:
            raise ValueError("No text embeddings found. Please encode texts first.")
        
        # Load and encode image
        image = Image.open(image_path).convert('RGB')
        
        with torch.no_grad():
            inputs = self.processor(images=[image], return_tensors="pt")
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            image_features = self.model.get_image_features(**inputs)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            query_embedding = image_features.cpu().numpy()
        
        # Calculate similarities
        similarities = cosine_similarity(query_embedding, self.text_embeddings).flatten()
        top_indices = similarities.argsort()[-top_k:][::-1]
        
        results = []
        for idx in top_indices:
            results.append({
                'document': self.documents[idx],
                'similarity_score': float(similarities[idx]),
                'rank': len(results) + 1
            })
        
        return results
    
    def cross_modal_fusion_search(self, text_query: str, image_query_path: str = None,
                                 top_k: int = 5, text_weight: float = 0.7):
        """Fusion search combining text and image queries"""
        
        scores = np.zeros(len(self.documents))
        
        # Text-based scores
        with torch.no_grad():
            inputs = self.processor(text=[text_query], return_tensors="pt",
                                   padding=True, truncation=True, max_length=77)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            text_features = self.model.get_text_features(**inputs)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
            text_embedding = text_features.cpu().numpy()
        
        text_similarities = cosine_similarity(text_embedding, self.text_embeddings).flatten()
        scores += text_weight * text_similarities
        
        # Image-based scores if provided
        if image_query_path and self.image_embeddings is not None:
            image = Image.open(image_query_path).convert('RGB')
            
            with torch.no_grad():
                inputs = self.processor(images=[image], return_tensors="pt")
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                image_features = self.model.get_image_features(**inputs)
                image_features = image_features / image_features.norm(dim=-1, keepdim=True)
                image_embedding = image_features.cpu().numpy()
            
            image_similarities = cosine_similarity(image_embedding, self.image_embeddings).flatten()
            scores += (1 - text_weight) * image_similarities
        
        # Get top results
        top_indices = scores.argsort()[-top_k:][::-1]
        
        results = []
        for idx in top_indices:
            results.append({
                'document': self.documents[idx],
                'fusion_score': float(scores[idx]),
                'text_similarity': float(text_similarities[idx]),
                'image_similarity': float(image_similarities[idx]) if image_query_path else None,
                'rank': len(results) + 1
            })
        
        return results
    
    def save_model(self, path: str = '../data/multimodal_search_engine.pkl'):
        """Save embeddings and metadata"""
        
        with open(path, 'wb') as f:
            pickle.dump({
                'image_embeddings': self.image_embeddings,
                'text_embeddings': self.text_embeddings,
                'documents': self.documents,
                'image_paths': self.image_paths
            }, f)
        print(f"💾 Multi-modal model saved to {path}")
    
    def load_model(self, path: str = '../data/multimodal_search_engine.pkl'):
        """Load embeddings and metadata"""
        
        print(f"📂 Loading multi-modal model from {path}...")
        with open(path, 'rb') as f:
            data = pickle.load(f)
            self.image_embeddings = data['image_embeddings']
            self.text_embeddings = data['text_embeddings']
            self.documents = data['documents']
            self.image_paths = data['image_paths']
        print(f"✅ Multi-modal model loaded successfully!")


# MAIN EXECUTION
if __name__ == "__main__":
    print("="*60)
    print("🎨 MULTI-MODAL SEARCH ENGINE - TRAINING")
    print("="*60)
    
    # Load documents from advanced search engine
    from search_engine_advanced import AdvancedSearchEngine
    
    print("\n📚 Loading documents...")
    text_engine = AdvancedSearchEngine()
    text_engine.load_model('../data/advanced_search_engine.pkl')
    documents = text_engine.documents[:100]  # Use first 100 for demo
    
    # Initialize multi-modal engine
    mm_engine = MultiModalSearchEngine()
    
    # Generate synthetic book covers (in real scenario, you'd have actual covers)
    image_paths = mm_engine.generate_synthetic_book_covers(documents)
    
    # Encode images and texts
    mm_engine.encode_images(image_paths)
    mm_engine.encode_texts(documents)
    
    # Save model
    mm_engine.save_model()
    
    # Test searches
    print("\n" + "="*60)
    print("🔍 TESTING MULTI-MODAL SEARCH")
    print("="*60)
    
    # Test 1: Text to Image Search
    query = "mystery detective crime"
    print(f"\n📝➡️🖼️ Text to Image Search: '{query}'")
    results = mm_engine.text_to_image_search(query, top_k=3)
    
    for r in results:
        print(f"  Rank {r['rank']}: Cover {os.path.basename(r['image_path'])}")
        if r['document']:
            print(f"    Book: {r['document']['title']}")
        print(f"    Score: {r['similarity_score']:.3f}")
    
    # Test 2: Cross-modal fusion
    print(f"\n🔀 Cross-Modal Fusion Search")
    results = mm_engine.cross_modal_fusion_search(
        text_query="science fiction space adventure",
        image_query_path=None,  # Would use an actual image in real scenario
        top_k=3
    )
    
    for r in results:
        print(f"  Rank {r['rank']}: {r['document']['title']}")
        print(f"    Fusion Score: {r['fusion_score']:.3f}")
        print(f"    Text Similarity: {r['text_similarity']:.3f}")
    
    print("\n" + "="*60)
    print("✅ MULTI-MODAL ENGINE TRAINING COMPLETE!")
    print("="*60)
