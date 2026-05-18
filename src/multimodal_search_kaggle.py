"""
Multimodal Search Engine using Kaggle Book Covers Dataset
WITH COLOR-ACCURATE VISUAL SEARCH
"""

import numpy as np
from PIL import Image
import torch
from transformers import CLIPProcessor, CLIPModel
import pickle
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans
import os
import pandas as pd
from tqdm import tqdm

class KaggleBookCoverSearch:
    """Multi-modal search engine with COLOR-ACCURATE visual search"""
    
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"ðŸ–¥ï¸ Using device: {self.device}")
        
        # Load CLIP model
        print("ðŸ“¦ Loading CLIP model...")
        self.model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        self.model.to(self.device)
        self.model.eval()
        
        # Storage
        self.image_embeddings = None
        self.text_embeddings = None
        self.color_features = None
        self.dominant_colors = None
        self.books_df = None
        
    def load_kaggle_dataset(self, csv_path, base_image_dir):
        """Load Kaggle book covers dataset"""
        print(f"ðŸ“š Loading Kaggle dataset from {csv_path}...")
        
        self.books_df = pd.read_csv(csv_path)
        print(f"   Total records in CSV: {len(self.books_df)}")
        
        self.books_df = self._fix_image_paths(self.books_df, base_image_dir)
        self.books_df = self.books_df[self.books_df['fixed_img_paths'].notna()].reset_index(drop=True)
        
        print(f"âœ… Loaded {len(self.books_df)} books with valid images")
        print(f"ðŸ“Š Categories: {self.books_df['category'].nunique()} unique")
        
        return self.books_df
    
    def _fix_image_paths(self, df, base_dir):
        """Fix image paths to match actual locations"""
        print("ðŸ”§ Fixing image paths...")
        fixed_paths = []
        
        for idx, row in df.iterrows():
            original_path = row['img_paths']
            parts = str(original_path).split('/')
            
            if len(parts) >= 2:
                category = parts[-2]
                filename = parts[-1]
                correct_path = os.path.join(base_dir, category, filename)
                
                if os.path.exists(correct_path):
                    fixed_paths.append(correct_path)
                else:
                    correct_path = os.path.join(base_dir, filename)
                    if os.path.exists(correct_path):
                        fixed_paths.append(correct_path)
                    else:
                        fixed_paths.append(None)
            else:
                fixed_paths.append(None)
            
            if (idx + 1) % 5000 == 0:
                print(f"  Processed {idx + 1} paths...")
        
        df['fixed_img_paths'] = fixed_paths
        valid = sum(1 for p in fixed_paths if p is not None)
        print(f"  âœ“ Found {valid}/{len(df)} valid images ({valid/len(df)*100:.1f}%)")
        
        return df
    
    def extract_color_features(self, image_path):
        """Extract COLOR HISTOGRAM features from image"""
        try:
            img = Image.open(image_path).convert('RGB')
            img_array = np.array(img.resize((100, 100)))
            
            hist_r = np.histogram(img_array[:,:,0], bins=16, range=(0,256))[0]
            hist_g = np.histogram(img_array[:,:,1], bins=16, range=(0,256))[0]
            hist_b = np.histogram(img_array[:,:,2], bins=16, range=(0,256))[0]
            
            color_histogram = np.concatenate([hist_r, hist_g, hist_b])
            color_histogram = color_histogram / (color_histogram.sum() + 1e-10)
            
            pixels = img_array.reshape(-1, 3)
            kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
            kmeans.fit(pixels)
            
            labels = kmeans.labels_
            label_counts = np.bincount(labels)
            dominant_label = label_counts.argmax()
            dominant_color = kmeans.cluster_centers_[dominant_label]
            
            return color_histogram, dominant_color
            
        except Exception as e:
            print(f"  âš ï¸ Error extracting color from {image_path}: {e}")
            return np.zeros(48), np.array([128, 128, 128])
    
    def encode_all_images(self, batch_size=32, max_images=None):
        """Encode all book cover images using CLIP + COLOR FEATURES"""
        print(f"ðŸ–¼ï¸ Encoding book cover images WITH COLOR FEATURES...")
        
        if max_images:
            df_subset = self.books_df.head(max_images)
            print(f"   Using first {max_images} images for testing")
        else:
            df_subset = self.books_df
        
        embeddings = []
        failed = 0
        
        for i in tqdm(range(0, len(df_subset), batch_size), desc="CLIP encoding"):
            batch_df = df_subset.iloc[i:i+batch_size]
            batch_images = []
            
            for idx, row in batch_df.iterrows():
                img_path = row['fixed_img_paths']
                try:
                    image = Image.open(img_path).convert('RGB')
                    batch_images.append(image)
                except Exception as e:
                    batch_images.append(Image.new('RGB', (224, 224), color=(128, 128, 128)))
                    failed += 1
            
            with torch.no_grad():
                inputs = self.processor(images=batch_images, return_tensors="pt", padding=True)
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                image_features = self.model.get_image_features(**inputs)
                image_features = image_features / image_features.norm(dim=-1, keepdim=True)
                embeddings.append(image_features.cpu().numpy())
        
        self.image_embeddings = np.vstack(embeddings)
        
        print(f"âœ… CLIP embeddings: {self.image_embeddings.shape}")
        if failed > 0:
            print(f"   âš ï¸ Failed to load {failed} images (using placeholders)")
        
        print(f"\nðŸŽ¨ Extracting color features (histograms + dominant colors)...")
        
        color_histograms = []
        dominant_colors = []
        
        for idx, row in tqdm(df_subset.iterrows(), total=len(df_subset), desc="Color extraction"):
            img_path = row['fixed_img_paths']
            hist, dom_color = self.extract_color_features(img_path)
            color_histograms.append(hist)
            dominant_colors.append(dom_color)
        
        self.color_features = np.array(color_histograms)
        self.dominant_colors = np.array(dominant_colors)
        
        print(f"âœ… Color features extracted:")
        print(f"   - Histograms: {self.color_features.shape}")
        print(f"   - Dominant colors: {self.dominant_colors.shape}")
        
        return self.image_embeddings, self.color_features
    
    def encode_all_texts(self, batch_size=64):
        """Encode book text descriptions using CLIP"""
        print(f"ðŸ“ Encoding book text descriptions...")
        
        texts = []
        for idx, row in self.books_df.iterrows():
            text = f"{row['name']} by {row['author']}. Category: {row['category']}"
            texts.append(text)
        
        embeddings = []
        
        for i in tqdm(range(0, len(texts), batch_size), desc="Encoding texts"):
            batch_texts = texts[i:i + batch_size]
            
            with torch.no_grad():
                inputs = self.processor(text=batch_texts, return_tensors="pt",
                                       padding=True, truncation=True, max_length=77)
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                text_features = self.model.get_text_features(**inputs)
                text_features = text_features / text_features.norm(dim=-1, keepdim=True)
                embeddings.append(text_features.cpu().numpy())
        
        self.text_embeddings = np.vstack(embeddings)
        
        print(f"âœ… Encoded {len(self.text_embeddings)} texts")
        print(f"   Embedding shape: {self.text_embeddings.shape}")
        
        return self.text_embeddings
    
    def search_by_visual_description(self, query, top_k=10):
        """Search for book covers using visual description WITH SMART QUERY PARSING"""
        if self.image_embeddings is None:
            raise ValueError("Please encode images first using encode_all_images()")
        
        print(f"\nðŸ” Searching for: '{query}'")
        
        color_keywords = {
            'red': np.array([200, 50, 50]),
            'green': np.array([50, 200, 50]),
            'blue': np.array([50, 50, 200]),
            'yellow': np.array([200, 200, 50]),
            'orange': np.array([255, 140, 0]),
            'purple': np.array([150, 50, 150]),
            'pink': np.array([255, 150, 200]),
            'brown': np.array([139, 90, 43]),
            'black': np.array([30, 30, 30]),
            'white': np.array([240, 240, 240]),
            'gray': np.array([128, 128, 128]),
            'grey': np.array([128, 128, 128]),
            'cyan': np.array([0, 200, 200]),
            'dark': np.array([50, 50, 50]),
            'bright': np.array([220, 220, 220]),
            'light': np.array([200, 200, 200])
        }
        
        query_lower = query.lower()
        detected_color = None
        color_name = None
        
        for color_key, rgb_value in color_keywords.items():
            if color_key in query_lower:
                detected_color = rgb_value
                color_name = color_key
                break
        
        query_words = query_lower.split()
        non_color_words = [w for w in query_words if w not in color_keywords.keys()
                           and w not in ['book', 'cover', 'colored']]
        
        has_specific_title = len(non_color_words) >= 2
        
        if detected_color is not None and has_specific_title:
            semantic_weight = 0.7
            color_weight = 0.3
            search_mode = "title+color"
            print(f"ðŸŽ¨ Color '{color_name}' detected in specific title query")
            print(f"   Using: 70% semantic (find book) + 30% color (filter by color)")
        elif detected_color is not None:
            semantic_weight = 0.4
            color_weight = 0.6
            search_mode = "color-focused"
            print(f"ðŸŽ¨ Color detected: '{color_name}' - applying color matching...")
            print(f"   Using: 40% semantic + 60% color matching")
        else:
            semantic_weight = 1.0
            color_weight = 0.0
            search_mode = "semantic-only"
            print(f"   Using: 100% semantic matching (no color keyword detected)")
        
        with torch.no_grad():
            inputs = self.processor(text=[query], return_tensors="pt",
                                   padding=True, truncation=True, max_length=77)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            query_features = self.model.get_text_features(**inputs)
            query_features = query_features / query_features.norm(dim=-1, keepdim=True)
            query_embedding = query_features.cpu().numpy()
        
        clip_similarities = cosine_similarity(query_embedding, self.image_embeddings).flatten()
        
        if detected_color is not None and self.dominant_colors is not None:
            color_distances = np.linalg.norm(self.dominant_colors - detected_color, axis=1)
            color_similarities = np.exp(-color_distances / 100)
            
            if clip_similarities.max() > 0:
                clip_similarities_norm = clip_similarities / clip_similarities.max()
            else:
                clip_similarities_norm = clip_similarities
            
            if color_similarities.max() > 0:
                color_similarities_norm = color_similarities / color_similarities.max()
            else:
                color_similarities_norm = color_similarities
            
            final_similarities = (semantic_weight * clip_similarities_norm +
                                color_weight * color_similarities_norm)
        else:
            final_similarities = clip_similarities
            color_similarities = np.zeros_like(clip_similarities)
        
        top_indices = final_similarities.argsort()[-top_k:][::-1]
        
        results = []
        for rank, idx in enumerate(top_indices, 1):
            book = self.books_df.iloc[idx]
            
            result = {
                'rank': rank,
                'title': book['name'],
                'author': book['author'],
                'category': book['category'],
                'image_path': book['fixed_img_paths'],
                'similarity_score': float(final_similarities[idx]),
                'clip_score': float(clip_similarities[idx]),
                'search_mode': search_mode,
                'isbn': book.get('isbn', 'N/A'),
                'rating': book.get('book_depository_stars', 'N/A'),
                'price': book.get('price', 'N/A')
            }
            
            if detected_color is not None and self.dominant_colors is not None:
                result['color_score'] = float(color_similarities[idx])
                result['dominant_rgb'] = self.dominant_colors[idx].tolist()
                result['detected_color'] = color_name
            
            results.append(result)
        
        return results
    
    def search_by_text(self, query, top_k=10):
        """Search for books using text query"""
        if self.text_embeddings is None:
            raise ValueError("Please encode texts first using encode_all_texts()")
        
        print(f"\nðŸ“– Searching for: '{query}'")
        
        with torch.no_grad():
            inputs = self.processor(text=[query], return_tensors="pt",
                                   padding=True, truncation=True, max_length=77)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            query_features = self.model.get_text_features(**inputs)
            query_features = query_features / query_features.norm(dim=-1, keepdim=True)
            query_embedding = query_features.cpu().numpy()
        
        similarities = cosine_similarity(query_embedding, self.text_embeddings).flatten()
        top_indices = similarities.argsort()[-top_k:][::-1]
        
        results = []
        for rank, idx in enumerate(top_indices, 1):
            book = self.books_df.iloc[idx]
            results.append({
                'rank': rank,
                'title': book['name'],
                'author': book['author'],
                'category': book['category'],
                'image_path': book['fixed_img_paths'],
                'similarity_score': float(similarities[idx]),
                'isbn': book.get('isbn', 'N/A'),
                'rating': book.get('book_depository_stars', 'N/A')
            })
        
        return results
    
    def search_by_image(self, image_path, top_k=10):
        """Reverse image search - find similar book covers"""
        if self.image_embeddings is None:
            raise ValueError("Please encode images first using encode_all_images()")
        
        print(f"\nðŸ–¼ï¸ Searching for similar covers to: {os.path.basename(image_path)}")
        
        image = Image.open(image_path).convert('RGB')
        
        with torch.no_grad():
            inputs = self.processor(images=[image], return_tensors="pt")
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            image_features = self.model.get_image_features(**inputs)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            query_embedding = image_features.cpu().numpy()
        
        similarities = cosine_similarity(query_embedding, self.image_embeddings).flatten()
        top_indices = similarities.argsort()[-top_k:][::-1]
        
        results = []
        for rank, idx in enumerate(top_indices, 1):
            book = self.books_df.iloc[idx]
            results.append({
                'rank': rank,
                'title': book['name'],
                'author': book['author'],
                'category': book['category'],
                'image_path': book['fixed_img_paths'],
                'similarity_score': float(similarities[idx])
            })
        
        return results
    
    def save_model(self, path='../data/kaggle_multimodal_search.pkl'):
        """Save embeddings and metadata"""
        print(f"ðŸ’¾ Saving model to {path}...")
        with open(path, 'wb') as f:
            pickle.dump({
                'image_embeddings': self.image_embeddings,
                'text_embeddings': self.text_embeddings,
                'color_features': self.color_features,
                'dominant_colors': self.dominant_colors,
                'books_df': self.books_df
            }, f)
        print(f"âœ… Model saved successfully!")
    
    def load_model(self, path='../data/kaggle_multimodal_search.pkl'):
        """Load embeddings and metadata"""
        print(f"ðŸ“‚ Loading model from {path}...")
        with open(path, 'rb') as f:
            data = pickle.load(f)
            self.image_embeddings = data['image_embeddings']
            self.text_embeddings = data['text_embeddings']
            self.color_features = data.get('color_features')
            self.dominant_colors = data.get('dominant_colors')
            self.books_df = data['books_df']
        print(f"âœ… Model loaded! Ready for search.")
        print(f"   Images: {len(self.image_embeddings)}")
        print(f"   Books: {len(self.books_df)}")
        print(f"   Color features: {'Available' if self.color_features is not None else 'Not available'}")


if __name__ == "__main__":
    print("="*70)
    print("ðŸŽ¨ KAGGLE BOOK COVER MULTIMODAL SEARCH - TRAINING (WITH COLOR)")
    print("="*70)
    
    search_engine = KaggleBookCoverSearch()
    
    CSV_PATH = '../data/main_dataset.csv'
    IMAGE_DIR = '../data/book-covers'
    
    if not os.path.exists(CSV_PATH):
        print(f"\nâŒ ERROR: CSV file not found at {CSV_PATH}")
        exit(1)
    
    if not os.path.exists(IMAGE_DIR):
        print(f"\nâŒ ERROR: Image directory not found at {IMAGE_DIR}")
        exit(1)
    
    df = search_engine.load_kaggle_dataset(CSV_PATH, IMAGE_DIR)
    
    if len(df) == 0:
        print("\nâŒ ERROR: No valid images found!")
        exit(1)
    
    print("\n" + "="*70)
    print("STEP 1: Encoding Images + Color Features")
    print("="*70)
    search_engine.encode_all_images(batch_size=32, max_images=4000)
    
    print("\n" + "="*70)
    print("STEP 2: Encoding Text Descriptions")
    print("="*70)
    search_engine.encode_all_texts(batch_size=64)
    
    print("\n" + "="*70)
    print("STEP 3: Saving Model")
    print("="*70)
    search_engine.save_model('../data/kaggle_multimodal_search.pkl')
    
    print("\n" + "="*70)
    print("ðŸ§ª TESTING COLOR-ACCURATE SEARCHES")
    print("="*70)
    
    print("\n--- Test 1: Specific Title with Color ---")
    results = search_engine.search_by_visual_description("starving the anxiety blue", top_k=5)
    for r in results:
        print(f"\n{r['rank']}. {r['title'][:50]}")
        print(f"   Overall: {r['similarity_score']:.3f}")
        if 'color_score' in r:
            print(f"   CLIP: {r['clip_score']:.3f} | Color: {r['color_score']:.3f}")
    
    print("\n--- Test 2: Pure Color Query ---")
    results = search_engine.search_by_visual_description("green book cover", top_k=5)
    for r in results:
        print(f"\n{r['rank']}. {r['title'][:50]}")
        if 'color_score' in r:
            print(f"   Color Score: {r['color_score']:.3f}")
    
    print("\n" + "="*70)
    print("âœ… TRAINING COMPLETE!")
    print("="*70)
