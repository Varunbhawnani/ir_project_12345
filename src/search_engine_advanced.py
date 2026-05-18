import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.naive_bayes import MultinomialNB
from sklearn.preprocessing import LabelEncoder
import pickle
import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import nltk
from sentence_transformers import SentenceTransformer
import warnings
import json
from datetime import datetime
from collections import defaultdict, Counter
import hashlib

warnings.filterwarnings('ignore')

# Download required NLTK data
try:
    nltk.data.find('corpora/stopwords')
except:
    nltk.download('stopwords')
    nltk.download('punkt')
    nltk.download('wordnet')

class QueryIntentClassifier:
    """Classify query intent to optimize search strategy"""
    
    def __init__(self):
        self.intent_patterns = {
            'navigational': ['isbn', 'find book', 'specific book', 'download', 'get book', 'book titled', 'exact'],
            'informational': ['what', 'how', 'why', 'explain', 'about', 'tell me', 'information'],
            'genre_browse': ['fiction', 'mystery', 'romance', 'thriller', 'fantasy', 'horror', 'biography', 'sci-fi', 'science fiction'],
            'recommendation': ['suggest', 'recommend', 'similar to', 'like', 'books about', 'looking for'],
            'author_search': ['by author', 'written by', 'books by', 'author:', 'works by', 'author name']
        }
    
    def classify(self, query):
        """Classify query intent"""
        import re
        query_lower = query.lower()
        
        # Check for ISBN (10 or 13 digits)
        if re.search(r'\b\d{10,13}\b', query):
            return 'navigational'
        
        # Check each intent pattern
        intent_scores = {}
        for intent, patterns in self.intent_patterns.items():
            score = sum(1 for pattern in patterns if pattern in query_lower)
            intent_scores[intent] = score
        
        # Return intent with highest score, default to informational
        if max(intent_scores.values()) > 0:
            return max(intent_scores, key=intent_scores.get)
        return 'informational'
    
    def get_search_weights(self, intent):
        """Get optimal VSM vs Semantic weights based on intent"""
        weights = {
            'navigational': (0.7, 0.3),  # More lexical matching
            'informational': (0.3, 0.7),  # More semantic understanding
            'genre_browse': (0.4, 0.6),   # Balanced
            'recommendation': (0.2, 0.8), # Heavy semantic
            'author_search': (0.8, 0.2)   # Heavy lexical
        }
        return weights.get(intent, (0.4, 0.6))

class PersonalizationEngine:
    """User personalization based on implicit feedback"""
    
    def __init__(self, embedding_dim=384, session_based=False):
        self.user_profiles = {}
        self.item_embeddings = {}
        self.interaction_history = defaultdict(list)
        self.embedding_dim = embedding_dim
        self.session_based = session_based  # NEW: flag for session-based mode
        
    def update_user_profile(self, user_id, doc_id, feedback_type='click', weight=1.0):
        """Update user profile based on interactions"""
        if user_id not in self.user_profiles:
            self.user_profiles[user_id] = {
                'embedding': np.zeros(self.embedding_dim),
                'genre_preferences': Counter(),
                'author_preferences': Counter(),
                'interaction_count': 0,
                'disliked_items': set()  # Track disliked items
            }
        
        profile = self.user_profiles[user_id]
        
        # Update embedding if available
        if doc_id in self.item_embeddings:
            if feedback_type == 'skip':
                # Negative feedback - subtract from profile and track
                feedback_weight = -weight * 0.5
                profile['disliked_items'].add(doc_id)
            else:
                # Positive feedback
                feedback_weight = weight
                # Remove from disliked if it was there
                profile['disliked_items'].discard(doc_id)
            
            profile['embedding'] += feedback_weight * self.item_embeddings[doc_id]
            
            # Normalize
            norm = np.linalg.norm(profile['embedding'])
            if norm > 0:
                profile['embedding'] /= norm
        
        profile['interaction_count'] += 1
        
        # Record interaction
        self.interaction_history[user_id].append({
            'doc_id': doc_id,
            'type': feedback_type,
            'timestamp': datetime.now().isoformat()
        })
    
    def get_personalized_score(self, user_id, doc_embedding, doc_metadata=None, doc_id=None):
        """Calculate personalized relevance score"""
        if user_id not in self.user_profiles:
            return 0.0
        
        profile = self.user_profiles[user_id]
        
        # Check if this item was explicitly disliked
        if doc_id is not None and doc_id in profile.get('disliked_items', set()):
            return -0.8  # Strong negative score for disliked items
        
        # Embedding similarity
        if profile['embedding'].any():
            sim_score = cosine_similarity(
                profile['embedding'].reshape(1, -1),
                doc_embedding.reshape(1, -1)
            )[0][0]
        else:
            sim_score = 0.0
        
        # Boost for preferred genres/authors if metadata available
        metadata_boost = 0.0
        if doc_metadata:
            if 'genre' in doc_metadata and doc_metadata['genre'] in profile['genre_preferences']:
                metadata_boost += 0.1 * profile['genre_preferences'][doc_metadata['genre']]
            if 'author' in doc_metadata and doc_metadata['author'] in profile['author_preferences']:
                metadata_boost += 0.15 * profile['author_preferences'][doc_metadata['author']]
        
        return sim_score + metadata_boost

class AdvancedSearchEngine:
    """Enhanced E-Library Search Engine with Advanced IR Features"""
    
    def __init__(self):
        # Basic components
        self.vectorizer = None
        self.tfidf_matrix = None
        self.semantic_embeddings = None
        self.documents = []
        self.stop_words = set(stopwords.words('english'))
        self.semantic_model = None
        
        # Advanced components
        self.intent_classifier = QueryIntentClassifier()
        self.personalization = PersonalizationEngine()
        self.query_cache = {}
        self.search_history = []
        
        # BM25 parameters
        self.avg_doc_length = 0
        self.doc_lengths = []
        self.doc_freqs = {}
        self.N = 0  # Total number of documents
        
        # Query expansion
        self.expansion_cache = {}
        
    def preprocess_text(self, text):
        """Clean and preprocess text"""
        if pd.isna(text):
            return ""
        text = str(text).lower()
        text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
        tokens = word_tokenize(text)
        tokens = [w for w in tokens if w not in self.stop_words and len(w) > 2]
        return ' '.join(tokens)
    
    def load_data(self, file_path, num_docs=5000):
        """Load book dataset from CSV with enhanced metadata extraction"""
        print(f"📚 Loading data from {file_path}...")
        
        try:
            df = pd.read_csv(file_path, encoding='utf-8', on_bad_lines='skip')
            df = df.head(num_docs)
            
            print(f"Columns found: {list(df.columns)}")
            
            for idx, row in df.iterrows():
                # Extract metadata
                title = str(row.get('title', row.get('Title', '')))
                author = str(row.get('authors', row.get('Author', 'Unknown')))
                
                # Extract ISBN (try multiple column names)
                isbn13 = str(row.get('isbn13', row.get('isbn', row.get('ISBN13', row.get('ISBN', '')))))
                isbn10 = str(row.get('isbn10', row.get('isbn', '')))
                
                # Extract genre if available
                genre = str(row.get('genre', row.get('categories', 'General')))
                
                # Extract year if available
                year = row.get('year', row.get('publication_year', 2020))
                
                # Extract rating if available
                rating = row.get('average_rating', row.get('rating', 3.5))
                
                # Get description/content
                content = ""
                for col in ['description', 'summary', 'Description', 'Summary']:
                    if col in df.columns and pd.notna(row.get(col)):
                        content = str(row[col])
                        break
                
                if not content or content == 'nan':
                    content = f"{title} by {author}"
                
                # Calculate document length for BM25
                doc_length = len(content.split())
                self.doc_lengths.append(doc_length)
                
                self.documents.append({
                    'id': idx,
                    'title': title[:200],
                    'content': content[:1000],
                    'author': author[:100],
                    'genre': genre,
                    'year': year,
                    'rating': float(rating) if rating else 3.5,
                    'doc_length': doc_length,
                    'isbn13': isbn13.replace('nan', '').replace('None', ''),
                    'isbn10': isbn10.replace('nan', '').replace('None', ''),
                    'isbn': isbn13.replace('nan', '').replace('None', '')  # Fallback field
                })
            
            self.N = len(self.documents)
            self.avg_doc_length = np.mean(self.doc_lengths)
            
            print(f"✅ Loaded {len(self.documents)} documents")
            return self.documents
            
        except Exception as e:
            print(f"❌ Error loading data: {e}")
            return []
    
    def build_vsm_index(self):
        """Build TF-IDF index with BM25 preparation"""
        print("\n🔨 Building VSM index with BM25 components...")
        
        texts = [self.preprocess_text(doc['title'] + ' ' + doc['content'])
                for doc in self.documents]
        
        self.vectorizer = TfidfVectorizer(
            max_features=5000,
            ngram_range=(1, 2),
            min_df=2,
            max_df=0.8,
            sublinear_tf=True
        )
        
        self.tfidf_matrix = self.vectorizer.fit_transform(texts)
        
        # Build document frequency dictionary for BM25
        feature_names = self.vectorizer.get_feature_names_out()
        for term_idx, term in enumerate(feature_names):
            df = np.sum(self.tfidf_matrix[:, term_idx].toarray() > 0)
            self.doc_freqs[term] = df
        
        print(f"✅ VSM index built: {self.tfidf_matrix.shape}")
        print(f"✅ Document frequencies calculated for BM25")
    
    def build_semantic_index(self):
        """Build semantic embeddings with personalization setup"""
        print("\n🧠 Loading semantic model...")
        
        self.semantic_model = SentenceTransformer('all-MiniLM-L6-v2')
        
        print("🔨 Building semantic index...")
        texts = [doc['title'] + ' ' + doc['content'][:500] for doc in self.documents]
        
        self.semantic_embeddings = self.semantic_model.encode(
            texts,
            show_progress_bar=True,
            batch_size=32
        )
        
        # Store embeddings for personalization
        for idx, embedding in enumerate(self.semantic_embeddings):
            self.personalization.item_embeddings[idx] = embedding
        
        print(f"✅ Semantic index built: {self.semantic_embeddings.shape}")
    
    def calculate_bm25_score(self, query, doc_idx, k1=1.2, b=0.75):
        """Calculate BM25 score for a document"""
        query_terms = self.preprocess_text(query).split()
        doc = self.documents[doc_idx]
        score = 0.0
        
        doc_length = doc['doc_length']
        
        for term in query_terms:
            if term in self.doc_freqs:
                # Term frequency in document
                tf = self.documents[doc_idx]['content'].lower().count(term)
                
                # Document frequency
                df = self.doc_freqs[term]
                
                # IDF calculation
                idf = np.log((self.N - df + 0.5) / (df + 0.5))
                
                # BM25 formula
                numerator = tf * (k1 + 1)
                denominator = tf + k1 * (1 - b + b * (doc_length / self.avg_doc_length))
                
                score += idf * (numerator / denominator)
        
        return score
    
    def query_expansion(self, query, top_k=3, num_terms=5, method='pseudo_relevance'):
        """Expand query using pseudo-relevance feedback or WordNet"""
        
        # Check cache
        cache_key = f"{query}_{top_k}_{num_terms}_{method}"
        if cache_key in self.expansion_cache:
            return self.expansion_cache[cache_key]
        
        expanded_terms = []
        
        if method == 'pseudo_relevance':
            # Get initial results
            initial_results = self.vsm_search(query, top_k)
            
            if initial_results:
                # Extract terms from top documents
                relevant_docs = ' '.join([r['document']['content'] for r in initial_results[:top_k]])
                relevant_processed = self.preprocess_text(relevant_docs)
                
                # Get term frequencies
                doc_vec = self.vectorizer.transform([relevant_processed])
                feature_names = self.vectorizer.get_feature_names_out()
                tfidf_scores = doc_vec.toarray()[0]
                
                # Get top expansion terms not in original query
                query_terms = set(self.preprocess_text(query).split())
                term_scores = [(feature_names[i], tfidf_scores[i])
                              for i in range(len(feature_names))
                              if tfidf_scores[i] > 0 and feature_names[i] not in query_terms]
                
                term_scores.sort(key=lambda x: x[1], reverse=True)
                expanded_terms = [term for term, score in term_scores[:num_terms]]
        
        # Build expanded query
        expanded_query = query + ' ' + ' '.join(expanded_terms)
        
        # Cache result
        self.expansion_cache[cache_key] = (expanded_query, expanded_terms)
        
        return expanded_query, expanded_terms
    
    def vsm_search(self, query, top_k=10):
        """VSM search with BM25 scoring option"""
        query_vec = self.vectorizer.transform([self.preprocess_text(query)])
        
        # TF-IDF similarity
        tfidf_similarities = cosine_similarity(query_vec, self.tfidf_matrix).flatten()
        
        # Calculate BM25 scores
        bm25_scores = np.array([self.calculate_bm25_score(query, i)
                                for i in range(len(self.documents))])
        
        # Normalize scores
        if bm25_scores.max() > 0:
            bm25_scores = bm25_scores / bm25_scores.max()
        
        # Combine TF-IDF and BM25 (weighted average)
        combined_scores = 0.6 * tfidf_similarities + 0.4 * bm25_scores
        
        top_indices = combined_scores.argsort()[-top_k:][::-1]
        
        results = []
        for idx in top_indices:
            if combined_scores[idx] > 0:
                results.append({
                    'document': self.documents[idx],
                    'score': float(combined_scores[idx]),
                    'tfidf_score': float(tfidf_similarities[idx]),
                    'bm25_score': float(bm25_scores[idx]),
                    'rank': len(results) + 1
                })
        return results
    
    def semantic_search(self, query, top_k=10):
        """Semantic search with query understanding"""
        query_embedding = self.semantic_model.encode([query])
        similarities = cosine_similarity(query_embedding, self.semantic_embeddings).flatten()
        top_indices = similarities.argsort()[-top_k:][::-1]
        
        results = []
        for idx in top_indices:
            results.append({
                'document': self.documents[idx],
                'score': float(similarities[idx]),
                'rank': len(results) + 1
            })
        return results
    
    def hybrid_search(self, query, top_k=10, vsm_weight=None, semantic_weight=None,
                 use_expansion=True, personalized_for_user=None):
        """Advanced hybrid search with all features"""
        
        # Classify query intent
        intent = self.intent_classifier.classify(query)
        
        # ===== HANDLE NAVIGATIONAL QUERIES (ISBN, specific book) =====
        # ===== HANDLE NAVIGATIONAL QUERIES (ISBN, specific book) =====
        if intent == 'navigational':
            # Extract ISBN if present
            import re
            isbn_match = re.search(r'\b\d{10,13}\b', query)
            if isbn_match:
                isbn = isbn_match.group()
                print(f"🔍 Searching for ISBN: {isbn}")
                
                # Direct ISBN lookup - check ALL possible ISBN fields
                for idx, doc in enumerate(self.documents):
                    # Check isbn13, isbn10, isbn fields and content
                    isbn_fields = [
                        str(doc.get('isbn13', '')),
                        str(doc.get('isbn10', '')),
                        str(doc.get('isbn', '')),
                        doc.get('title', ''),
                        doc.get('content', '')
                    ]
                    
                    # Check if ISBN appears in any field
                    for field in isbn_fields:
                        if isbn in str(field).replace('-', '').replace(' ', ''):
                            print(f"✅ Found match: {doc['title']}")
                            return [{
                                'document': doc,
                                'vsm_score': 1.0,
                                'bm25_score': 1.0,
                                'semantic_score': 1.0,
                                'hybrid_score': 1.0,
                                'rank': 1,
                                'query_intent': intent,
                                'expanded_terms': [],
                                'match_type': 'exact_isbn'
                            }]
                
                print(f"❌ No book found with ISBN: {isbn}")
            
            # Check for exact title match
            query_lower = query.lower()
            for idx, doc in enumerate(self.documents):
                if query_lower in doc['title'].lower():
                    return [{
                        'document': doc,
                        'vsm_score': 1.0,
                        'bm25_score': 1.0,
                        'semantic_score': 1.0,
                        'hybrid_score': 1.0,
                        'rank': 1,
                        'query_intent': intent,
                        'expanded_terms': [],
                        'match_type': 'exact_title'
                    }]
        
        # ===== HANDLE AUTHOR SEARCH =====
        if intent == 'author_search':
            query_lower = query.lower().replace('by author', '').replace('written by', '').replace('books by', '').replace('author:', '').strip()
            
            author_results = []
            for idx, doc in enumerate(self.documents):
                if query_lower in doc.get('author', '').lower():
                    # Calculate scores normally for ranking
                    query_vec = self.vectorizer.transform([self.preprocess_text(query)])
                    vsm_score = cosine_similarity(query_vec, self.tfidf_matrix[idx:idx+1]).flatten()[0]
                    
                    query_embedding = self.semantic_model.encode([query])
                    semantic_score = cosine_similarity(query_embedding, self.semantic_embeddings[idx:idx+1]).flatten()[0]
                    
                    author_results.append({
                        'document': doc,
                        'vsm_score': float(vsm_score),
                        'bm25_score': float(vsm_score),
                        'semantic_score': float(semantic_score),
                        'hybrid_score': 0.8 * float(vsm_score) + 0.2 * float(semantic_score),
                        'rank': len(author_results) + 1,
                        'query_intent': intent,
                        'expanded_terms': [],
                        'match_type': 'author_match'
                    })
            
            if author_results:
                # Sort by hybrid score
                author_results.sort(key=lambda x: x['hybrid_score'], reverse=True)
                # Update ranks
                for i, r in enumerate(author_results[:top_k], 1):
                    r['rank'] = i
                return author_results[:top_k]
        
        # Get optimal weights if not provided
        if vsm_weight is None or semantic_weight is None:
            vsm_weight, semantic_weight = self.intent_classifier.get_search_weights(intent)
        
        # Query expansion
        original_query = query
        if use_expansion and intent in ['informational', 'recommendation']:
            query, expansion_terms = self.query_expansion(query)
        else:
            expansion_terms = []
        
        # VSM scores (with BM25)
        query_vec = self.vectorizer.transform([self.preprocess_text(query)])
        vsm_scores = cosine_similarity(query_vec, self.tfidf_matrix).flatten()
        
        # BM25 scores
        bm25_scores = np.array([self.calculate_bm25_score(query, i)
                                for i in range(len(self.documents))])
        
        # Combine VSM and BM25
        if bm25_scores.max() > 0:
            bm25_scores = bm25_scores / bm25_scores.max()
        vsm_combined = 0.6 * vsm_scores + 0.4 * bm25_scores
        
        # Semantic scores
        query_embedding = self.semantic_model.encode([query])
        semantic_scores = cosine_similarity(query_embedding, self.semantic_embeddings).flatten()
        
        # Normalize scores
        if vsm_combined.max() > 0:
            vsm_combined = vsm_combined / vsm_combined.max()
        if semantic_scores.max() > 0:
            semantic_scores = semantic_scores / semantic_scores.max()
        
        # Combine scores
        hybrid_scores = (vsm_weight * vsm_combined) + (semantic_weight * semantic_scores)
        
        # Apply personalization if user is specified
        # Apply personalization if user is specified
        if personalized_for_user:
            for idx in range(len(hybrid_scores)):
                personal_score = self.personalization.get_personalized_score(
                    personalized_for_user,
                    self.semantic_embeddings[idx],
                    {'genre': self.documents[idx].get('genre'),
                     'author': self.documents[idx].get('author')},
                    doc_id=idx  # Pass document ID to check if disliked
                )
                # Blend personalization (20% weight)
                # personal_score can be negative for disliked items
                if personal_score < 0:
                    # For negative scores (dislikes), reduce the ranking
                    hybrid_scores[idx] = hybrid_scores[idx] * (1 + personal_score)
                else:
                    # For positive scores (likes), boost the ranking
                    hybrid_scores[idx] = 0.8 * hybrid_scores[idx] + 0.2 * personal_score
        
        # Get top results
        top_indices = hybrid_scores.argsort()[-top_k:][::-1]
        
        results = []
        for idx in top_indices:
            if hybrid_scores[idx] > 0:
                result = {
                    'document': self.documents[idx],
                    'vsm_score': float(vsm_scores[idx]),
                    'bm25_score': float(bm25_scores[idx]),
                    'semantic_score': float(semantic_scores[idx]),
                    'hybrid_score': float(hybrid_scores[idx]),
                    'rank': len(results) + 1,
                    'query_intent': intent,
                    'expanded_terms': expansion_terms
                }
                
                if personalized_for_user:
                    result['personalized'] = True
                
                results.append(result)
        
        # Log search
        self.search_history.append({
            'query': original_query,
            'expanded_query': query if use_expansion else original_query,
            'intent': intent,
            'timestamp': datetime.now().isoformat(),
            'num_results': len(results),
            'user': personalized_for_user
        })
        
        return results
    
    def get_facets(self, results):
        """Extract facets from search results for filtering"""
        facets = {
            'genres': Counter(),
            'authors': Counter(),
            'years': Counter(),
            'ratings': {'5_stars': 0, '4_stars': 0, '3_stars': 0, '2_stars': 0, '1_star': 0}
        }
        
        for result in results:
            doc = result['document']
            
            # Genre facets
            if 'genre' in doc:
                facets['genres'][doc['genre']] += 1
            
            # Author facets
            if 'author' in doc:
                facets['authors'][doc['author']] += 1
            
            # Year facets
            if 'year' in doc:
                year_range = f"{(doc['year'] // 10) * 10}s"
                facets['years'][year_range] += 1
            
            # Rating facets
            if 'rating' in doc:
                rating = doc['rating']
                if rating >= 4.5:
                    facets['ratings']['5_stars'] += 1
                elif rating >= 3.5:
                    facets['ratings']['4_stars'] += 1
                elif rating >= 2.5:
                    facets['ratings']['3_stars'] += 1
                elif rating >= 1.5:
                    facets['ratings']['2_stars'] += 1
                else:
                    facets['ratings']['1_star'] += 1
        
        return facets
    
    def more_like_this(self, doc_id, top_k=5):
        """Find similar documents to a given document"""
        if doc_id >= len(self.documents):
            return []
        
        # Use the document's embedding
        doc_embedding = self.semantic_embeddings[doc_id].reshape(1, -1)
        
        # Calculate similarities with all other documents
        similarities = cosine_similarity(doc_embedding, self.semantic_embeddings).flatten()
        
        # Exclude the document itself
        similarities[doc_id] = -1
        
        # Get top similar documents
        top_indices = similarities.argsort()[-top_k:][::-1]
        
        results = []
        for idx in top_indices:
            if similarities[idx] > 0:
                results.append({
                    'document': self.documents[idx],
                    'similarity_score': float(similarities[idx]),
                    'rank': len(results) + 1
                })
        
        return results
    
    def get_query_suggestions(self, partial_query, max_suggestions=5):
        """Generate query suggestions based on search history"""
        if not partial_query:
            return []
        
        partial_lower = partial_query.lower()
        suggestions = []
        
        # Get suggestions from search history
        for entry in self.search_history[-100:]:  # Last 100 searches
            if entry['query'].lower().startswith(partial_lower):
                suggestions.append(entry['query'])
        
        # Remove duplicates and limit
        suggestions = list(dict.fromkeys(suggestions))[:max_suggestions]
        
        return suggestions
    
    def save_model(self, path='../data/advanced_search_engine.pkl'):
        """Save trained model with all components (but NOT personalization data)"""
        with open(path, 'wb') as f:
            pickle.dump({
                'vectorizer': self.vectorizer,
                'tfidf_matrix': self.tfidf_matrix,
                'semantic_embeddings': self.semantic_embeddings,
                'documents': self.documents,
                'doc_freqs': self.doc_freqs,
                'avg_doc_length': self.avg_doc_length,
                'doc_lengths': self.doc_lengths,
                'N': self.N,
                'search_history': self.search_history,
                # NOTE: personalization is NOT saved - it's session-based only
            }, f)
        print(f"\n💾 Advanced model saved to {path}")
    
    def load_model(self, path='../data/advanced_search_engine.pkl'):
        """Load trained model with all components"""
        print(f"📂 Loading advanced model from {path}...")
        with open(path, 'rb') as f:
            data = pickle.load(f)
            self.vectorizer = data['vectorizer']
            self.tfidf_matrix = data['tfidf_matrix']
            self.semantic_embeddings = data['semantic_embeddings']
            self.documents = data['documents']
            self.doc_freqs = data.get('doc_freqs', {})
            self.avg_doc_length = data.get('avg_doc_length', 100)
            self.doc_lengths = data.get('doc_lengths', [])
            self.N = data.get('N', len(self.documents))
            self.search_history = data.get('search_history', [])
            # NOTE: personalization is NOT loaded - will be created fresh per session
        
        # Reinitialize components
        self.semantic_model = SentenceTransformer('all-MiniLM-L6-v2')
        self.intent_classifier = QueryIntentClassifier()
        # Create fresh personalization engine (will be overwritten in app for session-based)
        self.personalization = PersonalizationEngine()
        
        print(f"✅ Advanced model loaded successfully!")


# MAIN EXECUTION
if __name__ == "__main__":
    print("="*60)
    print("🚀 ADVANCED E-LIBRARY SEARCH ENGINE - TRAINING")
    print("="*60)
    
    # Initialize
    engine = AdvancedSearchEngine()
    
    # Load data
    engine.load_data('../data/books.csv', num_docs=5000)
    
    if len(engine.documents) == 0:
        print("❌ No documents loaded! Check your CSV file.")
        exit(1)
    
    # Build indexes
    engine.build_vsm_index()
    engine.build_semantic_index()
    
    # Save model
    engine.save_model()
    
    # Test advanced features
    print("\n" + "="*60)
    print("🔬 TESTING ADVANCED FEATURES")
    print("="*60)
    
    test_query = "mystery detective stories"
    
    # Test 1: Query Intent Classification
    print(f"\n📍 Query: '{test_query}'")
    intent = engine.intent_classifier.classify(test_query)
    print(f"🎯 Detected Intent: {intent}")
    
    # Test 2: Query Expansion
    expanded_query, expansion_terms = engine.query_expansion(test_query)
    print(f"📝 Expanded Terms: {expansion_terms}")
    
    # Test 3: Hybrid Search with all features
    results = engine.hybrid_search(test_query, top_k=3, use_expansion=True)
    
    print(f"\n🔍 Found {len(results)} results:")
    for r in results:
        print(f"\nRank {r['rank']}: {r['document']['title']}")
        print(f"  Author: {r['document']['author']}")
        print(f"  VSM: {r['vsm_score']:.3f} | BM25: {r['bm25_score']:.3f}")
        print(f"  Semantic: {r['semantic_score']:.3f} | Hybrid: {r['hybrid_score']:.3f}")
        print(f"  Intent: {r['query_intent']}")
    
    # Test 4: More Like This
    if results:
        print(f"\n📚 Books similar to '{results[0]['document']['title']}':")
        similar = engine.more_like_this(results[0]['document']['id'], top_k=3)
        for s in similar:
            print(f"  - {s['document']['title']} (similarity: {s['similarity_score']:.3f})")
    
    # Test 5: Facets
    facets = engine.get_facets(results)
    print(f"\n📊 Search Facets:")
    print(f"  Genres: {dict(facets['genres'])}")
    print(f"  Authors: {dict(facets['authors'])}")
    
    print("\n" + "="*60)
    print("✅ ADVANCED ENGINE TRAINING COMPLETE!")
    print("="*60)
