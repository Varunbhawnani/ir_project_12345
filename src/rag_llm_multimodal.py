from groq import Groq
from search_engine_advanced import AdvancedSearchEngine
import os

class MultimodalRAGWithLLM:
    """Enhanced RAG system with intelligent query routing"""
    
    def __init__(self, api_key=None):
        # Load text search engine (books.csv - 5000 books)
        print("📚 Loading text search engine (books.csv)...")
        self.search_engine = AdvancedSearchEngine()
        self.search_engine.load_model('../data/advanced_search_engine.pkl')
        print(f"   ✅ Loaded {len(self.search_engine.documents)} books from books.csv")
        
        # Load visual search engine (Kaggle dataset - 32000 books)
        print("🎨 Loading visual search engine (Kaggle dataset)...")
        try:
            from multimodal_search_kaggle import KaggleBookCoverSearch
            self.visual_engine = KaggleBookCoverSearch()
            self.visual_engine.load_model('../data/kaggle_multimodal_search.pkl')
            self.visual_available = True
            print(f"   ✅ Loaded {len(self.visual_engine.books_df)} books from Kaggle dataset")
        except Exception as e:
            print(f"   ⚠️ Visual search not available: {e}")
            self.visual_engine = None
            self.visual_available = False
        
        # Initialize Groq client
        if api_key is None:
            api_key = os.environ.get("GROQ_API_KEY", "")
        
        if not api_key:
            raise ValueError("⚠️ No API key provided!")
        
        self.client = Groq(api_key=api_key)
        print("✅ Multimodal RAG system initialized!\n")
    
    def analyze_query_type(self, query):
        """
        Analyze query to determine type:
        - Pure text (no color keywords)
        - Pure visual (only color keywords)
        - Hybrid (both content + color keywords)
        """
        color_keywords = [
            'blue', 'green', 'red', 'yellow', 'orange', 'purple',
            'pink', 'black', 'white', 'dark', 'bright', 'colorful',
            'cyan', 'gray', 'grey', 'brown', 'cover', 'design',
            'minimalist', 'vintage', 'modern', 'image'
        ]
        
        # Content keywords (genres, topics, etc.)
        content_keywords = [
            'mystery', 'thriller', 'horror', 'romance', 'science fiction',
            'fantasy', 'biography', 'history', 'adventure', 'detective',
            'crime', 'love', 'magic', 'space', 'war', 'business', 'self-help',
            'cooking', 'travel', 'poetry', 'drama', 'comedy', 'action', 'sci-fi',
            'fiction', 'non-fiction', 'philosophy', 'psychology', 'religion',
            'art', 'music', 'sports', 'health', 'technology', 'children',
            'young adult', 'ya', 'graphic novel', 'memoir', 'autobiography'
        ]
        
        query_lower = query.lower()
        
        # Check for color keywords
        has_color = False
        detected_colors = []
        for color in color_keywords:
            if color in query_lower:
                has_color = True
                detected_colors.append(color)
        
        # Check for content keywords
        has_content = False
        detected_content = []
        for content in content_keywords:
            if content in query_lower:
                has_content = True
                detected_content.append(content)
        
        # Determine query type
        if has_color and has_content:
            return 'hybrid', detected_colors, detected_content
        elif has_color:
            return 'visual', detected_colors, []
        else:
            return 'text', [], detected_content
    
    def retrieve_books(self, query, top_k=3, vsm_weight=None, semantic_weight=None, use_expansion=True):
        """
        IMPROVED: Intelligent dataset routing
        - Pure text → books.csv only
        - Pure visual → Kaggle only (visual matching)
        - Hybrid (text+visual) → Kaggle only (semantic + visual matching)
        """
        
        query_type, colors, content = self.analyze_query_type(query)
        
        print(f"🔍 Query Analysis:")
        print(f"   Type: {query_type}")
        if colors:
            print(f"   Colors detected: {colors}")
        if content:
            print(f"   Content detected: {content}")
        
        # ===== HYBRID QUERY: Use Kaggle with BOTH semantic + visual =====
        
        if query_type == 'hybrid' and self.visual_available:
            print(f"🔀 HYBRID QUERY → Using Kaggle dataset with SEMANTIC + VISUAL matching")
            print(f"   Kaggle will search for: content ({content}) + color ({colors})")
            
            # Search Kaggle with FULL query (includes both content and color)
            # The improved search_by_visual_description will handle both signals
            visual_results = self.visual_engine.search_by_visual_description(
                query,  # Pass full query: "horror book with blue cover"
                top_k=top_k
            )
            
            print(f"   ✅ Kaggle search used both semantic and visual signals")
            
            # Convert to RAG format
            rag_results = []
            for v_result in visual_results:
                rag_results.append({
                    'document': {
                        'id': v_result['rank'],
                        'title': v_result['title'],
                        'author': v_result['author'],
                        'content': f"Category: {v_result['category']}. "
                                   f"ISBN: {v_result.get('isbn', 'N/A')}. "
                                   f"Rating: {v_result.get('rating', 'N/A')}.",
                        'genre': v_result['category']
                    },
                    'hybrid_score': v_result['similarity_score'],
                    'clip_score': v_result.get('clip_score', 0),
                    'visual_score': v_result.get('color_score', 0),
                    'search_mode': v_result.get('search_mode', 'hybrid'),
                    'source': 'kaggle',
                    'detected_content': v_result.get('detected_content', content),
                    'image_path': v_result.get('image_path')
                })
            
            return rag_results
        
        # ===== PURE VISUAL QUERY: Use Kaggle (visual only) =====
        
        elif query_type == 'visual' and self.visual_available:
            print(f"🎨 PURE VISUAL → Using Kaggle dataset (visual matching only)")
            
            visual_results = self.visual_engine.search_by_visual_description(query, top_k=top_k)
            
            rag_results = []
            for v_result in visual_results:
                rag_results.append({
                    'document': {
                        'id': v_result['rank'],
                        'title': v_result['title'],
                        'author': v_result['author'],
                        'content': f"Category: {v_result['category']}. "
                                   f"ISBN: {v_result.get('isbn', 'N/A')}. "
                                   f"Rating: {v_result.get('rating', 'N/A')}.",
                        'genre': v_result['category']
                    },
                    'hybrid_score': v_result['similarity_score'],
                    'visual_score': v_result.get('color_score', v_result['similarity_score']),
                    'search_mode': v_result.get('search_mode', 'visual'),
                    'source': 'kaggle',
                    'image_path': v_result.get('image_path')
                })
            
            return rag_results
        
        # ===== PURE TEXT QUERY: Use books.csv =====
        
        else:
            print(f"📖 PURE TEXT → Using books.csv (content-based search)")
            
            text_results = self.search_engine.hybrid_search(
                query,
                top_k=top_k,
                vsm_weight=vsm_weight,
                semantic_weight=semantic_weight,
                use_expansion=use_expansion
            )
            
            # Mark source
            for result in text_results:
                result['source'] = 'books.csv'
            
            return text_results
    
    def generate_response(self, query, retrieved_books):
        """Generate natural language response using LLM"""
        
        if not retrieved_books:
            return "I couldn't find any relevant books for your query."
        
        # Check which dataset was used
        dataset_used = retrieved_books[0].get('source', 'unknown')
        search_mode = retrieved_books[0].get('search_mode', 'unknown')
        
        # Format book information
        books_context = ""
        for i, result in enumerate(retrieved_books, 1):
            doc = result['document']
            books_context += f"\n{i}. Title: {doc['title']}\n"
            books_context += f"   Author: {doc['author']}\n"
            books_context += f"   Category/Genre: {doc.get('genre', 'N/A')}\n"
            
            # Different formatting based on dataset
            if dataset_used == 'kaggle':
                books_context += f"   Relevance Score: {result.get('hybrid_score', 0):.3f}\n"
                
                if result.get('clip_score'):
                    books_context += f"   Semantic Match: {result['clip_score']:.3f}\n"
                if result.get('visual_score'):
                    books_context += f"   Visual Match: {result['visual_score']:.3f}\n"
                if result.get('search_mode'):
                    books_context += f"   Search Mode: {result['search_mode']}\n"
                if result.get('detected_content'):
                    books_context += f"   Matched Topics: {', '.join(result['detected_content'])}\n"
            else:  # books.csv
                books_context += f"   Description: {doc['content'][:300]}...\n"
                books_context += f"   Relevance Score: {result.get('hybrid_score', result.get('score', 0)):.3f}\n"
        
        # Create dataset-aware prompt
        if search_mode == 'hybrid (content + color)':
            dataset_note = "These books match BOTH your content preferences (genre/topic) AND visual appearance (cover color/design)."
        elif dataset_used == 'kaggle':
            dataset_note = "These books were found based on cover appearance and visual characteristics."
        else:
            dataset_note = "These books were found based on content, themes, and descriptions."
        
        prompt = f"""You are a helpful librarian assistant. A user asked: "{query}"

{dataset_note}

Here are the most relevant books:
{books_context}

Please provide a friendly, conversational response that:
1. Recommends the most relevant books (focus on top 2-3)
2. Explains why each book matches their query
3. For hybrid queries (content + visual), mention BOTH the genre/topic relevance AND the cover appearance
4. For pure visual queries, focus on the visual characteristics
5. For pure text queries, focus on content and themes
6. Asks if they'd like more specific recommendations

Keep your response concise (2-3 sentences per book) and engaging."""

        # Call Groq API
        try:
            response = self.client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[
                    {"role": "system", "content": "You are a knowledgeable and friendly librarian who helps users find books."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7,
                max_tokens=600
            )
            
            return response.choices[0].message.content
        
        except Exception as e:
            # Fallback response
            fallback = f"I found {len(retrieved_books)} relevant books for your query:\n\n"
            for i, result in enumerate(retrieved_books, 1):
                doc = result['document']
                fallback += f"{i}. **{doc['title']}** by {doc['author']}\n"
                fallback += f"   Category: {doc.get('genre', 'N/A')}\n"
                
                if dataset_used == 'kaggle' and result.get('detected_content'):
                    fallback += f"   Matches: {', '.join(result['detected_content'])}\n"
                elif dataset_used == 'books.csv':
                    fallback += f"   {doc['content'][:150]}...\n"
                
                fallback += "\n"
            
            fallback += f"\n(Note: LLM generation failed: {e})"
            return fallback
    
    def chat(self, query, vsm_weight=None, semantic_weight=None, use_expansion=True):
        """Complete multimodal RAG pipeline with intelligent query routing"""
        print(f"\n{'='*70}")
        print(f"🔍 Query: '{query}'")
        print(f"{'='*70}")
        
        # Retrieve books (intelligently routes to appropriate dataset)
        books = self.retrieve_books(
            query,
            top_k=3,
            vsm_weight=vsm_weight,
            semantic_weight=semantic_weight,
            use_expansion=use_expansion
        )
        
        if not books:
            return "Sorry, I couldn't find any relevant books for your query.", []
        
        dataset_used = books[0].get('source', 'unknown')
        search_mode = books[0].get('search_mode', 'unknown')
        print(f"✅ Found {len(books)} books from {dataset_used} (mode: {search_mode})")
        
        # Generate response
        print("🤖 Generating response with LLM...\n")
        response = self.generate_response(query, books)
        
        return response, books


# ============================================================
# TEST SCRIPT
# ============================================================
if __name__ == "__main__":
    print("="*70)
    print("🎨 IMPROVED MULTIMODAL RAG WITH HYBRID QUERY SUPPORT - TEST")
    print("="*70)
    
    api_key = os.environ.get("GROQ_API_KEY", "")
    
    # Initialize
    rag = MultimodalRAGWithLLM(api_key=api_key)
    
    # Test queries including HYBRID queries
    test_queries = [
        ("horror book with blue cover image", "HYBRID: Should find horror books with blue covers"),
        ("mystery thriller", "TEXT: Should use books.csv"),
        ("blue book cover", "VISUAL: Should use Kaggle (color only)"),
        ("romantic novel with pink cover", "HYBRID: Romance + pink cover"),
        ("science fiction space adventure", "TEXT: Should use books.csv"),
        ("dark cover mystery", "HYBRID: Mystery + dark cover"),
    ]
    
    for query, expected in test_queries:
        print(f"\n{'='*70}")
        print(f"TEST: {query}")
        print(f"EXPECTED: {expected}")
        print(f"{'='*70}")
        
        response, books = rag.chat(query)
        
        print(f"\n🤖 AI Librarian Response:")
        print(f"{response}\n")
        
        print(f"📚 SOURCE BOOKS:")
        for book in books:
            print(f"  • {book['document']['title'][:60]}")
            print(f"    Author: {book['document']['author'][:40]}")
            print(f"    Genre: {book['document'].get('genre', 'N/A')}")
            print(f"    Source: {book['source']}")
            print(f"    Mode: {book.get('search_mode', 'unknown')}")
            
            if 'clip_score' in book:
                print(f"    CLIP Score: {book['clip_score']:.3f}")
            if 'visual_score' in book:
                print(f"    Visual Score: {book['visual_score']:.3f}")
            if 'detected_content' in book:
                print(f"    Matched Content: {book['detected_content']}")
        
        input("\nPress Enter to continue to next test...")
    
    print("\n" + "="*70)
    print("✅ ALL TESTS COMPLETE!")
    print("="*70)
    print("\n📊 Summary:")
    print("• Pure text queries → books.csv (5,000 books)")
    print("• Pure visual queries → Kaggle (32,581 covers)")
    print("• Hybrid queries (content + color) → Kaggle with semantic + visual matching")
    print("\n🎯 The system now intelligently handles all query types!")
