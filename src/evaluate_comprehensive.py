"""
COMPREHENSIVE EVALUATION SCRIPT
Evaluates ALL components: Text Search, Visual Search, Multi-modal, RAG System
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from search_engine_advanced import AdvancedSearchEngine
import time
import pandas as pd
from collections import defaultdict, Counter
import json
import os

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (14, 10)

class ComprehensiveEvaluator:
    """Evaluates all system components"""
    
    def __init__(self):
        # Load text search engine
        print("📚 Loading text search engine...")
        self.text_engine = AdvancedSearchEngine()
        self.text_engine.load_model('../data/advanced_search_engine.pkl')
        print(f"   ✅ Loaded {len(self.text_engine.documents)} books from books.csv\n")
        
        # Load visual search engine
        print("🎨 Loading visual search engine...")
        try:
            from multimodal_search_kaggle import KaggleBookCoverSearch
            self.visual_engine = KaggleBookCoverSearch()
            self.visual_engine.load_model('../data/kaggle_multimodal_search.pkl')
            self.visual_available = True
            print(f"   ✅ Loaded {len(self.visual_engine.books_df)} books from Kaggle dataset\n")
        except Exception as e:
            print(f"   ⚠️ Visual search not available: {e}\n")
            self.visual_engine = None
            self.visual_available = False
        
        # Load RAG system
        print("💬 Loading RAG system...")
        try:
            from rag_llm_multimodal import MultimodalRAGWithLLM
            self.rag_system = MultimodalRAGWithLLM()
            self.rag_available = True
            print(f"   ✅ RAG system loaded\n")
        except Exception as e:
            print(f"   ⚠️ RAG system not available: {e}\n")
            self.rag_system = None
            self.rag_available = False
    
    # ============================================================
    # PART 1: TEXT SEARCH EVALUATION (Original)
    # ============================================================
    
    def evaluate_text_search(self):
        """Evaluate text-based search methods"""
        print("="*80)
        print("📊 PART 1: TEXT SEARCH EVALUATION")
        print("="*80)
        
        # Test queries with relevance judgments
        test_queries = {
            "mystery detective stories": {
                'keywords': ['mystery', 'detective', 'crime', 'investigation'],
                'anti_keywords': ['romance', 'comedy']
            },
            "science fiction space": {
                'keywords': ['science', 'fiction', 'space', 'alien', 'future'],
                'anti_keywords': ['historical', 'romance']
            },
            "romantic love stories": {
                'keywords': ['romance', 'love', 'relationship', 'heart'],
                'anti_keywords': ['horror', 'thriller']
            },
            "fantasy magic adventure": {
                'keywords': ['fantasy', 'magic', 'wizard', 'dragon', 'quest'],
                'anti_keywords': ['biography', 'history']
            },
            "horror scary thriller": {
                'keywords': ['horror', 'scary', 'terror', 'ghost', 'suspense'],
                'anti_keywords': ['comedy', 'children']
            }
        }
        
        methods = ['VSM', 'Semantic', 'Hybrid', 'Hybrid+QE', 'Hybrid+Personal']
        results_by_method = defaultdict(lambda: defaultdict(list))
        
        test_user_id = "eval_user_001"
        
        for query_idx, (query, judgment) in enumerate(test_queries.items(), 1):
            print(f"\n[{query_idx}/{len(test_queries)}] Query: '{query}'")
            
            # Test each method
            for method in methods:
                start_time = time.time()
                
                if method == 'VSM':
                    results = self.text_engine.vsm_search(query, top_k=20)
                elif method == 'Semantic':
                    results = self.text_engine.semantic_search(query, top_k=20)
                elif method == 'Hybrid':
                    results = self.text_engine.hybrid_search(query, top_k=20, use_expansion=False)
                elif method == 'Hybrid+QE':
                    results = self.text_engine.hybrid_search(query, top_k=20, use_expansion=True)
                else:  # Hybrid+Personal
                    # Build profile
                    if query_idx <= 2:
                        for r in self.text_engine.hybrid_search(query, top_k=3, use_expansion=False):
                            self.text_engine.personalization.update_user_profile(
                                test_user_id, r['document']['id'], 'click'
                            )
                    
                    results = self.text_engine.hybrid_search(
                        query, top_k=20, use_expansion=True,
                        personalized_for_user=test_user_id if query_idx > 2 else None
                    )
                
                exec_time = time.time() - start_time
                
                # Calculate relevance
                relevance_scores = []
                for result in results:
                    doc = result['document']
                    doc_text = (doc.get('title', '') + ' ' + doc.get('content', '')).lower()
                    
                    score = 0.0
                    for kw in judgment['keywords']:
                        if kw in doc_text:
                            score += 1.0
                    for anti_kw in judgment['anti_keywords']:
                        if anti_kw in doc_text:
                            score -= 0.5
                    
                    score = max(0, min(score / len(judgment['keywords']), 1.0))
                    relevance_scores.append(score)
                
                # Calculate metrics
                for k in [5, 10, 20]:
                    relevant = [1 if r >= 0.3 else 0 for r in relevance_scores[:k]]
                    results_by_method[method][f'P@{k}'].append(sum(relevant) / k)
                    results_by_method[method][f'NDCG@{k}'].append(self._ndcg(relevance_scores[:k]))
                
                results_by_method[method]['Time'].append(exec_time * 1000)
        
        # Aggregate results
        text_metrics = []
        for method in methods:
            text_metrics.append({
                'Method': method,
                'P@5': np.mean(results_by_method[method]['P@5']),
                'P@10': np.mean(results_by_method[method]['P@10']),
                'NDCG@10': np.mean(results_by_method[method]['NDCG@10']),
                'Time(ms)': np.mean(results_by_method[method]['Time'])
            })
        
        df_text = pd.DataFrame(text_metrics).round(4)
        print("\n" + "="*80)
        print("TEXT SEARCH RESULTS:")
        print("="*80)
        print(df_text.to_string(index=False))
        
        return df_text, results_by_method
    
    # ============================================================
    # PART 2: VISUAL SEARCH EVALUATION (NEW!)
    # ============================================================
    
    def evaluate_visual_search(self):
        """Evaluate visual/color-based search"""
        print("\n" + "="*80)
        print("🎨 PART 2: VISUAL SEARCH EVALUATION")
        print("="*80)
        
        if not self.visual_available:
            print("⚠️ Visual search not available. Skipping...")
            return None, None
        
        # Test queries for visual search
        visual_test_queries = [
            {
                'query': 'blue book cover',
                'expected_color': 'blue',
                'color_weight': 0.6
            },
            {
                'query': 'red colored book',
                'expected_color': 'red',
                'color_weight': 0.6
            },
            {
                'query': 'dark mysterious cover',
                'expected_color': 'dark',
                'color_weight': 0.5
            },
            {
                'query': 'green book',
                'expected_color': 'green',
                'color_weight': 0.6
            },
            {
                'query': 'bright colorful cover',
                'expected_color': 'bright',
                'color_weight': 0.5
            }
        ]
        
        visual_metrics = []
        
        for test in visual_test_queries:
            query = test['query']
            print(f"\n🔍 Testing: '{query}'")
            
            start_time = time.time()
            results = self.visual_engine.search_by_visual_description(query, top_k=10)
            search_time = time.time() - start_time
            
            # Analyze results
            if results and 'color_score' in results[0]:
                avg_color_score = np.mean([r.get('color_score', 0) for r in results])
                avg_clip_score = np.mean([r.get('clip_score', 0) for r in results])
                avg_overall_score = np.mean([r.get('similarity_score', 0) for r in results])
                
                search_mode = results[0].get('search_mode', 'unknown')
                
                visual_metrics.append({
                    'Query': query,
                    'Avg_Color_Score': avg_color_score,
                    'Avg_CLIP_Score': avg_clip_score,
                    'Avg_Overall': avg_overall_score,
                    'Search_Mode': search_mode,
                    'Time(ms)': search_time * 1000
                })
                
                print(f"   Color Score: {avg_color_score:.3f}")
                print(f"   CLIP Score: {avg_clip_score:.3f}")
                print(f"   Mode: {search_mode}")
            else:
                # Semantic-only results
                avg_score = np.mean([r.get('similarity_score', 0) for r in results])
                visual_metrics.append({
                    'Query': query,
                    'Avg_Color_Score': 0.0,
                    'Avg_CLIP_Score': avg_score,
                    'Avg_Overall': avg_score,
                    'Search_Mode': 'semantic-only',
                    'Time(ms)': search_time * 1000
                })
        
        df_visual = pd.DataFrame(visual_metrics).round(4)
        print("\n" + "="*80)
        print("VISUAL SEARCH RESULTS:")
        print("="*80)
        print(df_visual.to_string(index=False))
        
        return df_visual, visual_metrics
    
    # ============================================================
    # PART 3: RAG SYSTEM EVALUATION (NEW!)
    # ============================================================
    
    def evaluate_rag_system(self):
        """Evaluate RAG system and dataset switching"""
        print("\n" + "="*80)
        print("💬 PART 3: RAG SYSTEM EVALUATION")
        print("="*80)
        
        if not self.rag_available:
            print("⚠️ RAG system not available. Skipping...")
            return None
        
        # Test queries - mix of text and visual
        rag_test_queries = [
            {'query': 'mystery thriller books', 'expected_dataset': 'books.csv', 'has_color': False},
            {'query': 'blue book cover', 'expected_dataset': 'kaggle', 'has_color': True},
            {'query': 'science fiction space', 'expected_dataset': 'books.csv', 'has_color': False},
            {'query': 'dark cover mystery', 'expected_dataset': 'kaggle', 'has_color': True},
            {'query': 'romantic novels', 'expected_dataset': 'books.csv', 'has_color': False},
            {'query': 'green colored book', 'expected_dataset': 'kaggle', 'has_color': True}
        ]
        
        rag_metrics = []
        
        for test in rag_test_queries:
            query = test['query']
            expected_dataset = test['expected_dataset']
            
            print(f"\n💬 Query: '{query}'")
            print(f"   Expected Dataset: {expected_dataset}")
            
            start_time = time.time()
            
            try:
                response, books = self.rag_system.chat(query)
                retrieval_time = time.time() - start_time
                
                # Check dataset selection
                if books:
                    actual_dataset = books[0].get('source', 'unknown')
                    correct_dataset = (actual_dataset == expected_dataset)
                    
                    print(f"   Actual Dataset: {actual_dataset}")
                    print(f"   ✓ Correct: {correct_dataset}")
                    print(f"   Retrieved: {len(books)} books")
                    
                    rag_metrics.append({
                        'Query': query,
                        'Expected_Dataset': expected_dataset,
                        'Actual_Dataset': actual_dataset,
                        'Correct_Selection': correct_dataset,
                        'Num_Books': len(books),
                        'Time(ms)': retrieval_time * 1000
                    })
                else:
                    rag_metrics.append({
                        'Query': query,
                        'Expected_Dataset': expected_dataset,
                        'Actual_Dataset': 'none',
                        'Correct_Selection': False,
                        'Num_Books': 0,
                        'Time(ms)': retrieval_time * 1000
                    })
            
            except Exception as e:
                print(f"   ❌ Error: {e}")
                rag_metrics.append({
                    'Query': query,
                    'Expected_Dataset': expected_dataset,
                    'Actual_Dataset': 'error',
                    'Correct_Selection': False,
                    'Num_Books': 0,
                    'Time(ms)': 0
                })
        
        df_rag = pd.DataFrame(rag_metrics)
        
        # Calculate accuracy
        accuracy = df_rag['Correct_Selection'].mean()
        
        print("\n" + "="*80)
        print("RAG SYSTEM RESULTS:")
        print("="*80)
        print(df_rag.to_string(index=False))
        print(f"\n📊 Dataset Selection Accuracy: {accuracy:.2%}")
        
        return df_rag
    
    # ============================================================
    # PART 4: COMPREHENSIVE VISUALIZATION
    # ============================================================
    
    def generate_visualizations(self, df_text, df_visual, df_rag, results_by_method):
        """Generate comprehensive visualizations"""
        print("\n" + "="*80)
        print("📈 GENERATING VISUALIZATIONS")
        print("="*80)
        
        os.makedirs('../results', exist_ok=True)
        
        # 1. Text Search Performance Comparison
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # P@10 Comparison
        ax1 = axes[0, 0]
        methods = df_text['Method'].values
        p10_scores = df_text['P@10'].values
        bars = ax1.bar(methods, p10_scores, color='steelblue', alpha=0.8)
        ax1.set_ylabel('Precision@10', fontsize=12, fontweight='bold')
        ax1.set_title('Text Search: Precision@10 Comparison', fontsize=14, fontweight='bold')
        ax1.set_ylim([0, 1])
        for bar, val in zip(bars, p10_scores):
            ax1.text(bar.get_x() + bar.get_width()/2, val + 0.02, f'{val:.3f}',
                    ha='center', fontweight='bold')
        
        # NDCG@10 Comparison
        ax2 = axes[0, 1]
        ndcg_scores = df_text['NDCG@10'].values
        bars = ax2.bar(methods, ndcg_scores, color='darkorange', alpha=0.8)
        ax2.set_ylabel('NDCG@10', fontsize=12, fontweight='bold')
        ax2.set_title('Text Search: NDCG@10 Comparison', fontsize=14, fontweight='bold')
        ax2.set_ylim([0, 1])
        for bar, val in zip(bars, ndcg_scores):
            ax2.text(bar.get_x() + bar.get_width()/2, val + 0.02, f'{val:.3f}',
                    ha='center', fontweight='bold')
        
        # Response Time Comparison
        ax3 = axes[1, 0]
        times = df_text['Time(ms)'].values
        bars = ax3.bar(methods, times, color='green', alpha=0.8)
        ax3.set_ylabel('Response Time (ms)', fontsize=12, fontweight='bold')
        ax3.set_title('Text Search: Response Time', fontsize=14, fontweight='bold')
        for bar, val in zip(bars, times):
            ax3.text(bar.get_x() + bar.get_width()/2, val + 2, f'{val:.1f}',
                    ha='center', fontweight='bold')
        
        # Overall Performance Heatmap
        ax4 = axes[1, 1]
        heatmap_data = df_text[['P@5', 'P@10', 'NDCG@10']].values
        sns.heatmap(heatmap_data, annot=True, fmt='.3f', cmap='YlOrRd',
                   xticklabels=['P@5', 'P@10', 'NDCG@10'],
                   yticklabels=methods, ax=ax4, cbar_kws={'label': 'Score'})
        ax4.set_title('Text Search: Performance Heatmap', fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig('../results/text_search_comprehensive.png', dpi=300, bbox_inches='tight')
        print("   ✅ Saved: text_search_comprehensive.png")
        plt.close()
        
        # 2. Visual Search Performance
        if df_visual is not None:
            fig, axes = plt.subplots(1, 2, figsize=(16, 6))
            
            # Color vs CLIP scores
            ax1 = axes[0]
            x = np.arange(len(df_visual))
            width = 0.35
            ax1.bar(x - width/2, df_visual['Avg_Color_Score'], width, label='Color Score',
                   color='purple', alpha=0.8)
            ax1.bar(x + width/2, df_visual['Avg_CLIP_Score'], width, label='CLIP Score',
                   color='teal', alpha=0.8)
            ax1.set_xlabel('Query', fontsize=12, fontweight='bold')
            ax1.set_ylabel('Average Score', fontsize=12, fontweight='bold')
            ax1.set_title('Visual Search: Color vs CLIP Scores', fontsize=14, fontweight='bold')
            ax1.set_xticks(x)
            ax1.set_xticklabels(df_visual['Query'].str[:15], rotation=45, ha='right')
            ax1.legend()
            ax1.set_ylim([0, 1])
            
            # Search mode distribution
            ax2 = axes[1]
            mode_counts = df_visual['Search_Mode'].value_counts()
            ax2.pie(mode_counts.values, labels=mode_counts.index, autopct='%1.1f%%',
                   colors=['#FF6B6B', '#4ECDC4', '#45B7D1'])
            ax2.set_title('Visual Search: Mode Distribution', fontsize=14, fontweight='bold')
            
            plt.tight_layout()
            plt.savefig('../results/visual_search_performance.png', dpi=300, bbox_inches='tight')
            print("   ✅ Saved: visual_search_performance.png")
            plt.close()
        
        # 3. RAG System Performance
        if df_rag is not None:
            fig, axes = plt.subplots(1, 2, figsize=(16, 6))
            
            # Dataset selection accuracy
            ax1 = axes[0]
            correct = df_rag['Correct_Selection'].sum()
            incorrect = len(df_rag) - correct
            ax1.bar(['Correct', 'Incorrect'], [correct, incorrect],
                   color=['green', 'red'], alpha=0.8)
            ax1.set_ylabel('Count', fontsize=12, fontweight='bold')
            ax1.set_title('RAG: Dataset Selection Accuracy', fontsize=14, fontweight='bold')
            accuracy = correct / len(df_rag) * 100
            ax1.text(0, correct + 0.1, f'{correct}\n({accuracy:.1f}%)',
                    ha='center', fontweight='bold')
            ax1.text(1, incorrect + 0.1, f'{incorrect}\n({100-accuracy:.1f}%)',
                    ha='center', fontweight='bold')
            
            # Response time by dataset
            ax2 = axes[1]
            for dataset in df_rag['Actual_Dataset'].unique():
                if dataset not in ['none', 'error']:
                    mask = df_rag['Actual_Dataset'] == dataset
                    times = df_rag[mask]['Time(ms)']
                    ax2.scatter([dataset]*len(times), times, alpha=0.6, s=100, label=dataset)
            ax2.set_xlabel('Dataset', fontsize=12, fontweight='bold')
            ax2.set_ylabel('Response Time (ms)', fontsize=12, fontweight='bold')
            ax2.set_title('RAG: Response Time by Dataset', fontsize=14, fontweight='bold')
            ax2.legend()
            
            plt.tight_layout()
            plt.savefig('../results/rag_system_performance.png', dpi=300, bbox_inches='tight')
            print("   ✅ Saved: rag_system_performance.png")
            plt.close()
        
        # 4. Overall System Comparison
        fig, ax = plt.subplots(figsize=(14, 8))
        
        comparison_data = {
            'Text Search\n(Hybrid+QE)': {
                'Performance': df_text[df_text['Method'] == 'Hybrid+QE']['NDCG@10'].values[0],
                'Speed': df_text[df_text['Method'] == 'Hybrid+QE']['Time(ms)'].values[0]
            }
        }
        
        if df_visual is not None:
            comparison_data['Visual Search\n(Color+CLIP)'] = {
                'Performance': df_visual['Avg_Overall'].mean(),
                'Speed': df_visual['Time(ms)'].mean()
            }
        
        if df_rag is not None:
            comparison_data['RAG System\n(Multi-modal)'] = {
                'Performance': df_rag['Correct_Selection'].mean(),
                'Speed': df_rag['Time(ms)'].mean()
            }
        
        components = list(comparison_data.keys())
        performance = [comparison_data[c]['Performance'] for c in components]
        speed = [comparison_data[c]['Speed'] for c in components]
        
        x = np.arange(len(components))
        width = 0.35
        
        ax1 = ax
        bars1 = ax1.bar(x - width/2, performance, width, label='Performance Score',
                       color='steelblue', alpha=0.8)
        ax1.set_ylabel('Performance Score', fontsize=12, fontweight='bold', color='steelblue')
        ax1.tick_params(axis='y', labelcolor='steelblue')
        ax1.set_ylim([0, 1])
        
        ax2 = ax1.twinx()
        bars2 = ax2.bar(x + width/2, speed, width, label='Response Time (ms)',
                       color='darkorange', alpha=0.8)
        ax2.set_ylabel('Response Time (ms)', fontsize=12, fontweight='bold', color='darkorange')
        ax2.tick_params(axis='y', labelcolor='darkorange')
        
        ax1.set_xlabel('System Component', fontsize=12, fontweight='bold')
        ax1.set_xticks(x)
        ax1.set_xticklabels(components)
        ax1.set_title('Overall System Performance Comparison', fontsize=16, fontweight='bold')
        
        # Add value labels
        for bar in bars1:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                    f'{height:.3f}', ha='center', va='bottom', fontweight='bold')
        
        for bar in bars2:
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + 5,
                    f'{height:.1f}ms', ha='center', va='bottom', fontweight='bold')
        
        fig.legend(loc='upper left', bbox_to_anchor=(0.1, 0.95), fontsize=11)
        plt.tight_layout()
        plt.savefig('../results/overall_system_comparison.png', dpi=300, bbox_inches='tight')
        print("   ✅ Saved: overall_system_comparison.png")
        plt.close()
    
    # ============================================================
    # HELPER FUNCTIONS
    # ============================================================
    
    def _ndcg(self, relevance_scores, k=None):
        """Calculate NDCG@K"""
        if k:
            relevance_scores = relevance_scores[:k]
        
        if len(relevance_scores) == 0:
            return 0.0
        
        dcg = relevance_scores[0]
        for i in range(1, len(relevance_scores)):
            dcg += relevance_scores[i] / np.log2(i + 1)
        
        ideal_scores = sorted(relevance_scores, reverse=True)
        idcg = ideal_scores[0]
        for i in range(1, len(ideal_scores)):
            idcg += ideal_scores[i] / np.log2(i + 1)
        
        return dcg / idcg if idcg > 0 else 0.0
    
    # ============================================================
    # MAIN EVALUATION
    # ============================================================
    
    def run_comprehensive_evaluation(self):
        """Run all evaluations and generate report"""
        print("="*80)
        print("🔬 COMPREHENSIVE SYSTEM EVALUATION")
        print("="*80)
        print()
        
        results = {}
        
        # Part 1: Text Search
        df_text, results_by_method = self.evaluate_text_search()
        results['text_search'] = df_text
        
        # Part 2: Visual Search
        df_visual, visual_metrics = self.evaluate_visual_search()
        results['visual_search'] = df_visual
        
        # Part 3: RAG System
        df_rag = self.evaluate_rag_system()
        results['rag_system'] = df_rag
        
        # Generate visualizations
        self.generate_visualizations(df_text, df_visual, df_rag, results_by_method)
        
        # Save results to CSV
        print("\n" + "="*80)
        print("💾 SAVING RESULTS")
        print("="*80)
        
        df_text.to_csv('../results/text_search_metrics.csv', index=False)
        print("   ✅ Saved: text_search_metrics.csv")
        
        if df_visual is not None:
            df_visual.to_csv('../results/visual_search_metrics.csv', index=False)
            print("   ✅ Saved: visual_search_metrics.csv")
        
        if df_rag is not None:
            df_rag.to_csv('../results/rag_system_metrics.csv', index=False)
            print("   ✅ Saved: rag_system_metrics.csv")
        
        # Generate summary report
        self.generate_summary_report(results)
        
        print("\n" + "="*80)
        print("✅ COMPREHENSIVE EVALUATION COMPLETE!")
        print("="*80)
        
        return results
    
    def generate_summary_report(self, results):
        """Generate a summary report"""
        summary = {
            'evaluation_date': pd.Timestamp.now().isoformat(),
            'components_evaluated': []
        }
        
        # Text search summary
        if 'text_search' in results:
            df = results['text_search']
            best_method = df.loc[df['NDCG@10'].idxmax()]
            summary['components_evaluated'].append('text_search')
            summary['text_search'] = {
                'best_method': best_method['Method'],
                'best_ndcg': float(best_method['NDCG@10']),
                'best_precision': float(best_method['P@10']),
                'avg_response_time': float(df['Time(ms)'].mean())
            }
        
        # Visual search summary
        if 'visual_search' in results and results['visual_search'] is not None:
            df = results['visual_search']
            summary['components_evaluated'].append('visual_search')
            summary['visual_search'] = {
                'avg_color_score': float(df['Avg_Color_Score'].mean()),
                'avg_clip_score': float(df['Avg_CLIP_Score'].mean()),
                'avg_response_time': float(df['Time(ms)'].mean())
            }
        
        # RAG system summary
        if 'rag_system' in results and results['rag_system'] is not None:
            df = results['rag_system']
            summary['components_evaluated'].append('rag_system')
            summary['rag_system'] = {
                'dataset_selection_accuracy': float(df['Correct_Selection'].mean()),
                'avg_response_time': float(df['Time(ms)'].mean()),
                'total_queries_tested': len(df)
            }
        
        # Save summary
        with open('../results/evaluation_summary.json', 'w') as f:
            json.dump(summary, f, indent=2)
        
        print("   ✅ Saved: evaluation_summary.json")
        
        # Print summary to console
        print("\n" + "="*80)
        print("📋 EVALUATION SUMMARY")
        print("="*80)
        print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    print("="*80)
    print("🚀 COMPREHENSIVE E-LIBRARY SYSTEM EVALUATION")
    print("="*80)
    print()
    
    evaluator = ComprehensiveEvaluator()
    results = evaluator.run_comprehensive_evaluation()
    
    print("\n" + "="*80)
    print("📊 FINAL STATISTICS")
    print("="*80)
    
    # Text search stats
    if 'text_search' in results:
        df = results['text_search']
        print("\n📚 TEXT SEARCH:")
        print(f"   Best Method: {df.loc[df['NDCG@10'].idxmax(), 'Method']}")
        print(f"   Best NDCG@10: {df['NDCG@10'].max():.4f}")
        baseline = df[df['Method'] == 'VSM']['NDCG@10'].values[0]
        best = df['NDCG@10'].max()
        improvement = ((best - baseline) / baseline) * 100
        print(f"   Improvement over VSM: {improvement:.1f}%")
    
    # Visual search stats
    if 'visual_search' in results and results['visual_search'] is not None:
        df = results['visual_search']
        print("\n🎨 VISUAL SEARCH:")
        print(f"   Average Color Score: {df['Avg_Color_Score'].mean():.4f}")
        print(f"   Average CLIP Score: {df['Avg_CLIP_Score'].mean():.4f}")
        print(f"   Queries with Color Detection: {(df['Avg_Color_Score'] > 0).sum()}/{len(df)}")
    
    # RAG system stats
    if 'rag_system' in results and results['rag_system'] is not None:
        df = results['rag_system']
        accuracy = df['Correct_Selection'].mean() * 100
        print("\n💬 RAG SYSTEM:")
        print(f"   Dataset Selection Accuracy: {accuracy:.1f}%")
        print(f"   Correct Selections: {df['Correct_Selection'].sum()}/{len(df)}")
        print(f"   Average Response Time: {df['Time(ms)'].mean():.1f}ms")
    
    print("\n" + "="*80)
    print("✅ ALL EVALUATIONS COMPLETE!")
    print("="*80)
    print("\n📁 Results saved in: ../results/")
    print("   • text_search_metrics.csv")
    print("   • visual_search_metrics.csv")
    print("   • rag_system_metrics.csv")
    print("   • evaluation_summary.json")
    print("\n📊 Visualizations saved:")
    print("   • text_search_comprehensive.png")
    print("   • visual_search_performance.png")
    print("   • rag_system_performance.png")
    print("   • overall_system_comparison.png")
    print("\n" + "="*80)
