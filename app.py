from flask import Flask, render_template, request, jsonify, send_file,redirect,flash,url_for,session
import google.generativeai as genai
from typing import List, Dict, Optional
import arxiv
import requests
import re
import os
from concurrent.futures import ThreadPoolExecutor
import threading
import pypdf
import uuid,io
from nltk import FreqDist
from nltk.corpus import stopwords
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from datetime import datetime
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from io import BytesIO
import json
from collections import defaultdict
import plotly.graph_objects as go
from wordcloud import WordCloud
import pandas as pd
from serpapi import GoogleSearch
import time
from bs4 import BeautifulSoup
import logging
from openalex import OpenAlex
import PyPDF2,textwrap
from PyPDF2 import PdfReader  # For PDF text extraction
from werkzeug.utils import secure_filename
from google.api_core import exceptions as google_exceptions
from fuzzywuzzy import fuzz, process

UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'pdf','doc','docx'}

genai.configure(api_key='AIzaSyBpW7-VjXY74uhxz3ZjiS4rcespAQUXIhM')
model = genai.GenerativeModel('gemini-1.5-flash')

# API Keys (should be in environment variables in production)
SERPAPI_API_KEY = '40a56a489669a35ab2918d8c843d35137c9d59c9e71d0cc8df16dfac9ce09891'

class EnhancedResearchAssistant:
    def __init__(self):
        self.papers = {}  # In-memory storage of papers by ID
        self.user_profiles = {}  # Tracks user preferences
        self.author_graph = nx.Graph()  # Tracks co-authorship
        self.citation_graph = nx.DiGraph()  # Tracks citations
        self.topic_graph = nx.Graph()  # Tracks topic relationships
        self.executor = ThreadPoolExecutor(max_workers=8)  # For parallel processing
        self.lock = threading.Lock()  # For thread-safe operations
        self.vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)
        self.paper_texts = []
        self.paper_ids = []
        self.cached_reviews = {}  # Cache for literature reviews
        self.cached_gaps = {}  # Cache for research gaps
        self.cached_trends = {}  # Cache for trend analysis

    # ------------------- Core Paper Processing -------------------
    def fetch_papers(self, query: str, max_results: int = 10, user_id: str = None) -> List[Dict]:
        """Fetch papers from multiple sources in parallel"""
        futures = []
        results = []

        # Fetch from multiple sources simultaneously
        futures.append(self.executor.submit(self.fetch_arxiv_papers, query, max_results // 3, user_id))
        futures.append(self.executor.submit(self.fetch_openalex_papers, query, max_results // 3, user_id))
        futures.append(self.executor.submit(self.fetch_google_scholar_papers, query, max_results // 3, user_id))

        for future in futures:
            try:
                results.extend(future.result())
            except Exception as e:
                print(f"Error fetching papers: {e}")

        # Deduplicate results
        seen = set()
        unique_results = []
        for paper in results:
            paper_id = paper.get('id') or paper.get('doi') or paper.get('arxiv_id') or str(uuid.uuid4())
            if paper_id not in seen:
                seen.add(paper_id)
                unique_results.append(paper)

        return unique_results[:max_results]

    def process_paper(self, paper_data: Dict, source: str, query: str = None) -> Dict:
        """Standardize paper data from different sources"""
        paper_id = paper_data.get('id') or paper_data.get('doi') or str(uuid.uuid4())

        with self.lock:
            if paper_id in self.papers:
                return self.papers[paper_id]

        # Extract common fields
        title = paper_data.get('title', 'Untitled')
        abstract = paper_data.get('abstract', '')
        authors = paper_data.get('authors', [])
        publication_date = paper_data.get('publication_date', '')
        pdf_url = paper_data.get('pdf_url', '')
        citations = paper_data.get('citations', [])
        references = paper_data.get('references', [])
        topics = paper_data.get('topics', [])

        # Extract full text if available
        full_text = ""
        if pdf_url:
            try:
                response = requests.get(pdf_url, timeout=10)
                temp_filename = f"temp_{paper_id}.pdf"
                with open(temp_filename, "wb") as f:
                    f.write(response.content)

                with open(temp_filename, "rb") as file:
                    reader = pypdf.PdfReader(file)
                    full_text = " ".join(page.extract_text() for page in reader.pages if page.extract_text())

                os.remove(temp_filename)
            except Exception as e:
                print(f"Error processing PDF for {title}: {e}")

        # Calculate relevance if query provided
        relevance_score = self.calculate_relevance_score(query, title + " " + abstract) if query else 0.5

        # Basic entity extraction using regex (capitalized phrases)
        entities = []
        text = title + " " + abstract
        # Match capitalized words or phrases (e.g., "Machine Learning", "United States")
        entity_pattern = r'\b(?:[A-Z][a-z]*\s*)+[A-Z][a-z]*\b'
        matches = re.findall(entity_pattern, text)
        entities = [(match, "UNKNOWN") for match in matches]

        # Extract key phrases using Gemini
        key_phrases = []
        try:
            prompt = f"""
            Extract 5-10 key phrases from the following text:
            {text[:1000]}

            Return as a JSON list of phrases.
            """
            response = model.generate_content(prompt)
            key_phrases = json.loads(response.text)
        except Exception as e:
            print(f"Error extracting key phrases: {e}")

        # Standardized paper data
        processed_paper = {
            'id': paper_id,
            'title': title,
            'abstract': abstract,
            'authors': authors,
            'publication_date': publication_date,
            'full_text': full_text,
            'relevance_score': relevance_score,
            'search_queries': [query] if query else [],
            'last_accessed': datetime.now().strftime('%Y-%m-%d'),
            'source': source,
            'citations': citations,
            'references': references,
            'entities': entities,
            'key_phrases': key_phrases,
            'topics': topics,
            'pdf_url': pdf_url
        }

        with self.lock:
            self.papers[paper_id] = processed_paper
            self.paper_texts.append(title + " " + abstract)
            self.paper_ids.append(paper_id)

            # Update graphs
            for i, author1 in enumerate(authors):
                for author2 in authors[i + 1:]:
                    self.author_graph.add_edge(author1, author2, weight=1)

            for citation in citations:
                self.citation_graph.add_edge(paper_id, citation)

            for reference in references:
                self.citation_graph.add_edge(reference, paper_id)

            for topic in topics:
                self.topic_graph.add_node(topic)
                self.topic_graph.add_edge(paper_id, topic)

        return processed_paper

    # ------------------- Data Source Integrations -------------------
    def fetch_arxiv_papers(self, query: str, max_results: int = 10, user_id: str = None) -> List[Dict]:
        """Fetch papers from arXiv"""
        client = arxiv.Client()
        search = arxiv.Search(
            query=query,
            max_results=max_results,
            sort_by=arxiv.SortCriterion.Relevance
        )

        papers = []
        for result in client.results(search):
            paper_data = {
                'id': result.entry_id.split('/')[-1],
                'title': result.title,
                'abstract': result.summary,
                'authors': [author.name for author in result.authors],
                'publication_date': result.published.strftime('%Y-%m-%d'),
                'pdf_url': result.pdf_url,
                'source': 'arxiv'
            }
            processed = self.process_paper(paper_data, 'arxiv', query)
            papers.append(processed)

        return papers

    def fetch_openalex_papers(self, query: str, max_results: int = 10, user_id: str = None) -> List[Dict]:
        """Fetch papers from OpenAlex using openalex-python"""
        try:
            oa = OpenAlex()
            works = oa.works().search(query).get(per_page=max_results)

            papers = []
            for result in works[:max_results]:
                paper_data = {
                    'id': result.get('id', '').split('/')[-1],
                    'title': result.get('title', ''),
                    'abstract': result.get('abstract', ''),
                    'authors': [author['author']['display_name'] for author in result.get('authorships', [])],
                    'publication_date': result.get('publication_date', ''),
                    'doi': result.get('doi', ''),
                    'citations': [citation['id'].split('/')[-1] for citation in result.get('referenced_works', [])],
                    'references': [ref['id'].split('/')[-1] for ref in result.get('referenced_works', [])],
                    'topics': [concept['display_name'] for concept in result.get('concepts', [])],
                    'pdf_url': result.get('primary_location', {}).get('pdf_url', '') if result.get('primary_location') else '',
                    'source': 'openalex'
                }
                processed = self.process_paper(paper_data, 'openalex', query)
                papers.append(processed)

            return papers
        except Exception as e:
            print(f"Error fetching from OpenAlex: {e}")
            return []

    def fetch_google_scholar_papers(self, query: str, max_results: int = 5, user_id: str = None) -> List[Dict]:
        """Fetch papers from Google Scholar using SerpAPI"""
        params = {
            "engine": "google_scholar",
            "q": query,
            "api_key": SERPAPI_API_KEY,
            "num": max_results
        }

        try:
            search = GoogleSearch(params)
            results = search.get_dict()

            papers = []
            for result in results.get('organic_results', [])[:max_results]:
                paper_data = {
                    'title': result.get('title', ''),
                    'authors': result.get('publication_info', {}).get('authors', []),
                    'year': result.get('publication_info', {}).get('year', ''),
                    'source': 'google_scholar',
                    'link': result.get('link', '')
                }

                # Try to get more details
                try:
                    page_response = requests.get(result.get('link'), timeout=5)
                    soup = BeautifulSoup(page_response.text, 'html.parser')
                    abstract = soup.find('div', class_='gs_rs').text if soup.find('div', class_='gs_rs') else ''
                    paper_data['abstract'] = abstract
                except:
                    paper_data['abstract'] = ''

                processed = self.process_paper(paper_data, 'google_scholar', query)
                papers.append(processed)

            return papers
        except Exception as e:
            print(f"Error fetching from Google Scholar: {e}")
            return []

    # ------------------- Enhanced Features -------------------
    def visualize_citation_network(self, paper_id: str, depth: int = 1) -> BytesIO:
        """Generate visualization of citation network"""
        if paper_id not in self.citation_graph:
            return None

        # Get subgraph around the paper
        nodes = [paper_id]
        for _ in range(depth):
            new_nodes = []
            for node in nodes:
                new_nodes.extend(list(self.citation_graph.successors(node)) +
                                 list(self.citation_graph.predecessors(node)))
            nodes.extend(new_nodes)
            nodes = list(set(nodes))

        subgraph = self.citation_graph.subgraph(nodes)

        # Create plot
        plt.figure(figsize=(12, 12))
        pos = nx.spring_layout(subgraph, k=0.5, iterations=50)

        # Color nodes based on type
        node_colors = []
        for node in subgraph.nodes():
            if node in self.papers:
                node_colors.append('skyblue')
            else:
                node_colors.append('lightgreen')

        nx.draw_networkx_nodes(subgraph, pos, node_size=800, node_color=node_colors)
        nx.draw_networkx_edges(subgraph, pos, alpha=0.5, width=1)

        # Add labels for papers we know about
        labels = {}
        for node in subgraph.nodes():
            if node in self.papers:
                labels[node] = self.papers[node]['title'][:30] + '...'
        nx.draw_networkx_labels(subgraph, pos, labels, font_size=8)

        plt.title(f"Citation Network for Paper: {self.papers.get(paper_id, {}).get('title', paper_id)}")
        plt.axis('off')

        # Save to bytes
        buf = BytesIO()
        plt.savefig(buf, format='png', dpi=300, bbox_inches='tight')
        buf.seek(0)
        plt.close()

        return buf

    def generate_trend_analysis(self, topic: str, years_back: int = 10) -> Dict:
        """Analyze publication trends over time"""
        if topic in self.cached_trends:
            return self.cached_trends[topic]

        # First fetch papers on this topic
        papers = self.fetch_papers(topic, max_results=100)

        # Extract publication years
        year_counts = defaultdict(int)
        for paper in papers:
            if 'publication_date' in paper and paper['publication_date']:
                year = paper['publication_date'][:4]
                if year.isdigit():
                    year_counts[year] += 1

        # Sort by year
        sorted_years = sorted(year_counts.items(), key=lambda x: x[0])
        years = [y[0] for y in sorted_years]
        counts = [y[1] for y in sorted_years]

        # Generate interactive plot
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=years,
            y=counts,
            mode='lines+markers',
            name='Publications'
        ))

        fig.update_layout(
            title=f'Publication Trends for "{topic}"',
            xaxis_title='Year',
            yaxis_title='Number of Publications',
            hovermode='x unified'
        )

        # Generate word cloud of recent abstracts
        recent_papers = [p for p in papers if 'publication_date' in p and
                         p['publication_date'] and
                         p['publication_date'][:4].isdigit() and
                         int(p['publication_date'][:4]) >= (datetime.now().year - 5)]

        abstracts = ' '.join([p['abstract'] for p in recent_papers if p.get('abstract')])

        wordcloud = WordCloud(width=800, height=400, background_color='white').generate(abstracts)
        plt.figure(figsize=(10, 5))
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis('off')
        plt.title(f'Recent Research Themes in "{topic}"')

        # Save word cloud to bytes
        wc_buf = BytesIO()
        plt.savefig(wc_buf, format='png', bbox_inches='tight')
        wc_buf.seek(0)
        plt.close()

        # Generate topic evolution analysis
        prompt = f"""
        Analyze how research on {topic} has evolved over the past {years_back} years based on these publication counts:
        {sorted_years}

        Provide:
        1. Key phases in the research evolution
        2. Possible explanations for spikes/drops in publications
        3. Emerging subtopics in recent years
        4. Predictions for future directions
        """

        analysis = model.generate_content(prompt).text
        analysis=clean_markdown(analysis)

        result = {
            'plot': fig.to_json(),
            'wordcloud': wc_buf,
            'analysis': analysis,
            'yearly_counts': sorted_years
        }

        self.cached_trends[topic] = result
        return result

    def extract_key_concepts(self, paper_id: str) -> Dict:
        """Extract key concepts and relationships from a paper using Gemini"""
        if paper_id not in self.papers:
            return {}

        paper = self.papers[paper_id]
        text = paper['title'] + " " + paper['abstract'] + " " + paper.get('full_text', '')[:5000]

        # Use Gemini to extract entities and relationships
        prompt = f"""
        Analyze the following text and extract key concepts:
        {text[:2000]}

        Return in JSON format:
        {{
            "entities": {{"entity_name": "entity_type", ...}},
            "relationships": [["subject", "verb", "object"], ...],
            "key_phrases": ["phrase1", "phrase2", ...]
        }}
        """
        try:
            response = model.generate_content(prompt).text
            response = clean_markdown(response)
            result = json.loads(response)
        except Exception as e:
            print(f"Error extracting concepts: {e}")
            result = {
                'entities': {},
                'relationships': [],
                'key_phrases': paper.get('key_phrases', [])
            }

        return result

    def extract_svo(self, sentence: str) -> Optional[tuple]:
        """Extract subject-verb-object triples using Gemini"""
        prompt = f"""
        Extract a subject-verb-object (SVO) triple from the following sentence:
        {sentence}

        Return in JSON format:
        {{
            "subject": "subject_text",
            "verb": "verb_text",
            "object": "object_text"
        }}
        If no SVO is found, return null.
        """
        try:
            response = model.generate_content(prompt).text
            response = clean_markdown(response)
            result = json.loads(response)
            if result and all(key in result for key in ['subject', 'verb', 'object']):
                return (result['subject'], result['verb'], result['object'])
            return None
        except Exception as e:
            print(f"Error extracting SVO: {e}")
            return None

    def compare_papers(self, paper_ids: List[str]) -> Dict:
        """Compare multiple papers across various dimensions"""
        papers = []
        for pid in paper_ids:
            if pid in self.papers:
                papers.append(self.papers[pid])
            else:
                return {"error": f"Paper {pid} not found"}

        if len(papers) < 2:
            return {"error": "Need at least 2 papers to compare"}

        # Create comparison table
        comparison = []
        dimensions = ['title', 'authors', 'publication_date', 'relevance_score',
                      'source', 'key_phrases', 'topics']

        for dim in dimensions:
            row = {'dimension': dim.replace('_', ' ').title()}
            for i, paper in enumerate(papers):
                row[f'paper_{i + 1}'] = paper.get(dim, 'N/A')
            comparison.append(row)

        # Generate similarity matrix
        texts = [p['title'] + " " + p['abstract'] for p in papers]
        tfidf = self.vectorizer.fit_transform(texts)
        similarity = cosine_similarity(tfidf)

        # Generate radar chart for comparison
        categories = ['Novelty', 'Methodology', 'Impact', 'Clarity', 'Evidence']
        scores = []
        for paper in papers:
            text = paper['title'] + " " + paper['abstract']
            # Simple word-based metrics
            words = text.lower().split()
            novelty = len([w for w in words if w in ['new', 'novel', 'innovative']]) / 10
            methodology = len([w for w in words if w in ['method', 'approach', 'model']]) / len(words)
            impact = len([w for w in words if w in ['significant', 'impact', 'important']]) / len(words)
            clarity = 1 - (len(text) / 1000)  # Simpler is clearer
            evidence = len([w for w in words if w in ['show', 'prove', 'demonstrate']]) / len(words)
            scores.append([novelty, methodology, impact, clarity, evidence])

        fig = go.Figure()
        for i, paper in enumerate(papers):
            fig.add_trace(go.Scatterpolar(
                r=scores[i],
                theta=categories,
                fill='toself',
                name=paper['title'][:20] + '...'
            ))

        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 1]
                )),
            showlegend=True,
            title="Paper Comparison Radar Chart"
        )

        # Generate detailed comparison using LLM
        prompt = f"""
        Compare these research papers in detail:
        1. {papers[0]['title']} ({papers[0]['publication_date']})
        Abstract: {papers[0]['abstract'][:500]}

        2. {papers[1]['title']} ({papers[1]['publication_date']})
        Abstract: {papers[1]['abstract'][:500]}

        Provide a detailed comparison covering:
        - Research questions addressed
        - Methodologies used
        - Key findings
        - Strengths and weaknesses
        - Potential synergies between the approaches
        """

        if len(papers) > 2:
            prompt += f"""
            3. {papers[2]['title']} ({papers[2]['publication_date']})
            Abstract: {papers[2]['abstract'][:500]}
            """

        analysis = model.generate_content(prompt).text
        analysis=clean_markdown(analysis)

        return {
            'comparison_table': comparison,
            'similarity_matrix': similarity.tolist(),
            'radar_chart': fig.to_json(),
            'detailed_analysis': analysis
        }

    def recommend_papers(self, user_id: str, max_results: int = 5) -> List[Dict]:
        """Recommend papers based on user's reading history and preferences"""
        if user_id not in self.user_profiles:
            return []

        profile = self.user_profiles[user_id]
        viewed_papers = profile.get('viewed_papers', [])
        saved_papers = profile.get('saved_papers', [])

        if not viewed_papers and not saved_papers:
            return []

        # Find similar papers based on viewed papers
        recommendations = []
        for paper_id in viewed_papers + saved_papers:
            if paper_id in self.papers:
                paper = self.papers[paper_id]
                similar = self.find_similar_papers(paper['title'] + " " + paper['abstract'], top_n=2)
                recommendations.extend(similar)

        # Deduplicate and rank
        seen = set()
        unique_recs = []
        for rec in recommendations:
            if rec['id'] not in seen and rec['id'] not in viewed_papers and rec['id'] not in saved_papers:
                seen.add(rec['id'])
                unique_recs.append(rec)

        # Sort by similarity
        unique_recs.sort(key=lambda x: x['similarity'], reverse=True)

        return unique_recs[:max_results]

    def generate_research_timeline(self, topic: str) -> Dict:
        """Generate an interactive timeline of key papers in a field"""
        papers = self.fetch_papers(topic, max_results=50)

        if not papers:
            return {"error": "No papers found on this topic"}

        # Sort by date
        dated_papers = []
        for paper in papers:
            if paper.get('publication_date'):
                dated_papers.append(paper)

        dated_papers.sort(key=lambda x: x['publication_date'])

        # Create timeline visualization
        fig = go.Figure()

        for paper in dated_papers[-20:]:  # Show most recent 20
            fig.add_trace(go.Scatter(
                x=[paper['publication_date']],
                y=[0],
                mode='markers+text',
                name=paper['title'][:30] + '...',
                text=[paper['title'][:30] + '...'],
                textposition='top center',
                marker=dict(size=12),
                hovertext=f"{paper['title']}<br>{paper['publication_date']}<br>{paper['authors'][0] if paper['authors'] else ''}"
            ))

        fig.update_layout(
            title=f"Research Timeline for {topic}",
            xaxis_title="Year",
            showlegend=False,
            hovermode='closest',
            height=600
        )

        # Generate summary of key milestones
        prompt = f"""
        Based on these papers, identify the key milestones in research on {topic}:
        {[p['title'] + ' (' + p['publication_date'] + ')' for p in dated_papers[-10:]]}

        Provide:
        1. 3-5 major breakthroughs
        2. How the field has evolved
        3. Current state of research
        """

        milestones = model.generate_content(prompt).text
        milestones=clean_markdown(milestones)

        return {
            'timeline': fig.to_json(),
            'milestones': milestones,
            'papers': dated_papers[-20:]
        }

    def analyze_author_impact(self, author_name: str) -> Dict:
        """Analyze an author's impact and collaboration network"""
        # Find papers by this author
        author_papers = []
        for paper in self.papers.values():
            if author_name in paper['authors']:
                author_papers.append(paper)

        if not author_papers:
            return {"error": f"No papers found for author {author_name}"}

        # Calculate metrics
        total_citations = sum(len(paper.get('citations', [])) for paper in author_papers)
        h_index = self.calculate_h_index(author_papers)
        avg_citations = total_citations / len(author_papers) if author_papers else 0

        # Get co-authors
        co_authors = set()
        for paper in author_papers:
            co_authors.update(paper['authors'])
        co_authors.discard(author_name)

        # Create collaboration network visualization
        plt.figure(figsize=(10, 8))
        ego_graph = nx.ego_graph(self.author_graph, author_name, radius=1)
        pos = nx.spring_layout(ego_graph)

        node_sizes = [1000 if node == author_name else 500 for node in ego_graph.nodes()]
        node_colors = ['red' if node == author_name else 'skyblue' for node in ego_graph.nodes()]

        nx.draw(ego_graph, pos, with_labels=True, node_size=node_sizes,
                node_color=node_colors, font_size=8, alpha=0.8)
        plt.title(f"Collaboration Network for {author_name}")

        # Save to bytes
        net_buf = BytesIO()
        plt.savefig(net_buf, format='png', bbox_inches='tight')
        net_buf.seek(0)
        plt.close()

        # Generate impact summary
        prompt = f"""
        Analyze the research impact of {author_name} based on:
        - {len(author_papers)} papers
        - {total_citations} total citations
        - h-index of {h_index}
        - Collaboration with {len(co_authors)} co-authors

        Provide:
        1. Assessment of research productivity
        2. Key areas of contribution
        3. Comparison to peers in the field
        4. Suggestions for increasing impact
        """

        analysis = model.generate_content(prompt).text
        analysis=clean_markdown(analysis)

        return {
            'metrics': {
                'total_papers': len(author_papers),
                'total_citations': total_citations,
                'h_index': h_index,
                'average_citations': avg_citations,
                'co_authors': len(co_authors)
            },
            'network_visualization': net_buf,
            'impact_analysis': analysis,
            'recent_papers': sorted(author_papers, key=lambda x: x.get('publication_date', ''), reverse=True)[:5]
        }

    def calculate_h_index(self, papers: List[Dict]) -> int:
        """Calculate h-index for an author's papers"""
        citations = [len(paper.get('citations', [])) for paper in papers]
        citations.sort(reverse=True)

        h = 0
        for i, c in enumerate(citations):
            if c >= i + 1:
                h = i + 1
            else:
                break
        return h

    def extract_methodologies(self, paper_id: str) -> List[str]:
        """Extract research methodologies from a paper using Gemini"""
        if paper_id not in self.papers:
            return []

        paper = self.papers[paper_id]
        text = paper['abstract'] + " " + paper.get('full_text', '')[:2000]

        prompt = f"""
        Extract the research methodologies used in this paper text:
        {text}

        Return as a JSON list of methodologies, each with:
        - name: methodology name
        - description: brief description
        - confidence: 0-1 confidence score
        """

        try:
            response = model.generate_content(prompt).text
            response = clean_markdown(response)
            return json.loads(response)
        except:
            return []

    def generate_research_proposal(self, topic: str, template: str = "nsf") -> Dict:
        """Generate a research proposal outline"""
        # First get relevant papers
        papers = self.fetch_papers(topic, max_results=15)

        # Get abstracts
        abstracts = [p['abstract'] for p in papers if p.get('abstract')]

        # Generate gaps analysis
        gaps = self.identify_research_gaps(topic)

        prompt = f"""
        Based on current research in {topic} and identified gaps, generate a research proposal outline following the {template} template.

        Current research highlights:
        {abstracts[:3]}

        Identified research gaps:
        {gaps}

        Include:
        1. Specific aims
        2. Background and significance
        3. Research design and methods
        4. Expected outcomes
        5. Broader impacts
        """

        proposal = model.generate_content(prompt).text
        proposal=clean_markdown(proposal)

        return {
            'proposal': proposal,
            'references': papers[:5]
        }

    def summarize_conference(self, conference_name: str, year: str) -> Dict:
        """Summarize key papers from a conference"""
        query = f"{conference_name} {year}"
        papers = self.fetch_papers(query, max_results=30)

        if not papers:
            return {"error": f"No papers found for {conference_name} {year}"}

        # Group by topic
        topics = defaultdict(list)
        for paper in papers:
            if paper.get('topics'):
                for topic in paper['topics']:
                    topics[topic].append(paper)
            else:
                topics['General'].append(paper)

        # Generate summary for each topic
        topic_summaries = {}
        for topic, papers in topics.items():
            prompt = f"""
            Summarize the key contributions in {topic} from {conference_name} {year} based on these papers:
            {[p['title'] for p in papers[:5]]}

            Highlight:
            1. Main themes
            2. Notable innovations
            3. Potential impact
            4. Limitations
            """
            topic_summaries[topic] = model.generate_content(prompt).text
            topic_summaries[topic]=clean_markdown(topic_summaries[topic])

        # Generate overall trends
        prompt = f"""
        Analyze overall trends from {conference_name} {year} based on {len(papers)} papers.

        Identify:
        1. Emerging topics
        2. Declining topics
        3. Methodological shifts
        4. Notable collaborations
        """

        trends = model.generate_content(prompt).text
        trends=clean_markdown(trends)

        return {
            'topic_summaries': topic_summaries,
            'trends_analysis': trends,
            'papers': papers[:10]
        }

    # ------------------- Helper Methods -------------------
    def calculate_relevance_score(self, query: str, content: str) -> float:
        """Calculate relevance score between query and content using TF-IDF"""
        try:
            if not query or not content:
                return 0.0

            # Fit vectorizer if not already fitted
            if not hasattr(self.vectorizer, 'vocabulary_'):
                self.vectorizer.fit(self.paper_texts + [query, content])

            # Transform the query and content
            query_vec = self.vectorizer.transform([query])
            content_vec = self.vectorizer.transform([content])

            # Calculate cosine similarity
            return float(cosine_similarity(query_vec, content_vec)[0][0])
        except Exception as e:
            print(f"Error calculating relevance: {e}")
            return 0.0

    def find_similar_papers(self, query: str, top_n: int = 3) -> List[Dict]:
        """Find papers similar to the query using TF-IDF"""
        try:
            # Transform all papers and the query
            tfidf_matrix = self.vectorizer.transform(self.paper_texts)
            query_vec = self.vectorizer.transform([query])

            # Calculate similarities
            similarities = cosine_similarity(query_vec, tfidf_matrix).flatten()

            # Get top papers
            top_indices = similarities.argsort()[-top_n:][::-1]
            return [{
                'id': self.paper_ids[i],
                'title': self.papers[self.paper_ids[i]]['title'],
                'similarity': float(similarities[i])
            } for i in top_indices if i < len(self.paper_ids)]
        except Exception as e:
            print(f"Error finding similar papers: {e}")
            return []

    def identify_research_gaps(self, field: str, user_id: Optional[str] = None) -> str:
        """Identify under-explored areas in a research field"""
        logging.basicConfig(level=logging.INFO)
        logger = logging.getLogger(__name__)

        if not field or not isinstance(field, str):
            logger.error("Invalid field provided.")
            return "Error: Please provide a valid research field."

        if field in self.cached_gaps:
            logger.info(f"Using cached gaps for {field}")
            return self.cached_gaps[field]

        try:
            papers = self.fetch_papers(field, max_results=20, user_id=user_id)
            if not papers:
                logger.warning(f"No papers found for {field}")
                return "No papers found. Try another field."

            abstracts = []
            with self.lock:
                for paper in papers:
                    paper_id = paper.get('id')
                    if paper_id in self.papers and self.papers[paper_id].get('abstract'):
                        abstracts.append(self.papers[paper_id]['abstract'])
                    if len(abstracts) >= 5:
                        break

            if not abstracts:
                logger.warning(f"No abstracts found for {field}")
                return "No abstracts available."

            prompt = f"""
            Analyze these abstracts from {field} papers: {" ".join(abstracts)}
            Identify:
            1. 3-5 key research gaps.
            2. Reasons these gaps exist (e.g., data issues, methodological limits).
            3. Potential impacts of addressing them.
            4. Suggested methods to explore these gaps.
            Keep the response clear and concise.
            """

            result = model.generate_content(prompt).text
            result=clean_markdown(result)
            self.cached_gaps[field] = result
            logger.info(f"Gaps identified for {field}")
            return result

        except Exception as e:
            logger.error(f"Error for {field}: {str(e)}")
            return f"Error: {str(e)}"

    def summarize_paper(self, paper_id: str, user_id: str = None) -> str:
        """Summarize a paper's content"""
        if paper_id not in self.papers:
            return "Paper not found."

        paper = self.papers[paper_id]
        text = paper['title'] + " " + paper['abstract'] + " " + paper.get('full_text', '')[:5000]

        prompt = f"""
        Summarize the following research paper:
        Title: {paper['title']}
        Abstract: {paper['abstract']}

        Provide a concise summary (150-200 words) covering:
        1. Main research question or objective
        2. Methodology
        3. Key findings
        4. Significance or implications
        """

        try:
            summary = model.generate_content(prompt).text
            summary=clean_markdown(summary)

            with self.lock:
                if user_id and user_id in self.user_profiles:
                    self.user_profiles[user_id]['viewed_papers'].append(paper_id)
            return summary
        except Exception as e:
            print(f"Error summarizing paper: {e}")
            return "Error generating summary."

    def generate_literature_review(self, topic: str, user_id: str = None) -> str:
        """Generate a literature review for a given topic"""
        if topic in self.cached_reviews:
            return self.cached_reviews[topic]

        papers = self.fetch_papers(topic, max_results=15, user_id=user_id)
        if not papers:
            return "No papers found for this topic."

        abstracts = [p['abstract'] for p in papers if p.get('abstract')]

        prompt = f"""
        Generate a literature review for the topic "{topic}" based on these abstracts:
        {abstracts[:5]}

        Include:
        1. Overview of current research
        2. Key themes and findings
        3. Methodological approaches
        4. Identified gaps
        5. Future research directions
        Keep the review concise (300-400 words).
        """

        try:
            review = model.generate_content(prompt).text
            review=clean_markdown(review)
            self.cached_reviews[topic] = review
            return review
        except Exception as e:
            print(f"Error generating literature review: {e}")
            return "Error generating literature review."

# ------------------- Flask Application Setup -------------------
app = Flask(__name__)
app.config['SECRET_KEY'] = os.environ.get('FLASK_SECRET_KEY', 'default-secret-key')
assistant = EnhancedResearchAssistant()

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Utility Functions
def understand_query(user_message: str) -> Dict:
    """Enhanced query understanding with multi-intent detection"""
    prompt = f"""
    Analyze this research-related query and determine the appropriate action(s):
    "{user_message}"

    Possible actions (can be multiple):
    1. summarize - Paper summary (extract paper title/ID)
    2. gap - Research gaps (extract research field)
    3. search - Search papers (extract search query)
    4. review - Literature review (extract topic)
    5. visualize - Data visualization (extract what to visualize)
    6. compare - Compare papers (extract paper IDs/titles)
    7. recommend - Paper recommendations
    8. timeline - Research timeline (extract topic)
    9. author - Author analysis (extract author name)
    10. conference - Conference summary (extract conference and year)
    11. proposal - Research proposal (extract topic)
    12. method - Methodology extraction (extract paper ID)
    13. concepts - Key concepts (extract paper ID)
    14. trends - Research trends (extract topic)
    15. general - General research help

    Respond in JSON format with:
    {{
        "actions": ["list", "of", "actions"],
        "parameters": {{
            "action1": "parameter",
            "action2": ["multiple", "parameters"]
        }},
        "confidence": 0-1
    }}
    """

    try:
        response = model.generate_content(prompt).text
        response = clean_markdown(response)
        return json.loads(response)
    except Exception as e:
        logger.error(f"Error understanding query: {e}", exc_info=True)
        return {
            "actions": ["general"],
            "parameters": {},
            "confidence": 0.5
        }
def clean_markdown(text: str) -> str:
    """Remove markdown formatting from text"""
    # Remove bold and italic
    text = re.sub(r'\*\*(.*?)\*\*', r'\1', text)
    text = re.sub(r'\*(.*?)\*', r'\1', text)
    # Remove headers
    text = re.sub(r'#+\s*', '', text)
    # Remove links
    text = re.sub(r'\[(.*?)\]\(.*?\)', r'\1', text)
    # Remove code blocks
    text = re.sub(r'`{3}.*?`{3}', '', text, flags=re.DOTALL)
    text = re.sub(r'`(.*?)`', r'\1', text)
    # Remove blockquotes
    text = re.sub(r'^\s*>+\s*', '', text, flags=re.MULTILINE)
    # Remove lists
    text = re.sub(r'^\s*[\*\-+]\s+', '', text, flags=re.MULTILINE)
    text = re.sub(r'^\s*\d+\.\s+', '', text, flags=re.MULTILINE)
    # Remove horizontal rules
    text = re.sub(r'^\s*[-*_]{3,}\s*$', '', text, flags=re.MULTILINE)
    # Clean up multiple newlines
    text = re.sub(r'\n\s*\n', '\n\n', text)
    return text.strip()
def validate_user_id(user_id: str) -> str:
    """Ensure we have a valid user ID for personalization"""
    if not user_id or user_id == 'anonymous':
        return str(uuid.uuid4())
    return user_id

# Flask Routes
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/chatting')
def index():
    return render_template('chat.html')

@app.route('/chat', methods=['POST'])
def chat():
    user_message = request.form.get('message', '').strip()
    user_id = validate_user_id(request.form.get('user_id', 'anonymous'))

    if not user_message:
        flash('Please enter a valid query.', 'error')
        return redirect(url_for('index'))

    try:
        # Understand the user's intent
        intent = understand_query(user_message)

        # Track user query for personalization
        with assistant.lock:
            if user_id not in assistant.user_profiles:
                assistant.user_profiles[user_id] = {
                    'viewed_papers': [],
                    'saved_papers': [],
                    'search_history': []
                }
            assistant.user_profiles[user_id]['search_history'].append(user_message)

        # Process each detected action
        responses = []
        for action in intent.get('actions', ['general']):
            param = intent.get('parameters', {}).get(action)

            if action == "summarize":
                if param:
                    with assistant.lock:
                        paper = next((p for p in assistant.papers.values()
                                      if p['id'] == param or p['title'].lower() == param.lower()), None)

                    if paper:
                        summary = assistant.summarize_paper(paper['id'], user_id)
                        responses.append({
                            'type': 'summary',
                            'data': {
                                'paper_id': paper['id'],
                                'title': paper['title'],
                                'summary': summary
                            }
                        })
                    else:
                        papers = assistant.fetch_papers(param, max_results=1, user_id=user_id)
                        if papers:
                            summary = assistant.summarize_paper(papers[0]['id'], user_id)
                            responses.append({
                                'type': 'summary',
                                'data': {
                                    'paper_id': papers[0]['id'],
                                    'title': papers[0]['title'],
                                    'summary': summary
                                }
                            })
                        else:
                            responses.append({
                                'type': 'error',
                                'message': f"Paper '{param}' not found. Try searching for it first."
                            })
                else:
                    responses.append({
                        'type': 'error',
                        'message': "Please specify which paper to summarize (title or ID)."
                    })

            elif action == "gap":
                if param:
                    gaps = assistant.identify_research_gaps(param, user_id)
                    responses.append({
                        'type': 'research_gaps',
                        'data': {
                            'field': param,
                            'gaps': gaps
                        }
                    })
                else:
                    responses.append({
                        'type': 'error',
                        'message': "Please specify a research field to identify gaps."
                    })

            elif action == "search":
                if param:
                    papers = assistant.fetch_papers(param, max_results=5, user_id=user_id)
                    if papers:
                        responses.append({
                            'type': 'search_results',
                            'data': {
                                'query': param,
                                'papers': papers
                            }
                        })
                    else:
                        responses.append({
                            'type': 'error',
                            'message': f"No papers found for '{param}'. Try a different query."
                        })
                else:
                    responses.append({
                        'type': 'error',
                        'message': "Please specify a search query."
                    })

            elif action == "review":
                if param:
                    review = assistant.generate_literature_review(param, user_id)
                    responses.append({
                        'type': 'literature_review',
                        'data': {
                            'topic': param,
                            'review': review
                        }
                    })
                else:
                    responses.append({
                        'type': 'error',
                        'message': "Please specify a topic for the literature review."
                    })

            elif action == "visualize":
                if param:
                    if "citation" in param.lower():
                        paper_id = intent.get('parameters', {}).get('visualize_paper')
                        if paper_id:
                            img_bytes = assistant.visualize_citation_network(paper_id)
                            if img_bytes:
                                import base64
                                image_data = base64.b64encode(img_bytes.getvalue()).decode('utf-8')
                                responses.append({
                                    'type': 'image',
                                    'data': {
                                        'title': f"Citation Network for Paper {paper_id}",
                                        'image': image_data
                                    }
                                })
                            else:
                                responses.append({
                                    'type': 'error',
                                    'message': f"Could not generate visualization for paper {paper_id}"
                                })
                        else:
                            responses.append({
                                'type': 'error',
                                'message': "Please specify which paper's citation network to visualize."
                            })
                    elif "trend" in param.lower():
                        topic = intent.get('parameters', {}).get('visualize_topic')
                        if topic:
                            trend_data = assistant.generate_trend_analysis(topic)
                            import base64
                            wordcloud_data = base64.b64encode(trend_data['wordcloud'].getvalue()).decode('utf-8')
                            trend_data['wordcloud'] = wordcloud_data
                            responses.append({
                                'type': 'trend_visualization',
                                'data': trend_data
                            })
                        else:
                            responses.append({
                                'type': 'error',
                                'message': "Please specify which topic's trends to visualize."
                            })
                    else:
                        responses.append({
                            'type': 'error',
                            'message': "Available visualizations: citation network, research trends."
                        })
                else:
                    responses.append({
                        'type': 'error',
                        'message': "Please specify what to visualize (citation network, research trends)."
                    })

            elif action == "compare":
                if param and len(param) >= 2:
                    comparison = assistant.compare_papers(param)
                    responses.append({
                        'type': 'paper_comparison',
                        'data': comparison
                    })
                else:
                    responses.append({
                        'type': 'error',
                        'message': "Please specify at least 2 papers to compare."
                    })

            elif action == "recommend":
                recs = assistant.recommend_papers(user_id)
                if recs:
                    responses.append({
                        'type': 'recommendations',
                        'data': recs
                    })
                else:
                    responses.append({
                        'type': 'error',
                        'message': "No recommendations available yet. View or save some papers first."
                    })

            elif action == "timeline":
                if param:
                    timeline = assistant.generate_research_timeline(param)
                    responses.append({
                        'type': 'research_timeline',
                        'data': timeline
                    })
                else:
                    responses.append({
                        'type': 'error',
                        'message': "Please specify a topic for the timeline."
                    })

            elif action == "author":
                if param:
                    analysis = assistant.analyze_author_impact(param)
                    if 'network_visualization' in analysis:
                        import base64
                        analysis['network_visualization'] = base64.b64encode(analysis['network_visualization'].getvalue()).decode('utf-8')
                    responses.append({
                        'type': 'author_analysis',
                        'data': analysis
                    })
                else:
                    responses.append({
                        'type': 'error',
                        'message': "Please specify an author name."
                    })

            elif action == "conference":
                if param:
                    parts = param.rsplit(' ', 1)
                    if len(parts) == 2:
                        conf, year = parts
                        summary = assistant.summarize_conference(conf, year)
                        responses.append({
                            'type': 'conference_summary',
                            'data': summary
                        })
                    else:
                        responses.append({
                            'type': 'error',
                            'message': "Please specify conference and year (e.g., 'NeurIPS 2023')."
                        })
                else:
                    responses.append({
                        'type': 'error',
                        'message': "Please specify a conference and year."
                    })

            elif action == "proposal":
                if param:
                    proposal = assistant.generate_research_proposal(param)
                    responses.append({
                        'type': 'research_proposal',
                        'data': proposal
                    })
                else:
                    responses.append({
                        'type': 'error',
                        'message': "Please specify a research topic for the proposal."
                    })

            elif action == "method":
                if param:
                    methods = assistant.extract_methodologies(param)
                    responses.append({
                        'type': 'methodologies',
                        'data': {
                            'paper_id': param,
                            'methods': methods
                        }
                    })
                else:
                    responses.append({
                        'type': 'error',
                        'message': "Please specify a paper ID to extract methodologies."
                    })

            elif action == "concepts":
                if param:
                    concepts = assistant.extract_key_concepts(param)
                    responses.append({
                        'type': 'key_concepts',
                        'data': {
                            'paper_id': param,
                            'concepts': concepts
                        }
                    })
                else:
                    responses.append({
                        'type': 'error',
                        'message': "Please specify a paper ID to extract key concepts."
                    })

            elif action == "trends":
                if param:
                    trend_data = assistant.generate_trend_analysis(param)
                    import base64
                    trend_data['wordcloud'] = base64.b64encode(trend_data['wordcloud'].getvalue()).decode('utf-8')
                    responses.append({
                        'type': 'trend_visualization',
                        'data': trend_data
                    })
                else:
                    responses.append({
                        'type': 'error',
                        'message': "Please specify a topic for trend analysis."
                    })

            else:  # general
                prompt = f"""As an AI research assistant, respond to this query:
                "{user_message}"

                Provide:
                1. Helpful information for academic research
                2. Suggested related queries
                3. Available features they could try

                Respond conversationally with markdown formatting."""
                response = model.generate_content(prompt).text
                response=clean_markdown(response)
                responses.append({
                    'type': 'general',
                    'data': response
                })

        return render_template('response.html', responses=responses, user_id=user_id, message=user_message)

    except Exception as e:
        logger.error(f"Error processing query: {e}", exc_info=True)
        flash(f"An error occurred: {str(e)}", 'error')
        return redirect(url_for('index'))

@app.route('/paper/<paper_id>')
def get_paper(paper_id):
    """Get detailed paper information"""
    with assistant.lock:
        paper = assistant.papers.get(paper_id)

    if not paper:
        flash('Paper not found', 'error')
        return redirect(url_for('index'))

    return render_template('paper.html', paper=paper)

@app.route('/visualize/citation/<paper_id>', methods=['POST'])
def visualize_citation(paper_id):
    """Generate citation network visualization"""
    try:
        img_bytes = assistant.visualize_citation_network(paper_id)
        if not img_bytes:
            flash(f"Could not generate visualization for paper {paper_id}", 'error')
            return redirect(url_for('index'))

        import base64
        image_data = base64.b64encode(img_bytes.getvalue()).decode('utf-8')
        return render_template('response.html', responses=[{
            'type': 'image',
            'data': {
                'title': f"Citation Network for Paper {paper_id}",
                'image': image_data
            }
        }], user_id=request.form.get('user_id', 'anonymous'), message=f"Visualize citation network for {paper_id}")
    except Exception as e:
        logger.error(f"Error generating visualization: {e}", exc_info=True)
        flash(f"Error generating visualization: {str(e)}", 'error')
        return redirect(url_for('index'))

@app.route('/user/<user_id>/history')
def get_user_history(user_id):
    """Get user's search and paper history"""
    with assistant.lock:
        profile = assistant.user_profiles.get(user_id, {})

    return render_template('history.html', profile=profile, user_id=user_id)

@app.route('/user/<user_id>/save/<paper_id>', methods=['POST'])
def save_paper(user_id, paper_id):
    """Save a paper to user's profile"""
    try:
        with assistant.lock:
            if user_id not in assistant.user_profiles:
                assistant.user_profiles[user_id] = {
                    'viewed_papers': [],
                    'saved_papers': [],
                    'search_history': []
                }

            if paper_id not in assistant.user_profiles[user_id]['saved_papers']:
                assistant.user_profiles[user_id]['saved_papers'].append(paper_id)

        flash('Paper saved to your profile!', 'success')
    except Exception as e:
        logger.error(f"Error saving paper: {e}", exc_info=True)
        flash(f"Error saving paper: {str(e)}", 'error')
    return redirect(url_for('index'))


app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Ensure upload folder exists
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Download NLTK data
nltk.download('punkt')
nltk.download('stopwords')

# Research-specific configuration
RESEARCH_QUESTION_TYPES = [
    'context',  # What is the background/context of this research?
    'gap',  # What gap in knowledge does this research address?
    'methodology',  # What methodology was used in this research?
    'findings',  # What are the key findings of this research?
    'implications',  # What are the practical/theoretical implications?
    'limitations',  # What are the limitations of this research?
    'future'  # What future research directions are suggested?
]


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def extract_text_from_pdf(file):
    reader = PyPDF2.PdfReader(file)
    text = ""
    for page in reader.pages:
        text += page.extract_text()
    return text


@app.route('/research_sync', methods=['GET'])
def sync():
    return render_template('research_synk.html',
                           research_questions={q: [] for q in RESEARCH_QUESTION_TYPES})


def clean_text(text):
    """Clean text by removing markdown and extra whitespace"""
    text = re.sub(r'\*\*|\*', '', text)
    text = textwrap.fill(text, width=80)  # Format for better display
    return text


def generate_research_questions(document_text=None, topic=None):
    """Generate research-specific questions using a 7-question framework"""
    genai.configure(api_key='AIzaSyCOoAQyClkN6jGPl5iskpU0knbnERA-gVE')
    model = genai.GenerativeModel('gemini-1.5-flash')

    if document_text:
        # Document-specific questions
        prompt = f"""Analyze this research document and generate 7 specific questions that would help someone understand:
        1. The research context/background
        2. The knowledge gap being addressed
        3. The methodology used
        4. The key findings
        5. The implications of the findings
        6. The limitations of the research
        7. Future research directions suggested

        Document Content (excerpt):
        {document_text[:5000]}

        Format your response as:
        context|||What is the research context for...?
        gap|||What knowledge gap does this research address about...?
        methodology|||What methodology was used to investigate...?
        findings|||What are the key findings regarding...?
        implications|||What are the implications of finding...?
        limitations|||What limitations should be considered about...?
        future|||What future research does this work suggest about...?"""
    else:
        # Topic-based questions
        prompt = f"""Generate 7 research questions about: {topic}
        The questions should cover:
        1. Context/background of the topic
        2. Potential knowledge gaps
        3. Possible methodologies to study it
        4. Hypothetical findings one might expect
        5. Potential implications
        6. Possible limitations
        7. Future research directions

        Format your response as:
        context|||What is the current understanding of...?
        gap|||What is not yet known about...?
        methodology|||How could one investigate...?
        findings|||What might researchers discover about...?
        implications|||How might findings about... impact the field?
        limitations|||What challenges might researchers face studying...?
        future|||What future studies could build on knowledge of...?"""

    response = model.generate_content(prompt)
    questions = {q: [] for q in RESEARCH_QUESTION_TYPES}

    if response.text:
        for line in response.text.split('\n'):
            if '|||' in line:
                q_type, question = line.split('|||')
                q_type = q_type.strip().lower()
                if q_type in questions:
                    questions[q_type].append(question.strip())

    return questions


def analyze_research_question(question, document_text=None):
    """Provide a detailed, research-oriented response"""
    genai.configure(api_key='AIzaSyCOoAQyClkN6jGPl5iskpU0knbnERA-gVE')
    model = genai.GenerativeModel('gemini-1.5-flash')

    if document_text:
        # Document-based analysis
        prompt = f"""Provide a detailed, academic-style answer to this research question based on the provided document.
        Include relevant details and maintain scholarly tone.

        Question: {question}

        Document Content (excerpt):
        {document_text[:5000]}

        Structure your response with:
        1. Direct answer from the document
        2. Supporting evidence/excerpts
        3. Interpretation of the findings
        4. Relation to broader research context"""
    else:
        # General research knowledge analysis
        prompt = f"""Provide a comprehensive research-oriented answer to this question.
        Include academic references and maintain scholarly tone.

        Question: {question}

        Structure your response with:
        1. Current understanding in the field
        2. Key studies on this topic
        3. Methodological approaches used
        4. Open questions and limitations
        5. Future research directions"""

    response = model.generate_content(prompt)
    return clean_text(response.text) if response.text else "I couldn't generate a response to this research question."


@app.route('/chat_sync', methods=['POST'])
def chatting():
    if request.method == 'POST':
        try:
            data = request.get_json()
            user_message = data.get('message', '')
            is_research_query = 'research' in user_message.lower() or 'paper' in user_message.lower() or 'study' in user_message.lower()

            # Check if we're in document analysis mode
            if 'document_text' in session:
                # Answer question from document with research focus
                answer = analyze_research_question(user_message, session['document_text'])
                questions = generate_research_questions(document_text=session['document_text'])

                return jsonify({
                    'response': answer,
                    'questions': questions,
                    'document_mode': True,
                    'research_mode': True
                })
            elif is_research_query:
                # Research question without document
                answer = analyze_research_question(user_message)
                questions = generate_research_questions(topic=user_message)

                return jsonify({
                    'response': answer,
                    'questions': questions,
                    'document_mode': False,
                    'research_mode': True
                })
            else:
                # Normal chat mode
                genai.configure(api_key='AIzaSyCOoAQyClkN6jGPl5iskpU0knbnERA-gVE')
                model = genai.GenerativeModel('gemini-1.5-flash')

                prompt = f"Provide a detailed response to: {user_message}"
                response = model.generate_content(prompt)
                generated_text = clean_text(response.text)

                return jsonify({
                    'response': generated_text,
                    'questions': {q: [] for q in RESEARCH_QUESTION_TYPES},
                    'document_mode': False,
                    'research_mode': False
                })
        except Exception as e:
            print(f"Error: {str(e)}")
            return jsonify({
                'response': "Sorry, I encountered an error processing your request.",
                'questions': {q: [] for q in RESEARCH_QUESTION_TYPES},
                'document_mode': False,
                'research_mode': False
            }), 500


@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file_content = file.read()
        file_stream = io.BytesIO(file_content)
        try:
            document_text = extract_text_from_pdf(file_stream)
            session['document_text'] = document_text
            questions = generate_research_questions(document_text=document_text)

            # Extract paper metadata
            title = "Research Document"
            authors = ""
            abstract = ""

            # Simple extraction of title (first non-empty line)
            for line in document_text.split('\n'):
                if line.strip():
                    title = line.strip()
                    break

            welcome_message = (
                f"Research paper '{filename}' uploaded successfully.\n"
                f"Title: {title}\n"
                "You can now ask research-specific questions about this paper."
            )

            return jsonify({
                'success': True,
                'message': welcome_message,
                'questions': questions,
                'metadata': {
                    'title': title,
                    'authors': authors,
                    'abstract': abstract
                }
            })
        except Exception as e:
            return jsonify({'error': f'Error processing PDF: {str(e)}'}), 500
    else:
        return jsonify({'error': 'Allowed file type is PDF'}), 400


@app.route('/clear_document', methods=['POST'])
def clear_document():
    if 'document_text' in session:
        session.pop('document_text')
    return jsonify({'success': True, 'message': 'Document session cleared. You can upload a new document now.'})

if __name__ == '__main__':
    app.run(debug=True)