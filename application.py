from flask import Flask, render_template, request, jsonify, send_file, redirect, flash, url_for, session
import google.generativeai as genai
from typing import List, Dict, Optional
import arxiv
import requests
import re
import os,textwrap
from concurrent.futures import ThreadPoolExecutor
import threading
import pypdf
import uuid, io
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
import logging
from openalex import OpenAlex
import PyPDF2
from PyPDF2 import PdfReader
from werkzeug.utils import secure_filename
from google.api_core import exceptions as google_exceptions
from fuzzywuzzy import fuzz, process

UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'pdf', 'doc', 'docx'}

genai.configure(api_key='AIzaSyC9lsET5jCJJOZmoPQ8k8TeMqeYvTvhIfk')
model = genai.GenerativeModel('gemini-1.5-flash')

class EnhancedResearchAssistant:
    def __init__(self):
        self.papers = {}
        self.user_profiles = {}
        self.author_graph = nx.Graph()
        self.citation_graph = nx.DiGraph()
        self.topic_graph = nx.Graph()
        self.executor = ThreadPoolExecutor(max_workers=8)
        self.lock = threading.Lock()
        self.vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)
        self.paper_texts = []
        self.paper_ids = []
        self.cached_reviews = {}
        self.cached_gaps = {}
        self.cached_trends = {}

    def fetch_papers(self, query: str, max_results: int = 10, user_id: str = None) -> List[Dict]:
        """Fetch papers from arXiv and OpenAlex in parallel"""
        futures = []
        results = []

        # Fetch from arXiv and OpenAlex simultaneously
        futures.append(self.executor.submit(self.fetch_arxiv_papers, query, max_results // 2, user_id))
        futures.append(self.executor.submit(self.fetch_openalex_papers, query, max_results // 2, user_id))

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

        title = paper_data.get('title', 'Untitled')
        abstract = paper_data.get('abstract', '')
        authors = paper_data.get('authors', [])
        publication_date = paper_data.get('publication_date', '')
        pdf_url = paper_data.get('pdf_url', '')
        citations = paper_data.get('citations', [])
        references = paper_data.get('references', [])
        topics = paper_data.get('topics', [])

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

        relevance_score = self.calculate_relevance_score(query, title + " " + abstract) if query else 0.5

        entities = []
        text = title + " " + abstract
        entity_pattern = r'\b(?:[A-Z][a-z]*\s*)+[A-Z][a-z]*\b'
        matches = re.findall(entity_pattern, text)
        entities = [(match, "UNKNOWN") for match in matches]

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
        """Fetch papers from OpenAlex"""
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

    def visualize_citation_network(self, paper_id: str, depth: int = 1) -> BytesIO:
        """Generate visualization of citation network"""
        if paper_id not in self.citation_graph:
            return None

        nodes = [paper_id]
        for _ in range(depth):
            new_nodes = []
            for node in nodes:
                new_nodes.extend(list(self.citation_graph.successors(node)) +
                                 list(self.citation_graph.predecessors(node)))
            nodes.extend(new_nodes)
            nodes = list(set(nodes))

        subgraph = self.citation_graph.subgraph(nodes)

        plt.figure(figsize=(12, 12))
        pos = nx.spring_layout(subgraph, k=0.5, iterations=50)

        node_colors = []
        for node in subgraph.nodes():
            if node in self.papers:
                node_colors.append('skyblue')
            else:
                node_colors.append('lightgreen')

        nx.draw_networkx_nodes(subgraph, pos, node_size=800, node_color=node_colors)
        nx.draw_networkx_edges(subgraph, pos, alpha=0.5, width=1)

        labels = {}
        for node in subgraph.nodes():
            if node in self.papers:
                labels[node] = self.papers[node]['title'][:30] + '...'
        nx.draw_networkx_labels(subgraph, pos, labels, font_size=8)

        plt.title(f"Citation Network for Paper: {self.papers.get(paper_id, {}).get('title', paper_id)}")
        plt.axis('off')

        buf = BytesIO()
        plt.savefig(buf, format='png', dpi=300, bbox_inches='tight')
        buf.seek(0)
        plt.close()

        return buf

    def generate_trend_analysis(self, topic: str, years_back: int = 10) -> Dict:
        """Analyze publication trends over time"""
        if topic in self.cached_trends:
            return self.cached_trends[topic]

        papers = self.fetch_papers(topic, max_results=100)

        year_counts = defaultdict(int)
        for paper in papers:
            if 'publication_date' in paper and paper['publication_date']:
                year = paper['publication_date'][:4]
                if year.isdigit():
                    year_counts[year] += 1

        sorted_years = sorted(year_counts.items(), key=lambda x: x[0])
        years = [y[0] for y in sorted_years]
        counts = [y[1] for y in sorted_years]

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

        wc_buf = BytesIO()
        plt.savefig(wc_buf, format='png', bbox_inches='tight')
        wc_buf.seek(0)
        plt.close()

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
        analysis = clean_markdown(analysis)

        result = {
            'plot': fig.to_json(),
            'wordcloud': wc_buf,
            'analysis': analysis,
            'yearly_counts': sorted_years
        }

        self.cached_trends[topic] = result
        return result

    def extract_key_concepts(self, paper_id: str) -> Dict:
        """Extract key concepts and relationships from a paper"""
        if paper_id not in self.papers:
            return {}

        paper = self.papers[paper_id]
        text = paper['title'] + " " + paper['abstract'] + " " + paper.get('full_text', '')[:5000]

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
        """Extract subject-verb-object triples"""
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

        comparison = []
        dimensions = ['title', 'authors', 'publication_date', 'relevance_score',
                      'source', 'key_phrases', 'topics']

        for dim in dimensions:
            row = {'dimension': dim.replace('_', ' ').title()}
            for i, paper in enumerate(papers):
                row[f'paper_{i + 1}'] = paper.get(dim, 'N/A')
            comparison.append(row)

        texts = [p['title'] + " " + p['abstract'] for p in papers]
        tfidf = self.vectorizer.fit_transform(texts)
        similarity = cosine_similarity(tfidf)

        categories = ['Novelty', 'Methodology', 'Impact', 'Clarity', 'Evidence']
        scores = []
        for paper in papers:
            text = paper['title'] + " " + paper['abstract']
            words = text.lower().split()
            novelty = len([w for w in words if w in ['new', 'novel', 'innovative']]) / 10
            methodology = len([w for w in words if w in ['method', 'approach', 'model']]) / len(words)
            impact = len([w for w in words if w in ['significant', 'impact', 'important']]) / len(words)
            clarity = 1 - (len(text) / 1000)
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
        analysis = clean_markdown(analysis)

        return {
            'comparison_table': comparison,
            'similarity_matrix': similarity.tolist(),
            'radar_chart': fig.to_json(),
            'detailed_analysis': analysis
        }

    def recommend_papers(self, user_id: str, max_results: int = 5) -> List[Dict]:
        """Recommend papers based on user's reading history"""
        if user_id not in self.user_profiles:
            return []

        profile = self.user_profiles[user_id]
        viewed_papers = profile.get('viewed_papers', [])
        saved_papers = profile.get('saved_papers', [])

        if not viewed_papers and not saved_papers:
            return []

        recommendations = []
        for paper_id in viewed_papers + saved_papers:
            if paper_id in self.papers:
                paper = self.papers[paper_id]
                similar = self.find_similar_papers(paper['title'] + " " + paper['abstract'], top_n=2)
                recommendations.extend(similar)

        seen = set()
        unique_recs = []
        for rec in recommendations:
            if rec['id'] not in seen and rec['id'] not in viewed_papers and rec['id'] not in saved_papers:
                seen.add(rec['id'])
                unique_recs.append(rec)

        unique_recs = sorted(unique_recs, key=lambda x: x['similarity'], reverse=True)[:max_results]
        return unique_recs

    def find_similar_papers(self, text: str, top_n: int = 5) -> List[Dict]:
        """Find papers similar to given text using TF-IDF and cosine similarity"""
        if not self.paper_texts:
            return []

        texts = self.paper_texts + [text]
        tfidf_matrix = self.vectorizer.fit_transform(texts)
        similarities = cosine_similarity(tfidf_matrix[-1], tfidf_matrix[:-1]).flatten()

        similar_indices = similarities.argsort()[-top_n:][::-1]
        similar_papers = []
        for idx in similar_indices:
            paper_id = self.paper_ids[idx]
            if paper_id in self.papers:
                paper = self.papers[paper_id].copy()
                paper['similarity'] = similarities[idx]
                similar_papers.append(paper)

        return similar_papers

    def generate_literature_review(self, topic: str, max_papers: int = 10) -> Dict:
        """Generate a comprehensive literature review for a topic"""
        if topic in self.cached_reviews:
            return self.cached_reviews[topic]

        papers = self.fetch_papers(topic, max_papers)
        if not papers:
            return {"error": "No papers found for the topic"}

        texts = [f"Title: {p['title']}\nAbstract: {p['abstract']}" for p in papers]
        combined_text = "\n\n".join(texts)

        prompt = f"""
        Based on the following papers, generate a literature review for the topic "{topic}":
        {combined_text[:5000]}

        Structure the review as follows:
        1. Introduction to the topic
        2. Key themes and findings
        3. Methodologies used
        4. Gaps in the literature
        5. Future research directions

        Return the review as a markdown-formatted string.
        """
        try:
            review = model.generate_content(prompt).text
            review = clean_markdown(review)
        except Exception as e:
            print(f"Error generating literature review: {e}")
            review = "Unable to generate literature review due to an error."

        result = {
            'topic': topic,
            'review': review,
            'papers': papers
        }

        self.cached_reviews[topic] = result
        return result

    def identify_research_gaps(self, field: str) -> Dict:
        """Identify research gaps in a given field"""
        if field in self.cached_gaps:
            return self.cached_gaps[field]

        papers = self.fetch_papers(field, max_results=20)
        if not papers:
            return {"error": "No papers found for the field"}

        texts = [p['abstract'] for p in papers if p.get('abstract')]
        combined_text = " ".join(texts)

        prompt = f"""
        Based on recent papers in the field of {field}, identify research gaps:
        {combined_text[:5000]}

        Provide:
        1. Overview of current research focus
        2. Areas that are under-explored
        3. Specific questions that remain unanswered
        4. Recommendations for future studies

        Return as a markdown-formatted string.
        """
        try:
            gaps = model.generate_content(prompt).text
            gaps = clean_markdown(gaps)
        except Exception as e:
            print(f"Error identifying research gaps: {e}")
            gaps = "Unable to identify research gaps due to an error."

        result = {
            'field': field,
            'gaps': gaps
        }

        self.cached_gaps[field] = result
        return result

    def generate_research_proposal(self, topic: str) -> Dict:
        """Generate a structured research proposal"""
        papers = self.fetch_papers(topic, max_results=5)
        texts = [f"Title: {p['title']}\nAbstract: {p['abstract']}" for p in papers]
        combined_text = "\n\n".join(texts)

        prompt = f"""
        Generate a research proposal for the topic "{topic}" based on these papers:
        {combined_text[:5000]}

        Include:
        1. Title
        2. Introduction
        3. Research Objectives
        4. Methodology
        5. Expected Outcomes
        6. References

        Return as a markdown-formatted string.
        """
        try:
            proposal = model.generate_content(prompt).text
            proposal = clean_markdown(proposal)
        except Exception as e:
            print(f"Error generating research proposal: {e}")
            proposal = "Unable to generate research proposal due to an error."

        result = {
            'topic': topic,
            'proposal': proposal,
            'references': papers
        }
        return result

    def extract_methodologies(self, paper_id: str) -> Dict:
        """Extract research methodologies from a paper"""
        if paper_id not in self.papers:
            return {"error": "Paper not found"}

        paper = self.papers[paper_id]
        text = paper.get('full_text', paper['abstract'])

        prompt = f"""
        Extract research methodologies from the following paper:
        Title: {paper['title']}
        Text: {text[:5000]}

        Return as a JSON list of objects with:
        - name: Methodology name
        - description: Brief description
        """
        try:
            response = model.generate_content(prompt).text
            response = clean_markdown(response)
            methods = json.loads(response)
        except Exception as e:
            print(f"Error extracting methodologies: {e}")
            methods = []

        return {
            'paper_id': paper_id,
            'methods': methods
        }

    def analyze_author(self, author_name: str) -> Dict:
        """Analyze an author's impact and collaboration network"""
        papers = []
        for paper in self.papers.values():
            if any(fuzz.ratio(author_name.lower(), author.lower()) > 80 for author in paper['authors']):
                papers.append(paper)

        if not papers:
            return {"error": "No papers found for the author"}

        total_citations = sum(len(p.get('citations', [])) for p in papers)
        co_authors = set()
        for p in papers:
            co_authors.update(p['authors'])
        co_authors.discard(author_name)

        h_index = 0
        citation_counts = sorted([len(p.get('citations', [])) for p in papers], reverse=True)
        for i, citation in enumerate(citation_counts):
            if citation < i + 1:
                h_index = i
                break
        else:
            h_index = len(citation_counts)

        G = nx.Graph()
        for p in papers:
            authors = p['authors']
            for i, a1 in enumerate(authors):
                for a2 in authors[i + 1:]:
                    G.add_edge(a1, a2)

        plt.figure(figsize=(10, 10))
        pos = nx.spring_layout(G)
        nx.draw_networkx_nodes(G, pos, node_size=500, node_color='skyblue')
        nx.draw_networkx_edges(G, pos, alpha=0.5)
        nx.draw_networkx_labels(G, pos, font_size=8)
        plt.title(f"Collaboration Network for {author_name}")

        buf = BytesIO()
        plt.savefig(buf, format='png', dpi=300, bbox_inches='tight')
        buf.seek(0)
        plt.close()

        prompt = f"""
        Analyze the research impact of author "{author_name}" based on:
        - Total papers: {len(papers)}
        - Total citations: {total_citations}
        - h-index: {h_index}
        - Number of co-authors: {len(co_authors)}

        Provide:
        1. Summary of research contributions
        2. Influence in the field
        3. Collaboration patterns
        4. Potential areas of expertise
        """
        try:
            analysis = model.generate_content(prompt).text
            analysis = clean_markdown(analysis)
        except Exception as e:
            print(f"Error analyzing author: {e}")
            analysis = "Unable to analyze author due to an error."

        return {
            'author_name': author_name,
            'metrics': {
                'total_papers': len(papers),
                'total_citations': total_citations,
                'h_index': h_index,
                'co_authors': len(co_authors)
            },
            'network_visualization': buf,
            'impact_analysis': analysis
        }

    def summarize_conference(self, conference: str, year: str) -> Dict:
        """Summarize key papers and trends from a conference"""
        query = f"{conference} {year}"
        papers = self.fetch_papers(query, max_results=20)

        if not papers:
            return {"error": "No papers found for the conference"}

        texts = [p['abstract'] for p in papers if p.get('abstract')]
        combined_text = " ".join(texts)

        prompt = f"""
        Summarize the key papers from {conference} {year} based on:
        {combined_text[:5000]}

        Provide:
        1. Key topics covered
        2. Major findings
        3. Emerging trends
        4. Notable papers
        """
        try:
            summary = model.generate_content(prompt).text
            summary = clean_markdown(summary)
        except Exception as e:
            print(f"Error summarizing conference: {e}")
            summary = "Unable to summarize conference due to an error."

        topics = defaultdict(list)
        for p in papers:
            for topic in p.get('topics', []):
                topics[topic].append(p)

        topic_summaries = {}
        for topic, topic_papers in topics.items():
            prompt = f"""
            Summarize the research on "{topic}" from {conference} {year}:
            {[p['abstract'] for p in topic_papers][:2000]}

            Provide a brief summary (2-3 sentences).
            """
            try:
                topic_summary = model.generate_content(prompt).text
                topic_summaries[topic] = clean_markdown(topic_summary)
            except Exception as e:
                print(f"Error summarizing topic {topic}: {e}")
                topic_summaries[topic] = "Summary unavailable."

        return {
            'conference': conference,
            'year': year,
            'topic_summaries': topic_summaries,
            'trends_analysis': summary
        }

    def generate_research_timeline(self, topic: str) -> Dict:
        """Generate a timeline of research milestones"""
        papers = self.fetch_papers(topic, max_results=50)
        if not papers:
            return {"error": "No papers found for the topic"}

        papers = sorted(papers, key=lambda x: x['publication_date'] or '0000-00-00')
        milestones = []
        for p in papers[:10]:
            prompt = f"""
            Identify a key milestone from this paper:
            Title: {p['title']}
            Abstract: {p['abstract'][:1000]}
            Publication Date: {p['publication_date']}

            Return as JSON:
            {{
                "date": "YYYY-MM-DD",
                "milestone": "Brief description"
            }}
            """
            try:
                response = model.generate_content(prompt).text
                milestone = json.loads(clean_markdown(response))
                milestones.append(milestone)
            except Exception as e:
                print(f"Error generating milestone for {p['title']}: {e}")

        fig = go.Figure(data=[
            go.Scatter(
                x=[m['date'] for m in milestones],
                y=[1] * len(milestones),
                mode='markers+text',
                text=[m['milestone'][:50] + '...' for m in milestones],
                textposition='top center',
                marker=dict(size=10)
            )
        ])

        fig.update_layout(
            title=f"Research Timeline for {topic}",
            xaxis_title="Date",
            yaxis=dict(showticklabels=False),
            showlegend=False
        )

        return {
            'topic': topic,
            'timeline': fig.to_json(),
            'milestones': milestones
        }

    def calculate_relevance_score(self, query: str, text: str) -> float:
        """Calculate relevance score between query and text"""
        if not query or not text:
            return 0.0
        texts = [query, text]
        tfidf = self.vectorizer.fit_transform(texts)
        return cosine_similarity(tfidf[0], tfidf[1])[0][0]

    def summarize_paper(self, paper_id: str) -> Dict:
        """Generate a summary of a paper"""
        if paper_id not in self.papers:
            return {"error": "Paper not found"}

        paper = self.papers[paper_id]
        text = paper.get('full_text', paper['abstract'])

        prompt = f"""
        Summarize the following research paper:
        Title: {paper['title']}
        Abstract: {paper['abstract']}
        Full Text (excerpt): {text[:5000]}

        Provide a summary (150-200 words) covering:
        - Research problem
        - Methodology
        - Key findings
        - Implications
        """
        try:
            summary = model.generate_content(prompt).text
            summary = clean_markdown(summary)
        except Exception as e:
            print(f"Error summarizing paper: {e}")
            summary = "Unable to summarize paper due to an error."

        return {
            'paper_id': paper_id,
            'title': paper['title'],
            'summary': summary
        }

def clean_markdown(text: str) -> str:
    text = re.sub(r'\*\*(.*?)\*\*', r'\1', text)
    text = re.sub(r'\*(.*?)\*', r'\1', text)
    text = re.sub(r'#+\s*', '', text)
    text = re.sub(r'\[(.*?)\]\(.*?\)', r'\1', text)
    text = re.sub(r'`{3}.*?`{3}', '', text, flags=re.DOTALL)
    text = re.sub(r'`(.*?)`', r'\1', text)
    text = re.sub(r'^\s*>+\s*', '', text, flags=re.MULTILINE)
    text = re.sub(r'^\s*[\*\-+]\s+', '', text, flags=re.MULTILINE)
    text = re.sub(r'^\s*\d+\.\s+', '', text, flags=re.MULTILINE)
    text = re.sub(r'^\s*[-*_]{3,}\s*$', '', text, flags=re.MULTILINE)
    text = re.sub(r'\n\s*\n', '\n\n', text)
    return text.strip()

app = Flask(__name__)
app.config['SECRET_KEY'] = os.urandom(24).hex()
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

assistant = EnhancedResearchAssistant()


@app.route('/')
def home():
    return render_template('index.html')

@app.route('/research')
def index():
    user_id = session.get('user_id', str(uuid.uuid4()))
    session['user_id'] = user_id
    now = datetime.now()
    responses = []
    return render_template('home.html', user_id=user_id, now=now, responses=responses)

@app.route('/chat', methods=['POST'])
def chat():
    user_id = request.form.get('user_id')
    message = request.form.get('message')
    now = datetime.now()

    if not user_id or not message:
        flash('User ID and message are required.', 'error')
        return redirect(url_for('index'))

    if user_id not in assistant.user_profiles:
        assistant.user_profiles[user_id] = {
            'viewed_papers': [],
            'saved_papers': [],
            'search_history': []
        }

    assistant.user_profiles[user_id]['search_history'].append(message)

    responses = []
    response = {"type": "text", "data": ""}

    try:
        # Handle different types of requests
        if message.lower().startswith("summarize the paper:"):
            paper_title = message[20:].strip()
            paper_id = None
            for pid, paper in assistant.papers.items():
                if fuzz.ratio(paper['title'].lower(), paper_title.lower()) > 80:
                    paper_id = pid
                    break
            if paper_id:
                response = {
                    "type": "summary",
                    "data": assistant.summarize_paper(paper_id)
                }
                assistant.user_profiles[user_id]['viewed_papers'].append(paper_id)
            else:
                response["data"] = "Paper not found. Please provide a valid paper title."

        elif message.lower().startswith("what are research gaps in"):
            field = message[25:].strip()
            response = {
                "type": "research_gaps",
                "data": assistant.identify_research_gaps(field)
            }

        elif message.lower().startswith("generate literature review about"):
            topic = message[31:].strip()
            response = {
                "type": "literature_review",
                "data": assistant.generate_literature_review(topic)
            }

        elif message.lower().startswith("compare papers:"):
            titles = [t.strip() for t in message[14:].split(',')]
            paper_ids = []
            for title in titles:
                for pid, paper in assistant.papers.items():
                    if fuzz.ratio(paper['title'].lower(), title.lower()) > 80:
                        paper_ids.append(pid)
                        break
            if len(paper_ids) >= 2:
                response = {
                    "type": "paper_comparison",
                    "data": assistant.compare_papers(paper_ids)
                }
            else:
                response["data"] = "Please provide at least two valid paper titles."

        elif message.lower().startswith("show citation network for"):
            paper_title = message[24:].strip()
            paper_id = None
            for pid, paper in assistant.papers.items():
                if fuzz.ratio(paper['title'].lower(), paper_title.lower()) > 80:
                    paper_id = pid
                    break
            if paper_id:
                buf = assistant.visualize_citation_network(paper_id)
                if buf:
                    import base64
                    img_base64 = base64.b64encode(buf.getvalue()).decode('utf-8')
                    response = {
                        "type": "image",
                        "data": {
                            "title": "Citation Network",
                            "image": img_base64
                        }
                    }
                else:
                    response["data"] = "Unable to generate citation network."
            else:
                response["data"] = "Paper not found."

        elif message.lower().startswith("show trends in"):
            field = message[14:].strip()
            response = {
                "type": "trend_visualization",
                "data": assistant.generate_trend_analysis(field)
            }

        elif message.lower().startswith("analyze author"):
            author_name = message[14:].strip()
            response = {
                "type": "author_analysis",
                "data": assistant.analyze_author(author_name)
            }

        elif message.lower().startswith("summarize"):
            conf_year = message[9:].strip().split()
            if len(conf_year) >= 2:
                conference = " ".join(conf_year[:-1])
                year = conf_year[-1]
                response = {
                    "type": "conference_summary",
                    "data": assistant.summarize_conference(conference, year)
                }
            else:
                response["data"] = "Please specify conference and year."

        elif message.lower().startswith("generate proposal about"):
            topic = message[22:].strip()
            response = {
                "type": "research_proposal",
                "data": assistant.generate_research_proposal(topic)
            }

        elif message.lower().startswith("extract methodologies from"):
            paper_title = message[25:].strip()
            paper_id = None
            for pid, paper in assistant.papers.items():
                if fuzz.ratio(paper['title'].lower(), paper_title.lower()) > 80:
                    paper_id = pid
                    break
            if paper_id:
                response = {
                    "type": "methodologies",
                    "data": assistant.extract_methodologies(paper_id)
                }
            else:
                response["data"] = "Paper not found."

        elif message.lower().startswith("extract key concepts from"):
            paper_title = message[25:].strip()
            paper_id = None
            for pid, paper in assistant.papers.items():
                if fuzz.ratio(paper['title'].lower(), paper_title.lower()) > 80:
                    paper_id = pid
                    break
            if paper_id:
                response = {
                    "type": "key_concepts",
                    "data": assistant.extract_key_concepts(paper_id)
                }
            else:
                response["data"] = "Paper not found."

        elif message.lower().startswith("show timeline for"):
            topic = message[17:].strip()
            response = {
                "type": "research_timeline",
                "data": assistant.generate_research_timeline(topic)
            }

        elif message.lower().startswith("search for"):
            query = message[10:].strip()
            papers = assistant.fetch_papers(query, user_id=user_id)
            response = {
                "type": "search_results",
                "data": {
                    "query": query,
                    "papers": papers
                }
            }

        elif message.lower().startswith("recommend papers"):
            recommendations = assistant.recommend_papers(user_id)
            response = {
                "type": "recommendations",
                "data": recommendations
            }

        else:
            # General research help
            prompt = f"""
            You are an academic research assistant. Respond to the following query:
            {message}

            Provide a concise and informative response, including references to relevant papers if possible.
            """
            try:
                answer = model.generate_content(prompt).text
                response["data"] = clean_markdown(answer)
            except Exception as e:
                print(f"Error processing general query: {e}")
                response["data"] = "Unable to process your request at this time."

    except Exception as e:
        print(f"Error in chat route: {e}")
        response["data"] = "An error occurred while processing your request."

    responses.append(response)
    return render_template('home.html', user_id=user_id, now=now, responses=responses, message=message)

@app.route('/paper/<paper_id>')
def paper_details(paper_id):
    user_id = session.get('user_id', str(uuid.uuid4()))
    session['user_id'] = user_id
    now = datetime.now()

    if paper_id not in assistant.papers:
        flash('Paper not found.', 'error')
        return redirect(url_for('index'))

    paper = assistant.papers[paper_id]
    assistant.user_profiles.setdefault(user_id, {'viewed_papers': [], 'saved_papers': [], 'search_history': []})
    if paper_id not in assistant.user_profiles[user_id]['viewed_papers']:
        assistant.user_profiles[user_id]['viewed_papers'].append(paper_id)

    return render_template('home.html', user_id=user_id, now=now, paper=paper)

@app.route('/user/<user_id>/save/<paper_id>', methods=['POST'])
def save_paper(user_id, paper_id):
    if paper_id not in assistant.papers:
        flash('Paper not found.', 'error')
        return redirect(url_for('index'))

    assistant.user_profiles.setdefault(user_id, {'viewed_papers': [], 'saved_papers': [], 'search_history': []})
    if paper_id not in assistant.user_profiles[user_id]['saved_papers']:
        assistant.user_profiles[user_id]['saved_papers'].append(paper_id)
        flash('Paper saved successfully.', 'success')
    else:
        flash('Paper already saved.', 'info')

    return redirect(url_for('paper_details', paper_id=paper_id))

@app.route('/visualize/citation/<paper_id>', methods=['POST'])
def visualize_citation(paper_id):
    user_id = request.form.get('user_id')
    if paper_id not in assistant.papers:
        flash('Paper not found.', 'error')
        return redirect(url_for('index'))

    buf = assistant.visualize_citation_network(paper_id)
    if not buf:
        flash('Unable to generate citation network.', 'error')
        return redirect(url_for('paper_details', paper_id=paper_id))

    import base64
    img_base64 = base64.b64encode(buf.getvalue()).decode('utf-8')
    response = {
        "type": "image",
        "data": {
            "title": "Citation Network",
            "image": img_base64
        }
    }
    return render_template('home.html', user_id=user_id, now=datetime.now(), responses=[response])

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

@app.route('/research_sync')
def research_sync():
    return render_template('research_synk.html',
                           research_questions={q: [] for q in RESEARCH_QUESTION_TYPES})
def clean_text(text):
    text = re.sub(r'\*\*|\*', '', text)
    text = textwrap.fill(text, width=80)  # Format for better display
    return text


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

def extract_text_from_pdf(file):
    reader = PyPDF2.PdfReader(file)
    text = ""
    for page in reader.pages:
        text += page.extract_text()
    return text

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
    nltk.download('punkt')
    nltk.download('stopwords')
    app.run(debug=True, host='0.0.0.0', port=5000)