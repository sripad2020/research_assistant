import google.generativeai as genai
from typing import List, Dict, Optional
import pandas as pd
import numpy as np
from datetime import datetime
from collections import defaultdict
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.metrics.pairwise import cosine_similarity
import sqlite3
import networkx as nx
import pypdf
import tensorflow_hub as hub
import os
import uuid
import arxiv


genai.configure(api_key='YOUR_GEMINI_API_KEY')
model = genai.GenerativeModel('gemini-pro')

use_model = hub.load("https://tfhub.dev/google/universal-sentence-encoder/4")

class ResearchAssistant:
    def __init__(self, db_path: str = "papers.db"):
        self.conn = sqlite3.connect(db_path)
        self.create_tables()
        self.user_profile = {}  # Tracks user preferences
        self.citation_graph = defaultdict(list)  # Tracks citation relationships
        self.author_graph = nx.Graph()  # Tracks co-authorship
        self.hypotheses = {}  # Tracks crowdsourced hypotheses
        self.user_personalities = {}  # Tracks user personality traits
        self.gamification_scores = defaultdict(int)  # Tracks user points
        self.load_from_db()

    def create_tables(self):
        """Create SQLite tables for papers, hypotheses, and metadata"""
        cursor = self.conn.cursor()
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS papers (
                id TEXT PRIMARY KEY,
                title TEXT,
                abstract TEXT,
                authors TEXT,A
                publication_date TEXT,
                citations TEXT,
                references TEXT,
                topic INTEGER,
                topic_score REAL,
                embeddings TEXT,
                venue TEXT,
                full_text TEXT
            )
        ''')
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS hypotheses (
                id TEXT PRIMARY KEY,
                user_id TEXT,
                hypothesis TEXT,
                votes INTEGER,
                comments TEXT
            )
        ''')
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS user_data (
                user_id TEXT PRIMARY KEY,
                personality TEXT,
                mentorship_goals TEXT
            )
        ''')
        self.conn.commit()

    def load_from_db(self):
        """Load papers, hypotheses, and citation graph from database"""
        cursor = self.conn.cursor()
        cursor.execute("SELECT * FROM papers")
        for row in cursor.fetchall():
            paper = {
                'id': row[0],
                'title': row[1],
                'abstract': row[2],
                'authors': eval(row[3]) if row[3] else [],
                'publication_date': datetime.strptime(row[4], '%Y-%m-%d') if row[4] else None,
                'citations': eval(row[5]) if row[5] else [],
                'references': eval(row[6]) if row[6] else [],
                'topic': row[7],
                'topic_score': row[8],
                'embeddings': np.fromstring(row[9], sep=',') if row[9] else None,
                'venue': row[10],
                'full_text': row[11]
            }
            for cited_paper in paper['references']:
                self.citation_graph[cited_paper].append(paper['title'])
            for i, author1 in enumerate(paper['authors']):
                for author2 in paper['authors'][i + 1:]:
                    self.author_graph.add_edge(author1, author2)
        cursor.execute("SELECT * FROM hypotheses")
        for row in cursor.fetchall():
            self.hypotheses[row[0]] = {
                'user_id': row[1],
                'hypothesis': row[2],
                'votes': row[3],
                'comments': eval(row[4]) if row[4] else []
            }

    def fetch_arxiv_papers(self, query: str, max_results: int = 10, user_id: str = None) -> List[Dict]:
        """Fetch papers from arXiv and add to database"""
        client = arxiv.Client()
        search = arxiv.Search(
            query=query,
            max_results=max_results,
            sort_by=arxiv.SortCriterion.SubmittedDate
        )
        papers_added = []
        for result in client.results(search):
            authors = [author.name for author in result.authors]
            publication_date = result.published.strftime('%Y-%m-%d')
            title = result.title
            abstract = result.summary
            paper_id = result.entry_id.split('/')[-1]
            venue = "arXiv"
            citations = []  # arXiv API doesn't provide citations
            references = []  # arXiv API doesn't provide references
            pdf_url = result.pdf_url

            # Download PDF and extract text
            full_text = ""
            if pdf_url:
                try:
                    import requests
                    response = requests.get(pdf_url)
                    with open(f"temp_{paper_id}.pdf", "wb") as f:
                        f.write(response.content)
                    with open(f"temp_{paper_id}.pdf", "rb") as file:
                        reader = pypdf.PdfReader(file)
                        full_text = " ".join(page.extract_text() for page in reader.pages if page.extract_text())
                    os.remove(f"temp_{paper_id}.pdf")
                except Exception as e:
                    print(f"Error processing PDF for {title}: {e}")

            self.add_paper(
                title=title,
                abstract=abstract,
                authors=authors,
                publication_date=publication_date,
                citations=citations,
                references=references,
                venue=venue,
                pdf_path=None,  # PDF already processed
                user_id=user_id,
                full_text=full_text,
                paper_id=paper_id
            )
            papers_added.append({'title': title, 'id': paper_id})
        return papers_added

    def add_paper(self, title: str, abstract: str, authors: List[str],
                  publication_date: str, citations: List[str], references: List[str],
                  venue: str = "", pdf_path: Optional[str] = None, user_id: str = None,
                  full_text: str = "", paper_id: str = None):
        """Add a research paper to the system"""
        paper_id = paper_id or str(uuid.uuid4())
        if not full_text and pdf_path and os.path.exists(pdf_path):
            try:
                with open(pdf_path, 'rb') as file:
                    reader = pypdf.PdfReader(file)
                    full_text = " ".join(page.extract_text() for page in reader.pages if page.extract_text())
            except Exception as e:
                print(f"Error processing PDF: {e}")

        embeddings = use_model([abstract]).numpy()[0] if abstract else np.zeros(512)

        cursor = self.conn.cursor()
        cursor.execute('''
            INSERT OR REPLACE INTO papers (id, title, abstract, authors, publication_date, citations,
                                        references, topic, topic_score, embeddings, venue, full_text)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            paper_id, title, abstract, str(authors), publication_date, str(citations),
            str(references), None, None, ','.join(map(str, embeddings)), venue, full_text
        ))
        self.conn.commit()

        for cited_paper in references:
            self.citation_graph[cited_paper].append(title)
        for i, author1 in enumerate(authors):
            for author2 in authors[i + 1:]:
                self.author_graph.add_edge(author1, author2)

        if user_id:
            self.gamification_scores[user_id] += 10  # Award points for adding paper

    def summarize_paper(self, paper_title: str, user_id: str = None) -> str:
        """Generate concise summary of a research paper"""
        cursor = self.conn.cursor()
        cursor.execute("SELECT abstract, full_text FROM papers WHERE title = ?", (paper_title,))
        paper = cursor.fetchone()
        if not paper:
            return "Paper not found."
        abstract, full_text = paper
        prompt = f"""
        Summarize this research paper in 3 paragraphs highlighting:
        1. Key findings and contributions
        2. Methodology used
        3. Potential applications

        Paper title: {paper_title}
        Abstract: {abstract}
        Full text excerpt (if available): {full_text[:1000] if full_text else 'Not available'}
        """
        response = model.generate_content(prompt)
        if user_id:
            self.gamification_scores[user_id] += 5  # Award points for summarizing
        return response.text

    def cluster_topics(self, num_topics: int = 5) -> Dict:
        """Cluster papers into distinct themes using topic modeling"""
        cursor = self.conn.cursor()
        cursor.execute("SELECT abstract, id FROM papers")
        abstracts = []
        paper_ids = []
        for row in cursor.fetchall():
            if row[0]:
                abstracts.append(row[0])
                paper_ids.append(row[1])

        if not abstracts:
            return {}

        vectorizer = TfidfVectorizer(max_df=0.95, min_df=2, stop_words='english')
        tfidf = vectorizer.fit_transform(abstracts)

        lda = LatentDirichletAllocation(n_components=num_topics, random_state=42)
        lda.fit(tfidf)

        topic_dist = lda.transform(tfidf)
        for i, paper_id in enumerate(paper_ids):
            cursor.execute(
                "UPDATE papers SET topic = ?, topic_score = ? WHERE id = ?",
                (int(np.argmax(topic_dist[i])), float(np.max(topic_dist[i])), paper_id)
            )
        self.conn.commit()

        feature_names = vectorizer.get_feature_names_out()
        topics = {}
        for topic_idx, topic in enumerate(lda.components_):
            cursor.execute("SELECT title FROM papers WHERE topic = ?", (topic_idx,))
            topics[topic_idx] = {
                'keywords': [feature_names[i] for i in topic.argsort()[:-10 - 1:-1]],
                'papers': [row[0] for row in cursor.fetchall()]
            }
        return topics

    def semantic_search(self, query: str, top_k: int = 5) -> List[Dict]:
        """Find papers semantically similar to a query"""
        query_embedding = use_model([query]).numpy()[0]
        cursor = self.conn.cursor()
        cursor.execute("SELECT title, abstract, embeddings FROM papers")
        results = []
        for title, abstract, emb_str in cursor.fetchall():
            if emb_str:
                embeddings = np.fromstring(emb_str, sep=',')
                similarity = cosine_similarity([query_embedding], [embeddings])[0][0]
                results.append({'title': title, 'abstract': abstract, 'similarity': similarity})
        return sorted(results, key=lambda x: x['similarity'], reverse=True)[:top_k]

    def author_network_analysis(self) -> Dict:
        """Analyze co-authorship network"""
        degree_centrality = nx.degree_centrality(self.author_graph)
        betweenness_centrality = nx.betweenness_centrality(self.author_graph)
        return {
            'top_authors_by_degree': sorted(degree_centrality.items(), key=lambda x: x[1], reverse=True)[:5],
            'top_authors_by_betweenness': sorted(betweenness_centrality.items(), key=lambda x: x[1], reverse=True)[:5],
            'network_density': nx.density(self.author_graph)
        }

    def paper_quality_score(self, paper_title: str) -> Dict:
        """Assess paper quality based on citations and venue"""
        cursor = self.conn.cursor()
        cursor.execute("SELECT venue FROM papers WHERE title = ?", (paper_title,))
        venue = cursor.fetchone()
        if not venue:
            return {"error": "Paper not found"}
        venue = venue[0]
        citation_count = len(self.citation_graph.get(paper_title, []))
        venue_score = 0.8 if venue and ("arxiv" in venue.lower() or "nature" in venue.lower() or "science" in venue.lower()) else 0.4
        quality_score = (0.7 * min(citation_count / 10, 1)) + (0.3 * venue_score)
        return {
            'paper': paper_title,
            'citation_count': citation_count,
            'venue': venue,
            'quality_score': round(quality_score, 2)
        }

    def analyze_trends(self, field: str, years_back: int = 10) -> Dict:
        """Analyze publication trends over time"""
        cursor = self.conn.cursor()
        cursor.execute(
            "SELECT title, abstract, publication_date FROM papers WHERE publication_date >= ? AND abstract LIKE ?",
            (f"{datetime.now().year - years_back}-01-01", f"%{field.lower()}%")
        )
        recent_papers = [
            {'title': row[0], 'abstract': row[1], 'publication_date': datetime.strptime(row[2], '%Y-%m-%d')}
            for row in cursor.fetchall() if row[2]
        ]

        year_counts = defaultdict(int)

        for paper in recent_papers:
            year_counts[paper['publication_date'].year] += 1

        prompt = f"""
        Analyze these research paper abstracts from the last {years_back} years in {field}
        and identify the 5 most significant trends in the research focus.

        Abstracts:
        {" ".join([p['abstract'] for p in recent_papers])}
        """
        trends = model.generate_content(prompt).text
        return {
            'yearly_counts': dict(sorted(year_counts.items())),
            'trend_analysis': trends
        }

    def citation_analysis(self, paper_title: str) -> Dict:
        """Evaluate citation patterns for a paper"""
        citation_count = len(self.citation_graph.get(paper_title, []))
        prompt = f"""
        The paper "{paper_title}" has been cited {citation_count} times.
        Analyze the significance of this citation count in its field and
        identify which aspects of the paper are most frequently cited.
        """
        analysis = model.generate_content(prompt).text
        return {
            'citation_count': citation_count,
            'citation_analysis': analysis,
            'citing_papers': self.citation_graph.get(paper_title, [])
        }

    def analyze_sentiment(self, paper_title: str) -> Dict:
        """Assess the tone and sentiment of a paper"""
        cursor = self.conn.cursor()
        cursor.execute("SELECT abstract FROM papers WHERE title = ?", (paper_title,))
        abstract = cursor.fetchone()
        if not abstract:
            return {"error": "Paper not found"}
        prompt = f"""
        Analyze the sentiment and tone of this research paper:
        Title: {paper_title}
        Abstract: {abstract[0]}

        Provide:
        1. Overall sentiment (positive/neutral/negative)
        2. Confidence level of the sentiment
        3. Key phrases that contribute to the sentiment
        4. Assessment of how controversial the paper might be
        """
        response = model.generate_content(prompt)
        return {
            'paper': paper_title,
            'sentiment_analysis': response.text
        }

    def identify_research_gaps(self, field: str) -> str:
        """Identify under-explored areas in a field"""
        cursor = self.conn.cursor()
        cursor.execute("SELECT abstract FROM papers WHERE abstract LIKE ?", (f"%{field.lower()}%",))
        abstracts = [row[0] for row in cursor.fetchall() if row[0]]
        prompt = f"""
        Analyze these research papers in {field} and identify significant gaps in the research:
        {" ".join(abstracts)}

        Provide:
        1. 3-5 key research gaps
        2. Why these areas are under-explored
        3. Potential impact if these gaps were addressed
        """
        return model.generate_content(prompt).text

    def recommend_papers(self, user_id: str, num_recommendations: int = 5) -> List[str]:
        """Recommend papers based on user preferences"""
        if user_id not in self.user_profile:
            return "No user profile found. Please read some papers first."
        user_topics = set(p['topic'] for p in self.user_profile[user_id]['read_papers'] if p.get('topic') is not None)
        if not user_topics:
            return "No topic data available for recommendations."
        cursor = self.conn.cursor()
        cursor.execute("SELECT title FROM papers WHERE topic IN ({})".format(','.join('?' * len(user_topics))),
                       list(user_topics))
        recommended = [row[0] for row in cursor.fetchall() if
                       row[0] not in [p['title'] for p in self.user_profile[user_id]['read_papers']]]
        return recommended[:num_recommendations]

    def answer_question(self, paper_title: str, question: str) -> str:
        """Answer questions about a specific paper"""
        cursor = self.conn.cursor()
        cursor.execute("SELECT abstract, full_text FROM papers WHERE title = ?", (paper_title,))
        paper = cursor.fetchone()
        if not paper:
            return "Paper not found."
        abstract, full_text = paper
        prompt = f"""
        Answer this question about the research paper "{paper_title}":
        Question: {question}

        Use this information from the paper:
        Abstract: {abstract}
        Full text excerpt (if available): {full_text[:1000] if full_text else 'Not available'}

        If the question cannot be answered from the available information, state what additional
        information would be needed from the full paper.
        """
        return model.generate_content(prompt).text

    def generate_research_roadmap(self, field: str) -> str:
        """Generate a research roadmap for a field"""
        cursor = self.conn.cursor()
        cursor.execute("SELECT abstract FROM papers WHERE abstract LIKE ?", (f"%{field.lower()}%",))
        abstracts = [row[0] for row in cursor.fetchall() if row[0]]
        prompt = f"""
        Based on these research papers in {field}, generate a detailed research roadmap:
        {" ".join(abstracts)}

        The roadmap should include:
        1. Current state of research
        2. Immediate next steps
        3. Long-term research directions
        4. Potential breakthrough areas
        """
        return model.generate_content(prompt).text

    def generate_literature_review(self, topic: str) -> str:
        """Automate creation of literature review reports"""
        cursor = self.conn.cursor()
        cursor.execute("SELECT abstract FROM papers WHERE abstract LIKE ?", (f"%{topic.lower()}%",))
        abstracts = [row[0] for row in cursor.fetchall() if row[0]]
        prompt = f"""
        Write a comprehensive literature review on {topic} using these papers:
        {" ".join(abstracts)}

        Structure the review with:
        1. Introduction and background
        2. Key themes and findings
        3. Methodologies used
        4. Gaps in current research
        5. Conclusion and future directions
        """
        return model.generate_content(prompt).text

    def visualize_trends(self, field: str):
        """Generate visualization of publication trends"""
        trends = self.analyze_trends(field)
        plt.figure(figsize=(10, 6))
        sns.lineplot(x=list(trends['yearly_counts'].keys()),
                     y=list(trends['yearly_counts'].values()))
        plt.title(f"Publication Trends in {field}")
        plt.xlabel("Year")
        plt.ylabel("Number of Publications")
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig('trends.png')
        plt.close()

    def cross_disciplinary_fusion(self, field1: str, field2: str) -> str:
        """Generate novel research ideas by combining concepts from two fields"""
        cursor = self.conn.cursor()
        cursor.execute("SELECT abstract FROM papers WHERE abstract LIKE ?", (f"%{field1.lower()}%",))
        abstracts1 = [row[0] for row in cursor.fetchall() if row[0]]
        cursor.execute("SELECT abstract FROM papers WHERE abstract LIKE ?", (f"%{field2.lower()}%",))
        abstracts2 = [row[0] for row in cursor.fetchall() if row[0]]
        prompt = f"""
        Analyze papers from {field1} and {field2} to propose 3 novel research ideas by combining concepts:
        {field1} Abstracts: {" ".join(abstracts1)}
        {field2} Abstracts: {" ".join(abstracts2)}

        For each idea, provide:
        1. Hypothesis
        2. Potential impact
        3. Challenges
        """
        return model.generate_content(prompt).text

    def simulate_research_impact(self, proposal: str, years: int = 10) -> str:
        """Simulate the potential impact of a research proposal"""
        cursor = self.conn.cursor()
        cursor.execute("SELECT abstract FROM papers")
        all_abstracts = [row[0] for row in cursor.fetchall() if row[0]]
        prompt = f"""
        Simulate the potential impact of this research proposal over {years} years:
        Proposal: {proposal}
        Context from related papers: {" ".join(all_abstracts[:1000])}

        Provide:
        1. Academic impact (citations, publications)
        2. Industry adoption
        3. Societal implications
        4. Uncertainties
        """
        return model.generate_content(prompt).text

    def evaluate_ethical_risks(self, paper_title: str) -> Dict:
        """Evaluate ethical risks of a paper"""
        cursor = self.conn.cursor()
        cursor.execute("SELECT abstract, full_text FROM papers WHERE title = ?", (paper_title,))
        paper = cursor.fetchone()
        if not paper:
            return {"error": "Paper not found"}
        abstract, full_text = paper
        prompt = f"""
        Analyze this paper for potential ethical risks:
        Title: {paper_title}
        Abstract: {abstract}
        Full text excerpt: {full_text[:1000] if full_text else 'Not available'}

        Provide:
        1. Identified ethical risks
        2. Severity (low/medium/high)
        3. Mitigation strategies
        """
        response = model.generate_content(prompt).text
        return {
            'paper': paper_title,
            'ethical_analysis': response
        }

    def match_collaborators(self, user_id: str, field: str) -> List[Dict]:
        """Match collaborators based on expertise and personality"""
        if user_id not in self.user_personalities:
            return [{"error": "Please complete personality assessment"}]
        user_personality = self.user_personalities[user_id]
        cursor = self.conn.cursor()
        cursor.execute("SELECT authors FROM papers WHERE abstract LIKE ?", (f"%{field.lower()}%",))
        authors = set()