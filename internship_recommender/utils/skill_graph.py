import networkx as nx

# small example graph with co-occurrence / dependency edges
SKILL_EDGES = [
    ("Python", "Flask"),
    ("Python", "Django"),
    ("Python", "Pandas"),
    ("SQL", "Pandas"),
    ("Machine Learning", "Pandas"),
    ("Machine Learning", "TensorFlow"),
    ("JavaScript", "React"),
    ("HTML", "CSS"),
]

G = nx.Graph()
G.add_edges_from(SKILL_EDGES)

def rank_missing_skills(have_skills, missing_skills):
    """
    Use NetworkX centrality + proximity to user's existing skills to rank missing skills.
    """
    scores = {}
    for m in missing_skills:
        score = 0.0
        # centrality as baseline
        if m in G:
            score += nx.degree_centrality(G).get(m, 0)
        # proximity: if any neighbor is in have_skills add weight
        if m in G:
            for n in G.neighbors(m):
                if n in have_skills:
                    score += 0.5
        scores[m] = score
    # sort descending by score
    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return [r[0] for r in ranked]
