"""G0 stage 4 (knoll-bench-plan.md section 6.0): merge near-duplicate tasks
across sessions and rank the clusters, giving the ordered queue G1 and G2 work
through.

Statements are embedded with bge-m3 and clustered greedily: each task, oldest
first, joins the most similar existing cluster when the cosine similarity to
that cluster's mean vector is at least THRESHOLD, and starts a new cluster
otherwise. A cluster's score is

    sessions + QUEUE_BONUS[best queue] + RECENCY * exp(-age_days / RECENCY_DAYS)
             + DIFFICULTY * (max difficulty - 1)

so work that recurs across sessions leads, a verified code commit (g2) beats
an analysis unit (g1) beats an idea, and recent and harder work breaks ties.
The ranking is a work order for the next stages, not a filter. Results go to
<private>/rank/<scope>.jsonl, best first.
"""

import datetime
import hashlib
import json
import math

from . import llm
from .classify import DOMAINS, PROJECTS
from .digest import SOURCES
from .link import QUEUES

EMBEDDER = "embed/bge-m3"
# Distinct but closely related tasks (three FitDoublePeak features from one
# session) reach 0.69 under bge-m3, so 0.80 merges rephrasings of one task
# without merging neighbouring work.
THRESHOLD = 0.80
BATCH = 32
QUEUE_BONUS = {"g2": 1.0, "g1": 0.5, "idea": 0.0}
RECENCY = 1.0
RECENCY_DAYS = 120
DIFFICULTY = 0.25
ALL = "all"
SIZES = [(1, "1"), (2, "2"), (5, "3to5"), (None, "gt5")]
VOCAB = frozenset(
    list(SOURCES)
    + [ALL]
    + QUEUES
    + [label for _, label in SIZES]
    + DOMAINS
    + PROJECTS
    + [
        "scope",
        "threshold",
        "tasks",
        "clusters",
        "cluster_size",
        "multi_session_clusters",
        "multi_source_clusters",
        "cluster_queue",
        "cluster_project",
        "cluster_domain",
        "top_sizes",
    ]
)


def unit(vector):
    norm = math.sqrt(sum(x * x for x in vector)) or 1.0
    return [x / norm for x in vector]


def cosine(a, b):
    return sum(x * y for x, y in zip(a, b))


def most_common(values):
    return max(set(values), key=values.count)


def size_label(size):
    for limit, label in SIZES:
        if limit is None or size <= limit:
            return label
    return SIZES[-1][1]


def count(table, key):
    table[key] = table.get(key, 0) + 1


def run_stage(private, scope):
    sources = list(SOURCES) if scope == ALL else [scope]
    tasks = []
    for source in sources:
        path = private / "link" / ("%s.jsonl" % source)
        if path.exists():
            with open(path) as handle:
                tasks += [json.loads(line) for line in handle]
    tasks.sort(key=lambda task: task["time"] or "")

    model = llm.Model(EMBEDDER)
    vectors = []
    for start in range(0, len(tasks), BATCH):
        vectors += [unit(v) for v in model.embed(task["statement"] for task in tasks[start : start + BATCH])]

    clusters = []  # [members, summed vector]
    for task, vector in zip(tasks, vectors):
        best, similarity = None, THRESHOLD
        for cluster in clusters:
            score = cosine(unit(cluster[1]), vector)
            if score >= similarity:
                best, similarity = cluster, score
        if best is None:
            clusters.append([[task], list(vector)])
        else:
            best[0].append(task)
            best[1] = [x + y for x, y in zip(best[1], vector)]

    now = datetime.datetime.now(datetime.timezone.utc)
    ranked = []
    for members, _ in clusters:
        queue = min((task["queue"] for task in members), key=["g2", "g1", "idea"].index)
        times = [datetime.datetime.fromisoformat(t["time"]) for t in members if t["time"]]
        age = (now - max(times)).total_seconds() / 86400 if times else RECENCY_DAYS * 10
        difficulty = max(task["difficulty"] for task in members)
        sessions = len({task["session"] for task in members})
        score = (
            sessions
            + QUEUE_BONUS[queue]
            + RECENCY * math.exp(-age / RECENCY_DAYS)
            + DIFFICULTY * (difficulty - 1)
        )
        lead = max(members, key=lambda t: (t["queue"] == "g2", t["difficulty"], t["time"] or ""))
        ranked.append(
            {
                "cluster": hashlib.sha256(" ".join(sorted(t["task"] for t in members)).encode()).hexdigest()[:16],
                "score": round(score, 3),
                "size": len(members),
                "sessions": sessions,
                "sources": sorted({task["source"] for task in members}),
                "queue": queue,
                "project": most_common([task["project"] for task in members]),
                "domain": most_common([task["domain"] for task in members]),
                "difficulty": difficulty,
                "latest": max(times).isoformat() if times else None,
                "lead": lead,
                "members": [task["task"] for task in members],
            }
        )
    ranked.sort(key=lambda cluster: -cluster["score"])

    out_dir = private / "rank"
    out_dir.mkdir(mode=0o700, exist_ok=True)
    with open(out_dir / ("%s.jsonl" % scope), "w") as out:
        for rank, cluster in enumerate(ranked, 1):
            out.write(json.dumps({"rank": rank, **cluster}) + "\n")

    summary = {
        "scope": scope,
        "threshold": THRESHOLD,
        "tasks": len(tasks),
        "clusters": len(ranked),
        "cluster_size": {},
        "multi_session_clusters": sum(cluster["sessions"] > 1 for cluster in ranked),
        "multi_source_clusters": sum(len(cluster["sources"]) > 1 for cluster in ranked),
        "cluster_queue": {},
        "cluster_project": {},
        "cluster_domain": {},
        "top_sizes": [cluster["size"] for cluster in ranked[:10]],
    }
    for cluster in ranked:
        count(summary["cluster_size"], size_label(cluster["size"]))
        count(summary["cluster_queue"], cluster["queue"])
        count(summary["cluster_project"], cluster["project"])
        count(summary["cluster_domain"], cluster["domain"])
    return summary
