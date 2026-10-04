"""Thread-pool runner and small summary helpers shared by every bench."""

import concurrent.futures
import json
import math


def run(jobs, worker, concurrency, jsonl_path, interval=25):
    """Run `worker(job)` over `jobs` and write one JSON row per finished job.

    Rows are written in completion order to `<jsonl_path>` and returned in the
    same order.
    """
    rows = []
    total = len(jobs)
    with open(jsonl_path, "w") as handle:
        with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as pool:
            futures = [pool.submit(worker, job) for job in jobs]
            for future in concurrent.futures.as_completed(futures):
                row = future.result()
                handle.write(json.dumps(row) + "\n")
                handle.flush()
                rows.append(row)
                if len(rows) % interval == 0 or len(rows) == total:
                    print("  %d/%d" % (len(rows), total))
    return rows


def mean(values):
    values = [value for value in values if value is not None]
    return sum(values) / len(values) if values else None


def binomial_ci(score, n):
    if not n:
        return None
    return 1.96 * math.sqrt(score * (1 - score) / n)


def write_summary(path, summary):
    with open(path, "w") as handle:
        json.dump(summary, handle, indent=2)
