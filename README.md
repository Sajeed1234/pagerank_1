# Website Importance Ranking Using PageRank

Run in this order (Python 3.10+, Java 17+ for PySpark):

    pip install numpy scipy networkx pyspark matplotlib
    python3 generate_graph.py          # writes web_synth.txt (seed 42, SNAP edge-list format)
    python3 run_experiments.py         # NumPy vs NetworkX, in-degree, damping -> results_core.json
    python3 measure_memory.py save     # caches the cleaned edge list
    python3 measure_memory.py numpy    # peak memory, NumPy/SciPy
    python3 measure_memory.py networkx # peak memory, NetworkX
    python3 run_spark_benchmark.py     # PySpark vs NumPy vs NetworkX on 10k/25k/50k/100k pages
    python3 make_charts.py             # figures/*.png

To use a real SNAP graph (web-Stanford, web-Google), download the .txt file, rename it
web_synth.txt (same tab-separated "from to" format with # comment lines) and re-run.
Timings depend on your machine; the stored results/ files are from the run reported.
