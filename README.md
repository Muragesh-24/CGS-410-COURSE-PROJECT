<h1>From Structure to Prediction: Memory Constraints and Dependency Length in Human Language and LLMs</h1>

<h2>Project Overview</h2>

<p>
This project investigates whether <b>transformer-based Large Language Models (LLMs)</b> exhibit structural patterns similar to human language under memory constraints. 
We study whether LLMs follow the <b>Dependency Length Minimization (DLM)</b> principle and whether syntactic complexity increases prediction difficulty measured using surprisal.
</p>

<p>
Human languages tend to minimize dependency length due to cognitive memory limitations. This project tests whether similar constraints emerge in LLM-generated text.
</p>

<h2>Research Questions</h2>

<ul>
<li>Do LLMs exhibit dependency length patterns similar to human language?</li>
<li>Does increasing syntactic hierarchy increase prediction difficulty?</li>
<li>Is there a correlation between dependency length and surprisal?</li>
<li>Do LLMs show evidence of memory constraints similar to humans?</li>
</ul>

<h2>Project Structure</h2>

<h3>1. Human Corpus Analysis</h3>

<p>We analyze dependency structures from human language datasets to establish a baseline.</p>

<b>Metrics:</b>

<ul>
<li>Average Dependency Length (ADL)</li>
<li>Maximum Dependency Length</li>
<li>Dependency length distribution</li>
<li>Sentence length vs dependency length</li>
</ul>

<b>Dataset:</b>

<ul>
<li>Universal Dependencies English Treebank</li>
</ul>

<hr>

<h3>2. Controlled Sentence Generation</h3>

<p>
We generate sentences with controlled syntactic complexity to test how increasing structure affects processing difficulty.
</p>

<b>Increasing sentence length:</b>

<pre>
The dog is barking.
The dog near a house is barking.
The dog near a house on the street is barking.
</pre>

<b>Increasing syntactic embedding:</b>

<pre>
The dog that the cat chased is barking.
The dog that the cat that the boy saw chased is barking.
The dog that the cat that the boy that the teacher scolded saw chased is barking.
</pre>

<hr>

<h3>3. LLM Text Generation</h3>

<p>Text is generated using LLM prompts designed to produce different syntactic complexity.</p>

<b>Example prompts:</b>

<ul>
<li>Write simple sentences</li>
<li>Write complex academic sentences</li>
<li>Write sentences with nested clauses</li>
<li>Write conversational text</li>
</ul>

<hr>

<h3>4. Dependency Parsing of LLM Text</h3>

<p>Generated text is parsed to extract syntactic relationships.</p>

<pre>
LLM text → Dependency parser → Dependency trees → DL calculation
</pre>

<hr>

<h3>5. Surprisal Analysis</h3>

<b>Metrics:</b>

<ul>
<li>Token surprisal</li>
<li>Sentence perplexity</li>
<li>Prediction entropy</li>
</ul>

<p>Purpose: Measure prediction difficulty as syntactic complexity increases.</p>

<hr>

<h3>6. Memory Constraint Experiment</h3>

<p>We test whether increasing syntactic depth increases processing difficulty.</p>

<b>Hypothesis:</b>

<ul>
<li>Higher embedding → Higher dependency length</li>
<li>Higher embedding → Higher surprisal</li>
<li>Higher embedding → Higher prediction difficulty</li>
</ul>

<hr>

<h3>7. Correlation Analysis</h3>

<b>Correlations tested:</b>

<ul>
<li>Dependency Length vs Surprisal</li>
<li>Sentence Length vs Surprisal</li>
<li>Embedding Depth vs Surprisal</li>
</ul>

<hr>

<h3>8. Human vs LLM Comparison</h3>

<b>Metrics compared:</b>

<ul>
<li>Average dependency length</li>
<li>Maximum dependency length</li>
<li>Dependency distributions</li>
<li>Surprisal trends</li>
</ul>

<hr>

<h3>9. Visualization</h3>

<b>Plots generated:</b>

<ul>
<li>Dependency length histograms</li>
<li>Surprisal vs embedding depth</li>
<li>Sentence length vs dependency length</li>
<li>Human vs LLM comparison</li>
</ul>

<hr>

<h3>10. Conclusions</h3>

<p>We summarize whether:</p>

<ul>
<li>LLMs minimize dependency length</li>
<li>Hierarchy increases prediction difficulty</li>
<li>LLMs show human-like memory constraints</li>
</ul>

<hr>

<h2>Methods Used</h2>

<ul>
<li>Dependency parsing</li>
<li>Statistical analysis</li>
<li>Correlation analysis</li>
<li>Surprisal analysis</li>
</ul>

<h2>Tools</h2>

<ul>
<li>Python</li>
<li>spaCy / Stanza</li>
<li>Matplotlib</li>
<li>NumPy</li>
<li>Transformer models</li>
</ul>

<h2>Future Work</h2>

<ul>
<li>Multi-language comparison</li>
<li>More LLM models</li>
<li>Random baseline comparison</li>
<li>Tree depth analysis</li>
<li>Attention pattern analysis</li>
</ul>

<h2>Collaborators</h2>

<ul>
<li>Muragesh Channappa Nyamagoud</li>
<li>Palak Meena</li>
<li>Kovid Saksham Lohia</li>
<li>Kajal Sankhla</li>
</ul>
