# Steps to setup langchain

* Create virtual environment(>=Python3.10)
* Install requirements.
* Install ollama app in system.
* Search for llm models in ollama(llama3.2b is general purpose and light weight)
* ollama pull llama3.2:1b
* To check the model downloaded ollama list
* Modify llm_hello_world.py and run.

# LLM for Dummies:
## Intro
* LLM means large language model.
* LLMs are based on transformer arcitecture of NN.
* They only predicts the next word in a sentence. 
* They use auto regressive regression for completing sentences/paragraphs.
Example:
```
Give me a recipe of paneer tikka -> First
Give me a recipe of paneer tikka First -> Marinate
Give me a recipe of paneer tikka First Marinate -> the
Give me a recipe of paneer tikka First Marinate the -> paneer
.... so on
```
* In the world of conditional probability, P(A|B) => Probability of A assuming B has occured. Here is an exmaple for sentence:
```
The cat sleeps => P(the)
                  P(cat | The)
                  P(sleeps | The, Cat)

P(“The cat sleeps”) = P(the) X P(cat | The) X P(sleeps | The, Cat)
```
## LLM parameters:

| parameter | Explanation |
|-|-|
| Context window | Max area of context (containing prev questions and answers). Window size is fixed. Window moves and current response is based on all Q&A in current context.
|Temperature|Higher value means, all the prob of words for next output are nearer. Lower value means, all the prob of words for next outputs are far. High T -> More randomness. <= 0.5 means more predictability(coding, QA)  
| top-k |   <li> Find all the next words with probabilities.</li><li> Arrange them in desc order of probabilities.</li><li>Select the top k words and discard others.</li><li>Choose an answer based on the selected words.</li><li>Makes model more predictive(Best for code recommendation, summarization etc)</li>|
| top-p |   <li> Find all the next words with probabilities.</li><li> Arrange them in desc order of probabilities.</li><li>Add up the probabilities to reach threshold (given p)</li><li>Discard all other remaining words.</li><li>Choose an answer based on the selected words.</li><li>Makes model more creative(Best for writing poetry, etc)</li>|

## LLM Challenges:
* Hallucination:
   * It works on patterns, lack of training cases to predict responses which are not valid. Example: Wheat Grass Kheer.
* Cost vs ROI
* Security.


## Prompt:
* For LLMs prompt is very critical.
* Good prompts will ensure LLM model performance.
* Feature of good prompt:
   * Provide context.
   * Question with more clarity.
   * Provide tone, style and constaints.
   * Output format with examples.
      * Few shot/one shot/zero shot => Examples (>=2, ==1, == 0)
      * More examples better performance.

## Document similarity:
* Douments are represented via vector(Multi dimentional). Direction of the vector implies meaning of the document.
* Multiple algorithms are available for finding document similarity:
   * Euclidean Distance:
      * Pythagorian distance between the vectors.
      * Its sensitive to the manitude of the vectors.
   * Cosine similarities:
      * Cos of angle between the vectors.
      * Independent of the magnitude of the vectors.
      * Ideal candidate for fining similarity.
      * Meaning of the cosine similarities values:
      ```
      1 => identical 
      0.8 => similar
      0 => unrelated
      -1 => opposite
      ```

## Vector databases:

* In the vector database, data is saved as embeddings. 
* Embeddings are vectors of a data.
* A sentence is converted to a multi dimensional vector, the direction of the vector implies meaning of the sentence. 
* When we query in vector DB, it's done via document similarity algorithms.
* Search query is converted into embeddings and document similarity is calculated for records.
* Search results are sorted by descending order of document similarity.
* Example of vector database: chromaDB
