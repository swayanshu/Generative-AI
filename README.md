# Text Generation Using Generative AI

# Project Description:

Explore the capabilities of generative AI models for text generation using the GPT-2 model from the Hugging Face Transformers library. The project aimed to generate coherent and creative text based on user-provided prompts and assess the quality of the generated text.

# Methods:

1. Model Selection: I selected the GPT-2 model and configured it for text generation.
2. Text Generation: Developed a Python program that allowed users to input prompts, and the model generated text based on those prompts.
3. Analysis: I used perplexity as a metric to evaluate the generated text's quality. I also assessed the generated text qualitatively for coherence and relevance.
4. Visualization: To enhance the user experience, I incorporated word cloud visualization of frequently occurring words in the generated text.

# Model:

```ruby
import torch
!pip install transformers  #if not installed
import transformers
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from wordcloud import WordCloud
import matplotlib.pyplot as plt

# Load pre-trained GPT-2 model and tokenizer
model_name = "gpt2-medium"  # You can change this to other GPT-2 variants
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
model = GPT2LMHeadModel.from_pretrained(model_name)

def generate_text(prompt, max_length=100, num_return_sequences=1):
    input_ids = tokenizer.encode(prompt, return_tensors="pt")
    output_ids = model.generate(
        input_ids,
        max_length=max_length,
        num_return_sequences=num_return_sequences,
        no_repeat_ngram_size=2,
        top_k=50,
        top_p=0.95,
        temperature=0.7,
    )
    generated_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    return generated_text

#  Input provided by User for a prompt
prompt = "what is self supervised learning?"
generated_text = generate_text(prompt)

#Let's create a visualization of our result
wordcloud = WordCloud(width=800, height=400, background_color="white").generate(generated_text)
# Display generated text and word cloud
print("Generated Text:")
print(generated_text)
print(f"Perplexity Score: {perplexity}")

# Display the word cloud
plt.figure(figsize=(10, 5))
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis("off")
plt.title("Word Cloud of Generated Text")
plt.show()
```

# Results:

- The generative AI model consistently produced coherent and contextually relevant text based on user prompts. The quality of the generated text was assessed through a combination of perplexity scores and qualitative analysis.
- Word Cloud Visualization: The word cloud visualization highlighted frequently occurring words in the generated text, providing an insightful view of the text's content
