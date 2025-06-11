import streamlit as st
import nltk
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import wikipedia

# Download required NLTK data
nltk.download('averaged_perceptron_tagger')
nltk.download('punkt')
nltk.download('wordnet')

lemmatizer = WordNetLemmatizer()

def lemma_me(sent):
    sentence_tokens = nltk.word_tokenize(sent.lower())
    pos_tags = nltk.pos_tag(sentence_tokens)

    sentence_lemmas = []
    for token, pos_tag in zip(sentence_tokens, pos_tags):
        if pos_tag[1][0].lower() in ['n', 'v', 'a', 'r']:
            lemma = lemmatizer.lemmatize(token, pos_tag[1][0].lower())
            sentence_lemmas.append(lemma)

    return sentence_lemmas

def process(subject, question):
    try:
        text = wikipedia.page(subject).content
        sentence_tokens = nltk.sent_tokenize(text)
        sentence_tokens.append(question)

        tv = TfidfVectorizer(tokenizer=lemma_me)
        tf = tv.fit_transform(sentence_tokens)
        values = cosine_similarity(tf[-1], tf)
        index = values.argsort()[0][-2]
        values_flat = values.flatten()
        values_flat.sort()
        coeff = values_flat[-2]

        if coeff > 0.3:
            return sentence_tokens[index]
        else:
            return "I couldn't find a confident match. Please ask a different question."
    except wikipedia.exceptions.DisambiguationError as e:
        return f"Disambiguation Error. Try to be more specific. Options: {', '.join(e.options[:5])}"
    except wikipedia.exceptions.PageError:
        return "The subject you entered was not found on Wikipedia."
    except Exception as e:
        return f"An error occurred: {e}"

# Streamlit UI
st.title("Wikipedia Chatbot")
st.write("Ask questions about any topic from Wikipedia!")

subject = st.text_input("Enter a subject:", "Computer Science")
question = st.text_input(f"What do you want to know about {subject}?")

if st.button("Ask"):
    if subject and question:
        with st.spinner('Searching Wikipedia...'):
            answer = process(subject, question)
            st.success("Response:")
            st.write(answer)
    else:
        st.warning("Please enter both a subject and a question.")
