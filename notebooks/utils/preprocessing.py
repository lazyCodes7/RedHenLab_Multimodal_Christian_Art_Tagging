import pandas as pd
import numpy as np
from .iconclass_labels import get_iconclass_labels
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import nltk
import re
class Preprocessor:
    def __init__(self):
        self.STOP_WORDS = stopwords.words()
        self.wnl = WordNetLemmatizer()
        self.STOP_WORDS.remove('peter')
        self.labels = get_iconclass_labels()


    def preprocess_text(self, text):
        text = re.sub(r'\d+', "", text)

        # Removing hyperlinks
        text = re.sub('http://\S+|https://\S+', '', text)

        # Removing emojis
        emoji_pattern = re.compile("["
                            u"\U0001F600-\U0001F64F"
                            u"\U0001F680-\U0001F6FF"
                            u"\U0001F1E0-\U0001F1FF"
                            u"\U00002702-\U000027B0"
                            u"\U000024C2-\U0001F251"
                            "]+", flags=re.UNICODE)
        text = emoji_pattern.sub(r'', text)

        # Removing some common symbols
        text = re.sub(r'@\w+',  '', text).strip()
        text = re.sub("[^a-zA-Z0-9 ']", "", text)

        # Using a lemmatizer to get a final text
        text=' '.join([self.wnl.lemmatize(i) for i in text.lower().split()])

        # Tokenize the text
        text_tokens = word_tokenize(text)

        # Itertating through the word and if a word is not in the stop words then adding it to the list
        tokens_without_sw = [word for word in text_tokens if not word in self.STOP_WORDS]

        # Getting the filtered sentence
        filtered_sentence = (" ").join(tokens_without_sw)
        text = filtered_sentence

        # Returning the transformed/filtered text
        return text


    def check_iconclass(self, row):
        classes = []
        # Splitting words based on whitespace
        for word in row.lower().split(" "):
            index = None
            for i in range(len(self.labels)):
                # Retrieving the iconclass label
                # For eg: 11H(JOHN THE BAPTIST) -> john the baptist
                iconclass = self.labels[i].split("(")[1][:-1].lower().split(" ")

                # Next we check the words from the class with our label
                for class_words in iconclass:
                    # If found a match, set the index as i
                    if(class_words == word):
                        index = i
                      
                # If index found then we would stop from checking further labels.
                if(index is not None):
                    break

            if(index == None):
              pass

            elif(classes!=[]):
                # Resolving cases when labels originate from similar words
                # Eg: JOHN THE BAPTIST, JOHN THE EVANGELIST.
                flag = True
                for idx in range(len(classes)):
                    # Taking an iconclass
                    prev_iconclass = self.labels[classes[idx]].split("(")[1][:-1].lower().split(" ")
                    iconclass = self.labels[index].split("(")[1][:-1].lower().split(" ")

                    # If they originate from same word. Then change the index to current iconclass.
                    '''
                      Eg: John the Evangelist is smiling.
                      - Here John is common in both baptist, 
                      - We might store index of baptist first, when evangelist comes we update.
                    '''

                    if(index == classes[idx]):
                      classes.append(index)
                      break

                    elif(prev_iconclass[0] == iconclass[0] and index!=classes[idx]):
                      classes[idx] = index
                      flag = False
                      break
              
                if(flag):
                    classes.append(index)
                
            else:
                classes.append(index)

        # Return the possible to set of iconclass indexes
        return str(classes)

    def set_image_name(self, image_link):
        return "web_gallery_" + "".join(image_link.split("detail")[1].split("/"))

    def retrieve_images(self, url):
        domain = url.split("html")[0]
        endpoint = url.split("html")[1]
        image_name = endpoint.split("/")[-1]
        return domain + "detail" + endpoint + "jpg"



