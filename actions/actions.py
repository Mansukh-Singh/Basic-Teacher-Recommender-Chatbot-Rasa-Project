from typing import Any, Text, Dict, List
from rasa_sdk import Action, Tracker
from rasa_sdk.executor import CollectingDispatcher
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import json
import numpy as np

with open("teacher.json") as file:
    TEACHERS = json.load(file) 

def preprocess_user_query(user_query):
    # Implement any preprocessing steps required for user query
    # For example, you can convert the user query to lowercase, remove stopwords, etc.
    query = [user_query.lower()]

    return query

def extract_teacher_features(dataset):
    teacher_features = [
    "{} {} {} {}".format(teacher['qualifications'],teacher["subject"], teacher["experience"], teacher["grade_level"]).lower()
    for teacher in dataset
    ]
    return teacher_features

class ActionRecommendTeacher(Action):
    def name(self) -> Text:
        return "action_recommend_teacher"

    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        
        vectorizer = TfidfVectorizer() 

        # Retrieve user query
        user_query = preprocess_user_query(tracker.latest_message['text'])
        print(user_query)

        teacher_features = extract_teacher_features(TEACHERS)

        tfidf_matrix = vectorizer.fit_transform(teacher_features)

        query_vector = vectorizer.transform(user_query)

        similarity_scores = cosine_similarity(query_vector, tfidf_matrix)
        
        # Get the index of the most similar teacher
        most_similar_index = np.argmax(similarity_scores)

        # Get the most similar teacher
        most_similar_teacher = TEACHERS[most_similar_index]

        if most_similar_teacher:
            response = {
                "teacher_name": most_similar_teacher["name"],
                "subject": most_similar_teacher["subject"],
                "qualifications": most_similar_teacher["qualifications"]
            }
            dispatcher.utter_message(template="utter_recommend_teacher", **response)
        else:
            dispatcher.utter_message(text="I'm sorry, I couldn't find a teacher for that subject.")

        return []
    
# This files contains your custom actions which can be used to run
# custom Python code.
#
# See this guide on how to implement these action:
# https://rasa.com/docs/rasa/custom-actions


# This is a simple example for a custom action which utters "Hello World!"

# from typing import Any, Text, Dict, List
#
# from rasa_sdk import Action, Tracker
# from rasa_sdk.executor import CollectingDispatcher
#
#
# class ActionHelloWorld(Action):
#
#     def name(self) -> Text:
#         return "action_hello_world"
#
#     def run(self, dispatcher: CollectingDispatcher,
#             tracker: Tracker,
#             domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
#
#         dispatcher.utter_message(text="Hello World!")
#
#         return []
