# -*- coding: utf-8 -*-
"""
Created on Thu Aug  8 15:46:16 2024

@author: asiva
"""
#%%
#Data Collection phase

#importing the necessary libraries for data collection and use.
import praw            #Python Reddit API Wrapper
import pandas as pd     
import numpy as np


#authenticating usage of the reddit API
reddit = praw.Reddit(
    client_id = 'AcMQ2MCSOfbrnH_eytdelw',
    client_secret = 'DLBxbdIj_9RX9anTTodhRV5-QJeoVg',
    user_agent = 'my-app by Abhivanth',
    username = 'Effective_Insect9336',
    password = ''
    )


#Function call to serach for posts and return a data frame of the select features of the posts
def search_posts(query):
    search_results = reddit.subreddit("all").search(query,limit=20)
    list_titles = []
    list_selftexts = []
    list_scores = []
    list_upvote_ratios = []
    list_comment_counts = []
    list_link_flair_texts = []
    list_post_ids = []
    list_cds = [] 
    list_top_comments = []


     #replacing missing values with NaN values for numbers and empty strings for string values to be imputed later
    for post in search_results:
        list_titles.append(post.title if not None else '')
        list_selftexts.append(post.selftext if not None else '')
        list_scores.append(post.score if not None else np.nan)
        list_upvote_ratios.append(post.upvote_ratio if not None else np.nan)      
        list_comment_counts.append(post.num_comments if not None else np.nan)
        list_link_flair_texts.append(post.link_flair_text if not None else '')
        list_cds.append(post.created_utc if not None else np.nan)
        list_post_ids.append(post.id if not None else '')
        
        d = {
                 'Post ID': list_post_ids,
                 'Titles':list_titles,
                 'Text Description':list_selftexts,
                 'Scores':list_scores,
                 'Number of Comments':list_comment_counts,
                 'Upvote Ratio':list_upvote_ratios,
                 'Additional Description':list_link_flair_texts,
                 'Date Created (UTC)':list_cds
             }
        

        top_comments = []
        for comments in post.comments.list()[:4]:   #Getting the top 4 comments of the post
            if hasattr(comments,'body'):
                top_comments.append(comments.body)   #If more comments havent been loaded, to avoid the morecomments having no body error, checking if the comment has a body attribute in its method if not then appending the top_comments list with an empty string.
            else:
                top_comments.append('')
            
        
        
        list_top_comments.append(('|').join(top_comments))
        d['Top Comments'] = list_top_comments 
    return d


#20 sample queries for each class of analysis.    

#Declarative AI is negative statements to be classified into negative reviews about AI. Performance measure being AI is bad. Classified into Label (-1)
AI_IS_BAD_QUERIES = [
                        "AI is harmful",
                        "Negative impacts of AI",
                        "Why AI is dangerous",
                        "AI is a threat to humanity",
                        "AI causing job loss",
                        "AI is unethical",
                        "Risks of artificial intelligence",
                        "AI is out of control",
                        "AI and privacy concerns",
                        "Downsides of AI",
                        "AI causing economic inequality",
                        "AI is biased",
                        "AI leading to unemployment",
                        "AI in warfare risks",
                        "AI contributing to surveillance state",
                        "AI and loss of human touch",
                        "AI causing social isolation",
                        "Why AI should be regulated",
                        "AI is not trustworthy",
                        "AI will replace humans"
                    ]


#Declarative AI is beneficial statements to be classified into positive reviews about AI. Performance measure being AI is good. Classified into Label (1)
AI_IS_GOOD_QUERIES = [
                        "AI is beneficial",
                        "Positive impacts of AI",
                        "Why AI is revolutionary",
                        "AI is a boon to humanity",
                        "AI creating job opportunities",
                        "AI is ethical",
                        "Advantages of artificial intelligence",
                        "AI is under control",
                        "AI and enhanced privacy",
                        "Upsides of AI",
                        "AI reducing economic inequality",
                        "AI is unbiased",
                        "AI leading to employment growth",
                        "AI in peacekeeping efforts",
                        "AI enhancing global security",
                        "AI and the human touch",
                        "AI fostering social connection",
                        "Why AI should be embraced",
                        "AI is trustworthy",
                        "AI will augment human abilities"     
                     ]



#Declarative neutral statements to be classified into a mixed bag of positive and negative reviews about AI. Performance measure being AI is neural. Classified into Label (0)
NEUTRAL_QUERIES_ON_AI = [
                           "AI's impact on society",
                           "The influence of AI in various industries",
                           "AI's effects on employment",
                           "The balance between risks and rewards of AI",
                           "The role of AI in technology",
                           "AI and its potential challenges",
                           "AI's influence on daily life",
                           "The interaction between humans and AI",
                           "AI's implications in public and private sectors",
                           "General discussions on AI",
                           "AI and its ethical considerations",
                           "The dual impact of AI on society",
                           "AI in everyday technology",
                           "The societal impact of AI advancements",
                           "AI's influence on human behavior",
                           "The role of AI in shaping modern life",
                           "AI's place in contemporary society",
                           "AI and its impact on public opinion",
                           "AI's potential benefits and drawbacks",
                           "The evolving role of AI in society"    
                         ]


#Function to accumulate data for the given queries input
def accumulate_data(list_queries,label):
    # Initialize an empty dictionary to accumulate data
    accumulated_data = {
        'Post ID': [],
        'Titles': [],
        'Text Description': [],
        'Scores': [],
        'Number of Comments': [],
        'Upvote Ratio': [],
        'Additional Description': [],
        'Date Created (UTC)': [],
        'Top Comments': [],
        'Label':[]
        }

    for queries in list_queries:
        d = search_posts(queries)
        
        for key in accumulated_data:
            if(key != 'Label'):
                accumulated_data[key].extend(d[key])
                
                
   
    num_posts = len(accumulated_data['Post ID'])
    accumulated_data['Label'].extend([label] * num_posts)     #Assigning a class to each category of classification and filling it out the desired maximum number fo times to match the other features  
    df = pd.DataFrame(accumulated_data)
    return df


AI_GOOD = accumulate_data(AI_IS_GOOD_QUERIES,'1')
AI_GOOD.to_csv('AI_GOOD_TRAINING.csv')

AI_BAD = accumulate_data(AI_IS_BAD_QUERIES,'-1')
AI_BAD.to_csv('AI_BAD_TRAINING.csv')

AI_NEUTRAL = accumulate_data(NEUTRAL_QUERIES_ON_AI,'0')
AI_NEUTRAL.to_csv('AI_NEURAL_TRAINING.csv')


Final_df = pd.concat((AI_GOOD,AI_BAD,AI_NEUTRAL),axis=0)
Final_df.to_csv('TRAINING.csv')
#%%
# Data Preprocessing phase
import pandas as pd
import numpy as np

df = pd.read_csv('Training.csv')
df = df.rename(columns  = {'Unnamed: 0':'Index'})

Columns_to_clean = ['Titles','Text Description','Top Comments','Additional Description']

import re
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

def clean_column(col_name):
    corpus = []
    ps = PorterStemmer()
    all_stopwords = set(stopwords.words('english'))

    df[col_name] = df[col_name].fillna('')

    for i in range(len(df['Post ID'])):
        text = df[col_name][i]
        
        if(df[col_name][i] == ''):    
            if(df['Label'][i] == '-1'):
                df[col_name][i] = 'Bad'
            if(df['Label'][i] == '1'):
                df[col_name][i] = 'Good'
            else:
                df[col_name][i] = 'Neutral'
        
        review = re.sub('[^a-zA-Z]', ' ', text)
        review = review.lower().split()
        
        cleaned_review = [ps.stem(word) for word in review if word not in all_stopwords]

        cleaned_review = ' '.join(cleaned_review)
        corpus.append(cleaned_review)
    
    return corpus


for item in Columns_to_clean:
    df[item] = clean_column(item)
    
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features = 1200)
    
for columns in Columns_to_clean:
    df[columns] = cv.fit_transform(df[columns]).toarray()
    
    
    
X = df.iloc[:,[2,9]].values
y = df.iloc[:,-1].values
    
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.2,random_state = 0)



#from sklearn.naive_bayes import GaussianNB
#classifier = GaussianNB()
#classifier.fit(X_train,y_train)
#y_pred = classifier.predict(X_test)                   #73% accuracy

#from sklearn.ensemble import RandomForestClassifier
#classifier = RandomForestClassifier(n_estimators=500)
#classifier.fit(X_train,y_train)
#y_pred = classifier.predict(X_test)                     #77% accuracy


from sklearn.tree import DecisionTreeClassifier
classifier = DecisionTreeClassifier()
classifier.fit(X_train,y_train)
y_pred = classifier.predict(X_test)                     #73.5% accuracy


#from sklearn.linear_model import LogisticRegression
#classifier = LogisticRegression()
#classifier.fit(X_train,y_train)
#y_pred = classifier.predict(X_test) 

from sklearn.metrics import confusion_matrix,f1_score
cm = confusion_matrix(y_test,y_pred)
f1 = f1_score(y_test,y_pred,average = 'macro')



    




# DR: ₹12,904,787.38. 
# CR: ₹13,137,469.00.


    
    





                        
        
            
                                                   
                                                    
                                               





from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

ps = PorterStemmer()
print(ps.stem('beneficial'))

    
                
                
                

     
    
    
                
    


    



    
    
    


    



    
    
    
  
    
    
    
    
    








    
    
    


    
    



































































    












   




























    

