# CIKM CUP 2016 Track 1: Cross-Device Linking 

The goal of the competition is to find browsing logs which belong to the same user.

More details at http://cikmcup.org and https://competitions.codalab.org/competitions/11171

## My solution

- Convert user ids into integers so they occupy less RAM 
- Split the train data into 2 folds based on connected components
- Use Elastic Search and More-Like-This queries to find top pair candidates
- Split logs into sessions (using 30 minute intervals) and compute the user "profile" (log features):
    - Number of sessions
    - Clicks within session
    - Duration of breaks between sessions
    - Starts and ends of sessions 
    - Title-based, Domain-based and Url-based similarities within sessions
- For candidates retrieved with Elastic Search, compute the following features:
    - Absolute difference between the profile features
    - Cosine between domains, full urls and titles 
- Train an xgboost model for predicting if a candidate pair corresponds to the same user or not

## Files:

- `1_prepare_data.py`: preprocesses the data
- `2_data_to_elastic.py`: puts the log data to elastic search 
- `3_candidates_elastic.py`: uses elastic search for retrieving top 70 candidates for each user
- `4_session_vectorizers.py`: "trains" count vectorizers for urls, domains and titles for user sessions
- `5_user_profiles.py`: extracts profile information from each user log 
- `6_pair_features.py`: computes features for each candidate pair
- `7_model.py`: trains the xgb model and creates the submission file


## Presentation

- This solution was presented at Berlin Machine Learning meetup. See the slides [here](http://www.slideshare.net/AlexeyGrigorev/cikm-cup-2016-crossdevice-linking).
