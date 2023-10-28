```
{'company': string; name of the company involved in the interaction,
  'has_cause': bool; does the conversation have a cause?,
  'cause': (field only occurs if has_cause is True)
           {'cause_label': string; annotation for type of cause, 
            'char_offset': list(int); position begin and end char (inclusive-exclusive), 
            'tweet_id': int; position of the cause in list of tweets in this conversation},
  'id': int; ID of this conversation,
  'is_subjective': bool; is the conversation subjective (i.e., does it contain emotions)?,
  'turns': [{'char_offset': list(int); position begin and end char (inclusive-exclusive),
             'emotions': (field only occurs if is_made_by_customer is True)
                         {'arousal': int; score on 1-5 scale,
                          'dominance': int; score on 1-5 scale,
                          'emo_clusters': list(string); list of emotion clusters,
                          'emo_labels': list(string); list of emotion labels,
                          'valence': int; score on 1-5 scale},
             'is_made_by_customer': bool; is the turn made by customer?,
             'response_strats': (field only occurs if is_made_by_customer is False)
                                list(string); list of response strategies used by operator,
             'tweets': [{'char_offset': list(int); position begin and end char (inclusive-exclusive),
                         'emotions': (field only occurs if is_made_by_customer is True)
                                     {'arousal': int; score on 1-5 scale,
                                      'dominance': int; score on 1-5 scale,
                                      'emo_clusters': list(string); list of emotion clusters,
                                      'emo_labels': list(string); list of emotion labels,
                                      'valence': int; score on 1-5 scale},
                         'response_strats': (field only occurs if is_made_by_customer is False)
                                            list(string); list of response strategies used by operator,
                         'text': string; text in tweet (note that URLs and users are anonymized (the char_offsets were created before anonymization'}]}]}
```
