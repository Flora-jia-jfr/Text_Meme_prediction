# from transformers import RobertaTokenizerFast, TFRobertaForSequenceClassification, pipeline
#
# new_index_2_sentiment = ["joy", "anger", "sorrow", "fear", "hate", "surprise"]
#
# ekman_map = {
#     'anger': 'anger',
#     'annoyance': 'anger',
#     'disapproval': 'anger',
#     'disgust': 'disgust',
#     'fear': 'fear',
#     'nervousness': 'fear',
#     'joy': 'joy',
#     'amusement': 'joy',
#     'approval': 'joy',
#     'excitement': 'joy',
#     'gratitude': 'joy',
#     'love': 'joy',
#     'optimism': 'joy',
#     'relief': 'joy',
#     'pride': 'joy',
#     'admiration': 'joy',
#     'desire': 'joy',
#     'caring': 'joy',
#     'sadness': 'sadness',
#     'disappointment': 'sadness',
#     'embarrassment': 'sadness',
#     'grief': 'sadness',
#     'remorse': 'sadness',
#     'surprise': 'surprise',
#     'realization': 'surprise',
#     'confusion': 'surprise',
#     'curiosity': 'surprise',
#     'neutral': 'neutral'
# }
#
#
# def get_emotion_distribution_from_context(context, ekman_map):
#     # tokenizer = RobertaTokenizerFast.from_pretrained("arpanghoshal/EmoRoBERTa")
#     # model = TFRobertaForSequenceClassification.from_pretrained("arpanghoshal/EmoRoBERTa")
#
#     emotion = pipeline('sentiment-analysis', model='arpanghoshal/EmoRoBERTa', return_all_scores=True)
#     result_dict = emotion(context)[0]
#     # result_dict.sort(key=lambda item: item['score'], reverse=True)
#     ekman_result = {
#     "joy": 0,
#     "anger": 0,
#     "sadness": 0,
#     "fear": 0,
#     "disgust": 0,
#     "surprise": 0,
#     "neutral": 0
#     }
#     for emotion in result_dict:
#         detailed_label = emotion['label']
#         ekman_label = ekman_map[detailed_label]
#         ekman_result[ekman_label] += emotion['score']
#     print(ekman_result)
#     return ekman_result.values()
#
#
# context = "I like this one, but I hate that one. I don't know"
# ekman_result_list = get_emotion_distribution_from_context(context, ekman_map)
# print(ekman_result_list)


a = [1,2,3]
print(a[:10])