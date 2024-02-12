Special_Question_Prefix = ['what', 'why', 'when', 'where', 'who', 'how', 'which', 'whose']
General_Question_Prefix = ['do', 'does', 'did', 'am', 'is', 'are', 'were', 'was', 'can', 'could', 'will', 'would',
                           'have', 'has', 'had', 'should', 'shall']

# Take P1 as an example, the corresponding general question is:
# "$EVIDENCE$\nMessage: $MESSAGE$\n"+"Is the message is LABEL?"
# Init_Predicate = {'Special': {
#     'VENUE': "What is the source of the information?",
#     'POSTTIME': "When was it published?",
#     'PUBLISHER': "Who did publish it? ",
#     'REPUTATION': "How about the background and reputation of the publisher?",
#     'INTENT': "What is the intent of this message?",
#     'STATEMENTS': "What are the important statements of this message for verification?",
#     'EVIDENCES': "What are relevant evidences to predict the veracity label of STATEMENT",
#     'EVIDENCE': "What are relevant evidences to predict the veracity label of MESSAGE"
# },
#
#     'General': {
#         'P1': ["Is the message is true?", [("Background information", "EVIDENCE"), ("Message", "MESSAGE")]],
#         'P2': ["Did the message contain adequate background information?", [("Message", "MESSAGE")]],
#         'P3': ["Is the background information in message accurate and objective?", [("Message", "MESSAGE")]],
#         'P4': [
#             "Is there any content in message that has been intentionally eliminated with the meaning being distorted?",
#             [("Message", "MESSAGE")]],
#         'P5': ["Is there an improper intention (political motive, commercial purpose, etc.) in the message?",
#                [("Message", "MESSAGE"), ("Intent", "INTENT")]],
#         'P6': ["Does the publisher have a history of publishing information with improper intention?",
#                [("Publisher Reputation", "REPUTATION")]],
#         'P7': ["Is the statement is true?", [("Background information", "EVIDENCES"), ("Statement", "STATEMENTS")]],
#     },
#     'G_prefix': "Evidence:EVIDENCE\n\nMessage: CLAIM\n\n",
#     'Q': "Q: "
# }

# P2,P3,P4  Retrieving the contextual information
# P4<-P5,P6  Is there any content that has been intentionally eliminated with the meaning being distorted?
# P7<-fact checking

# Prompt_IE_FC = refer to the markdown file


Init_Predicate = {'Special': {
    'VENUE': "What is the source of the information?",
    'POSTTIME': "When was it published?",
    'PUBLISHER': "Who did publish it? ",
    'REPUTATION': "How about the background and reputation of the publisher?",
    'INTENT': "What is the intent of this message?",
    'STATEMENTS': "What are the important statements of this message for verification?",
    'EVIDENCES': "What are relevant evidences to predict the veracity label of STATEMENT",
    'EVIDENCE': "What are relevant evidences to predict the veracity label of MESSAGE"
},

    'General': {
        # each sample in dataset is a dict, EVIDENCE/Message (the second element of the tuple) is the key of the dict.
        'P1': ["Is the message true?", [("Background information", "EVIDENCE"), ("Message", "MESSAGE")]],
        'P2': ["Did the message contain adequate background information?", [("Message", "MESSAGE")]],
        'P3': ["Is the background information in message accurate and objective?", [("Message", "MESSAGE")]],
        'P4': [
            "Is there any content in message that has been intentionally eliminated with the meaning being distorted?",
            [("Message", "MESSAGE")]],
        'P5': ["Is there an improper intention (political motive, commercial purpose, etc.) in the message?",
               [("Message", "MESSAGE"), ("Intent", "INTENT")]],
        'P6': ["Does the publisher have a history of publishing information with improper intention?",
               [("Publisher Reputation", "REPUTATION")]],
        'P7': ["Is the statement true?", [("Background information", "EVIDENCES"), ("Statement", "STATEMENTS")]],
        'P8': ["Is the message false?", [("Background information", "EVIDENCE"), ("Message", "MESSAGE")]],
    },
    'G_prefix': "Evidence:EVIDENCE\n\nMessage: CLAIM\n\n",
    'Q': "Q: "
}

Evolving_Predicate = {
'General': {
    'P9': ["Is the news report based on facts or does it primarily rely on speculation or opinion?",  [("Background information", "EVIDENCE"), ("News Report", "MESSAGE")]],
    'P10':["Are the sources cited in the news report reputable and known for their accuracy?", [("News Report", "MESSAGE")]],
'P11':["Are there any logical fallacies or misleading arguments present in the news report?", [("News Report", "MESSAGE")]],
'P12':["Does the message provide a balanced perspective, or does it exhibit bias?", [("Message", "MESSAGE")]],
'P13':["Are there any grammatical or spelling errors in the news report that may indicate a lack of professional editing?", [("News Report", "MESSAGE")]],
'P14':["Does the news report cite verifiable sources or provide references for the information presented?",  [("News Report", "MESSAGE")]],
'P15':["Does the news report use inflammatory language or make personal attacks?", [("News Report", "MESSAGE")]]
}
}

