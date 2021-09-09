"""Options of join functions"""

autofj_lg = {
    "preprocess_methods":["lower", "lowerStem", "lowerRemovePunctuation",
                          "lowerRemovePunctuationStem"],
    "tokenize_methods": ["threeGram", "splitBySpace"],
    "token_weights": ["uniformWeight", "idfWeight"],
    "char_distance_functions": ["editDistance", "jaroDistance"], 
    "set_distance_functions": ["containJaccardDistance",
                               "containCosineDistance",
                               "containDiceDistance",
                               "intersectDistance",
                               "jaccardDistance",
                               "cosineDistance",
                               "diceDistance",
                               "maxincDistance"]
}

autofj_md = {
    "preprocess_methods":["lower", "lowerRemovePunctuationStem"],
    "tokenize_methods": ["threeGram", "splitBySpace"],
    "token_weights": ["uniformWeight", "idfWeight"],
    "char_distance_functions": ["editDistance", "jaroDistance"],
    "set_distance_functions": ["containJaccardDistance",
                               "containCosineDistance",
                               "containDiceDistance",
                               "intersectDistance",
                               "jaccardDistance",
                               "cosineDistance",
                               "diceDistance",
                               "maxincDistance"]
}

autofj_sm = {
    "preprocess_methods":["lower", "lowerRemovePunctuationStem"],
    "tokenize_methods": ["threeGram", "splitBySpace"],
    "token_weights": ["idfWeight"],
    "char_distance_functions": ["jaroDistance"],
    "set_distance_functions": ["containCosineDistance",
                               "jaccardDistance",
                               "maxincDistance"]
}