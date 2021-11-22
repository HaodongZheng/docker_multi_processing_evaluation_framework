from flask import Flask
import initialize_services
from initialize_services import extract_intent_and_semantic_tags_from_result, get_single_result, initialize_through_recalculating, method_evaluation_parsing_based_CKY_MED, semantic_grammar_parsing_general_idf
import socket
import threading
import json
import multiprocessing as mp
import nltk
app = Flask(__name__)

@app.route('/')
def get_result():
    nltk.download('wordnet')
    grammar_pieces_dict, grammars_idf_from_dict_2, normalized_idf_dict, states, A, B, vocab, vocab_document, intent_taglist_dict, pos_dict_for_grammar_terminal, lowest_grammar_idf = initialize_through_recalculating(cnf_option=True)
    test_sentences = []
    test_intent_labels = []
    with open("test_sentences.txt", "r") as f:
        for line in f.readlines():
            if line != "\n":
                test_sentence_pair = line.split("|")
                sentence = (
                    test_sentence_pair[0]
                    .replace(",", "")
                    .replace(".", "")
                    .replace("!", "")
                    .replace("?", "")
                )
                label = test_sentence_pair[1].replace(" ", "").replace("\n", "")
                test_sentences.append(sentence)
                test_intent_labels.append(label)
            # print(sentence + " | " + label)
    print("Number of sentences: ", len(test_sentences))
    print(test_sentences)
        # print(sentence + " | " + label)
    beam_size_list = [4]
    relative_threshold_list = [1000]
    nprocs = mp.cpu_count()-2
    print(f"Number of CPU cores: {nprocs}")
    pool = mp.Pool(processes=nprocs)
    result = pool.starmap(get_single_result, [(vocab_document,
                                               grammars_idf_from_dict_2,
                                               test_sentences,
                                               test_intent_labels,
                                               grammar_pieces_dict,
                                               vocab,
                                               states,
                                               A,
                                               B,
                                               lowest_grammar_idf,
                                               intent_taglist_dict,
                                               pos_dict_for_grammar_terminal,
                                               relative_matching_threshold,
                                               beam_size) for relative_matching_threshold in relative_threshold_list for beam_size in beam_size_list])
    return "Finish Running."
